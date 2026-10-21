# -*- coding: utf-8 -*-
"""
管线 A · 质量控制：感知哈希去重 + JSONL 格式校验 + 版本记录

- dHash 感知哈希：PIL + numpy 自实现（环境无 imagehash，按要求不 pip install）
- 去重：两图 dHash 汉明距离 <= 阈值视为重复，保留先出现者
- 校验：字段齐全、图像文件存在、instruction/response 非空、
  response 与结构化标注一致性（缺陷类型字符串出现）
- 版本记录：每次产出写入 versions/v<N>/（dataset.jsonl + manifest.json）
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .annotation import DEFECT_TYPE_ZH


# ---------------------------------------------------------------------------
# dHash 感知哈希（自实现）
# ---------------------------------------------------------------------------

def dhash(image_path: Path, hash_size: int = 8) -> np.ndarray:
    """计算图像 dHash，返回 shape=(hash_size*hash_size,) 的 bool 数组。

    步骤：灰度化 → 缩放到 (hash_size+1, hash_size) → 相邻像素差分。
    """
    img = Image.open(image_path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """两个等长哈希向量的汉明距离。"""
    return int(np.count_nonzero(h1 != h2))


def dedup_by_phash(records: list[dict], dataset_root: Path,
                   hash_size: int = 8, hamming_threshold: int = 5
                   ) -> tuple[list[dict], list[dict]]:
    """基于 dHash 的近似去重。

    逐条扫描，与已保留样本的最小汉明距离 <= 阈值则判为重复并剔除。
    返回 (kept, dropped)；dropped 中记录 dup_of（被判定与之重复的图像）。
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    kept_hashes: list[tuple[np.ndarray, str]] = []
    for rec in records:
        h = dhash(Path(dataset_root) / rec["image"], hash_size)
        dup_of = None
        for kh, kimg in kept_hashes:
            if hamming_distance(h, kh) <= hamming_threshold:
                dup_of = kimg
                break
        if dup_of is None:
            kept.append(rec)
            kept_hashes.append((h, rec["image"]))
        else:
            dropped.append({**rec, "dup_of": dup_of})
    return kept, dropped


# ---------------------------------------------------------------------------
# JSONL 格式校验
# ---------------------------------------------------------------------------

def validate_record(rec: dict, dataset_root: Path,
                    required_fields: list[str] | None = None,
                    consistency_check: bool = True) -> list[str]:
    """校验单条记录，返回错误信息列表（空列表表示通过）。

    校验项：
    1. 必填字段齐全且为字符串；
    2. 图像文件存在；
    3. instruction / response 非空；
    4. 一致性：response 应包含标注中的缺陷类型中文名（含缺陷时），
       正常样本的 response 应表明无缺陷。
    """
    required_fields = required_fields or ["image", "instruction", "response"]
    errors: list[str] = []
    for f in required_fields:
        if f not in rec or not isinstance(rec[f], str):
            errors.append(f"缺少字段或类型错误: {f}")
    if errors:
        return errors

    if not (Path(dataset_root) / rec["image"]).is_file():
        errors.append(f"图像文件不存在: {rec['image']}")
    if not rec["instruction"].strip():
        errors.append("instruction 为空")
    if not rec["response"].strip():
        errors.append("response 为空")

    if consistency_check and not errors:
        ann = (rec.get("meta") or {}).get("annotation") or {}
        defect_type = ann.get("defect_type", "")
        if defect_type and defect_type != "good":
            zh = DEFECT_TYPE_ZH.get(defect_type, defect_type)
            if zh not in rec["response"]:
                errors.append(
                    f"一致性校验失败: response 未包含缺陷类型「{zh}」")
        elif defect_type == "good":
            # 关键词集来自对真实 VLM 措辞的观察（2026-09-02，qwen3-vl-plus）：
            # VLM 描述正常样本常用"完好/良品/无明显缺陷"等自然措辞，
            # 不限于"无缺陷/正常/合格"三个词
            good_keywords = ("无缺陷", "正常", "合格", "完好", "良品",
                             "无异常", "未见异常", "未发现异常", "无明显",
                             "无可见", "符合质量")
            if not any(k in rec["response"] for k in good_keywords):
                errors.append("一致性校验失败: 正常样本 response 未表明无缺陷")
    return errors


def validate_records(records: list[dict], dataset_root: Path,
                     required_fields: list[str] | None = None,
                     consistency_check: bool = True
                     ) -> tuple[list[dict], list[dict]]:
    """批量校验。返回 (passed, failed)；failed 中记录 errors 列表。"""
    passed, failed = [], []
    for rec in records:
        errs = validate_record(rec, dataset_root, required_fields,
                               consistency_check)
        if errs:
            failed.append({**rec, "errors": errs})
        else:
            passed.append(rec)
    return passed, failed


# ---------------------------------------------------------------------------
# 版本记录
# ---------------------------------------------------------------------------

def config_hash(config_path: Path) -> str:
    """配置文件内容的 SHA256（前 16 位），用于 manifest 追溯。"""
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16]


def next_version_dir(versions_dir: Path) -> Path:
    """返回下一个版本目录 versions/v<N>/（N 从 1 递增）。"""
    versions_dir = Path(versions_dir)
    versions_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name[1:]) for p in versions_dir.iterdir()
                if p.is_dir() and p.name.startswith("v")
                and p.name[1:].isdigit()]
    n = max(existing, default=0) + 1
    vdir = versions_dir / f"v{n}"
    vdir.mkdir(parents=True)
    return vdir


def write_version(records: list[dict], versions_dir: Path,
                  manifest_extra: dict) -> tuple[Path, dict]:
    """写入一个版本：dataset.jsonl + manifest.json。返回 (版本目录, manifest)。"""
    vdir = next_version_dir(versions_dir)
    jsonl_path = vdir / "dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": vdir.name,
        "output": str(jsonl_path),
        "num_records": len(records),
        **manifest_extra,
    }
    with (vdir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return vdir, manifest


def load_config(config_path: Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
