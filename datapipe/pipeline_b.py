# -*- coding: utf-8 -*-
"""
管线 B · CLI 入口：合成数据质量过滤

    GAN 生成图像 → VLM 打分（自然度/融合度/可用性，各 1-5 分，结构化 JSON 输出）
    → 低于阈值过滤 → 过滤前后对比报告（Markdown）

降级说明：未配置 API key 时使用【启发式降级打分】——计算候选图像与参考
（真实正常图像）分布在亮度/对比度/边缘密度上的 z 分数差异并映射到 1-5 分。
启发式分数只是工程代理指标，报告中会明确标注为 heuristic_fallback，
绝不伪装成 VLM 打分。

checkpoint 状态：启动时检测 checkpoints/ 目录是否存在可用权重并如实记录；
当前仓库不含任何 checkpoint 时，报告写明“checkpoint 不可用，
下游（过滤后数据回灌训练的）对比实验为后续工作”。

用法：
    python -m datapipe.pipeline_b [--config datapipe/config.yaml] [--input <生成图目录>]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datapipe.quality import config_hash, load_config   # noqa: E402
from datapipe.rewrite import resolve_api_key, strip_code_fence  # noqa: E402

CKPT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


# ---------------------------------------------------------------------------
# checkpoint 状态检测
# ---------------------------------------------------------------------------

def check_checkpoints(ckpt_dir: Path) -> dict:
    """检测 GAN checkpoint 可用性，返回状态 dict（如实记录，不臆造）。"""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return {"available": False,
                "detail": f"checkpoint 目录不存在: {ckpt_dir}"}
    files = sorted(p.name for p in ckpt_dir.rglob("*")
                   if p.suffix.lower() in CKPT_SUFFIXES)
    if not files:
        return {"available": False,
                "detail": f"checkpoint 目录存在但无权重文件: {ckpt_dir}"}
    return {"available": True, "detail": f"发现 {len(files)} 个权重文件",
            "files": files}


# ---------------------------------------------------------------------------
# 图像统计特征（启发式降级打分用）
# ---------------------------------------------------------------------------

def image_stats(image_path: Path) -> dict:
    """亮度（灰度均值）、对比度（灰度标准差）、边缘密度（梯度幅值均值）。"""
    arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64) / 255.0
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    return {
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
        "edge_density": float((gx + gy) / 2),
    }


def reference_distribution(reference_root: Path) -> dict:
    """以参考目录中 good（正常）图像的特征均值/标准差作为真实分布基准。

    标准差设下界（鲁棒标准差）：当参考集本身方差极小（如合成 demo 中
    所有正常图纹理几乎一致）时，原始 z 分数会虚高，导致正常候选图被
    误杀。下界为工程经验值：亮度 0.05、对比度 0.01、边缘密度 0.005
    （均为 [0,1] 归一化灰度尺度）。
    """
    STD_FLOOR = {"brightness": 0.05, "contrast": 0.01, "edge_density": 0.005}
    reference_root = Path(reference_root)
    feats = []
    for p in sorted(reference_root.rglob("*.png")):
        parts = set(p.parts)
        if "good" in parts:  # 只用正常样本建立参考分布
            feats.append(image_stats(p))
    if not feats:  # 没有 good 目录时退化为使用全部图像
        feats = [image_stats(p) for p in sorted(reference_root.rglob("*.png"))]
    if not feats:
        raise FileNotFoundError(f"参考目录下没有可用图像: {reference_root}")
    keys = ["brightness", "contrast", "edge_density"]
    dist = {}
    for k in keys:
        vals = np.array([f[k] for f in feats])
        dist[k] = {"mean": float(vals.mean()),
                   "std": max(float(vals.std()), STD_FLOOR[k])}
    dist["num_reference"] = len(feats)
    return dist


def heuristic_score(stats: dict, ref: dict, z_full: float = 1.0) -> dict:
    """启发式降级打分（明确标注为 heuristic，非 VLM 输出）。

    各特征与参考分布的 z 分数 → 1-5 分：
    |z| <= z_full 记满分 5，之后每超出一个 z_full 扣 1 分，最低 1 分。
    - 自然度：亮度 + 对比度 的 z 组合
    - 融合度：边缘密度 的 z（缺陷与背景融合异常时边缘统计偏离）
    - 可用性：三维特征的 z 组合
    """
    def z(k):
        return abs(stats[k] - ref[k]["mean"]) / ref[k]["std"]

    def to_score(zv):
        return max(1, 5 - int(max(0.0, zv - z_full) / z_full + 1e-9))

    z_nat = (z("brightness") + z("contrast")) / 2
    z_fus = z("edge_density")
    z_use = (z_nat + z_fus) / 2
    return {
        "naturalness": to_score(z_nat),
        "fusion": to_score(z_fus),
        "usability": to_score(z_use),
        "reason": (f"启发式降级打分（非 VLM）：z_natural={z_nat:.2f}, "
                   f"z_fusion={z_fus:.2f}"),
    }


# ---------------------------------------------------------------------------
# VLM 打分（OpenAI 兼容接口，结构化 JSON 输出解析）
# ---------------------------------------------------------------------------

_VLM_SCORE_PROMPT = (
    "你是工业质检合成图像评审专家。这是一张 GAN 合成的工业缺陷图像。"
    "请从三个维度打分（各 1-5 整数分）："
    "naturalness（自然度：是否像真实拍摄）、"
    "fusion（融合度：缺陷与背景融合是否自然）、"
    "usability（可用性：是否适合用作训练数据）。"
    "只输出 JSON，形如 "
    '{"naturalness": 4, "fusion": 3, "usability": 4, "reason": "简要理由"}。'
)


def vlm_score(image_path: Path, vlm_cfg: dict, api_key: str) -> dict:
    """调用 VLM 对单张合成图打分并解析结构化 JSON 输出。"""
    import base64

    from openai import OpenAI  # 延迟导入

    client = OpenAI(base_url=vlm_cfg["base_url"], api_key=api_key,
                    timeout=vlm_cfg.get("timeout", 60))
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    resp = client.chat.completions.create(
        model=vlm_cfg["model"],
        messages=[
            {"role": "system", "content": _VLM_SCORE_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}"}}]},
        ],
        temperature=0.2,
    )
    text = strip_code_fence(resp.choices[0].message.content or "")
    data = json.loads(text)
    for k in ("naturalness", "fusion", "usability"):
        v = int(data[k])
        if not 1 <= v <= 5:
            raise ValueError(f"分数越界: {k}={v}")
        data[k] = v
    data.setdefault("reason", "")
    return data


# ---------------------------------------------------------------------------
# 过滤与报告
# ---------------------------------------------------------------------------

def score_all(images: list[Path], vlm_cfg: dict, ref: dict,
              z_full: float) -> tuple[list[dict], str]:
    """对全部候选图打分。返回 (scored, mode)，mode ∈ {"vlm", "heuristic_fallback"}。

    VLM 模式按 vlm_cfg["concurrency"] 并发调用（默认 4），输出顺序与输入一致。
    每条结果附带三维均分 mean。
    """
    api_key = resolve_api_key(vlm_cfg) if vlm_cfg.get("enable_vlm", True) else None
    concurrency = max(1, int(vlm_cfg.get("concurrency", 4)))
    scored: list[dict] = [None] * len(images)  # type: ignore

    def process(p: Path) -> dict:
        try:
            s = vlm_score(p, vlm_cfg, api_key)
            s["_mode"] = "vlm"
        except Exception as e:  # noqa: BLE001
            s = heuristic_score(image_stats(p), ref, z_full)
            s["_mode"] = "heuristic_fallback"
            s["reason"] += f"（VLM 调用失败回退: {type(e).__name__}）"
        s["image"] = str(p)
        return s

    if api_key:
        mode = "vlm"
        if concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(process, images))
            for i, s in enumerate(results):
                scored[i] = s
        else:
            for i, p in enumerate(images):
                scored[i] = process(p)
    else:
        mode = "heuristic_fallback"
        for i, p in enumerate(images):
            s = heuristic_score(image_stats(p), ref, z_full)
            s["_mode"] = "heuristic_fallback"
            s["image"] = str(p)
            scored[i] = s
    for s in scored:
        s["mean"] = round((s["naturalness"] + s["fusion"]
                           + s["usability"]) / 3, 2)
    if mode == "vlm":
        n_vlm = sum(1 for s in scored if s["_mode"] == "vlm")
        if n_vlm == 0:
            mode = "heuristic_fallback"
        elif n_vlm < len(scored):
            mode = "vlm_partial_fallback"
    return scored, mode


def write_report(scored: list[dict], mode: str, threshold: float,
                 ckpt_status: dict, input_dir: Path, ref_dist: dict,
                 config_path: str, reports_dir: Path) -> Path:
    """生成过滤前后对比 Markdown 报告。"""
    def mean_score(s):
        return (s["naturalness"] + s["fusion"] + s["usability"]) / 3

    for s in scored:
        s.setdefault("mean", round(mean_score(s), 2))
    kept = [s for s in scored if s["mean"] >= threshold]
    dropped = [s for s in scored if s["mean"] < threshold]
    filter_rate = len(dropped) / len(scored) if scored else 0.0

    dims = ["naturalness", "fusion", "usability"]
    dist_lines = []
    for d in dims:
        counts = {i: sum(1 for s in scored if s[d] == i) for i in range(1, 6)}
        dist_lines.append(
            f"| {d} | " + " | ".join(str(counts[i]) for i in range(1, 6)) + " |")

    mode_zh = {
        "vlm": "VLM 打分（OpenAI 兼容接口，全部样本成功）",
        "vlm_partial_fallback": "VLM 打分（部分样本失败，已回退启发式）",
        "heuristic_fallback": "启发式降级打分（无 API key 或全部回退，基于亮度/对比度/边缘密度分布差异，非 VLM 输出）",
    }.get(mode, mode)
    mode_counts = Counter(s.get("_mode", "unknown") for s in scored)
    mode_detail = (f"VLM {mode_counts.get('vlm', 0)} 张，"
                   f"启发式回退 {mode_counts.get('heuristic_fallback', 0)} 张")
    ckpt_detail = str(ckpt_status["detail"]).replace(
        str(PROJECT_ROOT), "<repo-root>").replace("\\", "/")
    ckpt_line = (f"✅ 可用：{ckpt_detail}" if ckpt_status["available"]
                 else f"❌ 不可用：{ckpt_detail}；"
                      f"下游（过滤后数据回灌训练的）对比实验为后续工作")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return p

    kept_lines = [f"- `{rel(s['image'])}` | {s['mean']}" for s in kept] or ["- （无）"]
    dropped_lines = [f"- `{rel(s['image'])}` | {s['mean']} | {s['reason']}"
                     for s in dropped] or ["- （无）"]
    lines = [
        f"# 管线 B 过滤前后对比报告（{ts}）",
        "",
        "## 运行状态",
        "",
        f"- 打分模式：**{mode_zh}**",
        f"- 打分明细：{mode_detail}",
        f"- checkpoint 状态：{ckpt_line}",
        f"- 输入目录：`{rel(str(input_dir))}`",
        f"- 过滤阈值：三维均分 < {threshold} 剔除",
        f"- 参考分布：{ref_dist['num_reference']} 张正常样本"
        f"（亮度 μ={ref_dist['brightness']['mean']:.3f}，"
        f"对比度 μ={ref_dist['contrast']['mean']:.3f}，"
        f"边缘密度 μ={ref_dist['edge_density']['mean']:.4f}）",
        f"- 配置哈希：`{config_hash(config_path)}`",
        "",
        "## 总体对比",
        "",
        "| 指标 | 过滤前 | 过滤后 |",
        "| --- | --- | --- |",
        f"| 样本数 | {len(scored)} | {len(kept)} |",
        f"| 自然度均分 | {_avg(scored, 'naturalness')} | {_avg(kept, 'naturalness')} |",
        f"| 融合度均分 | {_avg(scored, 'fusion')} | {_avg(kept, 'fusion')} |",
        f"| 可用性均分 | {_avg(scored, 'usability')} | {_avg(kept, 'usability')} |",
        "",
        f"**过滤率：{len(dropped)}/{len(scored)} = {filter_rate:.1%}**",
        "",
        "## 各维度分数段分布（过滤前）",
        "",
        "| 维度 | 1分 | 2分 | 3分 | 4分 | 5分 |",
        "| --- | --- | --- | --- | --- | --- |",
        *dist_lines,
        "",
        "## 保留样例（路径 | 均分）",
        "",
        *kept_lines,
        "",
        "## 剔除样例（路径 | 均分 | 理由）",
        "",
        *dropped_lines,
        "",
    ]
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"pipeline_b_report_{ts}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _avg(items: list[dict], key: str) -> str:
    return f"{np.mean([s[key] for s in items]):.2f}" if items else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="管线 B：合成数据质量过滤")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "datapipe" / "config.yaml"))
    parser.add_argument("--input", default=None,
                        help="GAN 生成图像目录（覆盖配置 paths.generated_root）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen_root = Path(args.input or cfg["paths"]["generated_root"])
    if not gen_root.is_absolute():
        gen_root = PROJECT_ROOT / gen_root
    ref_root = Path(cfg["paths"]["reference_root"])
    if not ref_root.is_absolute():
        ref_root = PROJECT_ROOT / ref_root
    threshold = float(cfg["scoring"]["filter_threshold"])
    z_full = float(cfg["scoring"]["heuristic_z_full_score"])

    ckpt_status = check_checkpoints(PROJECT_ROOT / cfg["paths"]["checkpoints_dir"])
    print(f"[0/3] checkpoint 状态: {ckpt_status['detail']}")

    images = sorted(p for p in gen_root.rglob("*.png"))
    if not images:
        print(f"输入目录无图像: {gen_root}", file=sys.stderr)
        return 1
    print(f"[1/3] 候选合成图像: {len(images)} 张")

    ref = reference_distribution(ref_root)
    print(f"[2/3] 参考分布: {ref['num_reference']} 张正常样本")

    scored, mode = score_all(images, cfg["vlm"], ref, z_full)
    kept = [s for s in scored if s["mean"] >= threshold]
    print(f"[3/3] 打分完成（{mode}）：保留 {len(kept)}/{len(scored)}，"
          f"过滤率 {(len(scored) - len(kept)) / len(scored):.1%}")

    report = write_report(scored, mode, threshold, ckpt_status, gen_root, ref,
                          args.config, PROJECT_ROOT / cfg["paths"]["reports_dir"])
    print(f"\n✅ 管线 B 完成 → 报告: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
