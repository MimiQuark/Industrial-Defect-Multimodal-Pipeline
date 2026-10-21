# -*- coding: utf-8 -*-
"""
管线 A · CLI 入口：指令数据集构建

    MVTec AD 图像 + ground truth mask
    → 程序化生成初始标注（annotation.py）
    → VLM / 模板降级改写（rewrite.py）
    → 感知哈希去重 + 格式校验 + 版本记录（quality.py）
    → 产出 {image, instruction, response} SFT 数据集（JSONL）

用法：
    python -m datapipe.pipeline_a [--config datapipe/config.yaml] [--root <数据集目录>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datapipe.annotation import scan_mvtec_dataset            # noqa: E402
from datapipe.quality import (config_hash, dedup_by_phash,    # noqa: E402
                              load_config, validate_records, write_version)
from datapipe.rewrite import rewrite_annotations              # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="管线 A：指令数据集构建")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "datapipe" / "config.yaml"),
                        help="配置文件路径")
    parser.add_argument("--root", default=None,
                        help="MVTec AD 数据集根目录（覆盖配置 paths.dataset_root）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(args.root or cfg["paths"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    seed = int(cfg.get("seed", 42))

    # 1. 程序化初始标注
    print(f"[1/4] 扫描数据集: {dataset_root}")
    annotations = scan_mvtec_dataset(dataset_root)
    n_defect = sum(1 for a in annotations if a.has_defect)
    print(f"      共 {len(annotations)} 张测试图像"
          f"（缺陷 {n_defect}，正常 {len(annotations) - n_defect}）")

    # 2. 改写为自然语言 {instruction, response}
    records, mode = rewrite_annotations(annotations, dataset_root,
                                        cfg["vlm"], seed)
    mode_zh = "VLM 改写" if mode == "vlm" else "确定性模板降级（无 API key）"
    print(f"[2/4] 改写模式: {mode_zh}，生成 {len(records)} 条记录")

    # 3. 感知哈希去重
    kept, dropped = dedup_by_phash(
        records, dataset_root,
        hash_size=int(cfg["dedup"]["hash_size"]),
        hamming_threshold=int(cfg["dedup"]["hamming_threshold"]))
    print(f"[3/4] dHash 去重: 剔除 {len(dropped)} 条重复"
          f"（阈值 hamming<={cfg['dedup']['hamming_threshold']}），保留 {len(kept)} 条")
    for d in dropped:
        print(f"      - 重复: {d['image']} ≈ {d['dup_of']}")

    # 4. 格式校验 + 版本记录
    passed, failed = validate_records(
        kept, dataset_root,
        required_fields=cfg["validation"]["required_fields"],
        consistency_check=bool(cfg["validation"]["consistency_check"]))
    pass_rate = len(passed) / len(kept) if kept else 0.0
    print(f"[4/4] 格式校验: 通过 {len(passed)}/{len(kept)}（{pass_rate:.1%}）")
    for f_ in failed:
        print(f"      - 校验失败: {f_['image']}: {f_['errors']}")

    vdir, manifest = write_version(passed, PROJECT_ROOT / cfg["paths"]["versions_dir"], {
        "pipeline": "A",
        "dataset_root": str(dataset_root),
        "data_basis": "synthetic demo" if "demo" in dataset_root.name else "mvtec_ad",
        "rewrite_mode": mode,
        "seed": seed,
        "config_hash": config_hash(args.config),
        "input_stats": {
            "num_images": len(annotations),
            "num_defect": n_defect,
            "num_good": len(annotations) - n_defect,
        },
        "dedup": {"removed": len(dropped), "kept": len(kept),
                  "removed_images": [d["image"] for d in dropped]},
        "validation": {"passed": len(passed), "failed": len(failed),
                       "pass_rate": pass_rate},
    })
    print(f"\n✅ 管线 A 完成 → {vdir}")
    print(f"   dataset.jsonl: {len(passed)} 条；manifest.json 已写入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
