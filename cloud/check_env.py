# -*- coding: utf-8 -*-
"""
云端环境自检脚本：训练前跑一遍，确认环境就绪。

用法：
    python cloud/check_env.py

检查项：Python 版本 / torch 及 CUDA / GPU 型号显存 / 关键依赖导入 /
数据集目录结构 / 磁盘剩余空间。全部打勾/叉输出，最后一行总结。
无 GPU 时仅警告不崩溃（本地预检也可用）。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (显示名, 导入名)——与 requirements.txt 对齐；torch/torchvision 单独检查
DEPENDENCIES = [
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("tqdm", "tqdm"),
    ("PyYAML", "yaml"),
    ("lpips", "lpips"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("optuna", "optuna"),
    ("matplotlib", "matplotlib"),
    ("albumentations", "albumentations"),
]

DATASET_REL = "datasets/mvtec_anomaly_detection"


def ok(msg: str) -> bool:
    print(f"  ✅ {msg}")
    return True


def fail(msg: str) -> bool:
    print(f"  ❌ {msg}")
    return False


def main() -> int:
    results: list[bool] = []

    print("[1/6] Python 版本")
    v = sys.version_info
    results.append(ok(f"Python {v.major}.{v.minor}.{v.micro}")
                   if v >= (3, 9) else fail(f"Python {v.major}.{v.minor} 过旧，建议 >= 3.9"))

    print("[2/6] PyTorch 与 CUDA")
    try:
        import torch
        ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            results.append(ok(f"CUDA 可用: {name}（{mem:.1f} GB）"))
            if mem < 12:
                print("     ⚠️  显存 <12GB，建议 batch_size 降到 16")
        else:
            results.append(fail("CUDA 不可用——云实例请选 GPU 版镜像；本地预检可忽略本项"))
    except ImportError:
        results.append(fail("torch 未安装（云平台镜像通常自带；否则见 cloud/README.md）"))

    print("[3/6] 关键依赖导入")
    missing = []
    for display, module in DEPENDENCIES:
        try:
            __import__(module)
        except ImportError:
            missing.append(display)
    results.append(ok(f"全部可导入（{len(DEPENDENCIES)} 项）") if not missing
                   else fail(f"缺失: {missing}，执行 pip install -r cloud/requirements-cloud.txt"))

    print("[4/6] 项目代码可导入")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from model.training.trainer import create_trainer_from_config  # noqa: F401
        results.append(ok("model.training.trainer 导入成功"))
    except Exception as e:
        results.append(fail(f"项目代码导入失败: {type(e).__name__}: {e}"))

    print("[5/6] 数据集目录")
    ds = PROJECT_ROOT / DATASET_REL
    if not ds.is_dir():
        results.append(fail(f"数据集不存在: {ds}（先下载 MVTec AD，见 cloud/README.md 第 3 节）"))
    else:
        cats = [p for p in ds.iterdir() if p.is_dir()]
        n_train = sum(len(list((c / "train" / "good").glob("*.png")))
                      for c in cats if (c / "train" / "good").is_dir())
        results.append(ok(f"{len(cats)} 个品类，训练集正常样本 {n_train} 张") if cats
                       else fail(f"数据集目录为空: {ds}"))

    print("[6/6] 磁盘剩余空间")
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024 ** 3
    # 数据集约 5GB + checkpoint 与 FID 缓存，预留 20GB 比较稳
    results.append(ok(f"剩余 {free_gb:.1f} GB") if free_gb >= 20
                   else fail(f"剩余 {free_gb:.1f} GB < 20 GB，可能不够用（数据集+checkpoint）"))

    n_fail = results.count(False)
    print("=" * 50)
    print(f"总结: {'✅ 环境就绪' if n_fail == 0 else f'❌ 缺 {n_fail} 项，按上面提示逐项解决'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
