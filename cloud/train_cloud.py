# -*- coding: utf-8 -*-
"""
云端训练启动脚本（上传即跑）

用法示例：
    # 单品类试水（强烈推荐先跑这个）
    python cloud/train_cloud.py --category bottle --epochs 30

    # 全品类训练
    python cloud/train_cloud.py --epochs 100

    # 断点续训（实例中断后重开）
    python cloud/train_cloud.py --category bottle --resume checkpoints/checkpoint_epoch_20.pth

说明：
- 本脚本是 model/training/trainer.py 的薄封装：现有工厂函数
  create_trainer_from_config 不接受 category 参数，本脚本复刻其装配逻辑，
  仅在构建 MVTecADDataset 时多传一个 category（数据集类本身支持该参数），
  不改动任何现有训练代码。
- --epochs 通过 trainer.train(n_epochs=...) 覆盖配置中的 training.epochs。
- checkpoint 保存在配置 training.checkpoint_dir（默认 ./checkpoints/）：
  checkpoint_epoch_N.pth / best_model.pth（FID 最优）/ final_model.pth。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focus-StyleGAN 云端训练启动脚本")
    parser.add_argument("--category", default=None,
                        help="MVTec AD 品类名（如 bottle）。不填则训练全部品类。"
                             "强烈建议先单品类试水验证链路，再全品类训练。")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "core" / "integrated_config.yaml"),
                        help="训练配置文件路径")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置中的 training.epochs（试水建议 20-30）")
    parser.add_argument("--resume", default=None,
                        help="从指定 checkpoint 续训，如 checkpoints/checkpoint_epoch_20.pth")
    return parser.parse_args()


def check_cuda():
    """检测 CUDA；无 GPU 时要求用户显式确认（云端按小时计费，CPU 训练是浪费钱）。"""
    import torch
    if not torch.cuda.is_available():
        print("⚠️  未检测到 CUDA GPU！云端实例按小时计费，用 CPU 训练 GAN 基本不可行。")
        ans = input("确认仍要按 CPU 继续？(输入 yes 继续): ").strip().lower()
        if ans != "yes":
            print("已取消。请检查实例镜像是否为 GPU 版、nvidia-smi 是否有输出。")
            sys.exit(1)
        return torch.device("cpu")
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"✅ 检测到 GPU: {name}（显存 {mem_gb:.1f} GB）")
    if mem_gb < 12:
        print(f"⚠️  显存 {mem_gb:.1f} GB < 12 GB：建议把 core/integrated_config.yaml "
              f"中 data.batch_size 从 32 降到 16，避免 OOM。")
    return torch.device("cuda")


def check_dataset(dataset_path: Path, category: str | None) -> None:
    """检查数据集目录结构符合 MVTec AD 布局，不符合时给出明确中文提示并退出。"""
    if not dataset_path.is_dir():
        print(f"❌ 数据集目录不存在: {dataset_path}")
        print("   请先下载 MVTec AD 并解压到 datasets/mvtec_anomaly_detection/")
        print("   下载地址: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        sys.exit(1)
    cats = sorted(p for p in dataset_path.iterdir() if p.is_dir())
    if not cats:
        print(f"❌ 数据集目录下没有任何品类子目录: {dataset_path}")
        sys.exit(1)
    if category is not None:
        if category not in [c.name for c in cats]:
            print(f"❌ 品类不存在: {category}，可用品类: {[c.name for c in cats]}")
            sys.exit(1)
        cats = [dataset_path / category]
    for cat in cats:
        good_train = cat / "train" / "good"
        if not good_train.is_dir() or not any(good_train.glob("*.png")):
            print(f"❌ 品类 {cat.name} 缺少训练集正常样本: {good_train} 不存在或无 PNG")
            sys.exit(1)
    total = sum(len(list((c / "train" / "good").glob("*.png"))) for c in cats)
    print(f"✅ 数据集就绪: {len(cats)} 个品类，训练集正常样本共 {total} 张")


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(PROJECT_ROOT / "train_cloud.log",
                                      encoding="utf-8")],
    )
    logger = logging.getLogger("train_cloud")

    import torch
    from model.utils.config import Config
    from model.utils.seed import set_seed
    from model.data.loader import MVTecADDataset
    from model.models.focus_stylegan import FocusStyleGAN
    from model.training.trainer import FocusStyleGANTrainer

    config = Config(args.config)
    dataset_path = Path(config.data.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    device = check_cuda()
    check_dataset(dataset_path, args.category)

    # 与 create_trainer_from_config 相同的装配流程，仅多传 category
    seed = config.training.get("seed", 42)
    set_seed(seed)
    dataset = MVTecADDataset(
        root_path=dataset_path,
        image_size=config.data.image_size,
        augment=config.data.augmentations.use,
        category=args.category,   # 数据集类原生支持；None 表示全品类
    )
    model = FocusStyleGAN(config=config.model).to(device)
    trainer = FocusStyleGANTrainer(model=model, dataset=dataset, config=config,
                                   device=device, logger=logger)

    n_epochs = args.epochs or config.training.epochs
    logger.info("开始训练: category=%s, epochs=%d, resume=%s",
                args.category or "全部", n_epochs, args.resume or "否")
    trainer.train(n_epochs=n_epochs, resume_from=args.resume)

    print("\n" + "=" * 60)
    print("训练完成")
    print(f"  checkpoint 目录: {trainer.checkpoint_dir}")
    print(f"  最佳 FID: {trainer.best_fid:.2f}"
          if trainer.best_fid != float("inf") else "  最佳 FID: 未计算（epoch 数不足 fid_interval）")
    print("  关键文件: best_model.pth（FID 最优）/ final_model.pth（最终轮）")
    print("  下一步: 将 checkpoints/ 拉回本地项目根目录，")
    print("          然后运行 python -m datapipe.pipeline_b 做合成数据质量过滤")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
