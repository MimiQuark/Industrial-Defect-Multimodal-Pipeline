# -*- coding: utf-8 -*-
"""
合成验证数据集生成器（demo / synthetic）

⚠️ 数据性质声明：本脚本生成的是【程序化合成的演示数据集】，
仅用于在没有真实 MVTec AD 的情况下跑通管线并产出可复现的验证数字。
它不是、也不能被表述为真实 MVTec AD 数据。

目录结构模拟 MVTec AD：
    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/*_mask.png

缺陷形状为几何图形（划痕=折线、污点=椭圆斑块），
mask 与缺陷区域严格一致（同一绘制指令同时落在图像与 mask 上）。
固定随机种子，输出完全可复现。

另外生成 datapipe/demo_generated/：模拟 GAN 产出的候选图像
（大部分分布正常 + 少量刻意劣化样本），供管线 B 演示过滤。

用法：
    python -m datapipe.make_demo_dataset [--root datasets/demo_mvtec_synthetic]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SIZE = 256
# 品类 → 背景基色（模拟不同工业材质）
CATEGORY_COLORS = {
    "bottle": (168, 178, 188),
    "cable": (150, 130, 110),
    "capsule": (190, 190, 170),
}


def make_texture(rng: np.random.RandomState, base_color: tuple[int, int, int],
                 size: int = IMAGE_SIZE) -> Image.Image:
    """程序化纹理背景：基色 + 低频起伏 + 细颗粒噪声。"""
    low = rng.rand(8, 8).astype(np.float32)
    low = np.asarray(
        Image.fromarray((low * 255).astype(np.uint8)).resize(
            (size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    low = (low - low.mean()) * 40  # 低频明暗起伏 ±20
    fine = rng.normal(0, 6, (size, size))  # 细颗粒噪声
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = base_color[c] + low + fine
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def draw_scratch(rng: np.random.RandomState, size: int = IMAGE_SIZE):
    """划痕：随机折线。返回施加到图像与 mask 上的绘制函数参数。"""
    n = int(rng.randint(3, 6))
    pts = [(int(rng.randint(0, size)), int(rng.randint(0, size))) for _ in range(n)]
    width = int(rng.randint(2, 5))
    return pts, width


def draw_stain(rng: np.random.RandomState, size: int = IMAGE_SIZE):
    """污点：随机椭圆斑块。"""
    cx, cy = int(rng.randint(size // 4, 3 * size // 4)), \
        int(rng.randint(size // 4, 3 * size // 4))
    rx, ry = int(rng.randint(12, 36)), int(rng.randint(12, 36))
    return (cx - rx, cy - ry, cx + rx, cy + ry)


def apply_defect(img: Image.Image, defect_type: str,
                 rng: np.random.RandomState) -> tuple[Image.Image, Image.Image]:
    """在图像上绘制缺陷并同步生成严格一致的 mask（前景=255）。"""
    img = img.copy()
    mask = Image.new("L", img.size, 0)
    d_img, d_mask = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    if defect_type == "scratch":
        pts, width = draw_scratch(rng, img.size[0])
        dark = tuple(int(c * 0.45) for c in img.getpixel(pts[0]))
        d_img.line(pts, fill=dark, width=width, joint="curve")
        d_mask.line(pts, fill=255, width=width, joint="curve")
    elif defect_type == "stain":
        box = draw_stain(rng, img.size[0])
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        dark = tuple(int(c * 0.55) for c in img.getpixel((cx, cy)))
        d_img.ellipse(box, fill=dark)
        d_mask.ellipse(box, fill=255)
    else:
        raise ValueError(f"未知缺陷类型: {defect_type}")
    return img, mask


def build_demo_dataset(root: Path, seed: int = 42,
                       categories: list[str] | None = None,
                       n_train_good: int = 3, n_test_good: int = 5,
                       n_per_defect: int = 4,
                       defect_types: tuple[str, ...] = ("scratch", "stain"),
                       with_duplicate: bool = True) -> dict:
    """生成演示数据集，返回统计信息 dict。重复运行会先清空目标目录。"""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    categories = categories or list(CATEGORY_COLORS)
    rng = np.random.RandomState(seed)
    stats = {"categories": {}, "total_images": 0, "total_masks": 0}

    for cat in categories:
        color = CATEGORY_COLORS.get(cat, (160, 160, 160))
        counts = {"train_good": 0, "test_good": 0}
        # train/good 与 test/good
        for split, n in (("train", n_train_good), ("test", n_test_good)):
            out = root / cat / split / "good"
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_texture(rng, color).save(out / f"{i:03d}.png")
                counts[f"{split}_good"] += 1
        # 一张精确重复图（验证 dHash 去重）
        if with_duplicate and n_test_good > 0:
            src = root / cat / "test" / "good" / "000.png"
            shutil.copyfile(src, src.with_name("999_duplicate.png"))
            counts["test_good"] += 1
        # 缺陷样本 + mask
        for dt in defect_types:
            img_dir = root / cat / "test" / dt
            mask_dir = root / cat / "ground_truth" / dt
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_defect):
                base = make_texture(rng, color)
                defected, mask = apply_defect(base, dt, rng)
                defected.save(img_dir / f"{i:03d}.png")
                mask.save(mask_dir / f"{i:03d}_mask.png")
                counts[dt] = counts.get(dt, 0) + 1
                stats["total_masks"] += 1
        stats["categories"][cat] = counts
        stats["total_images"] += sum(counts.values())
    stats["seed"] = seed
    stats["root"] = str(root)
    return stats


def build_fake_generated(out_dir: Path, seed: int = 123,
                         categories: list[str] | None = None,
                         n_ok: int = 4, n_bad: int = 2) -> dict:
    """生成“模拟 GAN 产出”候选图像，供管线 B 演示。

    每个品类 n_ok 张分布正常的伪合成图 + n_bad 张刻意劣化图
    （第 1 张亮度过曝、第 2 张对比度塌陷），预期被启发式过滤剔除。
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    categories = categories or list(CATEGORY_COLORS)
    rng = np.random.RandomState(seed)
    total = 0
    for cat in categories:
        color = CATEGORY_COLORS.get(cat, (160, 160, 160))
        d = out_dir / cat
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_ok):
            base = make_texture(rng, color)
            defected, _ = apply_defect(base, "scratch" if i % 2 == 0 else "stain", rng)
            defected.save(d / f"gen_ok_{i:03d}.png")
            total += 1
        # 劣化样本 1：整体过曝
        over = make_texture(rng, color).point(lambda v: min(255, v + 90))
        over.save(d / "gen_bad_overbright.png")
        # 劣化样本 2：强模糊导致对比度/边缘塌陷
        flat = make_texture(rng, color).filter(ImageFilter.GaussianBlur(12))
        flat = flat.point(lambda v: int(128 + (v - 128) * 0.15))
        flat.save(d / "gen_bad_flat.png")
        total += n_bad
    return {"root": str(out_dir), "total": total, "seed": seed}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成演示数据集（非真实 MVTec AD）")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "datasets" / "demo_mvtec_synthetic"))
    parser.add_argument("--gen-root", default=str(PROJECT_ROOT / "datapipe" / "demo_generated"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = build_demo_dataset(Path(args.root), seed=args.seed)
    print(f"✅ 演示数据集（synthetic，非真实 MVTec AD）→ {stats['root']}")
    print(f"   图像 {stats['total_images']} 张，mask {stats['total_masks']} 张，种子 {stats['seed']}")
    for cat, c in stats["categories"].items():
        print(f"   {cat}: {c}")

    gen = build_fake_generated(Path(args.gen_root))
    print(f"✅ 模拟 GAN 产出候选图 → {gen['root']}（{gen['total']} 张）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
