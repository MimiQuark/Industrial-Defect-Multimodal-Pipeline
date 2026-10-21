# -*- coding: utf-8 -*-
"""
管线 A · 程序化初始标注

扫描 MVTec AD 目录结构：
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/*_mask.png

从 ground truth mask 程序化提取结构化初始标注：
- 缺陷类型（来自目录名，如 scratch / stain / good）
- 位置（mask 质心 → 3×3 九宫格方位描述，如“右上区域”）
- 面积占比（mask 前景像素数 / 图像总像素数）
- bbox（mask 前景像素的外接矩形，注意：这是从 mask 程序化提取的，
  并非 VLM 检测输出；本管线不使用 VLM 做任何像素级定位）

正常样本（test/good）生成“无缺陷”标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# 缺陷类型英文目录名 → 中文描述
#
# 覆盖 MVTec AD 15 个品类的全部测试目录名。漏掉某个目录名时 VLM 改写会
# 回退成英文目录名，而 quality.validate_record 要求 response 中出现中文
# 缺陷名，导致该样本被静默判为不一致——因此新增目录名时必须同步本表。
DEFECT_TYPE_ZH = {
    "good": "无缺陷",
    # bottle
    "broken_large": "大面积破损",
    "broken_small": "小面积破损",
    # cable
    "bent_wire": "线材弯曲",
    "cable_swap": "线缆错位",
    "combined": "复合缺陷",
    "cut_inner_insulation": "内绝缘层割破",
    "cut_outer_insulation": "外绝缘层割破",
    "missing_cable": "线缆缺失",
    "missing_wire": "线材缺失",
    "poke_insulation": "绝缘层戳破",
    # capsule
    "crack": "裂纹",
    "faulty_imprint": "印刷瑕疵",
    "poke": "戳痕",
    "scratch": "划痕",
    "squeeze": "挤压变形",
    "squeezed": "挤压变形",
    # carpet / grid / leather / wood / zipper
    "color": "颜色异常",
    "cut": "切口",
    "hole": "孔洞",
    "metal_contamination": "金属污染",
    "thread": "线头",
    "bent": "弯曲变形",
    "broken": "破损",
    "glue": "溢胶",
    "fold": "褶皱",
    # hazelnut / pill
    "print": "印刷缺陷",
    "contamination": "污染",
    "pill_type": "药片类型错误",
    # metal_nut
    "flip": "翻转错误",
    # screw
    "manipulated_front": "前端篡改",
    "scratch_head": "头部划痕",
    "scratch_neck": "颈部划痕",
    "thread_side": "侧边线头",
    "thread_top": "顶部线头",
    # tile
    "cracked": "开裂",
    "glue_strip": "胶条缺陷",
    "gray_stroke": "灰色划痕",
    "oil": "油污",
    "rough": "表面粗糙",
    # toothbrush
    "defective": "缺陷",
    # transistor
    "bent_lead": "引脚弯曲",
    "cut_lead": "引脚割断",
    "damaged_case": "外壳损坏",
    "misplaced": "位置偏移",
    # wood
    "liquid": "液体污染",
    # zipper
    "broken_teeth": "断齿",
    "fabric_border": "布边缺陷",
    "fabric_interior": "布面内部缺陷",
    "split_teeth": "齿裂",
    "stain": "污点",  # 合成 demo 使用
}

# MVTec AD 官方 15 个品类的全部测试目录名（不含合成 demo 的 stain）。
MVTEC_DEFECT_TYPES = frozenset({
    "good",
    "broken_large", "broken_small",
    "bent_wire", "cable_swap", "combined", "cut_inner_insulation",
    "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation",
    "crack", "faulty_imprint", "poke", "scratch", "squeeze",
    "color", "cut", "hole", "metal_contamination", "thread",
    "bent", "broken", "glue", "fold",
    "print", "contamination", "pill_type",
    "flip",
    "manipulated_front", "scratch_head", "scratch_neck", "thread_side",
    "thread_top",
    "cracked", "glue_strip", "gray_stroke", "oil", "rough",
    "defective",
    "bent_lead", "cut_lead", "damaged_case", "misplaced",
    "liquid",
    "broken_teeth", "fabric_border", "fabric_interior", "split_teeth",
})


@dataclass
class Annotation:
    """单张图像的结构化初始标注"""

    image: str                      # 图像路径（相对数据集根目录）
    category: str                   # 品类，如 bottle
    defect_type: str                # 缺陷类型目录名，good 表示正常
    defect_type_zh: str             # 缺陷类型中文描述
    has_defect: bool                # 是否存在缺陷
    mask: Optional[str] = None      # mask 路径（相对数据集根目录），good 无 mask
    area_ratio: float = 0.0         # 缺陷面积占比 [0, 1]
    bbox: Optional[list] = None     # [x_min, y_min, x_max, y_max]，像素坐标
    centroid: Optional[list] = None # [cx, cy] 质心归一化坐标 [0, 1]
    position_zh: str = ""           # 九宫格方位中文描述，如“右上区域”
    image_size: Optional[list] = field(default=None)  # [W, H]

    def to_dict(self) -> dict:
        return asdict(self)


def position_zh_from_centroid(cx: float, cy: float) -> str:
    """质心归一化坐标 → 3×3 九宫格方位中文描述。

    cx, cy ∈ [0, 1]。列：左 / 中 / 右；行：上 / 中 / 下。
    """
    col = "左" if cx < 1 / 3 else ("右" if cx >= 2 / 3 else "中")
    row = "上" if cy < 1 / 3 else ("下" if cy >= 2 / 3 else "中")
    if col == "中" and row == "中":
        return "中央区域"
    if col == "中":
        return f"{row}部区域"
    if row == "中":
        return f"{col}侧区域"
    return f"{col}{row}区域"


def extract_annotation_from_mask(mask_path: Path,
                                expected_size: tuple[int, int] | None = None
                                ) -> dict:
    """从单张 mask 提取面积占比、bbox、质心、方位。

    mask 约定：前景（缺陷区域）为任意非零像素。
    返回 dict：area_ratio / bbox / centroid / position_zh。

    expected_size 为对应原图的 (W, H)。尺寸不一致或 mask 无前景时抛
    ValueError——缺陷样本的空 mask 属于数据错误，不能当成“有缺陷但
    area_ratio=0、位置=中央区域”继续产出标注。
    """
    with Image.open(mask_path) as mask_img:
        arr = np.asarray(mask_img.convert("L"))
    h, w = arr.shape
    if expected_size is not None and (w, h) != tuple(expected_size):
        raise ValueError(
            f"mask 尺寸 {w}x{h} 与图像尺寸 {expected_size[0]}x{expected_size[1]} "
            f"不一致: {mask_path}")
    fg = arr > 0
    total = h * w
    ys, xs = np.nonzero(fg)
    if len(xs) == 0:
        raise ValueError(f"缺陷 mask 中没有前景像素: {mask_path}")
    area_ratio = float(len(xs)) / float(total)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h
    return {
        "area_ratio": area_ratio,
        "bbox": bbox,
        "centroid": [round(cx, 4), round(cy, 4)],
        "position_zh": position_zh_from_centroid(cx, cy),
    }


def scan_mvtec_dataset(root: Path) -> list[Annotation]:
    """扫描 MVTec AD 目录结构，返回全部测试集图像的结构化初始标注。

    仅扫描 <category>/test/ 下样本；train/good 不参与标注。
    缺陷样本按文件名匹配 ground_truth/<defect_type>/<name>_mask.png；
    找不到 mask 时抛出异常（宁可报错也不产出无依据的标注）。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    annotations: list[Annotation] = []
    categories = sorted(p for p in root.iterdir() if p.is_dir())
    if not categories:
        raise FileNotFoundError(f"数据集根目录下没有任何 category 目录: {root}")

    for cat_dir in categories:
        test_dir = cat_dir / "test"
        if not test_dir.is_dir():
            continue
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            if defect_type not in DEFECT_TYPE_ZH:
                raise ValueError(
                    f"未收录的缺陷目录名 {defect_type!r}（{defect_dir}）。"
                    f"请先补充 DEFECT_TYPE_ZH，否则 VLM 一致性校验会静默丢弃样本。")
            defect_type_zh = DEFECT_TYPE_ZH[defect_type]
            for img_path in sorted(defect_dir.glob("*.png")):
                rel_img = str(img_path.relative_to(root)).replace("\\", "/")
                w, h = Image.open(img_path).size
                if defect_type == "good":
                    ann = Annotation(
                        image=rel_img, category=cat_dir.name,
                        defect_type="good", defect_type_zh="无缺陷",
                        has_defect=False, position_zh="",
                        image_size=[w, h],
                    )
                else:
                    mask_path = (cat_dir / "ground_truth" / defect_type
                                 / f"{img_path.stem}_mask.png")
                    if not mask_path.is_file():
                        raise FileNotFoundError(
                            f"找不到与 {rel_img} 对应的 mask: {mask_path}")
                    info = extract_annotation_from_mask(mask_path, (w, h))
                    ann = Annotation(
                        image=rel_img, category=cat_dir.name,
                        defect_type=defect_type, defect_type_zh=defect_type_zh,
                        has_defect=True,
                        mask=str(mask_path.relative_to(root)).replace("\\", "/"),
                        area_ratio=round(info["area_ratio"], 6),
                        bbox=info["bbox"], centroid=info["centroid"],
                        position_zh=info["position_zh"],
                        image_size=[w, h],
                    )
                annotations.append(ann)
    return annotations
