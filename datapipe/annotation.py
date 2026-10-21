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
DEFECT_TYPE_ZH = {
    "good": "无缺陷",
    "scratch": "划痕",
    "stain": "污点",
    "broken_teeth": "断齿",
    "bent": "弯曲变形",
    "broken_large": "大面积破损",
    "broken_small": "小面积破损",
    "cable_swap": "线缆错位",
    "combined": "复合缺陷",
    "contamination": "污染",
    "crack": "裂纹",
    "cut": "切口",
    "faulty_imprint": "印刷瑕疵",
    "flip": "翻转错误",
    "glue": "溢胶",
    "glue_strip": "胶条缺陷",
    "gray_stroke": "灰色划痕",
    "hole": "孔洞",
    "misplaced": "位置偏移",
    "missing_cable": "线缆缺失",
    "missing_teeth": "缺齿",
    "poke": "戳痕",
    "poke_insulation": "绝缘层戳破",
    "print": "印刷缺陷",
    "rough": "表面粗糙",
    "scratch_head": "头部划痕",
    "scratch_neck": "颈部划痕",
    "squeezed": "挤压变形",
    "thread": "线头",
    "thread_side": "侧边线头",
    "thread_top": "顶部线头",
    "transistor": "晶体管缺陷",
}


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


def extract_annotation_from_mask(mask_path: Path) -> dict:
    """从单张 mask 提取面积占比、bbox、质心、方位。

    mask 约定：前景（缺陷区域）为任意非零像素。
    返回 dict：area_ratio / bbox / centroid / position_zh。
    """
    arr = np.asarray(Image.open(mask_path).convert("L"))
    fg = arr > 0
    h, w = fg.shape
    total = h * w
    ys, xs = np.nonzero(fg)
    if len(xs) == 0:
        return {"area_ratio": 0.0, "bbox": None, "centroid": None,
                "position_zh": "中央区域"}
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
            defect_type_zh = DEFECT_TYPE_ZH.get(defect_type, defect_type)
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
                    info = extract_annotation_from_mask(mask_path)
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
