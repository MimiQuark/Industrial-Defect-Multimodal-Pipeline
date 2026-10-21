# -*- coding: utf-8 -*-
"""
管线 A · 标注改写：结构化标注 → {instruction, response}

两级实现（按优先级自动选择）：

1. VLM 改写：通过 OpenAI 兼容 chat API（默认 DashScope 兼容模式，
   base_url/model 见 config.yaml）把结构化标注连同图像发给 Qwen-VL，
   让它生成多样化的自然语言描述。
2. 确定性模板降级：未配置 API key（DASHSCOPE_API_KEY / OPENAI_API_KEY
   均未设置）或 enable_vlm=false 时自动启用。多套模板 + 可控随机种子
   保证可复现。

⚠️ 降级说明：模板降级产出的是【程序化模板文本】，不是 VLM 输出。
每条记录会带 meta.rewrite_mode 字段（"vlm" 或 "template_fallback"），
文档与报告中必须如实标注，绝不把模板文本伪装成 VLM 改写结果。
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
from pathlib import Path

from .annotation import Annotation

# ---------------------------------------------------------------------------
# 确定性模板降级（无 API key 时使用，随机种子可控、结果可复现）
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATES = [
    "请描述这张{category}质检图像中是否存在缺陷，若有请说明缺陷类型、位置与严重程度。",
    "检查这张{category}工业图像，给出缺陷分析结论。",
    "你是工业质检员，请分析这张{category}图像的缺陷情况。",
    "请评估这张{category}产品图像是否合格，并说明理由。",
    "观察这张{category}表面图像，报告你发现的任何异常。",
    "针对这张{category}图像做质量判定，缺陷需给出类型与方位。",
]

RESPONSE_DEFECT_TEMPLATES = [
    "该{category}图像存在{defect_zh}缺陷，位于{position}，"
    "缺陷区域约占图像面积的{area_pct}%，{severity_zh}级别，建议标记为不合格品。",
    "检测到{defect_zh}。缺陷集中在图像{position}，面积占比约{area_pct}%，"
    "属于{severity_zh}缺陷，该产品应判为不合格。",
    "质检结论：不合格。图像{position}可见{defect_zh}，"
    "覆盖约{area_pct}%的画面区域，严重程度为{severity_zh}。",
    "发现异常：{position}出现{defect_zh}，占比约{area_pct}%（{area_desc}），"
    "{severity_zh}级别，需剔除该样本。",
]

RESPONSE_GOOD_TEMPLATES = [
    "该{category}图像为正常样本，表面纹理均匀，未发现任何缺陷，判定合格。",
    "质检结论：合格。该{category}图像无缺陷，外观符合质量标准。",
    "未发现异常。这张{category}产品图像属于无缺陷正常样本，可放行。",
    "该{category}样本检测通过：无缺陷，图像中不存在划痕、污点等任何异常，判定合格。",
]


def area_description(area_ratio: float) -> tuple[str, str, str]:
    """面积占比 → (约 X% 的数值串, 粗略量级描述, 严重程度中文)"""
    pct = area_ratio * 100
    area_pct = f"{pct:.2f}"
    if pct < 1:
        return area_pct, "极小的局部区域（<1%）", "轻微"
    if pct < 5:
        return area_pct, "较小范围（1%~5%）", "轻度"
    if pct < 15:
        return area_pct, "中等范围（5%~15%）", "中度"
    return area_pct, "较大范围（>15%）", "重度"


def template_rewrite(ann: Annotation, rng: random.Random) -> dict:
    """确定性模板降级：结构化标注 → {instruction, response}。

    使用传入的 rng（由全局种子派生）挑选模板，保证可复现。
    """
    instruction = rng.choice(INSTRUCTION_TEMPLATES).format(category=ann.category)
    if not ann.has_defect:
        response = rng.choice(RESPONSE_GOOD_TEMPLATES).format(category=ann.category)
    else:
        area_pct, area_desc, severity = area_description(ann.area_ratio)
        response = rng.choice(RESPONSE_DEFECT_TEMPLATES).format(
            category=ann.category,
            defect_zh=ann.defect_type_zh,
            position=ann.position_zh,
            area_pct=area_pct,
            area_desc=area_desc,
            severity=severity,
            severity_zh=severity,
        )
    return {"instruction": instruction, "response": response}


# ---------------------------------------------------------------------------
# VLM 改写（OpenAI 兼容接口）
# ---------------------------------------------------------------------------

_VLM_SYSTEM_PROMPT = (
    "你是工业质检数据标注专家。我会给你一张工业质检图像和它的结构化标注"
    "（缺陷类型/位置/面积占比/bbox，均来自 ground truth mask 的程序化提取）。"
    "请把结构化标注改写成一段自然、多样的中文质检描述。"
    "要求：1) 不编造标注之外的缺陷；2) 只做定性描述，不输出像素级坐标或 bbox；"
    "3) 直接输出描述文本，不要附加解释；"
    "4) 必须原样使用结构化标注里的中文缺陷类型名称（如「划痕」「金属污染」），"
    "不要替换成近义词、英文或自造名称。"
)


def _encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def vlm_rewrite(ann: Annotation, image_abs_path: Path, vlm_cfg: dict,
                api_key: str, rng: random.Random) -> dict:
    """调用 OpenAI 兼容 VLM 接口改写单条标注。

    网络/解析失败时抛异常，由调用方决定回退策略。
    """
    from openai import OpenAI  # 延迟导入，模板降级模式无需该包可用

    client = OpenAI(
        base_url=vlm_cfg["base_url"],
        api_key=api_key,
        timeout=vlm_cfg.get("timeout", 60),
    )
    instruction = rng.choice(INSTRUCTION_TEMPLATES).format(category=ann.category)
    ann_text = json.dumps(ann.to_dict(), ensure_ascii=False)
    user_content = [
        {"type": "text", "text": f"结构化标注：{ann_text}\n请改写为自然语言质检描述。"},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{_encode_image_base64(image_abs_path)}"}},
    ]
    resp = client.chat.completions.create(
        model=vlm_cfg["model"],
        messages=[
            {"role": "system", "content": _VLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.8,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("VLM 返回空内容")
    return {"instruction": instruction, "response": text}


def resolve_api_key(vlm_cfg: dict) -> str | None:
    """按配置读取 API key；两个环境变量均未设置时返回 None（触发降级）。"""
    env_name = vlm_cfg.get("api_key_env", "DASHSCOPE_API_KEY")
    return os.environ.get(env_name) or os.environ.get("OPENAI_API_KEY")


def rewrite_annotations(annotations: list[Annotation], dataset_root: Path,
                        vlm_cfg: dict, seed: int) -> tuple[list[dict], str]:
    """批量改写标注。返回 (records, mode)。

    mode ∈ {"vlm", "vlm_partial_fallback", "template_fallback"}；部分样本
    VLM 调用失败时如实标为 vlm_partial_fallback，绝不整体标成 "vlm"。

    record = {image, instruction, response, meta}，meta 中含结构化标注与改写模式。
    未配置 API key 或 enable_vlm=false 时整体走模板降级；
    VLM 模式下若单条调用失败，该条回退模板并在 meta 中记录 fallback 原因。
    VLM 模式按 vlm_cfg["concurrency"] 并发调用（默认 4）；
    每条记录的随机种子由全局种子+序号派生，与并发调度无关，结果可复现。
    """
    rng = random.Random(seed)
    api_key = resolve_api_key(vlm_cfg) if vlm_cfg.get("enable_vlm", True) else None
    mode = "vlm" if api_key else "template_fallback"
    concurrency = max(1, int(vlm_cfg.get("concurrency", 4)))
    # 种子在派发到线程前串行派生，保证并发下结果仍可复现
    item_seeds = [rng.randrange(2 ** 31) + i for i in range(len(annotations))]

    def process(i: int, ann: Annotation) -> dict:
        item_rng = random.Random(item_seeds[i])
        if mode == "vlm":
            try:
                pair = vlm_rewrite(ann, Path(dataset_root) / ann.image,
                                   vlm_cfg, api_key, item_rng)
                rewrite_mode = "vlm"
                fallback_reason = None
            except Exception as e:  # noqa: BLE001 - 单条失败不应中断整批
                pair = template_rewrite(ann, item_rng)
                rewrite_mode = "template_fallback"
                fallback_reason = f"vlm_call_failed: {type(e).__name__}"
        else:
            pair = template_rewrite(ann, item_rng)
            rewrite_mode = "template_fallback"
            fallback_reason = "no_api_key" if vlm_cfg.get("enable_vlm", True) \
                else "vlm_disabled"

        return {
            "image": ann.image,
            "instruction": pair["instruction"],
            "response": pair["response"],
            "meta": {
                "annotation": ann.to_dict(),
                "rewrite_mode": rewrite_mode,
                **({"fallback_reason": fallback_reason} if fallback_reason else {}),
            },
        }

    records: list[dict] = [None] * len(annotations)  # type: ignore
    if mode == "vlm" and concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(process, i, ann): i
                       for i, ann in enumerate(annotations)}
            for fut in futures:
                records[futures[fut]] = fut.result()
    else:
        for i, ann in enumerate(annotations):
            records[i] = process(i, ann)

    # 单条 VLM 失败会回退模板；此时不能把整次运行标成 "vlm"。
    if mode == "vlm":
        n_vlm = sum(1 for r in records
                    if r["meta"]["rewrite_mode"] == "vlm")
        if n_vlm == 0:
            mode = "template_fallback"
        elif n_vlm < len(records):
            mode = "vlm_partial_fallback"
    return records, mode


def strip_code_fence(text: str) -> str:
    """去除 VLM 输出常见的 ```json 代码围栏，便于 JSON 解析。"""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()
