# -*- coding: utf-8 -*-
"""
datapipe：多模态数据构建与合成数据质量过滤管线

- 管线 A（pipeline_a.py）：MVTec AD 图像 + ground truth mask
  → 程序化初始标注（annotation.py）
  → VLM/模板改写为自然语言（rewrite.py）
  → 感知哈希去重 + 格式校验 + 版本记录（quality.py）
  → 产出 {image, instruction, response} SFT 数据集（JSONL）
- 管线 B（pipeline_b.py）：GAN 生成图像 → VLM 打分（自然度/融合度/可用性）
  → 低分过滤 → 过滤前后对比报告

说明：所有 VLM 调用均为 OpenAI 兼容接口（base_url/model 可配置，
默认指向 DashScope 兼容模式）。未配置 API key 时自动降级为
确定性模板/启发式实现，降级行为在输出与报告中明确标注，绝不伪装成 VLM 输出。
"""

__version__ = "0.1.0"
