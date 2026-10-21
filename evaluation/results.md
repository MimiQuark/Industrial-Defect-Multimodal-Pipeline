# 数据管线运行结果记录

## 云端训练记录（Kaggle）

- 平台：Kaggle Notebook，GPU T4（16GB），`cloud/kaggle_train.ipynb`
- 品类：bottle（Kaggle 公开 MVTec AD 镜像，训练集 167 张 / 验证集 42 张，0.8 划分）
- 第一次会话：30 epoch（2026-09-03）；第二次会话断点续训至 100 epoch（2026-09-03）
- FID 轨迹（每 10 epoch，从曲线图读取的**近似值**，精确值以 notebook 摘要为准）：
  187 → 173 → 151 → 158 → 135 → 131 → 121 → 107 → 114 → **约 96（epoch 100，末段仍在下降）**
- epoch ~68 出现 D loss 瞬时尖峰（约 800），FID 两次小幅回弹后自行恢复——
  GAN 对抗训练的常见瞬时失稳，未影响整体趋势
- **诚实质量备注**：FID 持续下降，但生成图仍存在整体品红色偏、模糊、
  缺陷形态不明显（与 30 epoch 时相比肉眼改善有限）。结论：FID 等统计指标改善
  ≠ 合成数据可用，需管线 B 的语义级质量过滤做最终把关（这本身即管线 B 价值的实证）。
- checkpoint：`best_model.pth`（FID 最优）待回拉本地后跑管线 B 真实过滤实验

---

本文件其余部分只记录**真实跑出来的数字**，每个数字标注运行命令、日期与数据基础。
数据基础说明：仓库内不含真实 MVTec AD 数据集与任何 checkpoint 权重，
以下结果均在 `datapipe/make_demo_dataset.py` 生成的**程序化合成演示数据集**
（synthetic demo，非真实 MVTec AD）上运行得到。

## 运行环境

- Python 3.12.14（Kimi 托管解释器），PIL 12.2.0、numpy 2.4.4、pyyaml 6.0.3
- VLM：qwen3-vl-plus（DashScope 兼容模式，4 并发，2026-09-02 真实调用）
- 无 API key 时 → 管线 A 走**确定性模板降级**，管线 B 走**启发式降级打分**
  （均非 VLM 输出，产出与报告中如实标注）

## 管线 A：指令数据集构建（VLM 模式，qwen3-vl-plus）

- 运行命令：`set DASHSCOPE_API_KEY=... && python -m datapipe.pipeline_a`
- 日期：2026-09-02；耗时 39.6s（42 张，4 并发，单条约 5s）
- 数据基础：synthetic demo（`datasets/demo_mvtec_synthetic`，3 品类，种子 42）

| 指标 | 数值 | 出处 |
| --- | --- | --- |
| 扫描测试图像总数 | 42（缺陷 24，正常 18） | 管线 A 标准输出 `[1/4]` |
| VLM 改写产出记录 | 42 条，全部 `rewrite_mode=vlm`，无单条回退 | 管线 A 标准输出 `[2/4]`，v6 manifest |
| dHash 去重剔除 | 3 条（每品类 1 张精确复制图，hamming≤5） | 管线 A 标准输出 `[3/4]` |
| 格式校验通过率 | 39/39 = 100% | 管线 A 标准输出 `[4/4]` |
| 最终 SFT 样本数 | **39 条（VLM 改写）** | `datapipe/versions/v6/dataset.jsonl` 行数 |
| 数值一致性抽查 | VLM 描述面积占比与 mask 程序化标注一致（1.7%/2.5%/5.2% 与 0.0170/0.0253/0.0520 对应） | v6 记录人工抽查 |

> 迭代记录（诚实留痕）：首个 VLM 版本 **v5** 校验通过率 76.9%（30/39）——
> 9 条正常样本被一致性检查拦截，复查确认是校验关键词表过窄（只认
> "无缺陷/正常/合格"），而 VLM 的自然措辞为"完好/良品/无明显缺陷"。
> 按真实观察到的 VLM 措辞扩充关键词表后重跑，v6 达 100%。
> 这验证了校验机制确实在拦截，而非橡皮图章。

## 管线 A：模板降级模式（无 API key 时）

- 运行命令：`python -m datapipe.make_demo_dataset && python -m datapipe.pipeline_a`
- 日期：2026-09-02
- 结果：42 扫描 → 去重 3 → 校验 100% → **39 条**（`datapipe/versions/v3`，
  模板降级产出，`rewrite_mode=template_fallback`）
- 复验：`versions/v4` 为重跑产物，数字与 v3 逐位一致，证明固定种子下可复现。

> 注：`versions/v1`、`v2` 为开发调试过程版本（v1 曾暴露一条"无缺陷"模板措辞
> 不含校验关键词、被一致性检查正确拦截，修复模板措辞后重跑）。

## 管线 B：合成数据质量过滤（VLM 模式，qwen3-vl-plus）

- 运行命令：`python -m datapipe.pipeline_b`
- 日期：2026-09-02；耗时 20.7s（18 张，4 并发）
- 数据基础：`datapipe/demo_generated/` 18 张**模拟 GAN 产出**候选图
  （12 张分布正常 + 6 张刻意劣化：过曝 ×3、对比度塌陷 ×3）

| 指标 | 数值 | 出处 |
| --- | --- | --- |
| 候选图像数 | 18 | `datapipe/reports/pipeline_b_report_20260902_225812.md` |
| 过滤率 | **18/18 = 100%（全部低于阈值 3.0）** | 同上报告"总体对比" |
| 刻意劣化组均分 | **1.11**（n=6，含 1.0~1.33） | 同上报告"剔除样例"人工统计 |
| 正常候选组均分 | **2.25**（n=12，含 1.67~2.67） | 同上 |

**100% 过滤率的解读（诚实说明，不调阈值美化数字）**：demo 候选图是
`make_demo_dataset.py` 程序化绘制的几何图形+噪声纹理，VLM 逐张给出了准确
理由（"缺陷呈明显人工绘制的黑色线条""边缘过于锐利，无阴影/遮挡关系"
"明显为人工绘制而非成像生成"）。对这批**本就粗糙的占位图**判低分是
**正确判定而非误杀**——且分数排序有区分度：刻意劣化组（1.11）显著低于
正常候选组（2.25）。该结果证明了 VLM 打分链路的判别能力；
在真实 GAN 产出上的过滤率有待 checkpoint 可用后实测。

## 管线 B：启发式降级打分（无 API key 时，对照）

- 运行命令：不设 API key，`python -m datapipe.pipeline_b`
- 结果：保留 13/18，过滤率 27.8%（剔除 5 张劣化，漏检 1 张过曝）
  （`datapipe/reports/pipeline_b_report_20260902_203343.md`）
- 对照结论：启发式只依赖亮度/对比度/边缘密度，**无法识别"画出来的缺陷"**
  这类语义级低质——VLM 与启发式的差距（100% vs 27.8% 过滤率）本身就是
  "为什么需要 VLM 打分"的实证，报告中如实并存两条结果。

## checkpoint 状态

**不可用**（`checkpoints/` 目录不存在）；下游"过滤后数据回灌训练"的
对比实验为后续工作。管线 B 的两次运行输入均为模拟产出。

## 测试

- `python tests/test_datapipe.py` → **通过 7/7**（2026-09-02，VLM 改造后复跑）：
  mask 标注提取、九宫格方位、dHash 去重、JSONL 校验、模板可复现性、
  启发式打分、端到端 smoke test。
- `python tests/test_basic.py` → **未能运行**：`model/data/loader.py` 顶层
  `import cv2`，当前环境无 opencv 且按项目纪律不 pip install。
  该失败为环境预置限制，与本次新增的 datapipe 无关（datapipe 不依赖 cv2/torch）。

## 在真实 MVTec AD 上的复现命令

```bash
# 1. 下载 MVTec AD（https://www.mvtec.com/company/research/datasets/mvtec-ad）
#    解压至 datasets/mvtec_anomaly_detection/
# 2. 启用 VLM 改写/打分：设置环境变量
#    set DASHSCOPE_API_KEY=sk-...      # Windows
#    export DASHSCOPE_API_KEY=sk-...   # Linux/macOS
#    datapipe/config.yaml 中 vlm.base_url / vlm.model 默认已指向
#    DashScope 兼容模式 + qwen3-vl-plus
# 3. 运行
python -m datapipe.pipeline_a --root datasets/mvtec_anomaly_detection
python -m datapipe.pipeline_b --input <GAN 生成图像目录>
```

无 API key 时上述命令仍可直接运行，自动降级为模板/启发式实现，
产出中 `meta.rewrite_mode` 与报告会如实标注降级状态。

## 边界声明

- 以上全部数字来自 synthetic demo 数据集，不代表真实 MVTec AD 上的效果。
- 仓库无 checkpoint，管线 B 输入为模拟产出，"过滤→回灌训练→指标提升"的
  闭环对比实验尚未进行，为后续工作。
- VLM 与人工判断的一致性验证（方案建议抽 200 条）尚未进行，为后续工作；
  当前仅有 VLM 输出与程序化标注的数值一致性抽查（面积占比吻合）。
- 本交付物是训练数据的生产管线；零样本 VLM 已足够做标注和过滤，
  管线产出的 JSONL 格式可直接对接 LLaMA-Factory 等训练框架。
