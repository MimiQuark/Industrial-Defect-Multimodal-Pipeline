# 管线 B 过滤前后对比报告（20260902_203203）

> 注：本次运行早于启发式打分实现变更（过滤率 83.3%）。当前实现的结果见 `pipeline_b_report_20260902_203343.md` 与 `pipeline_b_report_20260902_225812.md`（27.8%）。本报告仅作历史记录保留。

## 运行状态

- 打分模式：**启发式降级打分（无 API key，基于亮度/对比度/边缘密度分布差异，非 VLM 输出）**
- checkpoint 状态：❌ 不可用：checkpoint 目录不存在: checkpoints；下游（过滤后数据回灌训练的）对比实验为后续工作
- 输入目录：`datapipe/demo_generated`
- 过滤阈值：三维均分 < 3.0 剔除
- 参考分布：27 张正常样本（亮度 μ=0.649，对比度 μ=0.044，边缘密度 μ=0.0266）
- 配置哈希：`d168a1bdfeb78a97`

## 总体对比

| 指标 | 过滤前 | 过滤后 |
| --- | --- | --- |
| 样本数 | 18 | 3 |
| 自然度均分 | 2.06 | 2.67 |
| 融合度均分 | 1.67 | 4.33 |
| 可用性均分 | 1.50 | 3.67 |

**过滤率：15/18 = 83.3%**

## 各维度分数段分布（过滤前）

| 维度 | 1分 | 2分 | 3分 | 4分 | 5分 |
| --- | --- | --- | --- | --- | --- |
| naturalness | 11 | 1 | 2 | 2 | 2 |
| fusion | 14 | 0 | 2 | 0 | 2 |
| usability | 14 | 1 | 2 | 0 | 1 |

## 保留样例（路径 | 均分）

- `datapipe/demo_generated/bottle/gen_ok_003.png` | 4.67
- `datapipe/demo_generated/cable/gen_ok_003.png` | 3.0
- `datapipe/demo_generated/capsule/gen_ok_001.png` | 3.0

## 剔除样例（路径 | 均分 | 理由）

- `datapipe/demo_generated/bottle/gen_bad_flat.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=9.92, z_fusion=310.04
- `datapipe/demo_generated/bottle/gen_bad_overbright.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=9.41, z_fusion=245.34
- `datapipe/demo_generated/bottle/gen_ok_000.png` | 1.33 | 启发式降级打分（非 VLM）：z_natural=4.47, z_fusion=44.65
- `datapipe/demo_generated/bottle/gen_ok_001.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=8.19, z_fusion=5.44
- `datapipe/demo_generated/bottle/gen_ok_002.png` | 2.67 | 启发式降级打分（非 VLM）：z_natural=1.25, z_fusion=7.42
- `datapipe/demo_generated/cable/gen_bad_flat.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=10.27, z_fusion=310.07
- `datapipe/demo_generated/cable/gen_bad_overbright.png` | 2.33 | 启发式降级打分（非 VLM）：z_natural=1.49, z_fusion=9.33
- `datapipe/demo_generated/cable/gen_ok_000.png` | 1.67 | 启发式降级打分（非 VLM）：z_natural=3.23, z_fusion=12.83
- `datapipe/demo_generated/cable/gen_ok_001.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=5.03, z_fusion=7.61
- `datapipe/demo_generated/cable/gen_ok_002.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=5.76, z_fusion=26.20
- `datapipe/demo_generated/capsule/gen_bad_flat.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=10.05, z_fusion=310.22
- `datapipe/demo_generated/capsule/gen_bad_overbright.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=11.56, z_fusion=296.40
- `datapipe/demo_generated/capsule/gen_ok_000.png` | 2.0 | 启发式降级打分（非 VLM）：z_natural=2.83, z_fusion=15.62
- `datapipe/demo_generated/capsule/gen_ok_002.png` | 1.0 | 启发式降级打分（非 VLM）：z_natural=9.11, z_fusion=23.55
- `datapipe/demo_generated/capsule/gen_ok_003.png` | 1.67 | 启发式降级打分（非 VLM）：z_natural=7.03, z_fusion=3.31
