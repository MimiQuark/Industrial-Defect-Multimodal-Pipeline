# 管线 B 过滤前后对比报告（20260902_203255）

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
| 样本数 | 18 | 13 |
| 自然度均分 | 4.56 | 4.92 |
| 融合度均分 | 3.94 | 5.00 |
| 可用性均分 | 4.33 | 5.00 |

**过滤率：5/18 = 27.8%**

## 各维度分数段分布（过滤前）

| 维度 | 1分 | 2分 | 3分 | 4分 | 5分 |
| --- | --- | --- | --- | --- | --- |
| naturalness | 0 | 0 | 2 | 4 | 12 |
| fusion | 4 | 1 | 0 | 0 | 13 |
| usability | 0 | 2 | 3 | 0 | 13 |

## 保留样例（路径 | 均分）

- `datapipe/demo_generated/bottle/gen_ok_000.png` | 5.0
- `datapipe/demo_generated/bottle/gen_ok_001.png` | 5.0
- `datapipe/demo_generated/bottle/gen_ok_002.png` | 5.0
- `datapipe/demo_generated/bottle/gen_ok_003.png` | 5.0
- `datapipe/demo_generated/cable/gen_bad_overbright.png` | 5.0
- `datapipe/demo_generated/cable/gen_ok_000.png` | 5.0
- `datapipe/demo_generated/cable/gen_ok_001.png` | 5.0
- `datapipe/demo_generated/cable/gen_ok_002.png` | 5.0
- `datapipe/demo_generated/cable/gen_ok_003.png` | 5.0
- `datapipe/demo_generated/capsule/gen_ok_000.png` | 5.0
- `datapipe/demo_generated/capsule/gen_ok_001.png` | 5.0
- `datapipe/demo_generated/capsule/gen_ok_002.png` | 4.67
- `datapipe/demo_generated/capsule/gen_ok_003.png` | 5.0

## 剔除样例（路径 | 均分 | 理由）

- `datapipe/demo_generated/bottle/gen_bad_flat.png` | 2.67 | 启发式降级打分（非 VLM）：z_natural=2.61, z_fusion=5.29
- `datapipe/demo_generated/bottle/gen_bad_overbright.png` | 2.67 | 启发式降级打分（非 VLM）：z_natural=3.48, z_fusion=4.19
- `datapipe/demo_generated/cable/gen_bad_flat.png` | 2.33 | 启发式降级打分（非 VLM）：z_natural=2.79, z_fusion=5.29
- `datapipe/demo_generated/capsule/gen_bad_flat.png` | 2.67 | 启发式降级打分（非 VLM）：z_natural=2.61, z_fusion=5.30
- `datapipe/demo_generated/capsule/gen_bad_overbright.png` | 2.0 | 启发式降级打分（非 VLM）：z_natural=3.95, z_fusion=5.06
