# 云端训练操作手册（Kaggle 优先 · AutoDL 备选）

目标：在云 GPU 上训练 Focus-StyleGAN，把 checkpoint 拉回本地，
打通 datapipe 管线 B 的真实 GAN 输入，让"过滤前后回灌对比实验"从
"后续工作"变成真实数字。

**平台选择**：主用 **Kaggle Notebook**（免费 GPU T4/P100，配额约 30 小时/周，
单次会话约 12 小时，以 Kaggle 官方说明为准），训练全程在
`cloud/kaggle_train.ipynb` 中完成，loss/FID 曲线与生成样例全部 inline 输出。
Kaggle 额度耗尽或需要长时间连续训练（如全品类 100 epoch）时，再用
AutoDL 类付费平台（3090/4090 约 1.2-2.5 元/小时，2026 年行情，以官网实时价为准）。

---

# 第一部分：Kaggle 训练（推荐）

## 1. 准备两个输入数据集

1. **项目代码**：把本地项目 `Industrial-Defect-Multimodal-Pipeline` 打包 zip，
   在 Kaggle → Datasets → New Dataset 上传（私有即可）。
2. **MVTec AD**：Kaggle 搜索 `mvtec ad` 有公开镜像可直接 Add Input；
   也可官网注册下载原版（https://www.mvtec.com/company/research/datasets/mvtec-ad ）
   后自行上传。公开镜像非官方发布，notebook 第 3 节会自动校验目录结构，
   校验失败请换官网原版。

## 2. 导入并配置 notebook

1. Kaggle → Code → New Notebook → File → Import Notebook，导入 `cloud/kaggle_train.ipynb`。
2. 右侧 Settings：Accelerator 选 **GPU T4**；Internet 设为 **On**（需手机验证账号）。
3. 右侧 Add Input：挂载上面两个数据集。
4. 修改第 1 节参数区（`CATEGORY` / `EPOCHS` / `BATCH_SIZE`），然后 **Run All**。

## 3. 训练与产出

- notebook 自动完成：环境检查 → 定位代码与数据 → 装 `lpips`/`optuna` →
  生成 Kaggle 专用配置副本（不改动原 config）→ 训练 → loss/FID 曲线 →
  加载 best_model 生成伪异常样例 inline 展示 → 打包 `checkpoints_<品类>.zip`。
- 产出在右侧 Output 面板下载；notebook 末尾会打印一段可直接誊入
  `evaluation/results.md` 的训练摘要。

## 4. 断点续训（Kaggle 单次会话时长有限的标准做法）

1. 本次结束后 **Save Version**；
2. 把 Output 的 checkpoint zip 下载后上传为 Kaggle Dataset（如 `fsgan-ckpt-bottle`）；
3. 下次会话挂载它，参数区设
   `RESUME_CHECKPOINT = '/kaggle/input/fsgan-ckpt-bottle/checkpoint_epoch_30.pth'`，Run All。

## 5. checkpoint 回本地与管线衔接

把 `best_model.pth` 放入本地项目根 `checkpoints/` 目录，然后：

```bash
python core/integrated_augmentor.py --mode single --input <good.png> --variations 50
python -m datapipe.pipeline_b --input <生成图目录>
python -m datapipe.pipeline_a --root datasets/mvtec_anomaly_detection
```

## Kaggle 避坑要点

- **配额规划**：T4 上单品类 30 epoch 约数小时（估算），每周约 30h 配额够
  跑 3-5 个品类的试水或 1-2 个品类的完整训练；别在参数没调对时长训。
- **Internet 必须 On**：首次训练要下载 VGG / InceptionV3 / LPIPS 预训练权重，
  还要 pip 安装 `lpips`/`optuna`。
- **Save Version 才保留 Output**：直接关页面不保存版本，工作区文件会丢。
- 显存 16GB 的 T4/P100 用默认 `BATCH_SIZE=16`；若换 P100 仍 OOM 就降到 8。

---

# 第二部分：AutoDL 类付费平台（长训备选）

> 费用说明（2026 年年中公开报价，随活动浮动，**以平台官网实时价为准**）：
> 3090 约 1.2-1.8 元/小时，4090 约 1.2-2.5 元/小时。单品类试水训练
> 预计 12-36 小时（属估算，取决于品类与 epoch 数），费用约 20-80 元。
> 学生认证与新用户礼包通常能覆盖大部分成本。

## 1. 开卡与镜像选择

1. 注册 AutoDL（或趋动云/智星云等同类平台），完成学生认证享折扣。
2. 租 **单卡 RTX 3090 或 4090**（24GB 显存足够，batch_size=32 + AMP）。
3. 镜像选官方 **PyTorch 2.x + CUDA 12.1**（如 `PyTorch 2.1.0 / Python 3.10 / CUDA 12.1`）。
   镜像已自带 GPU 版 torch，`requirements-cloud.txt` 里特意不含 torch。
4. 计费模式选**按量计费**（用完即关，关机数据保留）。

## 2. 上传代码

三种方式任选其一（在**本地 Windows** 终端执行）：

```bash
# 方式 A：scp 整个项目（实例的 ssh 地址/端口在平台控制台复制）
scp -P <端口> -r "<repo-root>" root@<实例地址>:/root/

# 方式 B：项目若已推 Git，云端直接 git clone / git pull

# 方式 C：AutoDL 网页面板的文件上传（适合小改动）
```

上传后在云端实例里：

```bash
cd /root/Industrial-Defect-Multimodal-Pipeline
pip install -r cloud/requirements-cloud.txt
```

## 3. 获取 MVTec AD 数据集

1. 官网免费注册下载：https://www.mvtec.com/company/research/datasets/mvtec-ad
2. 建议**本地下载后上传**（官网对云机房 IP 的下载速度不稳定）；
   云实例带宽充足时也可直接在实例上 `wget`。
3. 解压到项目内 `datasets/mvtec_anomaly_detection/`，确认结构为：

```
datasets/mvtec_anomaly_detection/
  bottle/
    train/good/*.png
    test/good/*.png  test/broken_large/*.png  ...
    ground_truth/broken_large/*_mask.png  ...
  cable/ ...
  （共 15 个品类）
```

## 4. 环境自检

```bash
python cloud/check_env.py
```

全部打勾再继续；缺依赖按提示 `pip install`，CUDA 不可用先检查镜像是否 GPU 版。

## 5. 开始训练

**先单品类试水**（验证链路 + 控制成本）：

```bash
# 防止 ssh 断线杀死训练：先开 tmux
tmux new -s train

python cloud/train_cloud.py --category bottle --epochs 30
# 断线重连后：tmux attach -t train
```

确认 loss 正常下降、FID 在下降、`checkpoints/` 里有文件产出后，
再决定全品类长训：

```bash
python cloud/train_cloud.py --epochs 100
```

训练中断了可以续训：

```bash
python cloud/train_cloud.py --category bottle --resume checkpoints/checkpoint_epoch_20.pth
```

训练日志同时写到终端和项目根 `train_cloud.log`。

## 6. checkpoint 回拉本地

训练完成后**立即关机**（按量计费，关机停止计费，数据保留 15 天）。

在**本地 Windows** 终端执行：

```bash
scp -P <端口> -r root@<实例地址>:/root/Industrial-Defect-Multimodal-Pipeline/checkpoints "<repo-root>"
```

## 7. 与 datapipe 管线衔接（本地执行）

checkpoint 放回项目根 `checkpoints/` 后：

```bash
# 1. 用增广管线生成一批 GAN 缺陷图（任选模式，见根 README 快速开始）
python core/integrated_augmentor.py --mode single --input <good.png> --variations 50

# 2. 管线 B：对真实 GAN 产出做质量过滤（本地无 key 自动走启发式降级）
python -m datapipe.pipeline_b --input <生成图目录>

# 3. 管线 A：在真实 MVTec AD 上构建指令数据集
python -m datapipe.pipeline_a --root datasets/mvtec_anomaly_detection
```

然后把新数字补进 `evaluation/results.md`（命令、日期、数据基础、
checkpoint 来源都写清楚）。

## 省钱与避坑要点

- **训练完立即关机**；按量计费实例关机即停费。
- **始终用 tmux/screen 跑长任务**，ssh 断线不会杀死训练。
- **先单品类 30 epoch 试水**，确认 FID 下降再全品类长训——GAN 训练
  最常见的浪费是跑了几小时才发现配置有问题。
- 显存 <12GB 的卡（如 3080 10G）把 `core/integrated_config.yaml` 里
  `data.batch_size` 从 32 降到 16。
- 首次训练会下载 VGG/InceptionV3/LPIPS 预训练权重（几百 MB），
  云实例一般外网可达；也可先在本地 `weights/` 目录备好再随代码上传，
  项目支持离线加载（见根 README"环境安装"一节）。
- 训练时长与费用为**估算**，实际以 loss/FID 收敛情况为准。
