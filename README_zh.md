# CGDD-Net 论文代码

**Context-Guided Dynamic Detail Modeling for Retinal Vessel Segmentation**

作者：**Xincheng Li, Xiaoqi Sheng, Xinyu Zhang**。

本仓库提供与当前论文结构对应的 PyTorch 实现，包括：

- Context-Guided Scale-Adaptive Deformable Encoding（CSDE）；
- Spatially Adaptive Multi-Kernel Gating（SAMG）；
- Dynamic Cross-Scale Detail Fusion（DCDF）；
- 共享细节引导解码与选择性跳跃连接；
- D3 融合后的单个 Criss-Cross Attention（CCA）模块。

完整使用说明见 [英文 README](README.md)，结构细节见 [architecture.md](docs/architecture.md)，数据协议见 [data_protocol.md](docs/data_protocol.md)。

## 与论文一致的默认结构

- 输入：单通道灰度图；
- 五级编码通道：`8 / 16 / 32 / 64 / 128`；
- 共享细节通道：`8`；
- 完整模型参数量：**1,956,802（约 1.96 M）**；
- CSDE 注意力尺度：`1 / 3 / 5 / 7`，每个头使用 3×3 九点参考采样；
- SAMG 卷积核：`1 / 3 / 5 / 7`；
- 细节特征由 E2/E3/E4 经 SAMG 和 DCDF 融合，并复用于 D3/D2/D1；
- 直接编码器跳跃连接保留 E4 和 E1；
- 下采样和主路径上采样后的通道投影均采用 `1×1 Conv-BN-ReLU`；
- 最终预测头为裸 `1×1 Conv`，输出 logits。

七组累积消融为：

`baseline -> csde -> samg -> dcdf -> detail_decoder -> selective_skip -> full`

## 训练与推理

默认参考配置：

- Adam；
- 基础学习率 `0.001`；
- 训练 batch size `64`；
- 固定 `100` 轮，默认关闭 early stopping；
- 训练 patch `64×64`；
- 每轮采样 `150000` 个 patch；
- weight decay `0`；
- 5 轮线性 warmup，之后使用带重启的余弦调度；
- Dropout 为 `0.1`；仓库为新运行提供的验证集选模功能不重新定义原实验记录；
- 推理 patch `64×64`，stride `16`，推理 mini-batch 默认 `8`；
- patch logits 经 sigmoid 后，在重叠位置平均概率并重构整图，再以 `0.5` 二值化。

损失函数为 FOV 掩码加权的 BCEWithLogits：

```python
loss = (
    F.binary_cross_entropy_with_logits(logits, target, reduction="none") * fov
).sum() / fov.sum()
```

没有额外 Dice、边界、拓扑、类别重加权或深监督损失。

## 论文结果

| 数据集 | SE | SP | ACC | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| DRIVE | 0.8421 | 0.9768 | 0.9704 | 0.8323 | 0.9824 |
| CHASE_DB1 | 0.8600 | 0.9815 | 0.9752 | 0.8102 | 0.9938 |
| STARE | 0.8661 | 0.9812 | 0.9775 | 0.8510 | 0.9895 |
| HRF | 0.8362 | 0.9823 | 0.9711 | 0.8157 | 0.9874 |

作者已确认实际实验协议为 DRIVE `20/20`、CHASE_DB1 `20/8`、STARE `10/10`、HRF `25/20`（训练/测试）。所有 STARE 实验原本即采用 `10/10`，此前的五折描述是写作错误，而非实验划分改变。采集 FOV 分别为 45°、30°、35°、60°，与逐像素 FOV 掩码不同；来源与划分记录见 `docs/data_protocol.md`。`results/` 保存论文主结果、消融与跨数据集结果的机器可读转录。本次同步没有重跑实验或修改这些数值，新运行的输出仍另行保存。

## 使用方法

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke_test.py
```

训练示例：

```bash
python train.py --config configs/default.json \
  --train-manifest splits/new_drive/train.json \
  --val-manifest splits/new_drive/val.json \
  --data-root ../datasets \
  --output runs/drive_seed42 \
  --device cuda
```

评估示例：

```bash
python evaluate.py \
  --checkpoint runs/drive_seed42/best.pt \
  --manifest splits/new_drive/test.json \
  --data-root ../datasets \
  --output runs/drive_seed42/test \
  --device cuda
```

AUC 基于重构后的每张整图概率图计算；SE、SP、ACC、F1 基于相应二值图计算。指标在每张图的 FOV 内评估，再进行不按像素数加权的图像级平均。仓库可导出逐图指标、概率图、二值预测和运行配置。

作者确认计算使用 NVIDIA H200 141 GB、Python、PyTorch 和 CUDA；本地工作站为 Mac。Mac 的系统不能被直接写为 H200 计算主机的系统；原运行的具体系统及软件版本未在此断言。CPU 自动测试仅验证代码行为，不替代数据集实验。

## 数据与发布

DRIVE、STARE、CHASE_DB1 和 HRF 数据需从原始提供方获取，仓库不重新分发数据。数据划分通过 manifest 明确记录；新生成的实验划分会与论文结果转录区分。

模型参数和当前实现的部分算子统计可通过：

```bash
python scripts/profile_model.py --config configs/default.json --batch-size 64 --height 64 --width 64
```

该命令明确记录 `64×1×64×64` 的前向输入；安装 `.[profile]` 后追加 `--fvcore`，可输出支持算子的计数、每 patch 归一化结果、工具版本及不支持的算子。`--batch-size 1` 用于单 patch。该新命令不意味着已经确认 Table 10 原始 profiling 使用相同张量。

论文保留 **1.96 M** 参数及共同比较设置下的 **29.96 GFLOPs**。原统计输入形状、库版本及计数约定尚未从日志恢复。64 个 64×64 patch 与一张 512×512 图像仅像素总数相同，不能直接等同其网络计算量。`docs/profile_measured.json` 中的历史部分算子计数作为诊断记录保留，不替代论文数据，也不表示硬件时延。

Table 7 的一个观测代表一张完整测试图像的 AUC；DRIVE、CHASE_DB1、STARE 的向量长度分别为 `20`、`8`、`10`。均值和标准差描述图像间差异，p 值使用相对于 CGDD-Net 的 Wilcoxon rank-sum test，阈值为 `p < 0.05`。原值及星号保留，本次未独立复算。`exact_p_bounds.py` 仅计算理论下限，不替换实测 p 值；`summarize_runs.py` 用于新的重复训练分析，不重建 Table 7。

仓库许可证、上游工作流归属和第三方许可信息见 `LICENSE`、`NOTICE` 和 `LICENSES/`。
