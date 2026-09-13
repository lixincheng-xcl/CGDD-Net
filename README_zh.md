# CGDD-Net 论文代码

**Context-Guided Dynamic Detail Modeling for Retinal Vessel Segmentation**

作者：**Xincheng Li, Xinyu Zhang, Xiaoqi Sheng**。

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
- batch size `64`；
- 最多 `50` 轮；
- 训练 patch `64×64`；
- 每轮采样 `150000` 个 patch；
- weight decay `0`；
- 5 轮线性 warmup，之后使用带重启的余弦调度；
- 以验证集平均 AUC 选择最佳模型；
- 推理 patch `96×96`，stride `16`；
- patch logits 经 sigmoid 后，在重叠位置平均概率，再以 `0.5` 二值化。

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

`results/` 中保存论文主结果、消融结果和跨数据集结果的机器可读转录，便于核对论文数值。新运行产生的评估文件与这些论文结果转录分开保存。

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

指标在每张图像的 FOV 内计算，再进行不按像素数加权的图像级平均。仓库可导出逐图指标、概率图、二值预测和运行配置等可复现信息。

论文实验环境为 NVIDIA H200 141 GB、PyTorch 2.13.0、CUDA 13.2。仓库中的 CPU 自动测试用于验证代码行为，不替代论文中的数据集实验。

## 数据与发布

DRIVE、STARE、CHASE_DB1 和 HRF 数据需从原始提供方获取，仓库不重新分发数据。数据划分通过 manifest 明确记录；新生成的实验划分会与论文结果转录区分。

模型参数和当前实现的部分算子统计可通过：

```bash
python scripts/profile_model.py --config configs/default.json --height 96 --width 96
```

当前论文只使用已确认的 **1.96 M** 参数量，不将部分算子计数表述为完整 FLOPs 或硬件时延。

仓库许可证、上游工作流归属和第三方许可信息见 `LICENSE`、`NOTICE` 和 `LICENSES/`。
