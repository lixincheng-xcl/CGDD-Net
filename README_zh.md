# CGDD-Net 论文代码

这是在你指定的 `VesselSeg-Pytorch-master` 内整理的独立论文代码仓库。上传 GitHub 时，应以当前 `CGDD-Net` 文件夹为仓库根目录。外层原项目及其数据集保持原样。

## 已实现

- 按论文最终结构图实现 CSDE、SAMG、DCDF、细节引导解码、选择性跳跃连接和 CCA。
- 补齐图 3 的 GELU、BN 和两层逐点卷积；Q、K、V 均进入可变形采样。
- 五个阶段通道为 `8/16/32/64/128`，共享细节通道为 `8`；七种累积消融可切换。
- 训练使用 Adam，基础学习率为作者确认的 `0.001`，采用线性 warmup 与余弦退火重启，二值阈值为 `0.5`。
- 显式的数据划分清单、图像与受试者重叠检查、训练保存与恢复、滑窗推理、指标导出和统计工具。
- 依赖文件、Apache-2.0 许可证、原项目致谢、自动测试和 GitHub Actions 配置。

## 使用方法

完整命令见 [英文 README](README.md)。建议先执行：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/smoke_test.py
```

安装 GPU 版 PyTorch 时，请通过[官方安装页面](https://pytorch.org/get-started/locally/)选择与你的服务器相符的版本。当前本地验证环境为 Python 3.12.2 / PyTorch 2.3.1 / CPU；论文中的 H200 141 GB、PyTorch 2.13.0、CUDA 13.2 是你提供的服务器环境。

原始数据放在外层 `../datasets`。先运行数据检查，再使用真实实验划分清单；只有开始新实验时才使用 `generate` 命令创建新划分。程序会将新生成的划分明确标记，不会把它们当作原服务器实验记录。

## 数值与版本说明

你确认的论文数值保存在 `results/`，未被本地测试结果覆盖。本轮按图构建的默认模型实测为 **1,956,802 个参数**，论文参数量已同步为 **1.96 M**。**20.05 G FLOPs** 仍为早期服务器记录，尚未按当前代码重测；新实现的测试通过不代表已经复现论文分数。

`docs/profile_measured.json` 记录了新实现的真实参数计数和部分算子计算量。fvcore 未覆盖的算子也已列出，因此部分计数不冒充完整 FLOPs。需要得到与论文复杂度一致的归档版本时，应核对真实服务器代码、输入尺寸和计算约定。

训练产生的 `best.pt/latest.pt`、`config.json`、`history.csv`、数据清单和评估 CSV/JSON 是后续归档材料。不同图像之间的标准差与不同随机种子训练结果的标准差不能混用。清单中的“长期有效归档地址”如何准备，见 [中文归档说明](docs/reproducibility_zh.md)。

## 生成 GitHub 源码包

```bash
python scripts/build_release.py
```

程序仅打包代码、配置、文档、测试和论文数值转录，不包含外层原始图像、训练权重、虚拟环境或运行缓存。当前 GitHub 同步与旧接口迁移说明见 [更新记录](RELEASE_NOTES.md)。

学习率 `0.001` 已由作者确认用于论文服务器实验；其余尚未单独确认的超参数仍为发布默认设置。历史 `docs/profile_measured.json` 中的配置哈希保留测量当时的值，学习率更新未改变模型结构或参数量。
