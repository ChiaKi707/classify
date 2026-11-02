# 四种累积光谱表示消融协议

## 1. 研究问题

固定模型结构、原始实验拆分、训练参数和训练种子，只逐步增加输入表示：

```text
raw → raw_snv → raw_snv_d1 → raw_snv_d1_d2
```

相邻组分别检验 SNV、一阶导数和二阶导数在已有通道基础上的增量贡献。这是输入表示消融，不是网络模块消融，不能据此证明多尺度、残差、SE 或注意力池化各自有效。

同样的四表示消融分别在 `msnet`、`cnn1d` 和 `resnet1d` 上运行。这样既能比较输入表示，也能在完全相同的数据协议下比较模型结构。

`data_plain` 是额外的无增强训练对照。它与 `data_prepared` 使用同一份原始实验拆分，只改变训练集是否含离线增强副本；验证集和测试集保持相同。

## 2. 必须固定的条件

- 相同的 2001 条独立原始实验及三分类标签；
- 相同的 `data_prepared` 目录和文件哈希；
- 相同的两轮离线增强参数；
- 相同训练超参数；
- 正式实验使用相同种子集合 66、67、68；
- 每个组合从头初始化，不继承其他组合权重；
- 模型选择只使用验证数据，测试数据不用于调参。

拆分已在上游完成，训练配置不再包含验证比例、测试比例或拆分种子。所有实验直接共享 `split_manifest.csv`。

## 3. 数据协议

每一行原始 CSV 是一个独立实验，ID 为“源文件名 + 原始编号”。程序先在 `source_file × concentration_group` 层内随机拆分原始实验，再只增强训练集合。

默认 70/15/15 拆分和 4 份增强副本对应：

| 集合 | 独立原谱 | 增强谱 | 实际用途 |
|---|---:|---:|---|
| 训练 | 1400 | 5600 | 7000 行全部训练 |
| 验证 | 300 | 0 | 早停和选模 |
| 主测试 | 301 | 0 | 最终分类结果 |
| 鲁棒性 | 来自主测试 | 1204 | 扰动稳定性，不参与选模 |

每份增强谱从原谱重新开始，连续执行 2 轮“平移 → 增益 → 高斯噪声”。三种操作每轮执行概率均为 0.5，默认幅度为 ±1 点、0.98～1.02 和谱标准差的 0.005。

全局预处理中心和尺度只用训练集合中未增强原谱拟合，再应用于训练增强谱、验证谱、测试谱和鲁棒性谱。

## 4. 四组输入

| 配置 | 通道数 | 计算方式 | MSNet 参数量 |
|---|---:|---|---:|
| `raw` | 1 | 逐波数训练集标准化 | 72,088 |
| `raw_snv` | 2 | raw + 逐谱 SNV | 72,256 |
| `raw_snv_d1` | 3 | 前两者 + SNV 一阶导数 | 72,424 |
| `raw_snv_d1_d2` | 4 | 前三者 + SNV 二阶导数 | 72,592 |

参数量差异来自首层输入通道数，其余 MSNet 结构相同。

## 5. 训练、选择和报告

默认训练为 AdamW、学习率 3e-4、weight decay 1e-4、batch 32、最多 120 轮、Dropout 0.15、梯度裁剪 5、余弦退火、patience 20、min_delta 1e-4。

任何严格更低的验证损失都会替换 `best.pt`；`min_delta` 只决定是否重置早停计数。训练完成后恢复最低验证损失权重，再分别评价主测试和鲁棒性数据。

正式配置规模为：

```text
1 个结构 × 4 种表示 × 3 个训练种子 = 12 次训练
```

结构 × 表示组合按以下验证指标排名：

1. 验证正确率均值降序；
2. 验证 Macro-F1 均值降序；
3. 验证损失均值升序；
4. 参数量升序。

主测试和鲁棒性结果只用于报告，不参与选择。

## 6. 最小结果集

矩阵根目录保留六个研究文件：

| 文件 | 作用 |
|---|---|
| `results.csv` | 每个种子的验证、主测试和鲁棒性结果 |
| `summary.csv` | 各组合的均值、标准差和验证排名 |
| `comparison.png` | 各结构下四种表示的主测试正确率 |
| `selected_model.json` | 按验证规则选出的候选权重 |
| `split.csv` | 本轮使用的原始实验固定拆分 |
| `config.json` | 有效运行配置和准备数据目录 |

诊断某一次训练时再看该组合的：

- `metrics.json`：完整协议、指标和样本计数；
- `history.csv`：每轮训练损失和验证指标；
- `diagnostics.png`：损失曲线、验证曲线和两个混淆矩阵；
- `best.pt`：最低验证损失权重及预处理状态。

## 7. 如何判断第三种是否最佳

优先检查 `summary.csv`：第三种 `raw_snv_d1` 应在预设验证规则下排名第一，并且多个训练种子的标准差不能明显大于它相对其他组的优势。再查看 `results.csv`，确认其主测试正确率、Macro-F1、错误数和鲁棒性结果没有明显退化。

最后查看每个种子的 `diagnostics.png`：确认损失正常收敛、错误没有集中在某一类、优势不是单个幸运种子造成。

如果四组正确率都为 100%，应表述为“当前数据上四组性能饱和，三通道方案按预设验证规则作为默认方案”，不能写成第三种显著更好。内部拆分也不能替代跨日期、跨批次、跨基底或跨仪器盲测。

## 8. 可复现性与参考依据

- 数据拆分和泄漏控制参考 scikit-learn 的 [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 与[数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html)；本工程进一步保存拆分清单和所有准备文件哈希。
- 三个训练种子用于估计权重初始化和批次顺序带来的波动。程序同时固定 Python、NumPy、PyTorch 和 CUDA 随机状态，并启用确定性设置；PyTorch 官方说明见 [`torch.use_deterministic_algorithms`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) 和 [cuDNN backend](https://docs.pytorch.org/docs/stable/backends.html)。不同硬件或不同依赖版本仍可能出现细微数值差异，因此结果同时保存环境版本。
- CNN1D/ResNet1D 的公开来源见 [MODELS.md](MODELS.md)。四种累积光谱表示、验证集排序规则和“测试集只报告”的具体协议均由本项目预先定义。

## 9. 命令

```powershell
python augment_dataset.py
python run_experiments.py --config configs\msnet_single.json --dry-run
python run_experiments.py --config configs\msnet_single.json
python run_experiments.py --config configs\cnn1d_single.json
python run_experiments.py --config configs\resnet1d_single.json
python compare_models.py --config configs\compare_single.json
python run_experiments.py --config configs\msnet_multi.json
python run_experiments.py --config configs\cnn1d_multi.json
python run_experiments.py --config configs\resnet1d_multi.json
python compare_models.py --config configs\compare_multi.json
python run_experiments.py --config configs\msnet_multi.json --skip-existing
python run_experiments.py --config configs\msnet_multi.json --summary-only
python augment_dataset.py --copies 0 --output-dir data_plain
python run_experiments.py --config configs\msnet_multi.json --prepared-dir data_plain --run-id msnet_plain
python run_experiments.py --config configs\cnn1d_multi.json --prepared-dir data_plain --run-id cnn1d_plain
python run_experiments.py --config configs\resnet1d_multi.json --prepared-dir data_plain --run-id resnet1d_plain
python compare_models.py --config configs\compare_plain_multi.json
```
