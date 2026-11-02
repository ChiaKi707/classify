# 三个一维分类模型及比较方法

## 1. 选择依据

新增的两个模型不是为当前数据专门搜索出来的结构，而是时间序列分类中常用的公开基线：

- [Wang、Yan、Oates：Time Series Classification from Scratch](https://arxiv.org/abs/1611.06455)提出 FCN 与残差网络作为端到端时间序列分类基线；
- 作者团队公开的 [FCN 实现](https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py)使用 8/5/3 三层 Conv1D、BatchNorm 和全局平均池化；
- 同仓库的 [ResNet 实现](https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py)使用三个残差块，每块包含 8/5/3 三层 Conv1D；
- [Keras 官方时间序列分类示例](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)也采用 Conv1D、BatchNorm、ReLU 和全局平均池化组成一维分类器。

工程将这两个公开 TensorFlow/Keras 骨架等价改写为 PyTorch，并统一接入当前三分类、四种输入表示、离线增强、早停、指标和结果协议。

仓库没有复制或打包上述项目的源文件；本地实现保留其核心层级思想，但加入统一 Dropout、三分类输出、模型注册表和本工程初始化方式。因此结果应表述为“基于经典 FCN/ResNet 骨架的本地 PyTorch 基线”，而不是原仓库模型的原样复现。

## 2. `cnn1d`：经典全卷积一维 CNN

实现文件：`models/cnn1d.py`。

```text
输入 [B, C, L]
→ Conv1D(C→128, kernel=8) + BN + ReLU
→ Conv1D(128→256, kernel=5) + BN + ReLU
→ Conv1D(256→128, kernel=3) + BN + ReLU
→ Global Average Pooling
→ Dropout(0.15)
→ Linear(128→3)
```

特征：

- 没有残差、多尺度并行、SE 或注意力，适合作为直接 CNN 基线；
- 三种卷积核从较宽到较窄，先提取较大范围峰形，再组合局部变化；
- 全局平均池化避免依赖固定展平长度，减少全连接层参数；
- 除统一 Dropout 和三分类头外，保留公开 FCN 的核心结构。

四种输入下的可训练参数量约为：

| 通道数 | 参数量 |
|---:|---:|
| 1 | 264,579 |
| 2 | 265,603 |
| 3 | 266,627 |
| 4 | 267,651 |

## 3. `resnet1d`：经典时间序列 ResNet

实现文件：`models/resnet1d.py`。

```text
输入 [B, C, L]
→ ResidualBlock(C→64, kernels=8/5/3)
→ ResidualBlock(64→128, kernels=8/5/3)
→ ResidualBlock(128→128, kernels=8/5/3)
→ Global Average Pooling
→ Dropout(0.15)
→ Linear(128→3)
```

每个残差块为：

```text
主分支：Conv8 + BN + ReLU → Conv5 + BN + ReLU → Conv3 + BN
捷径支路：通道变化时使用 Conv1 + BN，否则使用 BN
两支相加 → ReLU
```

特征：

- 捷径连接让较深网络可以直接传播原始特征和梯度；
- 8/5/3 连续卷积逐步融合宽峰形与局部峰边缘；
- 三个残差块是经典时间序列分类 ResNet，而不是把二维图像 ResNet-18 机械改成一维；
- 不使用时间轴下采样，避免短光谱中的窄峰过早丢失。

四种输入下的可训练参数量约为：

| 通道数 | 参数量 |
|---:|---:|
| 1 | 503,235 |
| 2 | 503,811 |
| 3 | 504,387 |
| 4 | 504,963 |

## 4. 与当前 MSNet 的差异

| 模型 | 主要结构 | 研究作用 |
|---|---|---|
| `cnn1d` | 三层串行 Conv1D + GAP | 无残差、无注意力的经典基线 |
| `resnet1d` | 三个 8/5/3 残差块 + GAP | 验证普通残差网络的效果 |
| `msnet` | 多尺度残差块 + SE + 多种池化 | 当前主模型 |

如果 MSNet 在同一数据、通道和种子下稳定超过 CNN1D，可以支持“复杂特征提取相对普通串行卷积有收益”。如果同时超过 ResNet1D，才能进一步说明优势不只是来自残差连接。

参数量不同，因此不能只看正确率。还应同时比较验证稳定性、Macro-F1、鲁棒性、训练时间和参数量。

## 5. 分开训练

单种子版本用于快速初步比较，三个模型分别运行且互不覆盖：

```powershell
python run_experiments.py --config configs\msnet_single.json
python run_experiments.py --config configs\cnn1d_single.json
python run_experiments.py --config configs\resnet1d_single.json
```

每个单种子配置运行4次。正式实验使用多种子配置：

```powershell
python run_experiments.py --config configs\msnet_multi.json
python run_experiments.py --config configs\cnn1d_multi.json
python run_experiments.py --config configs\resnet1d_multi.json
```

每个多种子配置运行：

```text
1 个模型 × 4 种表示 × 3 个种子 = 12 次训练
```

三个模型合计36次正式训练。它们共享同一个 `data_prepared`、四种表示、三个训练种子和全部训练参数。

无增强对照仍分别训练三个模型，只切换准备数据目录和运行名：

```powershell
python run_experiments.py --config configs\msnet_multi.json --prepared-dir data_plain --run-id msnet_plain
python run_experiments.py --config configs\cnn1d_multi.json --prepared-dir data_plain --run-id cnn1d_plain
python run_experiments.py --config configs\resnet1d_multi.json --prepared-dir data_plain --run-id resnet1d_plain
python compare_models.py --config configs\compare_plain_multi.json
```

## 6. 只比较已经完成的结果

单种子模型都训练完成后执行：

```powershell
python compare_models.py --config configs\compare_single.json
```

多种子正式结果完成后执行：

```powershell
python compare_models.py --config configs\compare_multi.json
```

如果已有结果使用了其他 `run_id`，只需修改对应比较JSON中的目录；但三个模型必须具有相同的四种表示和训练种子覆盖。已有MSNet结果满足当前数据与共享训练代码协议时，不需要仅因为新增模型而重新训练。

该命令不会训练模型。它会重新读取三个实验目录中的 `metrics.json` 和 `best.pt`，并在比较前检查：

- 数据文件哈希、准备配置和拆分完全一致；
- 数据读取、训练、指标和早停等共享核心代码指纹一致；
- 测试样本数量及评价协议一致；
- 每个模型都有相同的四种表示和三个种子；
- 相同表示、相同种子下的训练超参数一致；
- 不存在重复或缺失的模型/表示/种子结果。

输出：

```text
output/model_compare_single/ 或 output/model_compare_multi/
├─ results.csv
├─ summary.csv
├─ model_summary.csv
├─ comparison.png
├─ confusion_matrices.png
├─ training_curves_msnet.png
├─ training_curves_cnn1d.png
├─ training_curves_resnet1d.png
├─ selected_model.json
└─ config.json
```

- `results.csv`：36 次训练的逐次结果；
- `summary.csv`：所有“模型 × 表示”组合的均值、波动和验证排名；
- `model_summary.csv`：先为每个模型按验证集选出最佳表示，再单独比较三个模型；
- `comparison.png`：同一表示下三个模型的测试正确率对比；
- `confusion_matrices.png`：三个模型 × 四种通道方案的全部测试混淆矩阵；多种子使用逐类归一化比例并标出累计预测次数；
- `training_curves_<architecture>.png`：每个模型单独一张训练图，四种通道方案用不同颜色；多种子时每个种子单独占一行；
- `selected_model.json`：只按验证数据选出的最终模型、表示和种子。

## 7. 正确结论方式

比较顺序必须是：

1. 每个模型内部先用验证集选择最佳通道表示；
2. 再用验证均值、Macro-F1、损失和参数量对三个模型排名；
3. 最后报告主测试和鲁棒性结果；
4. 测试结果不能反过来用于选择通道或模型。

如果三个模型在测试集都达到 100%，应报告当前测试集性能饱和，并比较参数量、训练时间和跨种子稳定性；不能仅凭相同的 100% 声称某个复杂模型更优秀。

## 8. 来源对照表

| 本地结构 | 论文/开源出处 | 保留的核心思想 | 本工程改动 |
|---|---|---|---|
| `cnn1d` | [Wang等FCN/ResNet论文](https://arxiv.org/abs/1611.06455)、[`fcn.py`](https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py) | 8/5/3串行卷积、BN、ReLU、全局平均池化 | PyTorch实现、可变输入通道、Dropout、三分类头 |
| `resnet1d` | [Wang等论文](https://arxiv.org/abs/1611.06455)、[`resnet.py`](https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py) | 三个8/5/3残差块、投影捷径、全局平均池化 | PyTorch实现、可变输入通道、Dropout、统一训练接口 |
| `msnet`残差部分 | [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | 残差捷径 | 改为一维多尺度光谱块 |
| `msnet`通道重标定 | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | 全局统计驱动的通道权重 | 嵌入每个一维多尺度残差块 |
| `msnet`其余部分 | 本工程 | — | 3/7/15并行卷积、两次下采样、平均/最大/注意力池化拼接 |
