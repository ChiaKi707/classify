# 模型结构扩展与后续实验设计

## 1. 当前接口

工程将三个实验维度明确分开：

```text
architecture     模型结构，例如 msnet、cnn1d、resnet1d
channel_set      raw、raw_snv、raw_snv_d1、raw_snv_d1_d2
training_seed    权重初始化和训练批次随机性
```

数据拆分与增强已经独立固化在准备数据目录，不属于单次训练参数。同一次模型比较中的每个模型、输入表示和种子必须读取相同目录：增强实验使用 `data_prepared/`，无增强对照使用 `data_plain/`。

当前实现包括：

- `representations.py`：四种光谱表示注册；
- `models/registry.py`：模型结构注册和统一构建；
- `research_protocol.py`：准备数据、代码和环境指纹；
- `run_experiments.py`：结构 × 表示 × 种子实验矩阵；
- `configs/*.json`：单种子、多种子正式实验和已有结果比较配置；
- 验证集聚合排名和 `selected_model.json`；
- 每次训练只输出 `metrics.json`、`history.csv`、`diagnostics.png` 和 `best.pt`。

当前已经注册 `msnet`、`cnn1d` 和 `resnet1d`。三个模型都使用相同训练入口和结果协议；结构开关和传统 SVM 尚未实现，因此目前不能报告这些未实现方案的结论。

## 2. 新增神经网络结构

新结构实现标准 PyTorch 模块，并提供以下构建函数：

```python
def build_custom_model(in_channels, dropout, num_classes, model_options):
    return CustomModel1D(
        in_channels=in_channels,
        dropout=dropout,
        num_classes=num_classes,
        **model_options,
    )
```

在 `models/registry.py` 登记：

```python
register_architecture(
    ArchitectureSpec(
        name="custom_model",
        display_name="自定义一维模型",
        backend="pytorch",
        builder=build_custom_model,
    )
)
```

然后在 JSON 的 `architectures` 中加入。`cnn1d` 和 `resnet1d` 已经按照这一接口接入，可直接参考 `models/registry.py`。

```json
{"name": "custom_model", "model_options": {}}
```

运行器会自动让新模型分别接收 1、2、3、4 个通道，并沿用相同数据、训练种子、评价和结果格式。

## 3. 已有模型与配置

| 模型 | 单种子配置 | 多种子配置 |
|---|---|---|
| MSNet | `configs/msnet_single.json` | `configs/msnet_multi.json` |
| CNN1D | `configs/cnn1d_single.json` | `configs/cnn1d_multi.json` |
| ResNet1D | `configs/resnet1d_single.json` | `configs/resnet1d_multi.json` |

模型分开训练后，分别使用 `compare_single.json` 或 `compare_multi.json` 对已有结果进行协议校验、合并和验证集排名，不会重新训练。

无增强训练通过 `--prepared-dir data_plain` 切换数据；单种子使用 `compare_plain_single.json`，正式多种子使用 `compare_plain_multi.json` 汇总，不复制六份几乎相同的训练配置。

## 4. 多模型配置示例

```json
{
  "run_id": "architecture_comparison",
  "prepared_dir": "data_prepared",
  "architectures": [
    {"name": "msnet", "model_options": {}},
    {"name": "cnn1d", "model_options": {}},
    {"name": "resnet1d", "model_options": {}}
  ],
  "channel_sets": [
    "raw",
    "raw_snv",
    "raw_snv_d1",
    "raw_snv_d1_d2"
  ],
  "training_seeds": [66, 67, 68],
  "training": {
    "epochs": 120,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "dropout": 0.15,
    "patience": 20,
    "min_delta": 0.0001,
    "grad_clip": 5.0,
    "preprocess_clip": 10.0,
    "device": "auto",
    "num_workers": 0
  }
}
```

此配置会产生：

```text
3 个结构 × 4 种表示 × 3 个种子 = 36 次训练
```

配置中不再放拆分比例、拆分种子或评价范围；这些内容由 `preparation_config.json` 和四个显式集合唯一确定。

## 5. 建议实验顺序

| 阶段 | 固定项 | 改变量 | 可支持的结论 |
|---|---|---|---|
| 输入消融 | MSNet、数据、训练参数 | 四种表示 | SNV、D1、D2 的条件增量 |
| 简单网络基线 | `raw_snv_d1`、数据、种子 | MSNet 与 CNN1D | 完整 MSNet 相对普通 CNN 的整体收益 |
| 残差网络基线 | `raw_snv_d1`、数据、种子 | MSNet 与 ResNet1D | MSNet 优势是否不只来自残差连接 |
| 模块消融 | `raw_snv_d1`、其余结构 | 去 SE、单尺度、去注意力等 | 单个网络模块的增量贡献 |
| 全矩阵复核 | 锁定数据与协议 | 候选模型 × 四种表示 | 模型与输入表示的交互 |
| 外部验证 | 锁定最终候选 | 新批次或新条件 | 泛化能力 |

结构消融应优先固定当前主表示 `raw_snv_d1`，避免同时改变网络和输入。只有某个结构本身得到支持后，再让它进入完整四表示矩阵。

## 6. 数据不断增加时

新增原始数据后，应使用新的输出目录或明确执行 `augment_dataset.py --overwrite`，然后重新运行完整矩阵。`preparation_config.json`、所有 CSV 哈希和代码指纹会进入每次训练结果，因此旧结果不会被 `--skip-existing` 错误复用。

若新增数据是此前未使用的真正外部批次，优先保留为外部盲测，不要立刻混回训练集。完成一次盲测报告后，再决定是否把它并入下一版训练数据。

## 7. 传统方法接入原则

RBF-SVM 等非神经网络不应伪装成 epoch 训练。后续可扩展独立 backend，但必须共享：

- `split_manifest.csv` 中的独立实验拆分；
- 只在训练数据拟合的预处理；
- 四种光谱表示定义；
- 干净主测试和独立鲁棒性评价；
- 核心分类指标与结果表结构。

SVM 的 C、gamma、PCA 和概率校准只能使用训练与验证数据选择。没有可比概率时，交叉熵和 Brier 分数应标记为不适用。

## 8. 公平比较要求

- 每组比较尽量只改变一个研究因素；
- 所有结构使用同一份准备数据和相同训练种子；
- 不根据测试结果反复修改结构；
- 报告参数量、训练时间和多种子波动；
- 模型选择只看验证集聚合结果；
- 主测试只用未增强原谱；
- 鲁棒性增强谱只作补充，不当作独立样本；
- 最终结论增加此前未使用的新批次盲测。

如果内部测试全部 100%，应如实报告性能饱和，并把研究重点转向外部批次、低信噪比、未知混合比例和真实干扰物，而不是反复查看测试集后继续调参。

## 9. 扩展设计的开源依据与边界

模型注册表和统一训练入口是本工程的软件设计，并非照搬某个仓库。新增模型时所要求的“相同拆分、相同种子、相同指标、只改结构”来自受控消融的基本原则。当前两个经典基线的网络骨架参考：

- [Wang、Yan、Oates：Time Series Classification from Scratch](https://arxiv.org/abs/1611.06455)；
- 作者团队公开的 [`dl-4-tsc`](https://github.com/hfawaz/dl-4-tsc)；
- [Keras 官方一维时间序列分类示例](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)。

本工程在这些骨架基础上独立改写 PyTorch 实现，并增加光谱表示注册、离线数据协议、跨结果哈希校验和统一比较图。未来模型可以参考其他开源实现，但必须在 `MODELS.md` 写清原论文、代码链接、借鉴范围和本地改动，不能只写模型名称。
