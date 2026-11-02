# MG1 / R6G 一维拉曼光谱三分类

本工程使用一维拉曼光谱识别三个互斥类别：`仅MG1`、`仅R6G`、`两者都有`。当前研究同时比较三个模型、四种累积光谱表示、增强/无增强两种数据协议以及单种子/多种子训练结果。

## 1. 当前研究矩阵

```text
数据协议 data_regime
├─ data_augment：四个原始CSV各自生成“原谱+全部增强副本”对应文件
├─ data_prepared：按父实验整体划分后的增强训练/验证/测试集
└─ data_plain：可选无增强对照，使用相同的父实验分组规则

模型 architecture
├─ msnet：多尺度残差 + SE + 平均/最大/注意力池化
├─ cnn1d：经典时间序列 FCN 基线
└─ resnet1d：经典时间序列 ResNet 基线

输入表示 channel_set
├─ raw                 1通道
├─ raw_snv             2通道
├─ raw_snv_d1          3通道
└─ raw_snv_d1_d2       4通道

训练种子 training_seed
├─ single：66
└─ multi：66、67、68
```

一次正式模型比较的规模是：

```text
3个模型 × 4种表示 × 3个种子 = 36次训练
```

增强与无增强都完成时合计72次正式训练。每个组合独立初始化，不继承其他组合的权重。

## 2. 数据含义与拆分

`data/` 中每一行表示一次独立实验。第二列是行描述，第一列是该实验的序号。父实验ID为：

```text
experiment_id = 源文件名（不含扩展名） + ":" + 行描述 + ":" + 实验序号
```

当前四个原始CSV共2001条独立实验。程序按 `source_file × concentration_group` 联合分层，以固定种子66进行70%/15%/15%拆分：

| 集合 | 独立原谱数 | 用途 |
|---|---:|---|
| train | 1400 | 更新模型权重 |
| validation | 300 | 早停、通道选择、模型选择 |
| test | 301 | 锁定模型后的最终报告 |

程序先对全部2001个父实验分别生成4份增强副本，再按父实验ID整体划分。默认情况下 `data_augment/` 共10005行；`train.csv`、`validation.csv`、`test.csv` 分别为7000、1500、1505行，对应1400、300、301个独立父实验。一个父实验的原谱和4份增强谱绝不会跨集合。

拆分实现使用 scikit-learn 的 [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。数据隔离原则参考其[数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html)，但本工程按用户指定采用“先全量增强、再按父实验组拆分”：防泄漏的关键是所有派生版本共享同一个父实验ID并整体进入同一集合。详细实现见 [DATA_PREPARATION_PLAN.md](DATA_PREPARATION_PLAN.md)。

## 3. 完整工程流程

```text
data/*.csv
   │
   ├─ python augment_dataset.py
   │      ├─ data_augment/：每个原始文件对应一份完整增强CSV
   │      └─ data_prepared/：按父实验组划分的train/validation/test
   │
   └─ python augment_dataset.py --copies 0 ...
          └─ data_plain_augment/ + data_plain/：无增强对照

准备数据目录
   │
   ├─ train.py：训练一个“模型 × 表示 × 种子”
   └─ run_experiments.py：运行完整实验矩阵并按验证集排名
          │
          ├─ output/<run_id>/：表格、曲线、混淆矩阵、选模记录
          └─ pt/<run_id>/：best.pt

三个模型各自训练完成
   │
   └─ compare_models.py：只读取已有结果，校验公平性并生成模型对比

selected_model.json 或 best.pt
   │
   └─ predict.py：预测未知CSV
```

预处理参数只由 `train.csv` 拟合。测试集不参与预处理拟合、权重更新、早停、通道选择或模型选择。

## 4. 工程目录

```text
R6gMg_classify/
├─ configs/                     实验与结果比较JSON
├─ models/
│  ├─ blocks.py                 MSNet多尺度残差与SE模块
│  ├─ net.py                    MSNet
│  ├─ cnn1d.py                  经典FCN基线
│  ├─ resnet1d.py               经典ResNet基线
│  └─ registry.py               模型注册表
├─ tests/                       接口、选模、早停回归测试
├─ augment_dataset.py           逐文件增强 + 父实验分组数据生成
├─ data.py                      CSV读取、数据协议校验、光谱预处理
├─ representations.py           四种累积输入表示注册表
├─ train.py                     单个组合训练入口
├─ run_experiments.py           模型 × 表示 × 种子矩阵入口
├─ compare_models.py            已完成模型结果比较入口
├─ predict.py                   未知光谱推理入口
├─ metrics.py                   指标与单次训练诊断图
├─ training_control.py          最佳权重与早停逻辑
├─ research_protocol.py         数据、代码、环境指纹
├─ check.py                     临时合成数据端到端自检
├─ README.md                    工程入口与最短运行说明
├─ DATA_PREPARATION_PLAN.md     数据拆分、增强和校验细节
├─ EXPERIMENTS.md               完整实验矩阵与评价协议
├─ MODELS.md                    三个模型、来源和比较解释
├─ EXPERIMENT_DESIGN.md         后续增加模型/数据的接口约束
└─ GIT_GITHUB.md                Git与GitHub发布步骤
```

以下目录只存在于本地并被 `.gitignore` 排除：`data/`、`data_augment*/`、`data_prepared*/`、`data_plain*/`、`pt*/`、`output*/`、虚拟环境和缓存。

## 5. 环境与数据准备

推荐Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果已经在项目现有的 Conda 环境中，命令行前缀显示 `(pytorch)` 后可直接使用 `python`。执行 `python --version` 和 `python -c "import torch; print(torch.__version__)"` 可确认没有误用未安装 PyTorch 的系统解释器；本次验证使用 Python 3.12.12 与 PyTorch 2.6.0。

生成增强数据：

```powershell
python augment_dataset.py
```

默认每条原谱生成4份增强谱，每份连续执行5轮平移、增益和高斯噪声；默认噪声标准差为当前谱标准差的0.010，终端按三个阶段和四个源文件打印简洁进度。

生成相同拆分的无增强对照：

```powershell
python augment_dataset.py --copies 0 --augmented-dir data_plain_augment --prepared-dir data_plain
```

目标目录已存在时，确认后添加 `--overwrite`。增强与无增强实验使用相同原始文件、比例和拆分种子时，`split_manifest.csv` 应完全一致。

## 6. 训练命令

### 6.1 单种子流程

增强数据：

```powershell
python run_experiments.py --config configs\msnet_single.json
python run_experiments.py --config configs\cnn1d_single.json
python run_experiments.py --config configs\resnet1d_single.json
python compare_models.py --config configs\compare_single.json
```

无增强数据：

```powershell
python run_experiments.py --config configs\msnet_single.json --prepared-dir data_plain --run-id msnet_plain_single
python run_experiments.py --config configs\cnn1d_single.json --prepared-dir data_plain --run-id cnn1d_plain_single
python run_experiments.py --config configs\resnet1d_single.json --prepared-dir data_plain --run-id resnet1d_plain_single
python compare_models.py --config configs\compare_plain_single.json
```

### 6.2 正式多种子流程

增强数据：

```powershell
python run_experiments.py --config configs\msnet_multi.json
python run_experiments.py --config configs\cnn1d_multi.json
python run_experiments.py --config configs\resnet1d_multi.json
python compare_models.py --config configs\compare_multi.json
```

无增强数据：

```powershell
python run_experiments.py --config configs\msnet_multi.json --prepared-dir data_plain --run-id msnet_plain
python run_experiments.py --config configs\cnn1d_multi.json --prepared-dir data_plain --run-id cnn1d_plain
python run_experiments.py --config configs\resnet1d_multi.json --prepared-dir data_plain --run-id resnet1d_plain
python compare_models.py --config configs\compare_plain_multi.json
```

`--skip-existing` 只复用数据哈希、代码指纹和参数都严格一致的完整结果；`--summary-only` 只重新汇总已有完整结果；`--dry-run` 只检查配置并打印将要执行的训练命令。

## 7. 输出与判断顺序

每个模型实验根目录：

```text
output/<run_id>/
├─ results.csv                 每个表示和种子的逐次指标
├─ summary.csv                 多种子均值、标准差和验证排名
├─ comparison.png              四种表示的测试正确率
├─ training_curves.png          四种表示用四种颜色叠加的训练过程
├─ selected_model.json         按验证规则选出的权重
├─ split.csv                   本轮使用的固定拆分
└─ config.json                 有效配置快照
```

每次训练：

```text
output/<run_id>/<architecture>/<channel_set>/seed_<n>/
├─ metrics.json
├─ history.csv
└─ diagnostics.png

pt/<run_id>/<architecture>/<channel_set>/seed_<n>/best.pt
```

三个模型的比较目录另外包含：

```text
├─ model_summary.csv
├─ confusion_matrices.png          3模型 × 4表示的12个测试混淆矩阵
├─ training_curves_msnet.png       四种表示不同颜色；多种子逐行显示
├─ training_curves_cnn1d.png
└─ training_curves_resnet1d.png
```

判断顺序必须是：

1. 用验证正确率、验证Macro-F1、验证损失和参数量完成排名；
2. 查看跨种子均值与标准差，排除幸运种子；
3. 再报告测试Accuracy、Macro-F1、错误数和混淆矩阵；
4. 同时报告测试行数和独立父实验数，不能把同一父实验的增强副本当作新的独立实验；
5. 若内部测试全部接近100%，应说明性能饱和，不能据此宣称复杂模型显著更优。

Macro-F1先分别计算三个类别的F1，再等权平均，因此能防止样本较多的“两者都有”类别掩盖其他类别的错误。

## 8. 模型来源与实现边界

`cnn1d` 与 `resnet1d` 参考 Wang、Yan、Oates 的[时间序列分类FCN/ResNet论文](https://arxiv.org/abs/1611.06455)以及作者团队的 [`dl-4-tsc`](https://github.com/hfawaz/dl-4-tsc) 开源实现，并在本工程中独立改写为PyTorch、统一使用三分类头和当前训练协议。Keras官方也提供了相近的[一维时间序列分类示例](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)。

MSNet为本工程主模型，其中残差连接参考[ResNet](https://arxiv.org/abs/1512.03385)，通道重标定参考[Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507)；多尺度卷积分支、三种池化组合及其光谱任务适配是本工程实现。更详细的结构差异与开源出处见 [MODELS.md](MODELS.md)。

## 9. 自检、预测与发布

软件自检不会读取真实数据，也不会运行完整训练：

```powershell
python check.py
```

预测未知光谱：

```powershell
python predict.py --input unknown.csv --selection output\msnet_multi\selected_model.json
```

上传新版本到当前GitHub仓库前，请先阅读 [GIT_GITHUB.md](GIT_GITHUB.md)。当前本地远端为 `git@github.com:ChiaKi707/classify.git`，通常只需要检查、提交并推送，不需要重新创建仓库。
