# MG1 / R6G 拉曼光谱三分类工程

## 1. 项目目标

本工程使用一维卷积神经网络分析拉曼光谱，将待测样品分为三个互斥类别：

1. 仅含MG1；
2. 仅含R6G；
3. 同时含MG1和R6G。

工程的业务前提是每个待测样品至少含一种目标试剂，因此不设置“二者都无”类别。混合比例1:1、1:2、2:1等均归入“两者都有”，模型不预测混合比例和浓度。

最终采用模型3，即“标准化原始光谱 + SNV光谱 + 一阶导数”三通道输入。工程同时保留模型1～4的通道递增实验，用于比较不同光谱表示的贡献。

## 2. 工程结构

~~~text
R6gMg_classify/
├─ models/
│  ├─ __init__.py        模型包入口
│  ├─ blocks.py          SE注意力和多尺度残差模块
│  └─ net.py             三分类网络ReagentClassMSNet
├─ config.py             路径、类别、通道和公共配置
├─ augment_dataset.py    原始光谱离线扩充脚本
├─ data.py               扩充CSV读取、来源绑定拆分和预处理
├─ train.py              单模型训练入口，默认模型3
├─ training_control.py   最低验证损失选优与独立早停监控
├─ predict.py            加载权重并预测未知光谱
├─ run_all.py            依次训练模型1～4并生成对比结果
├─ metrics.py            Softmax推理、分类指标和绘图
├─ check.py              独立临时目录中的完整流程自检
├─ requirements.txt      Python依赖版本
├─ README.md             工程运行说明
├─ EXPERIMENTS.md        四模型递增实验说明
├─ EXPERIMENT_DESIGN.md  网络结构消融和传统基线的扩展建议
├─ tests/                权重选优与早停的回归测试
├─ data/                 2001条原始实验光谱
├─ data_augmented/       离线扩充CSV、扩充清单和参数记录
├─ pt/                   训练权重目录
└─ output/               表格、图片和结果摘要目录
~~~

`models`保存网络结构代码，`pt`保存训练后的权重，`output`保存实验结果。相同模型再次训练时，同名权重和结果文件直接覆盖，不创建额外版本目录。

## 3. 环境配置

核验环境如下：

~~~text
Python 3.12
PyTorch 2.6.0
NumPy 2.3.4
pandas 2.3.3
scikit-learn 1.7.2
Matplotlib 3.10.6
~~~

Conda环境创建与依赖安装命令：

~~~powershell
conda create -n spectra python=3.12
conda activate spectra
python -m pip install -r requirements.txt
~~~

已有PyTorch环境时执行：

~~~powershell
conda activate pytorch
python --version
python -c "import torch; print(torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
~~~

默认参数`--device auto`在CUDA可用时使用显卡，否则使用CPU。

## 4. 数据组织

原始实验数据放在`data`：

~~~text
data/
├─ MG1.csv
├─ R6g1.csv
├─ R6g1_MG1.csv
└─ R6g1_MG2.csv
~~~

执行扩充后，脚本在`data_augmented`生成同名训练CSV：

~~~text
data_augmented/
├─ MG1.csv
├─ R6g1.csv
├─ R6g1_MG1.csv
├─ R6g1_MG2.csv
├─ augmentation_manifest.csv
└─ augmentation_config.json
~~~

`python train.py`和`python run_all.py`默认读取`data_augmented`，不直接读取原始`data`。

文件角色与三分类标签如下：

| 文件 | 角色 | 类别编号 | 类别名称 |
|---|---|---:|---|
| MG1.csv | mg | 0 | 仅MG1 |
| R6g1.csv | r6g | 1 | 仅R6G |
| R6g1_MG1.csv | both | 2 | 两者都有 |
| R6g1_MG2.csv | both | 2 | 两者都有 |

原始数据共2001条独立测量光谱，每条包含524个波数点：

| 类别 | 样本数 |
|---|---:|
| 仅MG1 | 500 |
| 仅R6G | 499 |
| 两者都有 | 1002 |

CSV格式为：

| 位置 | 第1列 | 第2列 | 第3列及之后 |
|---|---|---|---|
| 第一行 | 编号列标题 | 浓度列标题 | 数字波数轴 |
| 第二行起 | 样本编号 | 浓度文字 | 对应波数上的光谱强度 |

示例：

~~~csv
编号,浓度,600.85,603.14,605.43,607.72
1,MG1 10^-2,123.4,125.1,129.7,126.0
2,MG1 10^-2,122.7,124.8,130.2,126.5
~~~

浓度文字用于分组拆分和分浓度统计，不作为网络输入。类别由文件角色确定，同一文件只保存同一类别的样品。

## 5. 数据扩充

默认扩充命令：

~~~powershell
python augment_dataset.py
~~~

每条原始光谱保留1份，并生成4份彼此独立的增强光谱。每份增强光谱都从原始光谱重新开始，默认重复5轮增强链；每轮对三种操作分别进行一次独立概率抽取，默认执行概率均为0.5，执行顺序固定为：

~~~text
1. 波数轴平移：抽中后从-2、-1、0、1、2中等概率选择，边界最近值填充
2. 强度变化：抽中后整条光谱乘以[0.98, 1.02]均匀分布随机倍率
3. 加性高斯噪声：抽中后逐点加入零均值噪声，标准差为当前光谱标准差的1%
~~~

高斯噪声在每个波数点独立采样，期望值为0，因此不会有意整体抬高或降低光谱。它用于模拟探测器、电路和测量环境造成的小幅随机波动，使模型减少对单个尖点和微小数值差异的依赖，提高对重复测量噪声的鲁棒性。噪声过大会淹没低浓度特征峰，因此默认只取当前光谱标准差的1%，并且每轮仅有0.5概率执行。

`--steps`控制每份增强光谱重复增强链的轮数，默认5；`--copies`控制每条原始光谱生成的增强份数，默认4。两者含义不同。原始版本编号为`00023_0`，4份增强版本编号为`00023_1`～`00023_4`。

随机种子固定为66。同一输入和同一参数重复运行会覆盖同名扩充文件，并生成一致的数据。实际数据量为：

| 数据 | 数量 | 统计含义 |
|---|---:|---|
| 独立原始光谱 | 2001 | 真实实验样本量 |
| 新生成扩充光谱 | 8004 | 2001×4 |
| 扩充CSV总行数 | 10005 | 2001×5 |

`augmentation_manifest.csv`逐行记录原始文件、原始行号、缺失点修复数、扩充编号、三种操作的实际执行次数和5轮完整操作轨迹；`augmentation_config.json`记录随机种子、概率、幅度、轮数、数量及原始文件SHA-256指纹。

完整显式命令：

~~~powershell
python augment_dataset.py --copies 4 --steps 5 --shift-prob 0.5 --gain-prob 0.5 --noise-prob 0.5 --shift-points 2 --gain-min 0.98 --gain-max 1.02 --noise-ratio 0.01 --seed 66
~~~

训练期间不再随机改变光谱。离线扩充CSV是固定实验输入，可直接归档和复核。

自定义扩充文件路径命令：

~~~powershell
python train.py --data mg=data_augmented/MG1.csv --data r6g=data_augmented/R6g1.csv --data both=data_augmented/R6g1_MG1.csv --data both=data_augmented/R6g1_MG2.csv
~~~

手动指定`--data`后，mg、r6g、both三个类别必须同时存在。

## 6. 模型与输入通道

四个模型共用同一网络主体，仅输入通道不同：

| 模型 | 输入通道 | 参数量 |
|---|---|---:|
| 模型1 | 标准化原始光谱 | 72,088 |
| 模型2 | 原始 + SNV | 72,256 |
| 模型3 | 原始 + SNV + 一阶导数 | 72,424 |
| 模型4 | 原始 + SNV + 一阶导数 + 二阶导数 | 72,592 |

模型3的主要结构为：

~~~text
[B,3,524]
  → Conv1d(3→24, kernel=7) + BatchNorm + GELU
  → 多尺度残差块(24→32) + MaxPool
  → 多尺度残差块(32→48) + MaxPool
  → 多尺度残差块(48→64)
  → 平均池化 + 最大池化 + 注意力池化
  → 全连接层[B,64]
  → 三分类输出层[B,3]
  → Softmax类别概率
~~~

三个Softmax概率依次表示：

~~~text
P(仅MG1), P(仅R6G), P(两者都有)
~~~

三个概率之和为1，最大概率对应最终类别。成分存在概率按下式派生：

~~~text
P(MG1存在) = P(仅MG1) + P(两者都有)
P(R6G存在) = P(仅R6G) + P(两者都有)
~~~

模型概率是分类分数，不表示化学浓度或混合比例。

## 7. 数据拆分与预处理

默认拆分参数为：

~~~text
训练目标比例：0.65
验证目标比例：0.15
测试目标比例：0.20
连续采集块大小：10
随机种子：66
~~~

程序根据扩充样本编号恢复其原始光谱来源，在每个“源文件 × 浓度”内部将连续10条原始光谱组成一个采集块，再按固定随机种子分配整块。同一原始光谱及其4份扩充版本绝不会跨集合。

来源绑定只决定集合归属：同一原始光谱及其4份扩充版本绝不会跨集合。划分完成后，每一行均作为一个模型输入，独立参与对应集合的损失、准确率、F1和混淆矩阵计算，不按父光谱合并。当前实际使用数量为：

| 集合 | 样本数 | 用途 |
|---|---:|---|
| 训练集 | 6010 | 1202条父光谱×5，用于拟合预处理和训练 |
| 验证集 | 2000 | 400条父光谱×5，选择最佳权重和执行早停 |
| 测试集 | 1995 | 399条父光谱×5，权重固定后的最终评价 |

因此，测试混淆矩阵全部单元格之和应为1995。这里的1995是“评价行数”，对应399条独立测量父光谱及其扩充版本，不能写成1995次独立实验测量。

预处理顺序为：

1. 检查波数轴和强度数值；
2. 修复少量内部缺失点；
3. 必要时插值到第一份文件的参考波数轴；
4. 根据原始光谱来源分组拆分，阻止扩充版本跨集合；
5. 每个父光谱的全部5个版本进入同一集合，集合内各行独立参与计算；
6. 仅使用训练集拟合逐波数点均值和标准差；
7. 按模型编号生成原始标准化、SNV和导数通道；
8. 训练期间按固定CSV逐批读取，不再执行随机增强。

## 8. 工程自检

自检不读取真实实验数据，也不写入工程的`pt`和`output`目录：

~~~powershell
python check.py
~~~

自检内容包括：

- 源文件和Python语法；
- 四种通道的输入维度和网络参数量；
- CSV读取、缺失点修复和波数插值；
- 离线五倍扩充、来源追踪和防泄漏拆分；
- 预处理保存/恢复；
- 三分类指标、混淆矩阵和预测表；
- 四模型各1轮训练与结果汇总；
- 权重重新加载和未知样品预测；
- 数据或参数变化后禁止复用旧结果。
- 极小损失下仍更新最低损失权重、累计微小改善、平局和非法损失；
- 测试时恢复的权重、最佳轮次、训练记录和摘要保持一致。

## 9. 最终模型3训练

模型3为最终默认模型。正式训练命令为：

~~~powershell
python train.py
~~~

完整等价命令为：

~~~powershell
python train.py --model 3 --epochs 120 --batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --dropout 0.15 --patience 20 --min-delta 0.0001 --grad-clip 5 --val-fraction 0.15 --test-fraction 0.20 --group-size 10 --preprocess-clip 10 --seed 66 --device auto --num-workers 0
~~~

正式训练前的单轮流程检查命令：

~~~powershell
python train.py --epochs 1
~~~

单轮检查和正式训练均写入默认目录。后执行的正式训练覆盖单轮检查生成的同名文件：

~~~text
pt/model3/best.pt
output/model3/*
~~~

模型根据带类别权重的三分类交叉熵更新参数，使用AdamW、余弦退火和梯度裁剪。权重选择和早停分别维护两个损失值：

- **最低验证损失**：只要本轮验证交叉熵严格低于历史最低值，就复制本轮权重；相同值保留最早轮次，不受`--min-delta`限制。
- **早停参考损失**：只有本轮损失比上次有效改善的参考值下降**超过**`--min-delta`，才更新参考值并重置耐心计数；否则计数加1。连续20轮无有效改善时停止。

因此，即使损失已小于`1e-4`，更低损失仍会更新权重；小幅改善还可以相对同一个早停参考值累积。训练停止后恢复所有已完成轮次中验证损失最低的权重，再评价测试集。默认早停阈值保持`1e-4`，其含义仅为是否继续等待，不能当作权重更新门槛。

训练记录新增权重更新标记、早停有效改善标记、历史最低损失、早停参考值和连续无有效改善轮数；摘要、权重和四模型对比表分别记录最佳轮次与早停参考轮次。终端使用有效数字显示微小损失，训练图标记实际保留权重的轮次。

新实验记录`training_policy_version=2`。旧格式6权重仍可用于预测；旧策略实验不能通过`--skip-existing`复用为新策略结果。旧权重只保存了当时保留的一轮，无法仅靠训练记录恢复第26轮等未保存参数，必须重新训练才能生成新策略的真实结果。

## 10. 四模型通道递增实验

完整四模型实验命令：

~~~powershell
python run_all.py
~~~

完整显式命令：

~~~powershell
python run_all.py --epochs 120 --batch-size 32 --learning-rate 0.0003 --weight-decay 0.0001 --dropout 0.15 --patience 20 --min-delta 0.0001 --grad-clip 5 --group-size 10 --val-fraction 0.15 --test-fraction 0.20 --preprocess-clip 10 --seed 66 --device auto --num-workers 0
~~~

四模型单轮流程检查：

~~~powershell
python run_all.py --epochs 1
~~~

数据、参数、任务版本和结果文件均一致时复用完整结果：

~~~powershell
python run_all.py --skip-existing
~~~

`--skip-existing`只复用已经完整结束且训练策略版本一致的实验，不执行断点续训。未使用该选项时，模型1～4及汇总结果按同名文件覆盖。

四模型均使用完全相同的数据拆分和训练参数，每个模型独立随机初始化。相邻模型的比较含义为：

~~~text
模型2 - 模型1：SNV的增量贡献
模型3 - 模型2：一阶导数的增量贡献
模型4 - 模型3：二阶导数的增量贡献
~~~

## 11. 未知样品预测

未知样品CSV格式与训练数据相同，第二列记录`unknown`或实验编号。默认加载模型3权重：

~~~powershell
python predict.py --input data/unknown.csv
~~~

显式指定权重和输出文件：

~~~powershell
python predict.py --input data/unknown.csv --checkpoint pt/model3/best.pt --output output/predict/unknown_预测.csv
~~~

CPU预测命令：

~~~powershell
python predict.py --input data/unknown.csv --device cpu
~~~

预测程序自动恢复训练时保存的参考波数轴、标准化参数、通道组合和网络权重，不重新拟合未知数据。

## 12. 输出文件

单次模型3训练生成：

~~~text
pt/
└─ model3/
   └─ best.pt

output/
└─ model3/
   ├─ training_config.json
   ├─ test_metrics.json
   ├─ 结果摘要.txt
   ├─ tables/
   │  ├─ dataset_manifest.csv
   │  ├─ 数据拆分清单.csv
   │  ├─ training_history.csv
   │  ├─ 训练记录.csv
   │  ├─ test_predictions.csv
   │  ├─ 测试集逐条预测.csv
   │  ├─ test_metrics_by_concentration.csv
   │  ├─ 按浓度测试结果.csv
   │  ├─ test_metrics_by_role.csv
   │  └─ 按类别测试结果.csv
   └─ figures/
      ├─ training_history.png
      └─ test_confusion_matrix.png
~~~

运行四模型实验后另外覆盖生成：

~~~text
output/compare/
├─ tables/
│  ├─ ablation_summary.csv
│  ├─ 四模型对比结果.csv
│  ├─ ablation_by_concentration.csv
│  ├─ 各浓度对比结果.csv
│  ├─ ablation_by_role.csv
│  └─ 各类别对比结果.csv
└─ figures/
   └─ ablation_summary.png
~~~

`best.pt`保存任务格式版本、训练策略版本、模型结构参数、权重、训练集预处理参数、参考波数轴、类别顺序和训练指标。`training_config.json`保存完整参数、最低损失/早停监控摘要和数据文件SHA-256指纹。

## 13. 结果阅读顺序

### 13.1 结果摘要

~~~powershell
Get-Content -Encoding UTF8 output\model3\结果摘要.txt
~~~

摘要包含实际完成轮数、是否触发早停、最低损失轮次及数值、早停参考轮次及数值、测试交叉熵、完全正确率、三类别平均F1、成分平均F1和MG1/R6G召回率。

### 13.2 训练曲线

`output/model3/figures/training_history.png`左侧显示训练和验证交叉熵，右侧显示验证集正确率、三类别平均F1和成分平均F1。验证损失持续上升而训练损失继续下降表示过拟合风险。

### 13.3 混淆矩阵

`output/model3/figures/test_confusion_matrix.png`为3×3矩阵，类别顺序为仅MG1、仅R6G、两者都有。主对角线表示正确分类，非对角线表示具体误判方向。默认扩充和拆分参数下，矩阵全部单元格合计为1995，即测试集逐行评价数量；它们来自399条独立测量父光谱。

### 13.4 逐条预测

`output/model3/tables/测试集逐条预测.csv`包含：

~~~text
仅MG1类别概率
仅R6G类别概率
两者都有类别概率
MG1存在概率
R6G存在概率
预测含MG1
预测含R6G
预测类别
真实类别
是否正确
~~~

逐条预测用于定位错误样品、低置信度样品、原始文件和CSV行号。

### 13.5 分浓度和分类别结果

`按浓度测试结果.csv`用于检查10^-2～10^-6各浓度的召回和误报；`按类别测试结果.csv`用于检查仅MG1、仅R6G和混合类的表现。

### 13.6 四模型对比

`output/compare/tables/四模型对比结果.csv`用于比较最佳验证损失、测试交叉熵、三分类正确率、三类别平均F1和成分指标。模型版本依据验证集结果确定，测试集用于最终报告。

## 14. 常用命令

~~~powershell
# 查看离线扩充参数
python augment_dataset.py --help

# 查看训练参数
python train.py --help

# 查看四模型实验参数
python run_all.py --help

# 查看预测参数
python predict.py --help

# 工程自检
python check.py

# 从2001条原始光谱生成10005行离线扩充CSV
python augment_dataset.py

# 最终默认模型3训练
python train.py

# 分别训练模型1～4
python train.py --model 1
python train.py --model 2
python train.py --model 3
python train.py --model 4

# 四模型通道递增实验
python run_all.py

# 复用数据和参数完全一致的完整结果
python run_all.py --skip-existing

# 未知样品预测
python predict.py --input data/unknown.csv

# 强制使用CPU训练或预测
python train.py --device cpu
python predict.py --input data/unknown.csv --device cpu

~~~

## 15. 结果评价原则

模型评价同时考虑以下指标：

1. 最佳验证交叉熵：用于选择模型版本，越低越好；
2. 三分类完全正确率：表示测试样品分类正确比例；
3. 三类别Macro-F1：三个类别等权平均，避免混合类样本较多造成掩盖；
4. 成分平均F1：MG1和R6G存在判断的平均F1；
5. 召回率：衡量漏检情况；
6. 误报率：衡量不存在成分被错误检出的情况；
7. 分浓度指标：检查低浓度性能是否下降；
8. 混淆矩阵：分析具体类别之间的误判方向；
9. 训练曲线：检查收敛、波动和过拟合。

离线扩充后的10005行不能表述为10005条独立实验样本；独立测量样本量仍为2001条。验证集和测试集包含其父光谱被分入对应集合的原始行与扩充行，报告中的准确率和混淆矩阵属于“扩充行级评价”，不能等同于独立盲测性能。内部连续块与来源绑定拆分用于避免扩充版本跨集合的数据泄漏，但不能代替独立配液批次、不同日期或不同仪器上的盲测。当前模型属于封闭集合三分类模型，输入范围限定为仅MG1、仅R6G或两者混合。
