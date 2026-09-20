# MG1 / R6G 拉曼光谱分类工程

本工程从原项目的最新分类脚本中提取，专门判断光谱中是否含有MG1、R6G。默认使用模型3，即“原始StandardScaler光谱 + SNV + 一阶导数”，同时保留四个通道版本的训练、对比和未知样品预测。

工程没有浓度回归任务，也没有复制原始实验数据、历史权重和输出图表。整个 R6gMg_classify 文件夹可以直接复制到其他位置使用，不依赖上级项目。

- 本文件说明工程结构、安装、运行、模型本体和结果位置。
- [EXPERIMENTS.md](EXPERIMENTS.md) 详细说明四次递进通道实验的思想、实现、操作、历史结果及结论边界。

## 1. 文件结构

~~~text
R6gMg_classify/
├─ models/              模型结构代码，仅依赖PyTorch
│  ├─ __init__.py        对外提供统一导入入口
│  ├─ blocks.py          SE注意力、多尺度残差模块
│  └─ net.py             完整网络与前向传播
├─ config.py            标签、默认文件名、通道版本、公共设置
├─ data.py              CSV读取、修复、拆分、预处理与增强
├─ train.py             单模型训练，默认模型3
├─ predict.py           加载权重，预测未知光谱
├─ run_all.py           顺序训练四模型、检查拆分、汇总结果
├─ metrics.py           推理、阈值搜索、分类指标和绘图
├─ check.py             使用临时合成数据做完整性自检
├─ requirements.txt     依赖版本
├─ README.md            工程使用说明
├─ EXPERIMENTS.md       四模型实验实现与研究思想
├─ .gitignore           忽略数据、权重、结果与缓存
├─ data/                留空：放入实验CSV
├─ pt/                  留空：训练后生成.pt权重
└─ output/              留空：训练后生成表格、图片与摘要
~~~

models保存模型结构代码；pt保存训练出来的参数文件。两者职责不同。全工程统一采用 pt 文件夹、PT_DIR 配置常量、pt_dir 参数字段和 --pt-dir 命令行选项，权重后缀统一为 .pt。

data、pt、output交付时没有任何占位文件。Git不记录空文件夹；通过Git重新取得代码后，可自行新建data，训练程序会自动建立需要的pt和output子目录。

## 2. 安装环境

推荐使用Python 3.12。本机核验环境为Python 3.12.12、PyTorch 2.6.0、NumPy 2.3.4、pandas 2.3.3、scikit-learn 1.7.2、Matplotlib 3.10.6；requirements.txt按这些直接依赖固定版本。

在PyCharm中打开 R6gMg_classify 文件夹，选择已有PyTorch解释器，在该目录的终端执行：

~~~powershell
python --version
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
~~~

如果需要独立Conda环境：

~~~powershell
conda create -n spectra python=3.12
conda activate spectra
python -m pip install -r requirements.txt
~~~

默认 --device auto 会在CUDA可用时使用显卡，否则使用CPU；无显卡也能运行。已安装的torch 2.6.0+cu124满足torch==2.6.0。固定依赖有助于复现，但不保证不同硬件、计算后端的浮点运算逐位相同。

## 3. 先运行工程自检

不需要放入真实数据，就可以执行：

~~~powershell
python check.py
~~~

自检会确认源文件和文档完整、代码能导入、四种输入维度正确，并在系统临时目录建立一份独立源码副本和小型合成数据，完成四模型各1轮训练、结果汇总、权重加载和预测。副本所在路径包含中文和空格，用于检查路径兼容性。

自检还覆盖少量缺失点修复、实际CSV行号、预处理保存/恢复、阈值、空白标签、数据更换后禁止复用旧结果等。检查完成后自动清理临时文件，不会读取或改写工程data中的真实样品，也不会向工程pt/output写入检查结果。

自检是软件检查，不是检测性能实验，不能将合成数据分数写成真实样品结果。

## 4. 准备实验数据

默认读取以下四份文件，名称与原项目一致：

~~~text
data/
├─ MG1.csv
├─ R6g1.csv
├─ R6g1_MG1.csv
├─ R6g1_MG2.csv
└─ blank.csv          可选：真实空白样品
~~~

| 文件 | 角色 | 标签[MG1,R6G] |
|---|---|---|
| MG1.csv | mg | [1,0] |
| R6g1.csv | r6g | [0,1] |
| R6g1_MG1.csv | both | [1,1] |
| R6g1_MG2.csv | both | [1,1] |
| blank.csv | none | [0,0] |

四个默认目标文件必须齐全。存在blank.csv时自动加入；unknown.csv之类的未知样品不会自动加入训练。混合比例不由程序估计，两个混合文件都标记为both；具体比例及试剂先后顺序应以配液记录为准。

CSV采用原工程布局：

| 位置 | 第1列 | 第2列 | 第3列及之后 |
|---|---|---|---|
| 第一行 | 编号列标题 | 浓度列标题 | 数字波数轴 |
| 第二行起 | 样本编号 | 浓度文字 | 对应波数上的光谱强度 |

以下只演示格式，实际数据使用完整波数序列：

~~~csv
编号,浓度,600.85,603.14,605.43,607.72
1,MG1 10^-2,123.4,125.1,129.7,126.0
2,MG1 10^-2,122.7,124.8,130.2,126.5
~~~

第二列建议包含10^-2、10^-3、10^-4、10^-5、10^-6，空白可以写blank。浓度文字只用于拆分和统计，不输入神经网络；类别由--data指定的角色决定，同一文件不能混写不同存在状态的样品。

自定义文件名时，可以明确传入：

~~~powershell
python train.py --data mg=data/MG1.csv --data r6g=data/R6g1.csv --data both=data/R6g1_MG1.csv --data both=data/R6g1_MG2.csv
~~~

有空白时在命令末尾加 --data none=data/blank.csv。手动使用--data后，只使用手动列出的文件，不再自动补入默认文件或空白。

## 5. 第一次训练的步骤

以下命令均在 R6gMg_classify 目录执行。

### 第一步：用真实数据做短流程检查

~~~powershell
python train.py --epochs 1 --pt-dir pt_check --output-dir output_check
~~~

确认数据读取、训练、权重和输出均正常。短训练的分数不用于正式结论。检查使用独立目录，不覆盖正式模型。

### 第二步：正式训练模型3

~~~powershell
python train.py
~~~

等价于：

~~~powershell
python train.py --model 3 --epochs 120 --batch-size 32 --seed 66
~~~

正常完成后，权重位于pt/model3/best.pt，结果位于output/model3。满足早停条件时，不必跑满120轮。

### 第三步：查看结果

按以下顺序阅读：

1. output/model3/结果摘要.txt：最佳轮次、验证和测试损失、阈值、主要指标。
2. output/model3/figures/training_history.png：训练与验证曲线。
3. output/model3/figures/test_confusion_matrix.png：预测类别和真实类别的对应。
4. output/model3/tables/按浓度测试结果.csv：检查低浓度召回与误报。
5. output/model3/tables/测试集逐条预测.csv：定位具体错误光谱。

### 第四步：运行四模型实验

~~~powershell
python run_all.py
~~~

程序按模型1→2→3→4的顺序训练。每个模型都从头初始化；不是在上一模型的权重上继续训练。全部完成后检查数据拆分完全一致，并生成output/compare下的汇总表与对比图。

若部分模型已经完成，同样参数下可运行：

~~~powershell
python run_all.py --skip-existing
~~~

仅当权重、结果齐全且输入文件路径、文件内容指纹和训练参数一致时复用已完成模型。这不是断点续训；如果主动修改了网络或预处理代码，应换输出目录并重新训练，不使用该选项。

### 第五步：预测未知样品

将同格式unknown.csv放入data：

~~~powershell
python predict.py --input data/unknown.csv
~~~

默认使用pt/model3/best.pt，预测表保存到output/predict/unknown_预测.csv。未知样品第二列可填写unknown。

指定其他模型或保存位置：

~~~powershell
python predict.py --input data/unknown.csv --checkpoint pt/model2/best.pt --output output/predict/model2_预测.csv
~~~

可以直接加载原分类工程的best_model.pt，只需用--checkpoint指定它。网络层和权重字典键名保留，因此拆到models文件夹不需要转换已有权重。工程没有附带旧权重；仅加载自己训练或信任来源的文件。

## 6. 模型本体在哪里、怎样使用

网络主体位于[models/net.py](models/net.py)，类名为DualPresenceMSNet；基础模块位于[models/blocks.py](models/blocks.py)。统一导入：

~~~python
import torch
from models import DualPresenceMSNet

model = DualPresenceMSNet(in_channels=3)  # 默认就是模型3的三个通道
model.eval()
with torch.no_grad():
    logits = model(torch.zeros(2, 3, 524))
    probabilities = torch.sigmoid(logits)
print(probabilities.shape)  # torch.Size([2, 2])
~~~

该例只演示结构。随机初始化模型的输出没有化学检测意义；实际预测要加载训练好的权重，并执行一致的光谱预处理。

模型3的主要尺寸：

| 阶段 | 输出形状，B表示批量大小 |
|---|---|
| 三通道输入 | [B,3,524] |
| 卷积核7的初始卷积、BN、GELU | [B,24,524] |
| 多尺度残差块1 | [B,32,524] |
| 最大池化 | [B,32,262] |
| 多尺度残差块2、最大池化 | [B,48,131] |
| 多尺度残差块3 | [B,64,131] |
| 平均/最大/注意力池化后拼接 | [B,192] |
| 共享全连接层 | [B,64] |
| MG1/R6G两个检测头 | [B,2] |

每个残差块包含卷积核3、7、15的并行分支、1×1卷积融合、残差连接、SE通道注意力和Dropout。模型3共72,359个参数。

模型只输出两个logit，不直接输出浓度，也不直接设置“有/无”阈值。训练损失在train.py，Sigmoid推理和阈值搜索在metrics.py，CSV处理在data.py。因此可以单独复制models文件夹到其他PyTorch工程使用。

## 7. 从数据到结果的完整流程

1. 读取CSV，检查波数和数值。降序波数会连同强度反转；其他文件范围能覆盖参考轴时，必要时线性插值对齐。
2. 修复少量内部缺失点，并记录修复数量。超过max(3, round(点数×1%))个，或首尾缺失时拒绝继续。沿用原脚本，内部修复按采样点序号插值；原CSV不会被改写。
3. 按“源文件×浓度×连续采集块”拆分。默认每10条一组；组数不足3时会提示逐行退化，样本少于3条无法三路拆分。
4. 仅使用训练集拟合逐波数点均值和标准差。验证、测试和预测复用这组参数。
5. 按指定版本生成原始标准化、SNV、一阶导数、二阶导数通道。只对训练集进行轻微平移、增益和噪声增强。
6. 前向传播得到两个logit；计算带正样本权重的二元交叉熵；反向传播、梯度裁剪、AdamW更新。每轮用余弦退火调整学习率。
7. 根据验证集未加权BCE保留权重。更新条件为“本轮损失 < 上次保留损失 - min_delta”，默认min_delta=0.0001，连续20轮没有有效改善则早停。
8. 恢复保留的权重，在验证集对每个试剂分别搜索0.05～0.95、步长0.005的F1最优阈值；并列时取最接近0.5的候选值。
9. 权重和阈值固定后评估测试集，保存完整结果及权重。测试集不参与参数训练、权重选择和阈值搜索。

注意：带min_delta的保留规则不保证选中曲线上数值最低但改善极小的那一轮；这与原实验一致。严格按最小损失选择可设--min-delta 0，应当作为新实验记录。

最佳权重训练时先保存在内存，训练结束评估后写入best.pt。强行中断可能没有本次完整权重；本基础工程不实现断点续训。

## 8. 训练后产生的文件

~~~text
pt/
└─ model3/
   └─ best.pt

output/
├─ model3/
│  ├─ training_config.json
│  ├─ test_metrics.json
│  ├─ 结果摘要.txt
│  ├─ tables/
│  │  ├─ 数据拆分清单.csv
│  │  ├─ 训练记录.csv
│  │  ├─ 测试集逐条预测.csv
│  │  ├─ 按浓度测试结果.csv
│  │  ├─ 按类别测试结果.csv
│  │  └─ 对应的英文列名CSV
│  └─ figures/
│     ├─ training_history.png
│     └─ test_confusion_matrix.png
└─ compare/                     运行run_all.py后生成
   ├─ tables/
   │  ├─ 四模型对比结果.csv
   │  ├─ 各浓度对比结果.csv
   │  ├─ 各类别对比结果.csv
   │  └─ 对应的英文列名CSV
   └─ figures/
      └─ ablation_summary.png
~~~

英文列名文件供程序自动汇总，中文文件方便直接阅读。单模型英文表分别为dataset_manifest.csv、training_history.csv、test_predictions.csv、test_metrics_by_concentration.csv、test_metrics_by_role.csv；它们与中文表是同一次实验结果。

best.pt包含模型结构参数、权重字典、预处理参数、参考波数轴、两个阈值、标签顺序和训练记录信息。training_config.json另保存数据文件SHA-256指纹。JSON键名为便于编程保持英文，中文摘要和表格供人工查看。

拆分清单同时提供“样本行序号”（第一条光谱为1）和“CSV文件行号”（包含标题行，第一条光谱为2），避免检查原文件时错一行。

## 9. 常用参数与路径规则

| 参数 | 默认值 | 用途 |
|---|---:|---|
| --model | 3 | 单模型通道版本 |
| --epochs / --batch-size | 120 / 32 | 最大轮数与批量大小 |
| --learning-rate / --weight-decay | 0.0003 / 0.0001 | AdamW设置 |
| --dropout | 0.15 | 特征丢弃比例 |
| --patience / --min-delta | 20 / 0.0001 | 早停和保留权重条件 |
| --group-size | 10 | 连续采集块大小 |
| --val-fraction / --test-fraction | 0.15 / 0.20 | 按块取整前的目标比例 |
| --seed | 66 | 拆分和训练种子 |
| --pt-dir / --output-dir | 工程pt / output | 权重与结果根目录 |
| --device / --num-workers | auto / 0 | 计算设备与数据加载进程 |

默认data/pt/output总是相对于工程位置；手动指定的相对路径相对于当前终端位置。数据路径会统一解析为绝对路径，避免同一文件的不同写法导致拆分对比失败。

组数取整会使实际比例偏离65%/15%/20%。原2001条数据在默认规则下为1202条训练、400条验证、399条测试，应以本次日志和清单为准。

相同模型、相同输出目录再次运行会覆盖该模型结果。保留新实验时应同时换权重和结果目录：

~~~powershell
python run_all.py --seed 42 --pt-dir pt_seed42 --output-dir output_seed42
~~~

## 10. 本次整理修正了什么

- 模型源代码集中到models，权重集中到pt，结果集中到output，名称和职责对应。
- 所有权重路径和参数统一使用pt；默认单模型、模型构造器和预处理器统一为模型3配置。
- 单模型与四模型共用训练参数定义，避免学习率、min_delta等设置遗漏或不一致。
- 默认数据名与现有四文件一致，删除过时的“数据尚未放入”描述和机器专用路径。
- 明确区分样本行序号与CSV实际行号；过短光谱、非法训练参数提前报中文错误。
- 增加验证/测试BCE到汇总表；复用结果前检查数据指纹与参数，降低混用实验的风险。
- 保留原网络层名、预处理计算、训练优化和阈值选择方法，已有分类权重可继续加载。

## 11. 结果解释与常见问题

没有真实空白时，程序仍能输出“无”，但该类别没有得到训练和测试；纯MG1是R6G的阴性、纯R6G是MG1的阴性，二者不能替代真实空白。此时三类全对而固定四类别Macro-F1为0.75，是缺失类按0参与平均的结果，不表示25%的样品错分。应结合实际类别Macro-F1、完全正确率和逐试剂指标分析。

内部的连续块拆分能减少相邻复测跨集合，但不是独立批次验证。低浓度样品识别正确，也不能直接宣布达到对应检出限。完整研究解释见[四模型实验说明](EXPERIMENTS.md)。

遇到图片中文方框，在metrics.py的字体列表中加入本机已安装的中文字体；CUDA不可用可先加--device cpu；找不到文件时检查data文件名或--data路径；训练尚未完成时不要把旧目录中的best.pt当作本次结果。

## 12. 本次完整性核验记录

2026-09-20整理完成后，已执行以下检查：

1. check.py全部通过：合成数据在独立临时副本中完成四模型各1轮训练、相同拆分检查、图表汇总、结果复用和默认模型预测。
2. 读取原2001条真实光谱，与旧脚本比较原始数组、标签和1202/400/399拆分，结果一致；原文件中的两个内部缺失点按原规则处理。
3. 在固定的确定性CPU检查设置下，四种输入的预处理结果及同权重网络的前向输出与旧实现一致，网络参数字典兼容。
4. 使用真实数据完成模型3单轮训练，验证新路径下的权重、摘要、表格与图像保存。
5. 使用原模型3正式权重在新预测入口读取499条纯R6G光谱，成功完成预测；没有把旧权重复制进本工程。

上述训练只用于软件检查，没有重新进行120轮正式性能实验。临时产物均已清理，data、pt、output保持空目录。正常训练仍使用原计算设置；特定检查中的确定性CPU设置不代表跨硬件或任意重新训练均可逐位复现。
