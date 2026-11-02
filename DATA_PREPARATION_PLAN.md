# 全量离线增强与父实验分组实现说明

> 状态：已实现。当前流程固定为“逐文件增强全部原始实验 → 按父实验整体划分 → 输出训练、验证、测试三份CSV”。训练端不再读取旧的 `robustness.csv`，也不接受旧版准备数据协议。

## 1. 原始数据含义

原始CSV中：

- 文件名描述样本来源和类别；
- 第二列是行描述，当前主要表示浓度；
- 第一列是实验序号，表示同类条件下的第几次实验；
- 每一行是一条独立原始光谱。

程序为每一行建立可读且稳定的父实验ID：

```text
experiment_id = 文件名（无扩展名）:行描述:实验序号
```

例如：

```text
MG1:10^-2:1
R6g1:R6G 10^-3:127
```

`source_row` 和 `source_csv_line` 另外保存实际行位置。即使未来实验序号重复，程序也会检查完整父实验ID是否唯一并拒绝含糊数据。

## 2. 新的两阶段流程

一个 `augment_dataset.py` 命令依次完成两个阶段：

```text
data/四个原始CSV
        │
        ▼
阶段一：每个父实验生成原谱和全部增强副本
        │
        └─ data_augment/
           ├─ MG1.csv
           ├─ R6g1.csv
           ├─ R6g1_MG1.csv
           ├─ R6g1_MG2.csv
           ├─ augmentation_manifest.csv
           └─ augmentation_config.json
        │
        ▼
阶段二：只对父实验ID进行70/15/15分组
        │
        └─ data_prepared/
           ├─ train.csv
           ├─ validation.csv
           ├─ test.csv
           ├─ split_manifest.csv
           └─ preparation_config.json
```

阶段二从 `data_augment/` 读取实际增强文件，不在内存中绕过该目录。这样既能直接检查每个源文件的增强结果，也能保证训练使用的三份CSV确实来自已保存的增强数据。

## 3. 增强规则

默认每条原谱生成4份独立副本。每份副本都从原谱重新开始，连续执行5轮：

```text
水平平移 → 强度增益 → 高斯噪声
```

| 参数 | 默认值 |
|---|---:|
| `copies` | 4 |
| `steps` | 5 |
| `shift_prob` | 0.5 |
| `gain_prob` | 0.5 |
| `noise_prob` | 0.5 |
| `shift_points` | 每轮±2点 |
| `gain` | 每轮0.97～1.03 |
| `noise_ratio` | 当前光谱标准差的0.010 |
| 平移边界 | 最近边界值填充，不循环 |

增强种子由以下内容经SHA-256稳定派生：

```text
augmentation_seed + experiment_id + augmentation_index + offline_full_dataset
```

因此改变文件遍历顺序不会改变某个父实验的指定增强副本。`augmentation_manifest.csv` 为每一行保存五轮操作轨迹、净平移、累计增益、最大噪声标准差和父实验ID。

## 4. 分组规则

程序只使用每个父实验的 `augmentation_index=0` 原谱记录进行抽签，然后把拆分结果映射给该父实验的全部版本：

```text
父实验 A
├─ aug_0 原谱
├─ aug_1
├─ aug_2
├─ aug_3
└─ aug_4

以上5行只能共同进入 train、validation 或 test 中的一个集合。
```

默认比例是70%/15%/15%。程序用scikit-learn两阶段 `train_test_split`：

1. 从父实验中拆出70%训练和30%暂存；
2. 再把暂存父实验1:1分成验证和测试；
3. 两次都按 `source_file × concentration_group` 联合分层；
4. 最后把每个父实验的全部增强行整体带入对应集合。

拆分种子和增强种子相互独立：

```text
split_seed          决定父实验进入哪个集合
augmentation_seed   决定每份增强谱的具体扰动
```

如果任一联合层样本太少，程序直接停止并显示最小层计数，不静默改成无分层拆分。

## 5. 默认命令

```powershell
python augment_dataset.py
```

完整等价命令：

```powershell
python augment_dataset.py `
  --source-dir data `
  --augmented-dir data_augment `
  --prepared-dir data_prepared `
  --train-ratio 0.70 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --split-seed 66 `
  --augmentation-seed 66 `
  --copies 4 `
  --steps 5 `
  --shift-prob 0.5 `
  --gain-prob 0.5 `
  --noise-prob 0.5 `
  --gain-min 0.97 `
  --gain-max 1.03 `
  --shift-points 2 `
  --noise-ratio 0.010
```

两个输出目录任一已存在时，程序默认拒绝覆盖。确认替换后：

```powershell
python augment_dataset.py --overwrite
```

运行时只打印必要进度：

```text
[1/3] 读取原始数据和父实验数量
[2/3] 逐文件增强；显示4个源文件各自的父实验数和生成行数
[3/3] 按父实验整体分组并写出三份集合
完成：显示父实验总数、增强后总行数和三个集合行数
```

无增强对照仍使用相同脚本和分组算法：

```powershell
python augment_dataset.py `
  --copies 0 `
  --augmented-dir data_plain_augment `
  --prepared-dir data_plain
```

此时 `data_plain_augment/` 中的四个对应文件只含原谱，`data_plain/` 包含三份无增强集合。

## 6. CSV字段

`data_augment/` 的四个对应文件包含：

```text
sample_id, experiment_id, source_file, source_row, source_csv_line,
original_sample_id, row_description, role, class_id,
concentration_text, concentration_group, augmentation_index,
is_augmented, repaired_points, <波数强度列>
```

`data_prepared/` 的三份集合在上述字段中增加 `split`。其中：

- `experiment_id` 是不可拆分的父实验组；
- `sample_id` 唯一标识父实验的某个版本；
- `augmentation_index=0` 表示原谱；
- `augmentation_index=1..copies` 表示对应增强副本；
- `is_augmented` 与增强编号必须一致。

## 7. 当前数据的预期数量

当前四个原始文件合计2001个父实验。默认4份增强副本后：

| 目录或文件 | 父实验数 | 原谱行 | 增强行 | 总行数 |
|---|---:|---:|---:|---:|
| `data_augment/`合计 | 2001 | 2001 | 8004 | 10005 |
| `train.csv` | 1400 | 1400 | 5600 | 7000 |
| `validation.csv` | 300 | 300 | 1200 | 1500 |
| `test.csv` | 301 | 301 | 1204 | 1505 |

实际数量以 `preparation_config.json` 为准。70/15/15针对父实验数计算，最终CSV行数等于父实验数乘以 `copies+1`。

## 8. 自动校验

增强阶段检查：

1. 四个对应CSV和两份追踪文件完整；
2. 波数轴一致且严格递增；
3. 每个父实验恰有1条原谱和完整增强编号；
4. `sample_id` 全局唯一；
5. 类别、源文件名、布尔标记和光谱数值有效；
6. 增强清单与四个文件逐行对应；
7. 所有输出文件SHA-256与配置一致。

分组阶段检查：

1. train、validation、test都包含三个类别；
2. 三个集合的父实验ID两两不交叉；
3. 每个集合内的每个父实验都有完整原谱和增强副本；
4. `split_manifest.csv` 每个父实验一行并与三份CSV一致；
5. 数据文件哈希与 `preparation_config.json` 一致；
6. 不同父实验的完全相同原谱若跨集合会记录警告。

写入采用临时目录。全部校验通过后才替换目标目录，避免保留半成品。

## 9. 与训练端的连接

`train.py` 现在直接读取：

```text
data_prepared/train.csv
data_prepared/validation.csv
data_prepared/test.csv
```

训练端的变化：

- 不再查找或评价 `robustness.csv`；
- 允许验证集和测试集包含增强行；
- 预处理中心和尺度只使用整个训练CSV拟合；
- 验证集用于早停和模型选择；
- 测试集只在模型锁定后报告；
- 指标同时记录三个集合的原谱数、增强数、总行数和独立父实验数；
- 准备配置、拆分清单、增强来源文件哈希和代码指纹写入训练结果。

旧版 `data_prepared/` 的方法标记不是 `augment_all_then_group_split` 时，训练会拒绝读取，防止旧结果与新协议混用。

## 10. 评价口径限制

新流程满足“同一父实验的原谱和增强副本不跨集合”，因此不会发生同一原谱的派生版本同时进入训练和测试的直接泄漏。

但验证和测试中的增强行不是新的独立实验。默认Accuracy、Macro-F1和混淆矩阵按CSV行计算，因此一个测试父实验贡献5次高度相关评价。报告结果时必须同时写明：

```text
测试集：301个独立父实验，1505条光谱行（每个父实验1条原谱+4条增强谱）
```

这种指标衡量的是模型对父实验及预设扰动的综合稳定性，不能把1505写成1505次独立实验。若论文需要严格的原始实验泛化结论，应另外从 `test.csv` 筛选 `augmentation_index=0` 报告父实验原谱指标，或增加真正独立的新批次盲测。

## 11. 参考依据与本工程边界

- 随机分层实现使用 [scikit-learn `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。
- 父实验级隔离依据 [scikit-learn数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html) 和 [Braindecode按采集实体拆分示例](https://braindecode.org/stable/auto_examples/datasets_io/plot_split_dataset.html)。本工程的关键适配是把“文件名+行描述+实验序号”作为分组实体。
- 一维信号增强的工程组织参考 [`torch_ecg`](https://github.com/DeepPSP/torch_ecg) 与 [tsai transforms](https://timeseriesai.github.io/tsai/data.transforms.html)。位移、增益、噪声的具体顺序、幅度、五轮设置和逐行追踪均为本工程协议。

大多数防泄漏建议更常见的做法是先拆分父实验、只增强训练集。当前项目根据新的明确要求改为先生成全部增强数据，再按父实验组整体拆分。两种实现都必须保证同一个父实验及其派生版本不跨集合；当前协议还需承担“验证和测试含非独立增强行”的解释限制。
