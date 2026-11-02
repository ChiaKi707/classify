# 原始实验拆分、离线增强与无增强对照实现说明

> 状态：已实现。当前训练入口接受本协议生成的增强目录 `data_prepared/` 或无增强目录 `data_plain/`，不再兼容旧的“先增强再由训练脚本拆分”流程。

## 1. 数据含义和目标

用户提供的原始 CSV 中，每一行是一次独立实验。实验含义由源文件名和该行原始编号共同确定：

```text
experiment_id = source_file_stem + ":" + original_sample_id
```

不同序号是不同实验；相同序号出现在不同源文件中也是不同实验。程序不再把连续 10 行或其他固定行数解释为同一采集组。

数据准备由一个脚本完成：

```text
读取原始 CSV
→ 校验编号、波数轴和强度
→ 建立独立 experiment_id
→ 随机分层拆分原始实验
→ 只对训练实验离线增强
→ 对测试实验生成独立鲁棒性副本
→ 输出光谱 CSV、拆分清单、增强清单和配置
```

当 `--copies 0` 时，拆分步骤完全相同，但跳过训练增强与鲁棒性增强，只输出三个干净主集合。

## 2. 为什么先拆分再增强

同一原谱的增强版本与原谱高度相关。如果先增强再按行随机拆分，原谱可能进入训练集，而它的轻微扰动版本进入验证或测试集，造成数据泄漏和过高的准确率。

当前协议为：

```text
独立原始实验
├─ train：原谱 + 4 份增强谱，更新模型权重
├─ validation：只含原谱，早停和选模
├─ test：只含原谱，最终主指标
└─ robustness：由 test 原谱生成的增强谱，只做压力测试
```

这与一维信号开源项目常用的“按 subject / recording / 原始序列先拆分，只增强训练部分”原则一致。以下链接用于说明设计依据，不表示本工程直接复制其代码：

- [scikit-learn：避免数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn：分层分组交叉验证](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Braindecode：按 subject、session 或 recording 拆分](https://braindecode.org/stable/auto_examples/datasets_io/plot_split_dataset.html)
- [tsai：时间序列拆分](https://timeseriesai.github.io/tsai/data.validation.html)
- [tsai：一维信号变换](https://timeseriesai.github.io/tsai/data.transforms.html)
- [torch_ecg：一维信号训练和增强](https://github.com/DeepPSP/torch_ecg)
- [raman-dl：拉曼分类的固定拆分与训练折增强](https://github.com/grenzo/raman-dl)
- [raman-open-ml：拉曼增强实验](https://github.com/JoulesSpace/raman-open-ml)

`robustness` 不是额外独立样本，不参与模型选择，也不能加入主测试样本量。

## 3. 随机拆分实现

默认比例是 70% / 15% / 15%。拆分使用 scikit-learn 的两阶段 `train_test_split`：

1. 先拆出 70% train 和 30% held-out；
2. 再把 held-out 按 1:1 分成 validation 和 test。

两个阶段都启用 shuffle 和联合分层：

```text
source_file × concentration_group
```

这样尽量同时保持：

- 仅 MG1、仅 R6G、两者都有三类比例；
- 两个混合物来源文件比例；
- 不同浓度组比例。

如果某个联合层样本太少，scikit-learn 无法完成三路分层时，脚本停止并打印最小层计数，不静默退化为无分层拆分。

两个随机种子职责分离：

```text
split_seed          决定原始实验属于 train / validation / test
augmentation_seed   决定增强操作是否执行及具体参数
```

每个增强版本的随机流由以下信息经 SHA-256 稳定派生：

```text
augmentation_seed + experiment_id + augmentation_index + purpose
```

`purpose` 区分训练增强和鲁棒性增强。改变文件遍历顺序不会改变同一个实验、同一增强编号的结果。

## 4. 保留的三种保守增强

增强方式、顺序和幅度保持不变，只把连续增强轮数固定为 2：

```text
每条训练原谱生成 4 份副本
每份副本从原谱重新开始
每份连续执行 2 轮：平移 → 增益 → 高斯噪声
```

| 参数 | 默认值 |
|---|---:|
| `copies` | 4 |
| `steps` | 2 |
| `shift_prob` | 0.5 |
| `gain_prob` | 0.5 |
| `noise_prob` | 0.5 |
| `shift_points` | ±1 点/轮 |
| `gain` | 0.98～1.02/轮 |
| `noise_ratio` | 当前谱标准差的 0.005 |
| 边界处理 | 最近边界值填充，不循环 |

验证和主测试原谱不增强。测试父谱用同样参数生成 `robustness.csv`，用于检查模型对已知小扰动是否稳定。

## 5. 单脚本命令

默认命令：

```powershell
python augment_dataset.py
```

完整等价命令：

```powershell
python augment_dataset.py `
  --source-dir data `
  --output-dir data_prepared `
  --train-ratio 0.70 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --split-seed 66 `
  --augmentation-seed 66 `
  --copies 4 `
  --steps 2 `
  --shift-prob 0.5 `
  --gain-prob 0.5 `
  --noise-prob 0.5 `
  --gain-min 0.98 `
  --gain-max 1.02 `
  --shift-points 1 `
  --noise-ratio 0.005
```

目标目录已存在时，默认拒绝覆盖；确认需要替换后执行：

```powershell
python augment_dataset.py --overwrite
```

无增强对照使用同一个入口：

```powershell
python augment_dataset.py --copies 0 --output-dir data_plain
```

## 6. 输出文件

```text
data_prepared/
├─ train.csv
├─ validation.csv
├─ test.csv
├─ robustness.csv
├─ split_manifest.csv
├─ augmentation_manifest.csv
└─ preparation_config.json
```

四份光谱 CSV 使用统一元数据字段：

```text
sample_id, experiment_id, source_file, source_row, source_csv_line,
original_sample_id, role, class_id, concentration_text,
concentration_group, split, augmentation_index, is_augmented,
repaired_points, <各波数强度列>
```

`split_manifest.csv` 每个独立原始实验一行；`augmentation_manifest.csv` 每个原始或增强版本一行，并保存：

- 两轮完整操作轨迹；
- 净平移量；
- 累计增益；
- 最大噪声标准差；
- 父实验、用途和增强编号。

`preparation_config.json` 保存拆分、增强、分布、源文件哈希、输出文件哈希和跨集合完全相同原谱警告。

无增强目录没有 `robustness.csv`；其 `augmentation_manifest.csv` 只记录三个集合的原谱，所有 `is_augmented` 均为 false，便于继续使用统一的追踪与哈希协议。

## 7. 写出前后的自动检查

程序先写临时目录，再调用与训练完全相同的 `load_prepared_data` 检查：

1. 三份主数据、两份清单和配置全部存在；增强模式另要求鲁棒性数据；
2. 波数列严格递增且所有集合一致；
3. 强度全部是有限数；
4. `role` 与 `class_id` 一致；
5. train、validation、test 的实验 ID 两两不交叉；
6. validation 和 test 只含未增强原谱；
7. 增强模式下 robustness 只含 test 父实验的增强版本；无增强模式不生成该文件；
8. train 每个实验恰有 1 条原谱和配置数量的增强谱；
9. 增强模式下 robustness 每个测试实验具有完整增强编号；
10. 三个主集合均包含三个类别；
11. 所有数据 CSV 的 `sample_id` 全局唯一；
12. `split_manifest.csv` 与实际数据逐实验一致；
13. `augmentation_manifest.csv` 与所有生成数据行一一对应；
14. 正式读取时再次核对输出文件 SHA-256。

脚本还检查不同实验是否存在完全相同的原始强度并跨越集合。发现时写入警告，不自动删除，因为根据当前实验定义，不同原始编号仍是不同实验。

## 8. 与训练端的连接

训练端已同步修改：

- `train.py` 直接读取三个主集合，并在存在时读取鲁棒性集合，不再随机二次拆分；
- 训练配置已删除 `group_size`、`val_fraction` 和 `test_fraction`；
- 标准化参数只用 train 中 `augmentation_index=0` 的干净原谱拟合；
- `run_experiments.py` 记录全部准备数据文件、配置和清单哈希；
- 同一次比较中的所有通道、种子和未来模型共享同一份准备数据目录；
- 矩阵根目录的 `split.csv` 直接复制 `split_manifest.csv`；
- 数据或代码变化后，旧结果不会通过严格的 `--skip-existing` 检查。

旧 `data_augmented/` 只是历史数据，当前训练代码不读取。旧 `run_all.py` 和旧拆分兼容接口已经移除。

## 9. 当前数据的预期数量

2001 条原始实验按当前 scikit-learn 取整规则得到：

| 文件 | 原谱 | 增强谱 | 总行数 |
|---|---:|---:|---:|
| `train.csv` | 1400 | 5600 | 7000 |
| `validation.csv` | 300 | 0 | 300 |
| `test.csv` | 301 | 0 | 301 |
| `robustness.csv` | 0 | 1204 | 1204 |

实际生成后应以 `preparation_config.json` 的精确计数为准。主分类结论只基于 301 条独立测试原谱；1204 条鲁棒性谱是这些测试实验的派生版本。

同一拆分参数下，无增强目录包含 1400 条训练原谱、300 条验证原谱和 301 条测试原谱，没有任何增强行或鲁棒性文件。两种目录的 `split_manifest.csv` 应逐行一致。

## 10. 小样本和后续扩展

固定 70/15/15 的优点是所有模型共享一套清晰、可查看的评价集合。缺点是小数据下单次拆分方差可能较大，因此正式比较至少使用多个训练种子。

若未来需要交叉验证，应从原始实验或 `split_manifest.csv` 生成训练/验证折，并保持当前 test 锁定不变。每个折仍必须先确定训练父实验，再只增强该折训练部分，不能把现有增强行随机分折。

新增原始数据后应重新生成整个 `data_prepared` 并重新跑完整比较。若新增数据来自真正独立的新批次，优先作为外部盲测使用，以提供比内部随机拆分更有价值的泛化证据。

## 11. 参考来源与本工程改写

| 工程决策 | 参考来源 | 本工程的具体实现 |
|---|---|---|
| 随机分层拆分 | [scikit-learn `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) | 两阶段70/15/15拆分，分层键为 `source_file × concentration_group` |
| 防止数据泄漏 | [scikit-learn 常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html) | 先拆原始 `experiment_id`，再只增强训练父谱；验证和主测试保持干净 |
| 按采集实体隔离 | [Braindecode 数据集拆分示例](https://braindecode.org/stable/auto_examples/datasets_io/plot_split_dataset.html) | 本数据没有 subject 字段，因此使用“源文件名 + 原始编号”作为最小独立实验 |
| 一维信号增强的工程参考 | [`torch_ecg`](https://github.com/DeepPSP/torch_ecg) 与 [tsai transforms](https://timeseriesai.github.io/tsai/data.transforms.html) | 保留原项目三种保守操作，仅实现位移、增益、噪声；固定两轮并输出每次操作轨迹 |
| 拉曼任务参考 | [`raman-dl`](https://github.com/grenzo/raman-dl)、[`raman-open-ml`](https://github.com/JoulesSpace/raman-open-ml) | 只参考拉曼分类和增强的任务组织方式；拆分比例、幅度、清单和哈希协议均为本工程定义 |

增强参数不是上述仓库给出的通用最优值，也没有声称物理等价于新的独立测量。它们是当前项目沿用的保守扰动范围，必须通过干净测试集、鲁棒性集和未来外部批次共同检验。
