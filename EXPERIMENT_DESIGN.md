网络模块消融与传统基线的工程扩展建议

本文是后续实现方案。当前已经实现的改动是training_control.py中的权重选优/早停解耦及其训练、汇总、自检接入；下面的普通CNN、网络开关、SVM和统一实验入口尚未实现，也没有生成对应真实实验结果。

**应把输入表示、网络结构和实验配置作为三个独立维度。** 当前--model 1～4实际表达输入通道组合，四者使用同一ReagentClassMSNet。建议保留这个入口兼容原实验，新增明确的architecture和experiment_id，避免把model5、model6继续用于含义不同的比较。

建议最终组织如下；标有“拟新增”的文件均属于后续方案：

```text
config.py                         公共路径与任务定义
data.py                           共享读取、拆分清单和预处理
training_control.py               已实现：最低损失与早停监控
metrics.py                        共享标签指标、概率指标与绘图
train.py                          保留原单模型命令行入口
run_all.py                        保留原四通道实验入口
predict.py                        按保存的模型类型调度预测
models/
  blocks.py                       给现有块增加可配置开关
  net.py                          保留现有网络和默认配置
  plain_cnn.py                    拟新增：普通1D-CNN
  registry.py                     拟新增：按architecture构建神经网络
baselines/
  svm.py                          拟新增：SVM Pipeline和调参逻辑
engine/
  neural.py                       拟新增：从train.py提取共享神经网络训练循环
experiments/
  schema.py                       拟新增：ExperimentSpec及参数校验
  channel_ablation.json           拟新增：固定网络、改变通道
  structure_ablation.json         拟新增：固定通道、改变网络模块
  baselines.json                  拟新增：普通CNN、SVM与完整网络
run_experiments.py                拟新增：统一运行与结果汇总
tests/
  test_training_control.py        已实现：监控规则与权重/结果一致性回归
  test_model_variants.py          拟新增：结构变体与恢复一致性
```

规模暂小时可以先保留train.py中的循环，等第二个神经网络接入后再提取engine/neural.py；SVM应使用独立fit流程，共享数据协议和评价，不硬套epoch、AdamW或早停。所有神经网络变体共用现有ValidationMonitor，避免各写一份选优逻辑。

**建议的最小实验矩阵如下。**

| 实验组 | 变体 | 固定项 | 能回答的问题 |
|---|---|---|---|
| 通道 | raw、raw_snv、raw_snv_d1、raw_snv_d1_d2 | 完整网络、拆分和训练协议 | 在已有通道基础上新增一个表示是否有收益 |
| 结构 | 完整网络 | raw_snv_d1及全部实验协议 | 结构实验的参考点 |
| 结构 | 去SE | 其余网络相同 | SE通道重加权是否有增益 |
| 结构 | 注意力池化改为均匀权重 | 池化输出维度和全连接层相同 | 学习位置权重是否有增益 |
| 结构 | 三分支卷积核都改成7 | 分支数、融合维度、残差和池化相同 | 多种卷积尺度是否优于重复单尺度 |
| 结构，可选 | 去残差连接 | 主卷积分支相同 | 残差连接是否有增益 |
| 方法基线 | 普通1D-CNN | 同三通道输入、同拆分和神经网络训练策略 | 完整结构相对简单卷积模型的综合收益 |
| 方法基线 | RBF-SVM | 同拆分、同三通道表示及相同调参数据权限 | 深度模型相对传统方法的收益 |

每组比较只支持对应的结论。去SE并不同时验证注意力池化；普通CNN与完整网络差异较多，只能作为整体方法比较。各组件可能存在相互作用，单模块消融不能宣称每个模块在任意组合下都有相同收益。

去SE可使用nn.Identity保持张量形状；多尺度开关可通过branch_kernels=(3,7,15)与(7,7,7)表示。不同卷积核仍会造成参数量变化，应报告参数量，必要时另加容量相近对照。不要只把准确率变化解释为尺度的唯一因果效果。

注意力池化建议先用均匀权重替代learned attention，使第三个池化向量和192→64全连接层维度保持一致。均匀权重池化等价于平均池化，所以此控制会产生一个重复的平均向量，这是为控制维度而有意设置的参照。直接去掉第三路并改成128→64可作为实际简化模型，但会同时减少全连接参数，应另列并解释。

普通1D-CNN可采用若干单尺度Conv+BN+GELU、两次池化和全局平均池化，保持类似层数/宽度作为起点；训练超参数使用相同的预先规定方案，同时记录容量和开销。如为各方法分别调参，应给出相近搜索预算且只使用训练/验证数据，不能只为完整模型优化。

**实验配置要能完整描述一次运行。**

拟采用的配置示例（不是当前CLI已经支持的命令）：

```json
{
  "experiment_id": "structure_no_se",
  "family": "structure_ablation",
  "backend": "torch",
  "architecture": "msnet",
  "channel_set": "raw_snv_d1",
  "model_options": {
    "branch_kernels": [3, 7, 15],
    "use_se": false,
    "use_residual": true,
    "attention_mode": "learned"
  },
  "split_manifest": "splits/fixed_manifest.csv",
  "split_seed": 66,
  "training_seeds": [66, 67, 68]
}
```

模型结构参数必须写入权重，并由registry完整恢复；不能在predict中用默认结构替代训练结构。当前--seed同时决定拆分和初始化，后续应拆成split_seed与train_seed，保持同一划分比较训练随机性，并在另一组实验中单独研究划分敏感性。

原始实验批次与父光谱版本都需要绑定。建议主要验证/测试只使用未增强原谱；训练是否用增强、增强份数与参数在各模型间一致。增强测试可另列为鲁棒性实验。涉及多个采集日期、配液样品或基底时应按这些真实独立单元分组，预先保存拆分清单供各后端共享。

**SVM应共享评价协议，但保留适合它的训练与输出接口。**

三通道张量可以展开为[N,1572]特征；另做raw-only SVM时应明确标记，避免把输入表示差异混入方法对比。SVM参数如C、gamma应在训练集内部按真实分组交叉验证，或使用固定验证集选择；测试集用于最终报告。谱预处理、特征缩放和可选PCA都必须在各训练折内拟合，不能先用全训练集拟合后再进行内部交叉验证。[scikit-learn预处理与泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html)

建议把metrics拆出接受y_true/y_pred的基础分类评价，再把概率、交叉熵和成分概率作为可选评价。SVM的decision_function分数不能直接当作Softmax概率或化学成分存在概率。SVC的概率估计涉及额外交叉验证；有父谱增强或实验批次结构时应显式控制校准的分组，不能无说明地沿用默认内部折分。不需要概率时先报告准确率、Macro-F1、逐类召回和混淆矩阵即可。[SVC官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

建议统一后端结果协议为predict(X)返回类别，predict_proba(X)可选，metadata记录backend、类别顺序、通道、预处理、参数和版本。SVM保存完整Pipeline；神经网络保存model_config、state_dict和预处理。汇总中不可用的概率指标标为不适用，不制造替代分数来强行排名。

**结果目录应体现实验因素和运行版本。**

```text
output/<run_id>/<experiment_id>/seed_<train_seed>/...
pt/<run_id>/<experiment_id>/seed_<train_seed>/...
output/<run_id>/compare/channel_ablation.csv
output/<run_id>/compare/structure_ablation.csv
output/<run_id>/compare/baselines.csv
```

公共字段至少包含experiment_id、backend、architecture、channel_set、model_options、数据/拆分指纹、代码版本、训练策略版本、种子、参数量、训练耗时、推理耗时、最佳轮次、最低验证损失、早停参考信息和最终评价。普通SVM没有神经网络参数量/轮次，应记录支持向量数量等适用指标，并将不适用字段留空。

结果复用应核对数据、拆分、代码、模型选项和训练策略；不能仅因model3目录存在就跳过。神经网络的选优沿用已实现的策略版本2：任何严格更低的验证损失保留权重，min_delta只影响早停；SVM按预先声明的调参标准选参数。不同方法的主要比较使用统一的分类评价，不能强制用不具备可比概率的交叉熵排序。

若多个方法在内部测试都为100%，应披露饱和结果，重点增加独立批次、预设扰动和实际开销的比较。不能为了分出高低反复根据测试集修改方案。已有测试集参与多轮研发之后，最终研究结论宜增加此前未使用的新批次盲测。

建议实施顺序为：先增加模型开关和普通CNN，再提取共享神经网络循环及实验配置；接着接入SVM与可选概率指标；最后补齐多种子、模块消融和独立验证。每一步都保留当前train/run_all的兼容入口，并用小规模流程检查确认权重恢复、类别顺序和拆分一致后再开展真实实验。
