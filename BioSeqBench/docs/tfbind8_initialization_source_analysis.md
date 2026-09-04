# TFBind8：全空间随机起始与训练集 TOP 起始分析

## 结论

本分析只使用现有的 12 个 TFBind8 weak model、每个模型 5 个 replicate 和共享初始化数据，没有重新运行优化。

如果目标是**尽可能找到一条高活性序列**，现有证据更支持保留全空间随机起始：在全部 12 个 weak model 中，50 条全空间随机序列的最高真实活性，以及其中被 weak model 预测为最优的序列的真实活性，都高于 50 条训练集 TOP 序列。

如果关注的是**整个起始群体的平均活性**，结论存在清晰分界：

- weak model 使用最低 1%–50% 数据训练时，全空间随机起始更高；
- weak model 使用最低 60%–95% 数据训练时，训练集 TOP 起始更高；
- 平均活性的交叉点位于 50% 和 60% 之间。

因此，当前阶段不建议把所有算法统一改成单一来源。对 benchmark 最稳妥的主设置仍是相同的 `50 Train Top + 50 Full-space Random` 共享种子；如果必须只选一种，并且评价目标是发现最高活性序列，应优先选择全空间随机起始。

## 为什么会出现这个结果

这里的“Train TOP”是 weak model 可见训练池中的 TOP，而训练池本身由全空间中**真实活性最低的 x%**组成。它不是全空间的高活性 TOP：

- 在 1% weak model 中，Train Top-50 的平均真实活性为 -0.3374，而 Random-50 为 -0.0301；
- 在 50% weak model 中，Train Top-50 为 -0.0530，Random-50 为 -0.0282；
- 到 60% 时才反转为 Train Top-50 -0.0105，高于 Random-50 -0.0366；
- 在 95% weak model 中，Train Top-50 的平均真实活性达到 0.3006，明显高于 Random-50 的 -0.0026。

另一方面，随机抽取 50 条序列覆盖整个 65,536 序列空间，因此更有机会直接抽到高活性尾部。Random-50 每组最高真实活性的 replicate 均值约为 0.36–0.45，而 Train Top-50 从 -0.336 随训练比例增加到 0.301。即使在 95% weak model 中，Random-50 的最高值仍为 0.4474，高于 Train Top-50 的 0.3014。

![两种初始化来源的直接质量比较](../results/tfbind8_optimization_all_models_5seeds_shared_init/initialization_source_quality.png)

## 按指标比较

| weak model 训练比例 | 起始群体平均活性更高 | 最高起始活性更高 | weak model 所选最佳起点的真实活性更高 | 最终候选整体更接近 |
|---:|---|---|---|---|
| 1% | Random | Random | Random | Train TOP（差异小、40.1% 距离并列） |
| 5% | Random | Random | Random | Random |
| 10% | Random | Random | Random | Random |
| 20% | Random | Random | Random | Random |
| 30% | Random | Random | Random | Random |
| 40% | Random | Random | Random | Random |
| 50% | Random | Random | Random | Random |
| 60% | Train TOP | Random | Random | Random |
| 70% | Train TOP | Random | Random | Random |
| 80% | Train TOP | Random | Random | Random |
| 90% | Train TOP | Random | Random | Train TOP |
| 95% | Train TOP | Random | Random | Train TOP |

“最终候选更接近”按最终序列到两组初始化种子的最小 Hamming 距离判断，并对 8 个有种子的优化算法取平均。它是搜索轨迹的近似证据，不是起始来源的因果效应。

## 初始化对最终结果的影响

在去除不同 weak model 训练比例造成的均值差异后，Random-50 的平均真实活性与最终候选平均真实活性的相关系数为：

| 算法 | 相关系数 |
|---|---:|
| DEN | 0.605 |
| AdaLead | 0.320 |
| GA | 0.167 |
| ISM | 0.075 |
| SA | -0.017 |
| Vanilla activation maximization | -0.037 |
| CbAS | -0.192 |
| Fast SeqProp | -0.209 |

DEN 对随机起始群体质量最敏感，AdaLead 为中等正相关；其余算法的线性相关较弱。每个训练比例只有 5 个随机初始化 replicate，因此这些相关系数应作为敏感性线索，而不是统计因果结论。

![初始化来源对最终候选的影响证据](../results/tfbind8_optimization_all_models_5seeds_shared_init/initialization_source_influence.png)

## 实验限制

现有优化运行的 8 个 seeded algorithms 每次都同时接收相同的 50 条 Train Top 和 50 条 Full-space Random 序列，输出文件没有记录单一祖先谱系。因此，不能从当前结果严格回答“只给 100 条随机序列”和“只给 100 条训练 TOP 序列”哪个会获得更好的最终优化性能。

另外，Full-space Random 是从整个序列空间均匀抽样，而不是刻意抽取训练集外序列。随着训练比例增加，随机种子自然落入训练池的比例也增加；这符合当前 benchmark 的初始化定义。

若以后需要因果结论，应增加匹配消融：对每个 weak model 和 optimizer seed，分别运行 `100 Full-space Random`、`100 Train Top` 和当前 `50 + 50`，并保持 query budget、超参数和随机数流一致。该消融不属于本次“使用现有数据”的分析范围。

## 可复现文件

- 分析脚本：`BioSeqBench/evaluation/analyze_tfbind8_initialization_sources.py`
- 指标级结论：`initialization_source_decision.csv`
- 初始化序列逐条评分：`initialization_sequences_scored.csv`
- 初始化来源汇总：`initialization_source_summary.csv`
- 最终候选与种子距离：`final_candidate_seed_proximity.csv`
- 距离汇总：`final_candidate_seed_proximity_summary.csv`
- 随机种子质量相关性：`random_seed_quality_correlations.csv`

以上 CSV 均位于 `BioSeqBench/results/tfbind8_optimization_all_models_5seeds_shared_init/`。
