# Optimus 5′UTR：12 个 weak model 的序列设计 benchmark

> **已失效（weak-model architecture）**：本报告中的优化运行使用旧的
> Optimus5Prime weak models。统一公平协议现要求 `WeakSeqCNN-v1`，因此这些
> 数值只能作为历史记录；新模型训练完成后必须重新运行优化。

## 实验完成情况

本实验已经完成 `12 weak models × 10 methods × 5 replicates = 600` 个正式条件，共产生并用 frozen Optimus strong model 评分 60,000 条候选序列。

### Weak model

- 数据：Sample et al. 2019 的 260,000 条官方 train 50-nt 随机 5′UTR；目标为实测 mean ribosome load（MRL），越高越好。
- 训练池：实测 MRL 最低的 1%、5%、10%、20%、30%、40%、50%、60%、70%、80%、90%、95%。
- 模型：每个训练比例一个 Optimus 5-Prime CNN，共 12 个 weak models；每个模型为三层 `Conv1d(120, k=8, same)`、Dense-40 和线性输出。
- Validation：从当前 lower-tail 可见池内部固定抽取 10%，不使用高表达序列 early stopping。
- OOD：官方 train 中固定最高 5%，仅用于诊断。
- 标签标准化：只使用当前模型的 low-tail fit subset 均值和标准差。

### Strong oracle

- Checkpoint：`strong_test_guided_v1/utr5_mpra/optimus5prime/best.pt`。
- 模型只在候选生成完成后调用，优化算法看不到 strong score。
- 生成序列的百分位以 canonical 280k library 的 strong predictions 为参考，而不是以 weak prediction 或实测标签计算。

### 初始化与预算

每个 weak model/replicate 为 8 个 seeded optimizers 提供完全相同的 100 条序列：

- 50 条当前可见 lower-tail 训练池中实测 MRL 最高的 `Train TOP`；
- 50 条从 `ACGT^50` 均匀生成的 `Full-space Random`；
- Random 不根据训练集成员身份或 strong/weak score 进行筛选；在 `4^50` 空间中，它们没有与 260k train 发生精确重合。

所有优化算法的 weak-model query budget 均为 10,000，并返回 100 条唯一候选。`Random screening` 独立筛选 10,000 条随机序列；`Train Top-100` 是零 query 的观测数据对照，不属于优化算法。

## 算法总体结果

以下排名在每个 `weak model fraction × replicate` 条件内按 100 条候选的平均 strong score 排名，然后对 60 个条件取平均。Train Top-100 不参与算法排名。

| 算法 | 平均排名 | 第一名次数（/60） | 总体平均 strong MRL | 平均 Top-1% hits（/100） | 平均 Best MRL |
|---|---:|---:|---:|---:|---:|
| DEN | 2.70 | 19 | 7.962 | 71.60 | 8.592 |
| Fast SeqProp | 2.75 | 18 | 7.963 | 68.52 | 8.807 |
| Genetic algorithm | 4.12 | 10 | 7.636 | 57.03 | 8.242 |
| AdaLead | 4.13 | 5 | 7.737 | 56.82 | 8.456 |
| Simulated annealing | 5.27 | 2 | 7.684 | 36.22 | 8.685 |
| CbAS | 5.85 | 3 | 7.666 | 26.88 | 8.704 |
| Vanilla activation maximization | 6.42 | 0 | 7.401 | 39.75 | 8.678 |
| Random screening | 6.87 | 3 | 7.606 | 15.60 | 8.615 |
| ISM greedy | 6.90 | 0 | 7.404 | 17.48 | 8.354 |

综合平均活性和 Top-1% 命中率，DEN 与 Fast SeqProp 是第一梯队。Fast SeqProp 的平均 Best MRL 更高；DEN 的平均排名和 Top-1% hits 略好。

![四项性能曲线](../results/utr5_optimization_all_models_5seeds_shared_init/four_metric_curves.png)

![四项性能热图](../results/utr5_optimization_all_models_5seeds_shared_init/four_metric_heatmaps.png)

## 不同 weak model 下的最佳方法

| Weak train pool | 最高平均 MRL | 最高 Best MRL | 最多 Top-1% hits |
|---:|---|---|---|
| 1% | Random screening（6.436） | Random screening（8.313） | Fast SeqProp（3.8） |
| 5% | Random screening（6.231） | CbAS（8.591） | CbAS（6.2） |
| 10% | DEN（8.155） | Vanilla AM（8.801） | DEN（66.2） |
| 20% | Fast SeqProp（8.376） | Fast SeqProp（8.990） | Fast SeqProp（76.6） |
| 30% | DEN（8.455） | Fast SeqProp（8.933） | DEN（89.0） |
| 40% | Fast SeqProp（8.830） | Fast SeqProp（9.427） | Fast SeqProp（91.6） |
| 50% | DEN（8.987） | Fast SeqProp（9.594） | DEN（100.0） |
| 60% | Fast SeqProp（8.646） | Vanilla AM（9.350） | GA（99.6） |
| 70% | Fast SeqProp（8.871） | Fast SeqProp（9.530） | DEN（96.8） |
| 80% | DEN（8.612） | Fast SeqProp（9.268） | DEN（92.8） |
| 90% | DEN（8.613） | Fast SeqProp（9.050） | GA（99.6） |
| 95% | GA（9.062） | Fast SeqProp（9.560） | GA（100.0） |

1% 和 5% weak models 几乎没有高表达区排序能力，复杂优化常不如 Random screening；从 10% 开始，DEN、Fast SeqProp、GA 和 AdaLead 才稳定显示收益。

## 初始化来源比较

按 50 条种子的平均 strong score：

- 1%–30%：Full Random 高于 Train TOP；
- 40%–95%：Train TOP 高于 Full Random；
- 交叉点位于 30% 与 40% 之间。

Random-50 的平均 strong score基本保持在 6.47–6.69，因为 Optimus 的原始实验库本身就是随机 50-mer；Train Top-50 随训练比例从 3.43 增加到 7.91。

按 50 条种子中的最高 strong score，Random 在大部分训练比例上仍占优；Train TOP 只在 60% 和 95% 明确更高。Random 的优势来自 50 次全空间抽样的极值和多样性，而不是平均质量。

由 weak model 在两组种子中各自挑选最高预测序列后：Random 在 1%–40% 和 60%–80% 的真实 strong score更高；50% 基本相同，90%–95% Train TOP 更高。

![初始化来源质量](../results/utr5_optimization_all_models_5seeds_shared_init/initialization_source_quality.png)

不同 replicate 中 Random-50 的平均 strong score与最终候选平均 strong score的相关系数都很弱（绝对值不超过 0.18）。说明在当前 10,000 query budget 下，最终结果主要由 weak model 和优化算法决定，而不是随机种子恰好抽得好不好。

![初始化来源影响](../results/utr5_optimization_all_models_5seeds_shared_init/initialization_source_influence.png)

距离热图只是最终候选到两组种子的最小 Hamming 距离证据。因为所有正式运行同时使用 50 TOP + 50 Random，它不能替代 `100 TOP vs 100 Random` 的因果消融。

## Weak model 的高表达区外推

12 个 weak models 在固定实测 top-5% OOD 上的 Pearson 为 -0.017 至 -0.260，Spearman 接近 0；随着可见训练比例增加，OOD MSE 从 35.50 降到 0.67，但在窄高表达尾部内仍无法可靠排序。

这不是简单的训练崩溃：同一 strong Optimus 在全局 test 上 Pearson 为 0.961，但在 combined 280k 的实测 top-5%、top-1% 和 top-0.1% 内 Pearson 分别为 -0.177、-0.344 和 -0.276。全局高相关不能证明模型在最顶端仍具有可靠排序能力。

## Strong-oracle 外推风险

canonical 280k library 的 strong prediction 范围为 0.369–8.990，而优化候选最高达到 9.895。60,000 条候选中有 2,703 条（4.505%）超过参考库的 strong-prediction 最大值：

| 算法 | 超过参考最大值的候选（/6000） | 比例 | 最大 strong score |
|---|---:|---:|---:|
| Fast SeqProp | 1,075 | 17.92% | 9.833 |
| DEN | 481 | 8.02% | 9.895 |
| GA | 477 | 7.95% | 9.730 |
| AdaLead | 414 | 6.90% | 9.615 |
| Vanilla AM | 196 | 3.27% | 9.749 |
| SA | 53 | 0.88% | 9.593 |
| CbAS | 4 | 0.07% | 9.265 |
| Random screening | 3 | 0.05% | 9.363 |
| ISM | 0 | 0% | 8.894 |

把所有超过参考最大值的预测保守截断到 8.990 后，DEN 和 Fast SeqProp 仍然排名第一、第二，说明总体结论不完全由少量极端分数驱动；但这些高分序列仍属于 surrogate extrapolation，不能称为实验验证的 MRL 提升。

此外，均匀 Random-50 与 Optimus 原始随机 50-mer MPRA 库的序列分布匹配，不等价于真实天然长 5′UTR 背景。天然长序列中的高表达密度可能更低，Random 初始化的优势不能直接外推。

## 当前建议

1. 以 in-silico strong score 的平均质量和命中率看，优先保留 DEN 与 Fast SeqProp；GA 作为高数据比例下的重要 population baseline。
2. 1%–5% weak model 没有足够信号时，Random screening 是更可靠的对照，复杂优化结果不应解释为真实改进。
3. 主 benchmark 保留相同的 `50 Train TOP + 50 Full Random`，同时报告起始平均值、起始最大值和相对起始的增益。
4. 进入湿实验前，不要只选择 strong score最高的序列；应使用独立架构/ensemble 重新评分、过滤极端外推序列，并从 DEN、Fast SeqProp、GA、Random 和 Train TOP 中分层选取。
5. 若要严格比较初始化来源，需要追加 matched `100 Train TOP`、`100 Full Random` 和当前 `50+50` 三组消融；本次混合初始化结果只提供来源影响证据。

## 主要产物

- Weak-model checkpoints 与 OOD 结果：`BioSeqBench/results/utr5_weak_cnns/`
- 正式 benchmark：`BioSeqBench/results/utr5_optimization_all_models_5seeds_shared_init/`
- Weak 训练脚本：`BioSeqBench/training/train_utr5_weak_cnns.py`
- 优化实现：`BioSeqBench/optimization/utr5_benchmark.py`
- 优化 CLI：`BioSeqBench/optimization/run_utr5_benchmark.py`
- Strong-oracle audit：`strong_oracle_tail_diagnostics.csv`、`strong_oracle_extrapolation.csv`、`strong_oracle_conservative_ranking.csv`
- 总体算法排名：`overall_algorithm_ranking.csv`
- 初始化来源统计：`initialization_source_summary.csv`、`final_candidate_seed_proximity_summary.csv`、`random_seed_quality_correlations.csv`
