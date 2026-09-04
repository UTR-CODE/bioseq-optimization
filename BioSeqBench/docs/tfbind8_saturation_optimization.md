# TFBind8：按算法饱和预算的序列优化 benchmark

## 问题与评价

目标是在仅用低活性训练池训练的 CNN weak model 上优化 DNA 序列，并判断不同优化算法在足够 weak-model 计算量下能否找到更高真实活性的序列。TFBind8 的 `4^8 = 65,536` 条序列均有实验测量值，因此生成完成后直接使用真实 `e_score` 评价；没有使用 strong model 代替实验 oracle。

训练池为真实活性最低的 `1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95%`，每个训练池运行 5 个随机重复，合计 60 个 weak-model × seed 条件。每个条件输出 100 条互不重复、且不属于该 weak model 可见训练池的候选；`Train Top-100` 是训练数据内已观测序列的参考，因而不要求 novel。

## 初始化与方法

八个需要起始序列的优化器为 ISM、SA、GA、AdaLead、Vanilla activation maximization、Fast SeqProp、CbAS、DEN。对于同一训练池和 seed，它们使用**相同**的 100 条起始序列：

- 50 条该低活性训练池中真实活性最高的序列；
- 50 条从完整 TFBind8 空间均匀抽取的随机序列。

这 100 条序列是算法共同的初始化，而非随机地从训练池外抽取。`Random` 不使用这些 seed：它从候选空间随机筛选并按 weak prediction 选前 100。另设两个 Random 对照，其搜索量分别匹配同一条件下八个 seeded 优化器 q99 query count 的中位数与算术均值。

## 饱和与 q99 的操作化定义

每个 seeded 优化器维护“截至当前 query、训练池外 weak prediction 最高的 100 条序列”的均值轨迹。轨迹以约 1,000 次 weak-model 查询为间隔存档（ISM 的变异批次会造成轻微的非整千数对齐）。

1. 至少进行 10,000 次查询后，如果该 top-100 weak-score 均值在连续 10,000 次查询内没有增加至少 `1e-5`，运行停止，记为 plateau。
2. 记录饱和初值 `b` 与停止时观测 plateau 值 `p`，选出最早满足 `score >= b + 0.99 * (p-b)` 的 archive，称为 q99 final set。
3. 额外保存 Kneedle 风格的拐点诊断（最大 `y_norm - x_norm`）。它用于描述曲线形状；实际停止由更稳健的 post-knee plateau 条件决定。

初始硬上限 65,500 queries 的审计发现 9 个 GA 和 2 个 CbAS trajectory 尚未 plateau；仅对这些 trajectory 扩展到 200,000。最终 480/480 条 seeded trajectory 均以 plateau 停止，最大的实际收敛查询数为 93,000，故最终 q99 不依赖硬上限截断。

## 主要结果

下表只在八个 seeded 优化器之间按每个 `(training fraction, seed)` 的 100 条候选平均真实活性排序。`top-3` 是进入前三名的 60 个条件比例；真实活性均值是全部 60 条件的非加权平均。

| 方法 | 平均 rank（低为好） | top-3 | 夺冠次数 / 60 | 平均真实活性 | 中位 q99 queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| CbAS | 2.70 | 76.7% | 23 | 0.31457 | 16,375 |
| ISM | 2.85 | 73.3% | 15 | **0.31487** | **9,076** |
| SA | 3.12 | **80.0%** | 5 | 0.31470 | 14,000 |
| Fast SeqProp | 3.63 | 30.0% | 6 | 0.31220 | 13,000 |
| Vanilla AM | 4.98 | 20.0% | 7 | 0.29936 | 10,000 |
| GA | 5.13 | 10.0% | 3 | 0.30191 | 28,000 |
| AdaLead | 6.33 | 1.7% | 0 | 0.28798 | 13,200 |
| DEN | 7.25 | 8.3% | 1 | 0.25500 | 3,000 |

没有单一算法在所有 training fraction 都最好。按五个 seed 的平均真实活性，fraction 赢家为：

- 1%：GA；5%：Fast SeqProp；10%：Vanilla AM；
- 20%、30%、95%：ISM；
- 40%、50%、60%、70%、80%、90%：CbAS。

所以应当区分“总体稳定性”和“跨条件平均活性”。CbAS 的平均 rank 最好，ISM 的全条件平均真实活性略高且 q99 query cost 约为 CbAS 的一半，SA 的 top-3 概率最高。它们构成当前 TFBind8 的稳定第一梯队，而不是一个无条件通吃的算法。

两个 Random q99 对照的平均真实活性分别为 0.30392（median-budget）和 0.30535（mean-budget），均低于 ISM、SA、CbAS 和 Fast SeqProp 的平均值，但其“best of 100”可因极值效应更高。这说明应同时看平均/中位候选质量、top-1% hit 数与 best activity，不能仅以 best 一项判定赢家。

## 交付物与可复核性

结果目录 `BioSeqBench/results/tfbind8_optimization_saturation_5seeds/` 包含：

- `protocol.json`：完整配置和操作性定义；
- `summary.csv`：660 个 algorithm-condition 结果；
- `all_candidates.csv`：66,000 条候选及其真实活性；
- `convergence_summary.csv` 和逐运行 `*_trajectory.csv`：480 条 seeded weak-score 轨迹；
- `saturation_optimizer_ranking.csv` 与 `saturation_optimizer_ranking_by_fraction.csv`：排名表；
- `saturation_four_exact_metrics.png`、`saturation_rank_stability.png`、`saturation_query_cost.png`：主图（另有 PDF）。

完整性检查验证：每个 candidate group 有且仅有 100 条唯一序列；非 `Train Top-100` group 均不含可见训练序列；每个 `(fraction, seed)` 都有八条 seeded q99；两个 Random budget 与这八个 q99 的中位数/算术均值逐条件精确一致。

## 限制与下一步

该结论仅适用于 TFBind8、该 CNN weak-model 训练方案和上述超参数。TFBind8 的极短全排列空间使得局部搜索尤其容易达到高分区域；不能直接推广到长序列空间。下一步应以同一饱和协议运行 TFBind10（也可用真实 oracle）及长序列 MPRA 数据集（需要独立 strong-model / 实验复测），并报告各任务的 Pareto 前沿而非宣称单一通用冠军。
