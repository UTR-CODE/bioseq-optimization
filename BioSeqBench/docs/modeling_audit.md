# 原始论文建模审计

审计日期：2026-08-20。`原论文 DL` 只指该数据集所对应的原始论文是否训练了端到端深度神经网络；树模型、线性/热力学模型不计入。`实现状态` 将“原论文确有 DL”与“当前 PyTorch 文件已经逐层复现”严格区分，避免把方便优化的替代 oracle 误标为论文复现。

| 标准化任务 | 原论文 DL | 原论文建模 | 当前 BioSeqBench 处理与实现状态 |
|---|---:|---|---|
| `tfbind8` | 否 | Barrera et al. 给出 PBM 结合谱，而非该 8-mer 任务的神经预测器。 | 全 65,536 个 8-mer；SHA-256 80/10/10 split。`TFBind8DeepBind` 是无预训练 DeepBind 风格 CNN，**后续替代模型**。 |
| `gb1` | 否 | Wu et al. 测量四位点的完整 GB1 适应度景观并分析进化路径，没有训练 DL surrogate。 | **已从当前 core benchmark 移除**（用户决定放弃蛋白任务）；原始/标准化文件暂保留，未再训练或评估。 |
| `gfp` | 否 | Sarkisyan et al. 是 GFP 突变景观实验与上位性分析，没有端到端 DL。 | **已从当前 core benchmark 移除**（用户决定放弃蛋白任务）；原始/标准化文件暂保留，未再训练或评估。 |
| `utr5_mpra` | 是 | Optimus-5Prime：50-nt one-hot，3 × Conv1D(120, k=8, same)+ReLU，flatten，Dense(40)+ReLU，dropout 0.2，线性 MRL。 | 使用 `egfp_unmod_1` 原始 GEO 表：按 `total_reads` 稳定降序取前 280,000；第 1–20,000 条为官方 test，其余 260,000 为官方 train；训练 MRL 仅以 train 均值/方差标准化；原论文无 val。`Optimus5PrimePaper` 为**逐层复现**。 |
| `deepstarr` | 是 | DeepSTARR 是 Basset 派生的双任务 CNN：4 个 Conv/BN/ReLU/MaxPool block（256/60/60/120 filters；7/3/5/3 kernels）、2 × Dense(256)，同时预测 developmental 与 housekeeping。 | 249-bp one-hot、两个标签、作者 chr2R train/val/test split。发布的 parquet 镜像将每个 split 的正向序列和相同标签的 reverse-complement augmentation 串接；benchmark 仅保留前半的独立 assay 序列（242,026 条，QC 后 242,015 条）。`DeepSTARRPaper` 为**逐层复现**；训练时可另行增加 RC augmentation。 |
| `cetnar2024_mrna_stability` | 是（LightGBM；非深度学习） | 论文使用 LightGBM：5 个主 isoform 的 RppH/结构/翻译/80-nt categorical 特征；先预测 T0/T2/T4/T8/T16，再将五个预测值加入 `k_fit` decay model。`paper_lightgbm` 按作者 `ModelTrainingTesting.py` 复现 6 个模型和固定 tree 参数（LightGBM 4.7.0 版本；论文为 3.3.2），36,219/9,064 author split 上 decay R2=0.4254，论文值 0.43。 | 统一版本以作者 feature-table 成员和固定 train/test 为准，过滤 103 个非正 `k_fit` 后为 36,138 train + 9,042 test = 45,180 条，排除未进入作者建模表的 12,600 条正标签低覆盖记录。`sequence_cnn_bilstm` 是另行实现的无预训练 sequence-only CNN–BiLSTM；group-held-out validation 后 test R2=0.5479（排除 21 个跨 split 相同 sequence 后 0.5477）。 |
| `kosuri2013_composability` | 否 | 使用可解释的组合性/因子化模型比较 promoter 与 RBS 对 RNA、蛋白输出的贡献。 | 11,696 条 QC 后构建体按原有 split 分为 9,372/1,152/1,172。`KosuriTwoBranchCNN` 是无预训练的 promoter/RBS 独立编码器加联合双回归头，以 train-only 标准化的 `log10` RNA/蛋白标签训练；epoch 136 早停。test 上蛋白 Pearson/Spearman/R²=0.955/0.949/0.912，RNA=0.985/0.987/0.970（均为 log10 尺度）。 |
| `zhao2022_ptrc_promoters` | 否（原文为 XGBoost） | 原文在 3,665 个 Ptrc 变体上比较机器学习方法并选择 XGBoost；其 Figure 4C 在 `log10(fluorescence/OD600)` 尺度报告 127 个设计序列 R²=0.8846、MAE=0.15、Pearson=0.94。论文未释放精确切分、超参数或模型文件。 | benchmark 对 QC 后 3,664 条用 SHA-256 80/10/10（2,883/391/390）拆分，以位置 A/C/G/T one-hot、验证集选参完成透明的功能级 XGBoost 复现：test Pearson=0.837、R²=0.698、MAE=0.239（log10 尺度）。同时训练无预训练 103,137 参数 `PtrcPromoterCNN`，test Pearson=0.848、R²=0.706、MAE=0.240。 |
| `zhou2026_l3_promoters` | 是 | 论文以 CNN 预测 L3 strength（论文 Pearson r=0.84），并以 balanced GAN with gradient penalty 生成定强度启动子。逐层 CNN 配置和权重未公开。 | L3 汇总表含 20,799 条 29-nt 序列，但将四个已公开的 `s1`--`s4` source sublibraries 合并而未标注。benchmark 从相应工作表无推断地恢复该 assay context，作为 `cell_type=L3_s1...L3_s4`（不是生物 cell type），并保持 SHA-256 80/10/10 的 16,603/2,101/2,095 sequence split。最终 `zhou_l3_conditional_cnn` 为位置保留 CNN + 子库条件头，test Pearson=0.792、Spearman=0.715、R²=0.626。去掉该 context 的纯 sequence-only CNN 仅 Pearson≈0.31，作为诊断而非 canonical surrogate；GAN 不属于预测 surrogate。 |
| `agarwal2025_lentimpra_{hepg2,k562,wtc11}` | 是 | 论文对 HepG2、K562、WTC11 各自训练 MPRAnn 与 MPRALegNet，并以十折 CV 评估；不是一个跨细胞 multi-task 模型。 | 使用作者 `human_legnet` 发布的完整 230-nt construct 训练表：HepG2 245,852、K562 393,328、WTC11 92,370 条。`MPRALegNet` 为**逐层 PyTorch 移植**：A/G/C/T+方向第五通道、EfficientNetV2-like block、shift/RC augmentation、AdamW + OneCycle。已完成 practical 10-fold（每个 outer test fold 固定下一个 fold 为 val，而非原文完整 10×9 sweep）：TTA test Pearson=0.798±0.008 / 0.823±0.004 / 0.712±0.008，Spearman=0.783±0.005 / 0.770±0.006 / 0.619±0.013，分别对应 HepG2/K562/WTC11。原先 55,338 条 joint library 只作为跨细胞扩展集，不能代替主建模库。 |
| `yeast_gpra_glucose` | 是 | Vaishnav et al. 从头训练 CNN 与 Transformer promoter-oracle，并用于序列设计；无语言模型预训练。 | complex-glucose 条件独立成任务；原始训练表没有本项目可复用的三分 split，采用固定 SHA-256 80/10/10。`YeastGPRACNN` 为从头 CNN 基线；若要逐层复现应补入论文的 Transformer oracle，当前状态为**近似实现**。 |
| `yeast_gpra_sc_ura` | 是 | 同上。 | SC-Ura 条件独立成任务；固定 SHA-256 80/10/10；当前从头 CNN 为**近似实现**。 |
| `parm_v1_phy3` | 是 | PARM 是按细胞/条件分别训练的 sequence-only CNN；论文从约千万个 promoter fragment 学习，并用遗传算法设计序列。 | 用 hg19、按链重建原始片段；保留论文 folds（fold 0 val、fold 1–4 train、官方 held-out test），全局精确去重优先级 test > val > train。`PARMCNN` 是轻量 PARM-style CNN，**尚非论文逐层复现**。 |
| `parm_v2_mh4` | 是 | 同上。 | 同一处理与 published folds；`PARMCNN` 为轻量近似实现。 |

## 5′UTR 的可复现性约定

`utr5_mpra` 的 canonical `sequence` 是**50-nt 随机可变区**，不是整个 75-nt reporter 5′UTR。若与外部表比较，应先去除固定 25-nt 前缀 `GGGACATCGTAGAGAGTCGTACTTA`。与 `/home/hexin/MPRA_dataset/MPRA_random_280k.1.csv.gz` 对照时，剥离该前缀后，序列集合、行顺序、`rl` 以及前 20,000 个 test 条目完全一致。

## 原始资料

- [Barrera et al. 2016](https://doi.org/10.1126/science.aad2257), [Wu et al. 2016](https://doi.org/10.7554/eLife.16965), [Sarkisyan et al. 2016](https://doi.org/10.1038/nature17995)
- [Sample et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7100133/) 及 [作者训练 notebook](https://github.com/pjsample/human_5utr_modeling/blob/master/modeling/training_MRL_CNN.ipynb)
- [DeepSTARR](https://www.nature.com/articles/s41588-022-01048-5), [Cetnar et al. 2024](https://www.nature.com/articles/s41467-024-54059-7), [Kosuri et al. 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3752251/)
- [Zhao et al. 2022](https://pubs.acs.org/doi/10.1021/acssynbio.1c00117), [Zhou et al. 2026](https://pubs.acs.org/doi/10.1021/acssynbio.5c00765)
- [Agarwal et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11903340/), [Vaishnav et al. 2022](https://doi.org/10.1038/s41586-022-04506-6), [PARM](https://pmc.ncbi.nlm.nih.gov/articles/PMC13017510/)
