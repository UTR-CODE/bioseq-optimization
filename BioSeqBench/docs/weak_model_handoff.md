# BioSeqBench weak-model 建模交接说明

## 项目与目标

项目根目录：`/mnt/data/home/hexin/MPRA_benchmark`  
BioSeqBench：`/mnt/data/home/hexin/MPRA_benchmark/BioSeqBench`

主要目标是比较哪些序列优化/生成算法能够找到活性更高的序列。在已标准化的 biological sequence-function 数据上，用逐渐扩大的低活性数据训练 CNN weak surrogate，再让 random search、GA、BO、CbAS、Fast SeqProp、GFlowNet、diffusion 等方法基于相同 weak model 和查询预算产生候选。候选最终由论文级 strong surrogate 打分；TFBind8/TFBind10 因具有完整排列 landscape，直接使用真实实验值 lookup，不使用 strong model 代替真实值。

统一数据格式：`dataset/<dataset>/{train.csv,val.csv,test.csv,metadata.json}`。基本字段为 `sequence_id, sequence, alphabet, length, label, label_type, cell_type, organism, source_reference`；任务可带有 TF identity、assay、多输出标签或 promoter/RBS 等附加字段。

所有数据均已完成字母表 QC、去重、长度/GC/2-mer/熵统计。全局 QC 在 `BioSeqBench/reports/dataset_statistics.html`，机器可读统计在 `BioSeqBench/reports/dataset_statistics.json`。

## Core weak-model 范围

按实验来源为 11 个核心来源；Agarwal 2025 lentiMPRA 的三个细胞系应视为三个独立任务。因此实际 weak-model task instances 为 13 个。

| 数据集 | 总数 | train / val / test | 任务与优化方向 | strong model 状态 |
|---|---:|---:|---|---|
| `tfbind8` | 65,536 | 52,317 / 6,669 / 6,550 | 8-mer DNA → TF binding；越高越好 | 不需要 full-data strong；已有 weak 实验 |
| `tfbind10` | 1,048,576 | 838,146 / 105,405 / 105,025 | 10-mer Pho4 → `-ddG`；越高越好 | 尚未训练 |
| `utr5_mpra` | 280,000 | 260,000 / 0 / 20,000 | human 5′UTR → MRL；越高越好 | 已训练 Optimus |
| `deepstarr` | 242,015 | 201,137 / 20,285 / 20,593 | enhancer → DEV/HK activity；越高越好 | 已训练 DeepSTARR CNN |
| `cetnar2024_mrna_stability` | 45,180 | 36,138 / 0 / 9,042 | RNA → decay rate；稳定性优化使用 `-decay` | 已训练 LightGBM 与 CNN-BiLSTM |
| `yeast_gpra_dream2022` | 6,786,293 | 6,043,623 / 671,567 / 71,103 | yeast promoter → activity；越高越好 | 已训练 Autosome/LegNet |
| `parm_v2_mh4` | 4,633,147 | 3,100,172 / 761,652 / 771,323 | regulatory DNA → log2RPM；越高越好 | 只训练 K562，PARM CNN |
| `agarwal2025_lentimpra_hepg2` | 122,926 | 98,342 / 12,292 / 12,292 | sequence → log2 RNA/DNA；越高越好 | 已训练 MPRALegNet CV ensemble |
| `agarwal2025_lentimpra_k562` | 196,664 | 157,332 / 19,666 / 19,666 | sequence → log2 RNA/DNA；越高越好 | 已训练 MPRALegNet CV ensemble |
| `agarwal2025_lentimpra_wtc11` | 46,185 | 36,948 / 4,617 / 4,620 | sequence → log2 RNA/DNA；越高越好 | 已训练 MPRALegNet CV ensemble |
| `kosuri2013_composability` | 11,696 | 9,372 / 1,152 / 1,172 | promoter+RBS → protein expression；越高越好 | 已训练双分支 CNN |
| `zhao2022_ptrc_promoters` | 3,664 | 2,883 / 391 / 390 | Ptrc promoter → fluorescence/OD；越高越好 | 已训练 XGBoost 与 CNN |
| `zhou2026_l3_promoters` | 20,799 | 16,603 / 2,101 / 2,095 | L3 promoter → strength；越高越好 | 已训练 conditional CNN |

## 明确不纳入 core weak benchmark 的数据

- `gb1`（149,361）和 `gfp`（51,715）：用户明确放弃所有蛋白数据集，不训练。
- OpenVaccine：已删除。
- Farasat 2014 Salis RBS：因数据量过小已删除。
- viral RNA stability/translation atlas：未获得预期约 20 万条序列，已删除。
- `yeast_gpra_glucose`（31,328,941）与 `yeast_gpra_sc_ura`（21,036,578）：已标准化，但与 yeast promoter/DREAM 高度重叠，暂不作为 core。
- `parm_v1_phy3`（11,136,682）：已标准化，当前不作为 core；只使用 PARM v2 MH4 K562。
- 父级 `agarwal2025_lentimpra`（55,338）：当前 core 使用 HepG2/K562/WTC11 三个细胞系任务，不重复纳入。

## 关键数据处理事实

### TFBind8

- `BioSeqBench/dataset/tfbind8/`。
- SIX6_REF_R1 8-mer landscape，完整覆盖 `4^8 = 65,536`。
- 可用作 exact oracle，不需要传统 full-data strong model。
- 已有 weak 实验：
  - `BioSeqBench/training/tfbind8_low_percentile_experiment.py`
  - `BioSeqBench/results/tfbind8_low_percentile_dynappo_cnn/`
- 历史实验协议是最低 1/5/10/20/50% 训练、最高 50% 测试；新统一协议改为最低 1/5/10/20/30/.../90/95% 训练，并使用固定最高 5% 做 weak-model OOD 诊断。

### TFBind10

- `BioSeqBench/dataset/tfbind10/`。
- 原始文件：`BioSeqBench/raw_data/tfbind10/TFBind10-Exact-v0.npz`。
- 1,048,576 个唯一 DNA 10-mer，完整覆盖 `4^10`。
- TF 为 Pho4。
- NPZ 的 `x` 为 `(1048576, 10)`，编码为 `0=A,1=C,2=G,3=T`；`y` 为 corrected higher-is-better `-ddG`。
- 标签范围约 `[-1.3636, 1.3532]`。
- Loader：`BioSeqBench/preprocessing/loaders/tfbind10.py`；配置：`BioSeqBench/configs/tfbind10.json`。
- 当前尚未训练 predictor。

### Human 5′UTR

- 采用 Sample et al. 2019 作者 notebook 的选择逻辑：按 total reads 稳定排序，保留 280k；最高 reads 的 20k 为 test、其余 260k 为 train。
- 原始序列前 25 nt 是固定 reporter prefix；标准化 `sequence` 是后 50 nt 可变 UTR。
- 目标是 MRL。论文 Optimus 模型使用 sklearn 标准化标签。

### DeepSTARR

- 242,015 是移除 forward/reverse-complement augmentation 后的实际规模；此前 484,030 是 RC 增广重复造成。
- 具有 `label_developmental` 和 `label_housekeeping`；weak 模型要么保留 multitask，要么明确拆分 DEV/HK。

### Cetnar 2024

- 作者训练数据基准为 45,180 条，而不是先前较大的表。
- label 是 fitted kinetic decay rate `k_fit`。
- 回归可预测原始 decay label；稳定性优化的统一分数必须设为 `optimization_score = -label`。

### DREAM、PARM、lentiMPRA、Kosuri

- DREAM 使用 step 而不是 epoch；其隐藏评估含 8 个加权子类别，不能只看全局 Pearson。诊断输出：`BioSeqBench/results/dream2022_oracle_diagnostics.{csv,json}`。
- PARM 当前只选择 v2 MH4 的 K562；使用论文 PARM CNN 五折 ensemble。不要混入 v1 phy3，除非重新决定。
- lentiMPRA 必须按 HepG2/K562/WTC11 分开。论文协议为每细胞系 `10 outer folds × 9 models = 90` 权重；三细胞系合计 270。论文 OOF 评估必须平均两个实际测量的 physical orientations，不能仅 reverse-complement TTA。
- WTC11 上游表有 `seq10002` 只有一个 orientation 的不规则项；聚合脚本应该 warning 而不是因要求恰好两个 orientation 而崩溃。
- Kosuri 为 promoter-RBS 组合；不能把它当作普通单段序列任务，沿用现有双分支模型。

## 已完成 strong/参考模型结果

不同数据集的 split 与 protocol 不同，数值不能直接横向比较。

| 任务 | 结果 |
|---|---|
| TFBind8 DynAPPO weak（1/5/10/20/50%） | Pearson/Spearman：`0.0346/0.0417`、`0.1345/0.1440`、`0.2489/0.2340`、`0.2338/0.2205`、`0.5190/0.5009` |
| TFBind8 unified CNN weak（1/5/10/20/30/40/50/60/70/80/90/95%，固定最高 5% OOD） | Pearson：`0.1024/-0.0668/0.0831/0.1139/0.1240/0.1389/0.2018/0.2235/0.2340/0.3432/0.4652/0.5631`；Spearman：`0.1128/-0.0596/0.0831/0.1065/0.1078/0.1292/0.1870/0.2031/0.2054/0.3054/0.4359/0.5329` |
| UTR5 Optimus extended/test-guided | P `0.96115`，S `0.94555`，MSE `0.26275`，R² `0.92270`；该数值不是完全无偏，因为用了 test early stopping |
| DeepSTARR CNN | DEV P/S `0.66685/0.64156`；HK `0.74847/0.56594` |
| Cetnar LightGBM | P/S/MSE/R² `0.65841/0.64071/0.03529/0.43237` |
| Cetnar CNN-BiLSTM | `0.74055/0.71058/0.02811/0.54791` |
| DREAM Autosome LegNet | P `0.97307`，S `0.97639`；MSE/R² 受 MAUDE affine scale 影响，不宜和论文直接比较 |
| PARM v2 K562 ensemble | promoter P/S/MSE/R² `0.90184/0.85626/0.27479/0.81303`；fragment `0.85689/0.75010/0.98721/0.73401` |
| lentiMPRA paper-orientation OOF | HepG2 P/S `0.82959/0.81907`；K562 `0.86205/0.81490`；WTC11 `0.78846/0.69837` |
| Zhao Ptrc XGBoost | P/S/MSE/R² `0.83666/0.82716/0.12751/0.69834` |
| Zhao Ptrc CNN | `0.84793/0.82178/0.12437/0.70577` |
| Zhou L3 CNN | `0.79179/0.71472/0.05255/0.62552` |
| Kosuri dual-branch CNN | RNA log10 P/S `0.98492/0.98655`；protein log10 `0.95518/0.94902` |

## 统一 weak-model 协议建议

目标是测试“只接触低质量/低活性离线数据时，surrogate 能否对高活性候选进行可靠排序”，而不是重复 full-data strong training。

建议抽象统一脚本：`BioSeqBench/training/train_weak_surrogates.py`，主要参数：

```text
--dataset
--model
--train-fractions 0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95
--ood-test-fraction 0.05
--ensemble-size
--seed
--max-train-samples
--device
--num-workers
```

建议协议：

1. 每个任务定义统一的 `optimization_score`，约定越高越好；Cetnar 使用 `-decay_rate`，其余 core 任务通常使用原 label。
2. 依 optimization score 排序。
3. weak train pool 分别取最低 1%、5%、10%、20%、30%、40%、50%、60%、70%、80%、90%、95%。
4. 固定最高 5% 仅用于所有训练比例间可比较的 weak-model OOD 诊断；它与最大 95% train pool 仍不重叠。
5. validation 只能从 low-score training subset 内切分，不能以 high-score test early stop。
6. 每种比例训练 ensemble，保存 member checkpoint、ensemble 预测、预测 CSV 和 protocol。
7. weak-model 诊断统一输出 Pearson、Spearman、MSE、R²；这些指标不是最终算法排名依据。
8. 优化算法必须使用相同 weak checkpoint、初始数据、随机种子、候选数和 surrogate-query budget。最终比较候选的 strong/oracle score、best score、top-k mean、全局百分位、成功率、多样性和相对 weak train pool 的新颖性。
9. TFBind8/TFBind10 的最终分数直接查完整实验 landscape；其他数据集才使用 strong model 评分。

weak 模型统一使用 CNN 架构，只改变可见的低活性训练数据比例；strong model 仍可使用各任务已验证的论文架构。当前优先完成 TFBind8/TFBind10 的统一 CNN：

- TFBind8/10：统一 two-convolution CNN ensemble；
- UTR5：Optimus-5Prime CNN；
- DeepSTARR：原始 multitask CNN；
- Cetnar：CNN-BiLSTM，另保留 LightGBM baseline；
- DREAM：Autosome/LegNet；
- PARM：PARM CNN；
- lentiMPRA：MPRALegNet；
- Kosuri：双分支 promoter-RBS CNN；
- Ptrc：CNN，XGBoost 作为参考；
- L3：conditional CNN。

不要使用蛋白预训练模型或 DNA/RNA foundation model 微调。

## 大规模任务注意事项

- DREAM、PARM 和 lentiMPRA 不应完整读入 pandas 后再筛选；应使用 streaming/chunked sampling 或预计算 quantile threshold + hash 选样。
- 可以添加 `--max-train-samples` 控制计算预算，但不能改变低分样本选择逻辑。
- 短序列 CNN 可以从 batch size 1024、2048 或 4096 开始。
- lentiMPRA 曾出现 `OSError: [Errno 24] Too many open files`；DataLoader 先用 `num_workers=0` 或 2，并确保 fold 间正确释放。

## 关键目录与文件

```text
BioSeqBench/dataset/                         # 标准化数据
BioSeqBench/preprocessing/                   # 预处理、loader、schema
BioSeqBench/models/                          # 模型
BioSeqBench/training/                        # 训练
BioSeqBench/evaluation/                      # 评估
BioSeqBench/results/                         # 结果
BioSeqBench/configs/sources.json             # 数据来源
BioSeqBench/docs/modeling_audit.md           # 论文建模审计

BioSeqBench/training/tfbind8_low_percentile_experiment.py
BioSeqBench/results/tfbind8_low_percentile_dynappo_cnn/

BioSeqBench/training/train_tfbind_weak_cnns.py
BioSeqBench/oracles/tfbind.py
BioSeqBench/models/tfbind8/ensemble.py
BioSeqBench/evaluation/evaluate_tfbind_candidates.py
BioSeqBench/results/tfbind_weak_cnns/tfbind8/

BioSeqBench/preprocessing/loaders/tfbind10.py
BioSeqBench/configs/tfbind10.json
BioSeqBench/raw_data/tfbind10/download_receipt.json

BioSeqBench/models/agarwal2025_lentimpra/mpralegnet.py
BioSeqBench/training/train_agarwal_mpralegnet.py
BioSeqBench/evaluation/aggregate_agarwal_mpralegnet_paper_cv.py
scripts/train_agarwal2025_mpralegnet_paper_cv.sh
```

测试命令：

```bash
python -m pytest -q
```

当前状态：18 passed。TFBind8 的 12 个训练池 × 10-member CNN ensemble 已完成，共 120 个 checkpoint。

## 推荐的新窗口首个任务

先完成 TFBind8/TFBind10 的统一 CNN weak training、checkpoint inference、优化算法接口与真实 landscape 评分。在统一预算下比较首批优化算法后，再接入其他任务。
