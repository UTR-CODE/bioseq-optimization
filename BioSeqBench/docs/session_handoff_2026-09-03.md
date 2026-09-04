# 会话交接摘要（2026-09-03）

本文档用于在新窗口继续工作，记录本次会话的研究方向、代码、结果和已知限制。

## 1. 研究方向变化

最初目标是做多数据集优化器 benchmark，比较不同 sequence optimization algorithm 找到高活性序列的能力。已尝试的算法包括：

- ISM greedy search
- simulated annealing
- genetic algorithm
- AdaLead
- vanilla categorical activation maximization
- Fast SeqProp
- CbAS
- DEN
- Random screening 和 Train Top-100 作为对照

后来发现：不同 threshold、weak-model seed、optimizer seed 和初始化来源会导致算法排名变化，因此不存在明显的 universal winner。当前研究重点从“寻找唯一最优优化器”转向“建立可复用的通用序列优化软件”。

最终软件目标：

> 数据/模型和优化算法完全解耦。用户只需把自己的模型包装成统一 scorer shell，指定序列空间、初始序列和 query budget，即可选择任意优化算法优化序列。

## 2. 数据集现状

项目核心 task instances（来自 `BioSeqBench/docs/weak_model_handoff.md`）：

- TFBind8：65,536 个 8-mer，全排列，可用真实 landscape lookup。
- TFBind10：1,048,576 个 10-mer，全排列；weak model/optimization 尚未完整完成。
- UTR5 MPRA：280,000 个 50-nt 5′UTR，strong Optimus。
- DeepSTARR：enhancer activity。
- Cetnar2024 mRNA stability：RNA decay/stability。
- DREAM 2022 yeast GPRA：6,786,293 个 yeast promoter，strong Autosome/LegNet。
- PARM v2 MH4：regulatory DNA activity。
- Agarwal lentiMPRA：HepG2、K562、WTC11 三个 task。
- Kosuri promoter/RBS composability。
- Zhao Ptrc promoter。
- Zhou L3 promoter。

明确不作为当前 core 的包括蛋白 GB1/GFP、OpenVaccine、病毒数据、Salis RBS、`yeast_gpra_glucose`/`sc_ura`、PARM v1，以及 Agarwal 父级混合数据。

当前最完整的优化实验数据集：TFBind8、UTR5 MPRA、DREAM。

## 3. WeakSeqCNN-v1

统一 weak architecture 写在根目录 `weak_model_description.txt`，实现于：

- `BioSeqBench/models/weakseqcnn.py`

结构：

- 输入 DNA/RNA one-hot，统一 A/C/G/T，RNA U→T；
- 1×1 input projection 4→64；
- 2 个相对位置通道；
- 4 个 residual dilated CNN blocks，channels=64，kernel=5，dilation=1/2/4/8；
- GroupNorm(8)、GELU、Dropout=0.1；
- masked mean/max/attention pooling；
- head 192→128→1；
- 支持 variable length mask、soft one-hot gradient、[B,L,4] 和兼容 [B,4,L]。

参数量：

- WeakSeqCNN-v1：194,882
- Optimus5PrimePaper：474,681
- WeakSeqCNN 约为 Optimus 的 41%。

## 4. UTR5 weak model 训练

脚本：

- `BioSeqBench/training/train_utr5_weak_cnns.py`

结果：

- `BioSeqBench/results/utr5_weak_cnns/`
- 12 个 threshold：1%、5%、10%、20%、30%、40%、50%、60%、70%、80%、90%、95%；
- 每个 threshold 5 个 seed，共 60 个 checkpoint；
- `batch_size=1024`；
- 每个 threshold visible pool 内部 90% fit / 10% validation；
- `train.csv` top 5% 作为固定 OOD test，共 13,000 条；
- 官方 `test.csv` 未用于该 weak-model benchmark；
- 全项目测试曾通过 `40 passed`。

报告：

- `BioSeqBench/docs/utr5_weakseqcnn_training.md`
- `BioSeqBench/results/utr5_weak_cnns/results.csv`
- `BioSeqBench/results/utr5_weak_cnns/surrogate_results.csv`
- `BioSeqBench/results/utr5_weak_cnns/protocol.json`

## 5. DREAM weak model 训练

脚本：

- `BioSeqBench/training/train_yeast_dream2022_weak_cnns.py`

缓存：

- `BioSeqBench/results/dream2022_weak_cnn_cache/`

结果：

- `BioSeqBench/results/dream2022_weak_cnns/`
- 12 threshold × 5 seed，共 60 个 WeakSeqCNN-v1；
- DREAM `train.csv` 有 6,043,623 条 FACS promoter observations；
- 每个 threshold visible pool 内部 90% fit / 10% validation；
- `train.csv` top 5% 为固定 test，共 302,181 条；
- 序列最大长度 142，优化时后来固定使用 110 bp（官方 DREAM test 长度）；
- 官方 DREAM test 是 MAUDE affine scale，与 FACS train scale 不同，因此不能直接用 raw MSE/R² 横比；
- 训练结果 `results.csv` 已完整，全部 60 个模型可用；
- 全项目测试最后通过 `40 passed`。

## 6. 旧的 UTR5/DREAM saturation 优化实验

### UTR5 saturation90

结果目录：

- `BioSeqBench/results/utr5_optimization_saturation90_all60/`

特点：

- 60 个 weak checkpoint 独立运行；
- 每个条件 50 条实际 fit split top + 50 条全空间 random；
- 8 个算法；
- q90 saturation 选择；
- strong Optimus 评分；
- 660 个条件、66,000 条候选。

总体平均（旧 saturation 运行）：Fast SeqProp、Vanilla AM、DEN、GA 较强，但没有 universal winner。

### DREAM promoter saturation

实现和结果：

- `BioSeqBench/optimization/dream_benchmark.py`
- `BioSeqBench/oracles/dream.py`
- `BioSeqBench/results/dream2022_optimization_benchmark/`

旧 DREAM saturation 结果：480 条条件（12 threshold × 5 seed × 8 algorithm），48,000 候选。总体平均：Vanilla AM、DEN、GA、Fast SeqProp 较强，但 threshold 依赖明显。

## 7. Fixed-budget 三数据集实验

用户随后决定改用相同 query budget，主实验预算选为 50,000 queries。目录：

- `BioSeqBench/results/fixed_budget_50k/tfbind8/`
- `BioSeqBench/results/fixed_budget_50k/utr5/`
- `BioSeqBench/results/fixed_budget_50k/dream/`

每个数据集均有：

- 12 threshold；
- 5 replicate；
- 8 个优化器；
- 每个算法严格 50,000 query；
- 每组 100 条候选。

### 已观察的总体均值

- TFBind8：SA 和 ISM 约 0.3153，整体差异很小；
- UTR5：AdaLead 约 6.9903，Fast SeqProp 约 6.9660；
- DREAM：Fast SeqProp 约 13.6299，DEN 约 13.6058。

详细旧总结：

- `BioSeqBench/docs/fixed_budget_50k_cross_dataset_summary.md`

### 重要限制

这批 fixed-budget 结果不是最终严格公平版本：

- TFBind8 和 UTR5 的旧 runner 是 threshold-level weak ensemble；
- DREAM runner 是每个 replicate 绑定一个独立 weak checkpoint；
- 没有统一包含 Random/Train Top-100 对照；
- 因此只能作为初步 fixed-budget 结果，不能作为最终严格跨数据集结论。

建议新窗口不要继续扩展这批旧 benchmark，而是优先完成通用优化软件架构。

## 8. 已建立的旧优化模块

旧 dataset-specific 代码仍保留，主要包括：

- `BioSeqBench/optimization/tfbind8_benchmark.py`
- `BioSeqBench/optimization/utr5_benchmark.py`
- `BioSeqBench/optimization/utr5_saturation_benchmark.py`
- `BioSeqBench/optimization/dream_benchmark.py`
- 各自的 `run_*.py`

这些模块服务历史 reproducibility，不建议新 API 继续依赖它们。

## 9. 新的独立软件目录

本次会话新建了：

```text
sequence_optimization/
├── __init__.py
├── api.py
├── algorithms.py
├── objectives.py
├── seeds.py
├── cli.py
├── README.md
└── examples/
    ├── optimize_example.py
    └── models/
        ├── README.md
        ├── __init__.py
        ├── catalog.py
        ├── deepstarr.py
        ├── utr5.py
        ├── tfbind8.py
        └── dream_legnet.py
```

`pyproject.toml` 已将 `sequence_optimization*` 加入 package discovery。

### 核心 API

`sequence_optimization/api.py` 提供：

- `SequenceScorer` protocol：`score(sequences) -> ScoreBatch`；
- `ScoreBatch`：scores + optional uncertainties；
- `CallableScorer`：包装任意 Python callable；
- `TorchSequenceScorer`：包装 PyTorch model + 用户 encoder；
- `CachedScorer`：重复序列缓存与 query budget；
- `SequenceSpace`：长度/字母表验证、encode/decode、random、single_mutants、mutate_one；
- `OptimizationConfig`；
- `OptimizationResult`：algorithm、sequences、scores、uncertainties、query_count、metadata、trajectory。

2026-09-03 后续更新：

- `ScoreBatch` 现支持 `[N]`、`[N, 1]` 和 `[N, tasks]` activity；
- `TargetObjective(mode="maximize")` 只最大化指定 target，忽略其他输出；
- `TargetObjective(mode="specificity")` 实现 CRESted-style target-specific utility，最大化 target 并惩罚其他 tasks；
- `OptimizationResult` 同时保留 scalar utility `scores` 和原始 `activities`，`to_frame()` 输出每个 activity 列；
- 便捷函数包括 `ISM_optimize`、`genetic_algorithm_optimize`、`simulated_annealing_optimize`、`adalead_optimize` 和 `random_optimize`。

### 算法注册系统

`sequence_optimization/algorithms.py` 提供：

- `register_algorithm(name)`；
- `available_algorithms()`；
- `run(name, scorer, space, config, initial=None, seed=0)`；
- `run_many(...)`。

当前独立包已实现：

- `random`
- `ism_greedy`
- `simulated_annealing`
- `genetic_algorithm`
- `adalead`

用户可以：

- 自己提供 exactly `num_candidates` 条 seed sequence；
- 用 `load_seed_sequences(path, space)` 从文本文件读取；
- 不提供 initial，使用 `random_seed_sequences(space, count, seed)` 自动随机生成。

### CLI

```bash
python -m sequence_optimization.cli \
  --scorer my_module:predict \
  --algorithm genetic_algorithm \
  --length 50 \
  --alphabet ACGT \
  --budget 50000 \
  --num-candidates 100 \
  --seed-file seeds.txt \
  --task-names developmental,housekeeping \
  --target developmental \
  --objective specificity \
  --output designs.csv
```

不提供 `--seed-file` 时，CLI 会自动生成随机 seed。

### Examples model catalog

`sequence_optimization/examples/models/catalog.py` 是 lazy metadata，不复制模型实现，登记了：

- WeakSeqCNN-v1；
- Optimus5Prime；
- TFBind8；
- DREAM LegNet；
- DeepSTARR；
- PARM；
- Agarwal MPRALegNet；
- Kosuri；
- Zhao Ptrc；
- Zhou L3。

真正的 model implementation 仍在 `BioSeqBench/models/`，软件层只通过 scorer shell 连接。

`Optimus5PrimeExample.from_checkpoint()` 会加载 strong Optimus checkpoint 并恢复训练时的 target scale；
`DeepSTARRExample.from_checkpoint()` 会加载 two-head checkpoint，并命名为
`developmental`/`housekeeping`。已实际完成 Optimus ISM（302 queries）和 DeepSTARR
specificity GA（8 queries）smoke test。

## 10. 测试状态

本次最后一次全量测试：

```text
55 passed
```

其中新增的通用 API 测试在：

- `tests/test_sequence_optimization.py`

旧的 `BioSeqBench/optimization/core.py` 也有一套重复的通用 API 草稿和测试，但新窗口应以顶层 `sequence_optimization/` 为主，后续可以删除或标记旧草稿，避免两套 API 并存。

## 11. 已知问题和下一步优先级

### 高优先级

1. **统一 API 中目前只有 5 个 black-box 算法。**
   - Vanilla AM、Fast SeqProp、CbAS、DEN 尚未完整迁移到独立包；
   - 需要定义 gradient/proposal-specific scorer protocol；
   - 不要把这些算法硬塞进纯 black-box scorer，应该增加可选 capability protocol。

2. **算法实现需要进一步审计。**
   - 检查所有算法在 budget 恰好耗尽时的 query_count；
   - 处理有限序列空间耗尽、重复候选、无法产生足够 unique candidates 的情况；
   - 对 GA/AdaLead 的结果和算法定义写更清楚的文档；
   - 增加 timeout/early-stop policy，但不能悄悄改变预算。

3. **模型 examples 仍需扩展。**
   - Optimus 和 DeepSTARR 已有 checkpoint-aware wrapper，Optimus 固定 50 bp，DeepSTARR 固定 249 bp；
   - TFBind8 wrapper 需要确认 TFBindCNN lazy/input shape；
   - DREAM LegNet example 当前只包装 scalar score，没有自动 reverse-complement averaging；
   - 其他模型的 checkpoint calibration、label normalization 和 strong/weak 选择仍应在具体 wrapper 中显式配置。

4. **包命名和向后兼容。**
   - `BioSeqBench/optimization/__init__.py` 仍导出旧 benchmark 和旧 core；
   - 新软件应从顶层 `sequence_optimization` 导入；
   - 后续可将 `BioSeqBench/optimization` 降级为 legacy compatibility namespace。

### 中优先级

5. 增加 `BaseOptimizer`/class-based interface，支持 state、checkpoint、日志和 resume。
6. 增加统一 `Constraint` interface，例如排除已知训练集、GC content、motif、Hamming distance、固定位置。
7. 增加 `CandidateArchive` 公共类型和候选去重/过滤策略。
8. 增加 scorer batch/concurrency/remote API 支持。
9. 增加统一评估工具：mean、median、best、top-k、percentile、diversity、query efficiency。
10. 为每个模型 example 提供 checkpoint loader 和 README。
11. 添加 Python package entry point，例如 `seqopt` CLI。

## 12. 推荐的新窗口启动方式

新窗口首先读取本文档，然后执行：

```bash
cd /mnt/data/home/hexin/MPRA_benchmark
python -m py_compile sequence_optimization/*.py sequence_optimization/examples/models/*.py
/home/hexin/miniconda3/envs/UTR/bin/python -m pytest -q
```

然后优先完成：

1. `sequence_optimization` 的 API 设计定稿；
2. 把 gradient-based 算法定义为可选 capability，而不是黑盒强行兼容；
3. 补充约束、archive、timeout 和 resume；
4. 将历史模型作为 examples/documentation，而不是重新做 benchmark；
5. 最后再决定是否对全部数据集提供统一 CLI examples。

## 13. 关键路径速查

- 规范：`weak_model_description.txt`
- 项目交接旧文档：`BioSeqBench/docs/weak_model_handoff.md`
- 新软件 README：`sequence_optimization/README.md`
- 新软件 API：`sequence_optimization/api.py`
- 新软件算法：`sequence_optimization/algorithms.py`
- 新软件多任务 objective：`sequence_optimization/objectives.py`
- 新软件 seeds：`sequence_optimization/seeds.py`
- 新软件 CLI：`sequence_optimization/cli.py`
- 新软件模型 examples：`sequence_optimization/examples/models/`
- UTR5 weak 训练：`BioSeqBench/training/train_utr5_weak_cnns.py`
- DREAM weak 训练：`BioSeqBench/training/train_yeast_dream2022_weak_cnns.py`
- 三数据 fixed-budget 旧结果：`BioSeqBench/results/fixed_budget_50k/`
