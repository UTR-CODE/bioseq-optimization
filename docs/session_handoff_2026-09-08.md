# Biosequence Optimization 交接文档（2026-09-08）

> 后续更新：本文第 6 节记录的 NumPy/`ScoreBatch` 接口问题已经解决。
> 最新验证与当前范围见第 9 节。

## 1. 当前状态

- 本地仓库：`/mnt/data/home/hexin/MPRA_benchmark`
- GitHub：`git@github.com:UTR-CODE/bioseq-optimization.git`
- 分支：`main`
- 编写本文档前的提交：`90c6f7b`
- Python 包：`bioseq-optimization==0.1.0`
- 导入名称：`sequence_optimization`
- 模型仓库：`/mnt/data/home/hexin/bioseq-models`
- 模型仓库 GitHub：`git@github.com:UTR-CODE/bioseq-models.git`
- 已检查的模型仓库提交：`f462dc7`

优化仓库已经从原来的 BioSeqBench 中拆出。当前 Git 历史中仍能看到旧的
BioSeqBench 文件，但这些训练、数据处理、模型架构和权重已经不属于本仓库的
当前代码。

## 2. 项目边界

本仓库只负责序列优化：

- 定义待设计的 DNA/RNA 序列空间；
- 接收外部模型或用户函数作为 activity scorer；
- 支持 seed 序列和随机初始化；
- 将单任务或多任务预测转成优化目标；
- 执行黑盒、生成式或梯度优化；
- 统计模型查询次数；
- 返回候选序列、目标分数、原始 activity 和优化轨迹。

本仓库不应重新加入：

- 模型训练代码；
- 具体论文模型的网络架构；
- checkpoint 或其他大文件；
- BioSeqBench 数据集和数据预处理；
- 针对某个具体模型硬编码的输入适配逻辑。

开箱即用的模型由独立的 `bioseq-models` 仓库提供。两个 Python 包之间不应形成
反向依赖：`bioseq-models` 不依赖优化仓库，优化仓库也不把它设为强制依赖，
但必须兼容它公开的 `score()` 接口。

## 3. 已实现的核心能力

### 序列和 scorer

主要实现在 `sequence_optimization/api.py`：

- `SequenceSpace`：固定生物序列长度及字母表；
- `normalize_nucleotide_sequences()`：转大写并将 `U` 转成 `T`；
- `pad_nucleotide_sequences()`：在送入模型前从左侧或右侧补 `N`；
- `CallableScorer`：包装普通 Python 预测函数；
- `TorchSequenceScorer`：包装离散输入的 PyTorch 模型；
- `DifferentiableTorchSequenceScorer`：额外提供 `score_soft()`；
- `CachedScorer`：缓存重复序列并统计 query budget；
- `ScoreBatch`：优化器内部使用的预测容器；
- `OptimizationConfig` 和 `OptimizationResult`：统一配置与结果格式。

每次优化的 `SequenceSpace.length` 是真实生物序列长度。固定长度模型需要的 `N`
padding 属于 scorer 的输入适配，不会进入被优化或返回的序列。所有返回序列均使用
ACGT，不返回 U。

### 优化目标

`sequence_optimization/objectives.py` 中的 `TargetObjective` 支持：

- `mode="maximize"`：只最大化指定输出，忽略其他输出；
- `mode="specificity"`：最大化指定输出，同时惩罚其他输出；
- 使用输出索引或 scorer 的 `task_names` 选择目标；
- 用 `penalty` 和 `other_weights` 控制非目标任务权重。

`OptimizationResult.scores` 是标量化后的优化目标，原始模型输出保存在
`OptimizationResult.activities`，列名保存在 `task_names`。

### 已注册算法

无需梯度、只要求离散序列预测：

- `random`
- `ism_greedy`
- `simulated_annealing`
- `genetic_algorithm`
- `adalead`
- `beam_search`
- `cross_entropy_method`
- `cbas`（内部训练自回归 proposal generator）

要求 scorer 实现可微的 `score_soft()`：

- `vanilla_activation_maximization`
- `fast_seqprop`
- `den`

每种算法既有 `run("name", ...)` 注册接口，也有相应的
`*_optimize()` 便捷函数。`ISM_optimize` 是为了兼容用户指定拼写而保留的别名。

### 查询预算和饱和停止

- `budget` 表示 activity model evaluation 数量，不是训练 epoch；
- 黑盒算法默认通过 `CachedScorer` 只计算新出现的序列；
- CbAS 和三个梯度/生成式算法默认把重复 proposal 也计入预算；
- `run_many()` 给每个算法独立预算和缓存；
- `optimize_to_saturation()` 支持 plateau 检测并返回 q80、q90 等指定位置的候选集；
- `maximum_queries` 是饱和优化的安全上限，不是正常停止目标。

### 命令行

`sequence_optimization/cli.py` 当前通过
`--scorer package.module:function` 加载用户预测函数，支持固定 budget 和
`--budget unlimited` 饱和模式，并写出 CSV 和 JSON metadata。

CLI 目前不能直接通过模型名称加载 `bioseq-models` 模型。

## 4. 代码入口

- `sequence_optimization/api.py`：公共类型、scorer、padding、query cache；
- `sequence_optimization/objectives.py`：单任务和多任务目标；
- `sequence_optimization/algorithms.py`：算法注册、离散算法和统一运行入口；
- `sequence_optimization/advanced.py`：beam、CEM、CbAS 和梯度算法；
- `sequence_optimization/saturation.py`：plateau 与 q-fraction 选择；
- `sequence_optimization/seeds.py`：随机 seed 和文本 seed 文件；
- `sequence_optimization/cli.py`：命令行入口；
- `tests/test_sequence_optimization.py`：当前测试；
- `sequence_optimization/examples/external_model.py`：计划中的 Optimus 联动示例，
  但受下述接口问题影响，当前不能成功运行。

## 5. 已验证结果

在具备本项目依赖的现有环境中执行：

```bash
cd /mnt/data/home/hexin/MPRA_benchmark
/home/hexin/miniconda3/envs/UTR/bin/python -m pytest -q
/home/hexin/miniconda3/envs/UTR/bin/python -m ruff check .
```

结果：

- 14 项现有单元测试全部通过；
- Ruff 0.13.0 检查通过；
- 当前测试覆盖所有 11 种算法的基本结果契约、budget/cache、多任务目标、RNA
  归一化、左右 padding、可微 soft input 和饱和停止；
- 当前测试使用的是小型合成 scorer，没有覆盖拆仓后的真实 `bioseq-models` 联动。

`requirements.txt` 同时列出运行和开发依赖；`pyproject.toml` 只列运行依赖。
项目只提供 pip 安装方式，不再维护 conda environment 文件。

## 6. 必须优先解决的接口问题

### `bioseq-models` 不能直接传给优化器

这是拆仓后尚未完成的主要问题。

当前 `bioseq-models` 的统一契约是：

```python
prediction = model.score(sequences)
# prediction 是形状 [N, tasks] 的 numpy.ndarray
```

当前优化器的 `ObjectiveScorer.score()` 却把返回值当成 `ScoreBatch`：

```python
result = self.scorer.score(values)
activities = result.scores
```

因此 README 和 `external_model.py` 中下面这种预期用法会失败：

```python
model = load_model("optimus5prime")
genetic_algorithm_optimize(model, SequenceSpace(50), ...)
```

2026-09-08 已用最新本地 `bioseq-models` 和 Optimus checkpoint 实际复现：

```text
AttributeError: 'numpy.ndarray' object has no attribute 'scores'
```

不要修改 `bioseq-models.score()` 让它重新返回包装对象。用户已经明确要求该包的
预测结果直接是 NumPy 数组。兼容逻辑应放在优化仓库内。

推荐修复方式：

1. 在 `api.py` 增加一个集中转换函数，接受 `ScoreBatch`、NumPy/array-like，或
   `(scores, uncertainties)`；
2. 让 `ObjectiveScorer.score()` 使用该函数，而不是直接访问 `.scores`；
3. 保留现有 `CallableScorer`、`TorchSequenceScorer` 和用户自定义 `ScoreBatch`
   的兼容性；
4. 更新 `SequenceScorer` Protocol，使类型声明与实际允许的返回类型一致；
5. 从原始 scorer 读取 `task_names`，确保 DeepSTARR 等多任务模型仍可按名称选择输出。

必须增加的回归测试：

- 普通对象的 `score()` 返回 `[N]` NumPy 数组；
- 普通对象的 `score()` 返回 `[N, tasks]` NumPy 数组；
- 返回 `ScoreBatch` 的旧接口继续工作；
- Optimus + `genetic_algorithm_optimize()` 真实端到端测试；
- DeepSTARR + `TargetObjective(target="developmental")` 真实多任务测试；
- 最终结果同时包含标量 `scores` 和原始 `activities`。

## 7. 其他待处理事项

1. README 指向
   `bioseq-models/examples/custom_cnn.py`，但当前模型仓库中没有该文件，链接无效。
   自定义模型示例更适合放在本优化仓库，但不能混入已复现论文模型的源码或权重。
2. `external_model.py` 当前没有集成测试，并会因 NumPy/ScoreBatch 接口问题失败。
3. 三个梯度算法已经实现，但 `bioseq-models` 的公共模型对象只保证离散
   `score()`；目前不能直接用这些模型执行 Gradient/Fast SeqProp/DEN。
   应先完成黑盒联动，再单独定义通用的可微 scorer 接口和示例。
4. 当前 CLI 只支持 Python callable；未来是否增加 `--model optimus5prime` 应在
   保持 `bioseq-models` 为可选依赖的前提下决定。
5. 使用 Ruff 0.16.6 会报告 17 项新版本风格规则问题；这些不是功能错误。若清理，
   应单独提交并避免与接口修复混在一起。
6. 本地存在 Git 忽略的旧 `__pycache__`、egg-info 和 `tmp/` 数据文件。它们不是
   当前仓库内容，不要在没有确认用途时删除或提交。

## 8. 下一阶段建议顺序

1. 修复 NumPy 输出到 `ScoreBatch` 的统一转换；
2. 添加 Optimus 和 DeepSTARR 的真实端到端测试；
3. 修复 README 与 `external_model.py`，保证文档示例可以原样运行；
4. 用所有黑盒算法对一个低 budget 的 Optimus 示例做 smoke test；
5. 构建 wheel，在只通过 pip 安装的临时环境中验证；
6. 再处理可微模型适配和梯度算法验证；
7. 所有修改完成后提交并 push 到 `main`。

第一阶段的完成标准是：

```python
from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize

model = load_model("optimus5prime", device="cpu")
result = genetic_algorithm_optimize(
    model,
    SequenceSpace(50),
    num_candidates=4,
    budget=20,
    seed=7,
)
```

这段代码无需额外包装即可运行，返回4条唯一ACGT序列，并正确记录
`activities`、`scores`、`task_names` 和 `query_count`。

## 9. 后续完成情况

2026-09-09 后续：新增 `basesearch` / `basesearch_optimize()`，算法总数为 13
（九种黑盒、四种梯度）。实现基于 BaseSearch 论文方法描述，使用优先级衰减、
单点/多点突变、全局去重与队列裁剪；具体本地参数选择见 README。

2026-09-09 后续：新增注册算法 `ledidi` 和 `ledidi_optimize()`，算法总数为 12。
实现为 Ledidi 参数化与 hard Gumbel-softmax 的单输出最大化适配，要求可微 scorer；
默认编辑惩罚 0.1，结果仍按 activity 排序。真实 Optimus 与 Mouse Heart 模型
已验证，详见 README 的 Ledidi 示例。此前 32 模型矩阵不包含 Ledidi。

2026-09-09：已完成全部 32 个单输出模型的算法兼容性测试，详见
[测试报告](single_task_model_compatibility_2026-09-09.md)。全部模型通过八种黑盒
算法；28 个模型通过可微 wrapper 支持三种梯度算法（Cetnar 使用 CPU）。
DREAM、PARM、UTR-CODE、CDS Stability 尚需专门的可微输入适配。

- 优化仓库已增加统一 scorer 输出转换，接受 NumPy `[N]`、`[N, 1]`、
  `(scores, uncertainties)` 和原有 `ScoreBatch`；
- `bioseq-models` 的单输出模型现在可以直接传给全部八种黑盒算法；
- Optimus 已对全部 11 种注册算法完成真实 checkpoint 测试；其中三种梯度算法
  需要用 `DifferentiableTorchSequenceScorer` 暴露 soft one-hot 输入；
- 当前产品范围是每次优化一个 activity scorer。不同细胞类型的单输出模型可以
  分别优化，但不组合多个模型对象进行联合优化；
- `tests/test_bioseq_models_integration.py` 会检查算法注册表中的每一种算法，避免
  新增算法后遗漏真实 Optimus 验证；
- 文档提供两个可运行示例：
  `sequence_optimization/examples/custom_cnn.py` 和
  `sequence_optimization/examples/external_model.py`。

## 10. 2026-09-09 multi-task 目标

新增 `MultiTaskObjective`，在同一个模型内选择 high/low 输出，支持 worst、mean、
smoothmin 聚合、固定 centers/scales，以及完整的 lower_bounds/upper_bounds。
NumPy 和 PyTorch 使用一致的目标定义；Ledidi 已解除单输出限制。
结果保留原始 activities，并通过 metadata 和 to_frame 展示 margins；有阈值时
还展示每个任务和全部任务是否达标。未设阈值时不声称达到绝对高低标准。

README 单任务示例为 Optimus；multi-task 示例为 DeepSTARR 249 nt 的
developmental/housekeeping 两个 enhancer reporter 输出，涵盖双高和两个方向的
一高一低。可运行文件：`sequence_optimization/examples/deepstarr_multitask.py`。
这两个输出不是两个细胞系；不组合多个独立模型。

真实 checkpoint 集成测试覆盖 13 种算法 × 三种 DeepSTARR 目标，并保留全部
Optimus 检查。低预算测试验证接口、查询预算和原始预测复算，不代表设计效果 benchmark。
本次完整测试结果为 114 passed（一个已有 PyTorch same-padding warning）；
Optimus 和 DeepSTARR 可运行示例均已执行成功。

### 简化后的公开用法

直接传入 `objective=["maximize", "minimize", "ignore"]`，按模型输出列顺序
表示更高、更低、不参与。列表长度必须匹配输出数，至少一个任务参与优化。
同一接口适用于单输出（`["maximize"]`）、多输出、黑盒和梯度算法及 saturation。
忽略列仍保留原始预测，但不影响目标和梯度。不依赖模型名称或任务名称；
bioseq-models 仅在示例和集成测试中加载，优化核心没有模型专用分支。
保留 MultiTaskObjective 高级用法；README 以简短 Python 列表示例为主，不提供 CLI。
列表接口完成全部 13 算法的 DeepSTARR 验证，完整测试 120 passed。

## 11. 多模型组合

新增 `CombinedScorer(models=[...], names=[...])`：每个子 scorer 单输出，
同一批序列依次交给各 scorer，按列表顺序拼接输出。模型编码、padding 和上下文
由子 scorer 负责；优化核心不导入 bioseq-models，不含模型名称分支。
复用 `objective=["maximize", "minimize", "minimize"]` 和现有 13 种算法。
梯度模式要求每个子 scorer 提供 score_soft，保留跨设备转换的梯度链。

示例 `sequence_optimization/examples/multiple_models.py` 提供 lentiMPRA
K562/HepG2/WTC11（230 nt）和 DeepSTARR Mouse heart/limb/midbrain activity
（1001 nt）。设计序列和预测保存在 `docs/multiple_model_examples_2026-09-09.json`。
100-query Mouse 示例没有证明 heart 特异性；文档明确区分执行兼容与设计效果。
模型调用计数为累计实际序列评估数；重复候选缓存命中时可能只增加优化预算计数。
README 只展示 Python 接口，已移除旧 CLI 用法章节。
验证：原有 120 项测试通过；新增 30 项组合测试通过，包括两组真实模型各 13 种
算法的预测复算、预算及梯度检查。最初四项预算断言混淆缓存命中与模型实际调用，
修正后完整重跑新增测试为 30 passed。

## 12. 简化公开接口

所有公开优化函数、run/run_many、saturation 和 seed helpers 改为接收 `length`，
可选 `alphabet="ACGT"`，不要求调用者构造 SequenceSpace。内部继续使用该数据类型
完成变异与合法性检查，旧的 positional SequenceSpace 调用保留兼容。
README 和所有 Python 示例均使用 length；真实模型集成测试也使用整数长度。

输入转换函数命名为 `onehot`，CNN wrapper 默认处理输入转换。
文档保持简短，不单独介绍转换函数、实现来源或性能对照。
完整测试 175 passed；简化后的 custom CNN 示例运行成功。
