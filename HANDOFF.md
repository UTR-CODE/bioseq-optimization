# bioseq-optimization 开发交接

本文面向接手开发、维护和发布工作的工程人员，描述仓库当前结构与稳定契约。

## 1. 项目定位

`bioseq-optimization` 是一个与数据集和模型结构解耦的 DNA/RNA 序列优化库。
优化器只依赖统一的序列评分接口，不保存模型权重、训练数据或生物任务专用训练代码。

- Git 仓库：`git@github.com:UTR-CODE/bioseq-optimization.git`
- 主分支：`main`
- Python distribution：`bioseq-optimization`
- Python import：`sequence_optimization`
- `pyproject.toml` 版本：`0.1.0`
- Python 要求：`>=3.10`
- 运行依赖：NumPy、pandas、PyTorch
- 可选模型仓库：`bioseq-models`，Python import 为 `bioseq_models`

项目职责包括：

- 固定长度核酸序列空间及种子序列管理；
- 单任务、多任务和多个独立模型的目标标量化；
- 9 个黑盒优化器和 4 个梯度优化器；
- 固定预算与饱和停止；
- 硬约束、概率约束及多约束交集筛选；
- 同义密码子范围内的 CDS 遗传算法；
- 模型适配器、结果对象、命令行入口和测试。

模型实现、checkpoint 下载、模型专用编码器和训练流程属于
`bioseq-models`。大型数据、临时结果和构建产物不属于本仓库。

## 2. 建议阅读顺序

接手开发时按以下顺序阅读：

1. [仓库规则](AGENTS.md)：文档、示例、检查和提交要求。
2. [README](README.md)：公共用法和主要示例。
3. [公共 API](sequence_optimization/__init__.py)：公开导出边界。
4. [核心数据结构](sequence_optimization/api.py)：scorer、sequence space、配置、缓存和结果。
5. [算法调度](sequence_optimization/algorithms.py)：注册表、公共入口和基础算法。
6. [高级算法](sequence_optimization/advanced.py)：生成式、队列式和梯度算法。
7. [目标函数](sequence_optimization/objectives.py)、[约束](sequence_optimization/constraints.py)和
   [饱和停止](sequence_optimization/saturation.py)。
8. [测试目录](tests)及[算法参数索引](docs/algorithms.md)。

## 3. 目录结构

```text
bioseq-optimization/
├── sequence_optimization/        # 可发布 Python 包
│   ├── __init__.py               # 公共导出
│   ├── api.py                    # scorer、配置、缓存、序列空间、结果
│   ├── algorithms.py             # 调度、公共 optimizer、基础黑盒算法
│   ├── advanced.py               # 高级黑盒、生成式和梯度算法
│   ├── objectives.py             # TargetObjective、MultiTaskObjective
│   ├── constraints.py            # 用户约束标准化、筛选和统计
│   ├── saturation.py             # 饱和监测和 q-fraction 结果
│   ├── combined.py               # 多个独立单输出 scorer 的组合
│   ├── codons.py                 # 标准遗传密码表和同义 CDS 空间
│   ├── onehot.py                 # A/C/G/T/U/N 向量化编码
│   ├── seeds.py                  # 随机种子和文件种子
│   ├── cli.py                    # python -m 命令行入口
│   └── examples/                 # 可运行示例和模型任务配方
├── docs/
│   ├── algorithms/               # 每个 optimizer 的参数页
│   ├── models/                   # bioseq-models 任务用法
│   ├── tasks/                    # 多任务、多模型和同义 CDS 用法
│   ├── constraints.md
│   ├── custom_models.md
│   ├── objectives.md
│   └── search_workflow.md
├── tests/                        # 单元、契约及可选真实模型测试
├── scripts/                      # 开发者兼容性检查脚本
├── pyproject.toml
├── requirements.txt
└── README.md
```

`.gitignore` 排除缓存、构建目录、本地结果、临时文件和旧的本地 benchmark
数据。开发脚本生成的报告应写入 `tmp/` 或其他不提交的工作目录。

## 4. 调用链与模块边界

```text
*_optimize(...)
    │
    ▼
optimize(...)
    ├── 固定预算 ──► OptimizationConfig ──► run(...)
    └── saturation ─► _optimize_with_saturation(...)
                                              │
SequenceSpace / SynonymousCodonSpace          │
              │                               │
              ▼                               ▼
        初始序列校验  ──► ObjectiveScorer ──► CachedScorer
                                               │
                              constraint 筛选 ─┤
                                               ▼
                                    注册表中的算法实现
                                               │
                                               ▼
                              archive ──► OptimizationResult
```

主要边界：

- optimizer 不解析模型结构，只调用 `score` 或 `score_soft`；
- `ObjectiveScorer` 保存原始 activity，并向算法提供一维优化分数；
- `CachedScorer` 负责去重、预算计数、约束筛选和模型调用缓存；
- 算法只维护候选状态和一维 objective score；
- `OptimizationResult` 在搜索结束后附加原始 activity、任务名、约束概率和元数据；
- 饱和模式复用同一算法注册表，仅替换 scorer 的监测与停止行为。

## 5. 公共评分契约

### 5.1 黑盒 scorer

对象必须实现：

```python
scores = scorer.score(sequences)
```

`sequences` 是一维字符串集合。允许的返回值：

- `[N]` 数组；
- `[N, T]` 多任务数组；
- `ScoreBatch(scores, uncertainties)`；
- `(scores, uncertainties)` 二元组。

每个输入必须对应一行输出，所有 score 和 uncertainty 必须为有限值。
uncertainty 若存在，其形状必须与 score 完全一致。

### 5.2 PyTorch 适配器

`SequencePredictor` 默认把字符串转换为 float32 `[N, 4, L]`，通道顺序为
A/C/G/T。可通过 `encoder` 或 `onehot` 自定义布局和通道顺序。

`DifferentiableSequencePredictor` 额外提供：

```python
activities, uncertainties = scorer.score_soft(probabilities)
```

optimizer 提供的标准软输入形状为 `[N, L, 4]`，通道顺序为 A/C/G/T。
`soft_input_transform` 必须在不 detach、不转 NumPy、不离散化的前提下转换为
模型需要的布局。包装后模型参数被冻结，但输入梯度保留。

### 5.3 多个独立模型

`CombinedScorer` 要求每个子 scorer 只返回一个输出，并按 `names` 顺序拼成
`[N, T]`。每个子模型可以拥有独立的编码、padding、上下文和设备。

- `model_query_counts`：各子模型累计离散序列调用数；
- `model_soft_query_counts`：各子模型累计软输入调用数；
- 任一子模型不提供 uncertainty 时，组合 uncertainty 为 `None`；
- 梯度优化要求所有子模型都实现 `score_soft`。

## 6. 序列空间和输入规则

`SequenceSpace` 负责长度、字母表、随机采样、编码、解码和单碱基突变。

- 默认优化字母表为 `ACGT`；
- 输入统一转大写，`U` 归一化为 `T`；
- 候选必须等长且只能包含配置字母表中的字符；
- `N` 只用于模型输入 padding，不属于默认优化候选；
- 当前优化空间只包含替换，不包含插入或删除；
- `random_seed_sequences` 返回可复现且唯一的随机种子；
- `load_seed_sequences` 读取每行一条序列，忽略空行和 `#` 注释行。

`pad_nucleotide_sequences` 支持左侧或右侧补 `N`。离散 one-hot 中 `N` 为全零；
软输入 padding 使用等价的全零向量。

`diversity` 默认为 5，规定多个返回序列之间的最小两两 Hamming distance；设为 0
可关闭该筛选。无法从已评分 archive 选足 `num_candidates` 时，optimizer 明确报错。

## 7. 目标函数

### 7.1 方向列表

公共 optimizer 的 `objective` 可以直接接收与模型输出顺序一致的列表：

- `maximize`：提高该输出；
- `minimize`：降低该输出；
- `ignore`：不参与优化，但仍预测并返回。
- 有限数值：该输出的 raw activity 下限；达到后成为后续筛选门槛。

方向列表内部转换为 `MultiTaskObjective`，默认使用 active signed outputs 的等权平均。
`model_weights` 可按模型输出顺序设置正的相对权重，并使用归一化加权平均。
只有数值目标和 `ignore` 时，同一候选达到全部数值目标即停止；数值目标不与
`saturation` 同时使用。

### 7.2 `TargetObjective`

用于指定单个目标输出：

- `mode="maximize"`：只最大化目标；
- `mode="specificity"`：目标减去其他输出的加权惩罚；
- `target` 可用列索引或 scorer 的任务名；
- `other_weights` 可用列表或索引/名称映射。

### 7.3 `MultiTaskObjective`

提供 `worst`、`mean` 和 `smoothmin` 聚合，以及固定 `centers`、`scales`、
`lower_bounds`、`upper_bounds`。使用 bounds 时，`worst_margin >= 0` 才表示所有选中
目标同时满足阈值。归一化尺度必须来自固定参考集，不能随候选 batch 重算。

## 8. 候选约束

每个公开 optimizer 的 `constraint` 都接受一个 callable 或 callable 列表。每个函数
接收一条或一组序列，并为每条序列返回 `[0, 1]` 内的有限通过概率。

- `0` 必定拒绝，`1` 必定通过；
- 中间值执行一次 Bernoulli 筛选；
- 多个函数独立执行，只有全部通过的交集进入模型和下一轮；
- 同一唯一序列保留第一次筛选结果，避免通过重复提议反复抽签；
- constraint 使用由 optimizer `seed` 派生的独立随机数生成器；
- 无效形状、非有限值、越界概率及空函数列表立即报错；
- 初始候选不足时，从相同 sequence space 补充通过约束的唯一候选；
- 梯度算法先离散化候选，只有通过者参与该轮可微 objective。

结果中的相关字段：

- `constraint_scores`：最终候选的原始通过概率，形状 `[N, C]`；
- `constraint_names`：约束列名；
- `metadata["constraint"]`：逐约束通过统计、交集统计、筛选计数和模型计数；
- `to_frame()`：生成 `constraint_<name>` 列。

完整用法见[约束文档](docs/constraints.md)。

## 9. optimizer 清单

| 注册名 | 公共函数 | 类型 | 实现文件 | 关键状态 |
|---|---|---|---|---|
| `random` | `random_optimize` | 黑盒 | `algorithms.py` | 均匀随机筛选 |
| `genetic_algorithm` | `genetic_algorithm_optimize` | 黑盒 | `algorithms.py` | elite、交叉、突变 |
| `ism_greedy` | `ism_optimize` | 黑盒 | `algorithms.py` | 单碱基邻域贪心扫描 |
| `simulated_annealing` | `simulated_annealing_optimize` | 黑盒 | `algorithms.py` | 温度控制接受率 |
| `adalead` | `adalead_optimize` | 黑盒 | `algorithms.py` | elite 亲本局部突变 |
| `beam_search` | `beam_search_optimize` | 黑盒 | `advanced.py` | 共享 top-k beam |
| `cross_entropy_method` | `cross_entropy_optimize` | 黑盒 | `advanced.py` | 位置独立 categorical 分布 |
| `basesearch` | `basesearch_optimize` | 黑盒 | `advanced.py` | 优先队列和 priority decay |
| `cbas` | `cbas_optimize` | 黑盒生成式 | `advanced.py` | 自回归 GRU proposal |
| `vanilla_activation_maximization` | `vanilla_activation_maximization_optimize` | 梯度 | `advanced.py` | soft position logits |
| `fast_seqprop` | `fast_seqprop_optimize` | 梯度 | `advanced.py` | 标准化 logits 和 straight-through sample |
| `den` | `den_optimize` | 梯度 | `advanced.py` | noise-to-sequence generator |
| `ledidi` | `ledidi_optimize` | 梯度 | `advanced.py` | seed prior、edit penalty、Gumbel sampling |

`ISM_optimize` 是 `ism_optimize` 的公开别名。新增代码应使用小写名称。

### 9.1 注册和调度

算法实现通过 `@register_algorithm("name")` 写入 `_REGISTRY`。`advanced.py` 在
`algorithms.py` 末尾导入，导入包时完成高级算法注册。`available_algorithms()` 返回
稳定排序后的注册名。

`run(name, scorer, length, config, ...)` 是配置对象入口；各 `*_optimize` 包装函数走
`optimize(...)`，由它构造 `OptimizationConfig` 并选择固定预算或饱和模式。

### 9.2 查询计数

默认情况下，离散算法按未缓存唯一序列计数。以下算法按 proposal 计数，即使候选
重复也消耗预算：

- `cbas`；
- `vanilla_activation_maximization`；
- `fast_seqprop`；
- `den`；
- `ledidi`。

其他算法可通过 `count_repeated=True` 改为 proposal 计数。梯度算法额外的软 forward/
backward 不计入离散 `query_count`。`budget` 至少为 `num_candidates`；某些算法可能因
空间耗尽、重复停滞或不足以展开完整邻域而提前停止。

## 10. 固定预算和饱和停止

固定预算调用省略 `budget` 时默认使用 10,000，返回 `OptimizationResult`。

饱和调用提供 `saturation=SaturationConfig(...)` 并省略 `budget`，返回
`SaturationResult`。两者同时提供会报错。

饱和监测使用当前最优 `num_candidates` 个唯一序列的平均 objective：

- `minimum_queries`：允许判定 plateau 前的最小查询数；
- `plateau_patience_queries`：无 material improvement 的容忍查询数；
- `maximum_queries`：安全上限；
- `material_improvement`：有效提升阈值；
- `fractions`：从首个完整 archive 到最终观测 plateau 的增益比例。

`SaturationResult.selections[fraction]` 和 `.at(fraction)` 返回对应最早 snapshot。
`stop_reason` 为 `plateau`、`maximum_queries` 或 `algorithm_stopped`。该机制描述观测到的
搜索轨迹，不保证全局最优。

## 11. 同义 CDS 优化

同义 CDS 搜索只允许 `genetic_algorithm_optimize`。设置：

- `reference_cds`：完整、frame 内的 ACGT CDS；
- `synonymous_codons`：可选的氨基酸到允许密码子列表；
- `initial`：明确准备的唯一同义 CDS 列表。

`SynonymousCodonSpace` 使用 NCBI 标准遗传密码表 1，并校验：

- 首密码子是 `ATG`；
- 末密码子是 `TAA`、`TAG` 或 `TGA`；
- 内部无终止密码子；
- 初始和生成候选与 reference 翻译结果一致；
- 自定义 codon 表不改变氨基酸身份。

交叉和突变以完整密码子为单位。其他 optimizer 收到 `reference_cds` 时主动报错。
详见[同义 CDS 文档](docs/tasks/synonymous_cds.md)。

## 12. 结果契约

`OptimizationResult` 的主要字段：

| 字段 | 含义 |
|---|---|
| `algorithm` | 注册算法名 |
| `sequences` | 唯一且按 objective 排序的最终序列 |
| `scores` | 一维标量 objective |
| `uncertainties` | 与 score 对齐的一维不确定度；未知时填零供算法内部使用 |
| `activities` | 原始模型输出 `[N, T]` |
| `task_names` | activity 列名 |
| `query_count` | 按当前计数模式累计的离散查询 |
| `metadata` | objective、constraint、算法和停止信息 |
| `trajectory` | 算法迭代或饱和监测轨迹 |
| `constraint_scores` | 最终候选的约束通过概率 |
| `constraint_names` | 约束名称 |

`to_frame()` 是面向表格输出的稳定入口，包含 sequence、score、uncertainty、activity、
constraint 以及多任务 margin/satisfaction 列。算法专用诊断仍从 `metadata` 和
`trajectory` 读取。

## 13. 命令行入口

仓库未声明独立 console script；使用模块入口：

```bash
python -m sequence_optimization.cli \
  --scorer my_package.predictors:score_sequences \
  --constraint my_package.constraints:gc_filter \
  --constraint my_package.constraints:motif_filter \
  --algorithm genetic_algorithm \
  --length 50 \
  --budget 1000 \
  --num-candidates 100 \
  --output results/designs.csv
```

scorer 和 constraint 必须是当前 Python 环境可导入的 callable。重复 `--constraint`
表示独立约束交集。`--budget unlimited` 启用饱和模式。固定预算输出 CSV 和相邻 JSON
元数据；饱和模式在 CSV 中增加 `saturation_fraction`。

## 14. 与 bioseq-models 的协作边界

基础安装不依赖 `bioseq-models`。需要预训练模型时单独安装：

```bash
python -m pip install git+ssh://git@github.com/UTR-CODE/bioseq-models.git
```

`bioseq-models` 应负责：

- checkpoint 定位、下载和校验；
- 模型结构和权重加载；
- 任务名、期望序列长度及固定上下文；
- 字符串编码和批推理；
- 可行时提供与离散输入一致的可微输入路径。

本仓库应负责：

- 接收已经实现评分契约的模型；
- 目标、约束、候选生成、预算和停止；
- 与模型无关的结果结构；
- 通用 `SequencePredictor` 和 `CombinedScorer`。

模型专用适配逻辑不要写入 optimizer 内核。模型输入约束应记录在
[模型输入要求](docs/model_input_requirements.md)和对应任务页。

## 15. 开发环境和验证

### 15.1 安装

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install pytest ruff
```

### 15.2 核心测试

不需要预训练模型仓库：

```bash
python -m pytest -q \
  --ignore=tests/test_bioseq_models_integration.py \
  --ignore=tests/test_combined_models_integration.py
```

### 15.3 可选模型契约测试

安装 `bioseq-models` 后运行：

```bash
python -m pytest -q \
  tests/test_bioseq_models_integration.py \
  tests/test_combined_models_integration.py \
  tests/test_model_task_recipes.py
```

真实 checkpoint 的小预算兼容性矩阵由下列脚本生成：

```bash
python scripts/check_single_task_models.py \
  --device cpu \
  --output tmp/model_compatibility.json
```

该脚本检查接口和结果契约，不评估优化效力。

### 15.4 格式和静态检查

```bash
python -m ruff format --check sequence_optimization tests scripts
python -m ruff check sequence_optimization tests scripts
git diff --check
```

文档修改还应解析所有 Python fenced code block，并检查相对链接存在。README 只保留
使用方式、参数、示例和必要限制；开发记录与验证结果不要作为用户文档提交。

## 16. 测试文件职责

| 文件 | 覆盖范围 |
|---|---|
| `test_sequence_optimization.py` | 公共结果契约、预算、算法集合、饱和模式 |
| `test_constraints.py` | 单/批约束、概率、交集、统计和全部 optimizer 参数 |
| `test_multitask.py` | 方向、聚合、bounds、scale 和可微标量化 |
| `test_combined.py` | 多模型拼接、设备、shape 和计数器 |
| `test_synonymous_cds.py` | CDS 边界、翻译一致性、codon 表和算法限制 |
| `test_onehot_and_length.py` | 编码、padding、长度和所有算法的输入边界 |
| `test_sequence_predictor.py` | PyTorch 离散/软适配器 |
| `test_basesearch.py` | 优先队列、负分、stall 和 metadata |
| `test_ledidi.py` | Ledidi 预算、梯度、编辑惩罚和错误路径 |
| `test_custom_cnn_tasks.py` | 三种 custom CNN 示例 |
| `test_multitask_cnn_example.py` | 三输出 CNN 示例 |
| `test_model_task_recipes.py` | 模型任务参数和固定上下文 wiring |
| `test_bioseq_models_integration.py` | 单模型真实 checkpoint 契约 |
| `test_combined_models_integration.py` | 三模型真实 checkpoint 契约 |

## 17. 扩展流程

### 17.1 新增 optimizer

1. 在 `algorithms.py` 或 `advanced.py` 实现统一内部签名：
   `initial, scorer, space, config, rng`。
2. 使用 `@register_algorithm("name")` 注册。
3. 所有离散评分通过 `_score`，确保 cache、constraint 和 archive 语义一致。
4. 需要保留候选状态时，显式排除未通过 constraint 的候选。
5. 在 `OptimizationConfig` 添加并校验专用参数。
6. 在 `optimize` 转发参数，并新增显式 `*_optimize` 公共包装函数。
7. 从 `sequence_optimization.__init__` 导出。
8. 在 README 算法表、`docs/algorithms.md` 和独立参数页登记；README 主优化示例仍使用遗传算法。
9. 增加公共契约、边界、约束、固定预算和饱和测试。
10. 如需真实模型，扩展可选 integration matrix。

### 17.2 新增模型支持

1. 优先在 `bioseq-models` 实现模型、权重和字符串 scorer。
2. 明确长度、通道顺序、padding、固定上下文、任务名和 uncertainty。
3. 梯度支持必须验证离散 `score` 与 `score_soft` 在 one-hot 输入上的预测一致。
4. 在本仓库增加对应 `docs/models/<task>.md`，逐个展示满足输入要求的 optimizer 调用。
5. 模型专有限制留在任务页，不向通用 optimizer 添加模型名称判断。

### 17.3 修改结果或预算语义

这两类修改会影响所有算法。至少同步检查：

- `_score`、`_finalize` 和 `CachedScorer`；
- 固定预算与 saturation 两条路径；
- `OptimizationResult.to_frame()`；
- `CombinedScorer` 累计计数；
- constraints 的 screened、counted 和 model query 统计；
- 黑盒、生成式和梯度算法测试。

## 18. 发布和提交

仓库使用 setuptools 和 `pyproject.toml`，没有 lockfile、自动版本生成或 console-script
声明。仓库也未配置 `.github/workflows` 或 `LICENSE`；若需要自动发布或对外分发，先由
维护者确定 CI、版本和许可证策略。发布前流程：

1. 确认公共 API、README、独立参数页和示例一致；
2. 按改动风险运行核心测试及可选模型测试；
3. 运行 Ruff、文档代码块/链接检查和 `git diff --check`；
4. 若准备新版本，更新 `pyproject.toml` 中的 version；
5. 检查 `git status`，不提交 checkpoint、生成结果、cache、build 或临时数据；
6. 提交到 `main` 并推送 `origin`。

## 19. 设计限制

- 候选空间为固定长度字符串，只实现替换搜索；
- 基础 optimizer 不负责模型训练或效力评估；
- 梯度 optimizer 依赖完整可微的 `score_soft` 路径；
- Python constraint 作用于离散序列，本身不可微；
- 概率 constraint 对每个唯一序列只抽样一次；
- saturation 是观测停止规则，不是全局最优证明；
- 同义 CDS 使用标准遗传密码表 1，且只由遗传算法处理；
- `CombinedScorer` 每个子模型必须是单输出；多输出单模型直接作为一个 scorer 使用；
- 基础依赖不安装预训练模型包，因此真实模型测试属于可选层；
- 公共 API 尚未提供向后兼容版本策略，修改导出名称和结果字段时需谨慎。

## 20. 维护原则

- 通用搜索逻辑留在本仓库，模型和数据逻辑留在各自项目；
- 所有算法共享 scorer、objective、constraint、budget 和 result 契约；
- 不绕过缓存评分入口复制一套查询统计；
- 不把任务专用规则硬编码进通用算法；CDS sequence space 是明确的结构化例外；
- 用户文档描述当前可用接口，不记录开发过程；
- 示例优先短小、可执行，并显式展示 `initial`、`objective`、预算或 saturation；
- 任何新增能力都同时覆盖固定预算、饱和模式和错误路径。
