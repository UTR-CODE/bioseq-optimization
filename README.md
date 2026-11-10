# bioseq-optimization

整合多种优化算法设计调控序列，再根据 **motif组成与位置的多样性** 选择实验候选。

预测模型来自独立的 [bioseq-models](https://github.com/UTR-CODE/bioseq-models/tree/main/bioseq_models)，统一通过 `from bioseq_models import load_model` 调用。本仓库维护优化、候选记录、motif距离和候选选择，不携带模型结构或权重。

```text
100条相同起始序列
        ↓
11种优化算法，各自stop on plateau，无默认查询budget
        ↓
各取90%增益档的100条候选 → 1,100条来源记录
        ↓
合并相同DNA，保留algorithm provenance
        ↓
贡献分 → seqlets → motif catalog → FIMO重新扫描
        ↓
质量门槛 → motif组合Max-Min → 位点位置Max-Min → 预测分数
        ↓
Top N候选、FASTA、预测值、motif位置与选择过程
```

1,100是**来源记录数**；不同算法可能得到相同DNA，去重后唯一序列数可能少于1,100。Random Search不包含在这11种算法中，可另外预测200,000条随机序列作参考。90%指本次搜索观察到的**增益比例**，不是预测概率、90%分位数或已知全局最优的90%。

## 从哪里开始

| 文档 | 内容 |
|---|---|
| [安装与模型接口](docs/installation.md) | 安装、环境、模型下载、输入输出约定 |
| [11算法生成1,100条候选](docs/candidate-generation.md) | 完整优化循环、增益档、平台期、查询量、时间统计 |
| [Enhancer逐步示例](docs/examples/enhancer.md) | Malinois；K562高于HepG2和SKNSH；切换目标细胞；motif筛选 |
| [Promoter逐步示例](docs/examples/promoter.md) | DREAM LegNet酵母promoter；80 nt插入片段；固定实验上下文 |
| [5′UTR逐步示例](docs/examples/5utr.md) | Optimus-5′ Prime；50 nt；提高MRL；Top10与uAUG检查 |
| [贡献分、seqlets与motif发现](docs/motif-discovery.md) | ISM/DeepLIFTShap、tangermeme、STREME、独立支持与验证 |
| [Motif diversity选择Top N](docs/motif-diversity.md) | PWM相似性、Soft-Jaccard、Hungarian、分层Max-Min；含免GPU真实数据示例 |
| [多进程与GPU队列](docs/parallel-scoring.md) | 多个CPU优化进程、单GPU批量评分、梯度请求和退出处理 |
| [算法与停止规则](docs/algorithms.md) | 11种算法名称、梯度需求、饱和/增益参数 |
| [验证与结果边界](docs/validation.md) | 自动测试、真实模型检查、参考数据来源与未覆盖部分 |

上述文档按步骤给出可以逐段执行的代码。三个主要示例也提供逐步Notebook：[enhancer](examples/enhancer.ipynb)、[promoter](examples/promoter.ipynb)、[5′UTR](examples/5utr.ipynb)。另有可直接执行的[免GPU motif选择Notebook](examples/motif_diversity.ipynb)。Enhancer的motif步骤需要先按发现文档准备catalog；按章节顺序执行代码块；示例中的生产规模为11×100，运行时间取决于模型、长度和平台期。Notebook的输出目录必须为空，重复实验请更改目录。

## 安装

```bash
git clone https://github.com/UTR-CODE/bioseq-optimization.git
cd bioseq-optimization
python -m pip install -e '.[analysis,motifs,dev]'
python -m pip install 'bioseq-models @ git+https://github.com/UTR-CODE/bioseq-models.git'
```

MEME Suite的FIMO/STREME/Tomtom以及R属于可选外部工具，安装方法见[安装文档](docs/installation.md)。普通优化不需要这些工具；扫描和发现motif时才需要。可选C扩展在安装时编译，没有C编译器时保留Python实现。

## 最小接口

```python
from bioseq_models import load_model
from sequence_optimization import random_seed_sequences
from sequence_optimization.workflows import generate_candidates

model = load_model("optimus5prime", device="cuda", batch_size=512)
initial = random_seed_sequences(50, 100, seed=42)
records, times = generate_candidates(
    model, 50, initial, ["maximize"],
    output_dir="outputs/utr_run1", num_candidates=100,
)
assert len(records) == 1100
print(times[["algorithm", "seconds", "queries", "stop_reason"]])
```

完整示例解释每个参数、保留原始预测值，并给出后续选择与导出步骤。没有可靠motif catalog时，promoter和5′UTR示例明确使用DNA序列多样性；它不等同于motif机制多样性。模型高分属于预测结果，需要实验验证。
