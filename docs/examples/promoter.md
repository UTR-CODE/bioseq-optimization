# Promoter：DREAM LegNet高活性酵母启动子

[返回主文档](../../README.md) · [Notebook](../../examples/promoter.ipynb) · [候选生成](../candidate-generation.md)

这个例子是**酵母promoter**。优化80 nt插入片段，固定实验接头；不是人类promoter，也不把接头当成可优化区域。

| 区域 | 长度 | 内容 |
|---|---:|---|
| 左实验接头 | 17 | `TGCATTTTTTTCACATC` |
| 可优化插入片段 | 80 | 候选DNA |
| 右实验接头 | 13 | `GGTTACGGCTGTT` |
| 导出的完整实验构建 | 110 | 左接头+插入+右接头 |

官方adapter在内部把93 nt区域转换成150 nt模型输入。包装只补充固定右接头和梯度路径的同样转换，保留AGCT通道、singleton通道以及正反链平均。

## 1. 模型、种子与输入检查

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bioseq_models import load_model
from sequence_optimization import random_seed_sequences
from sequence_optimization.models import PromoterScorer, validate_scorer, PROMOTER_LEFT, PROMOTER_RIGHT
from sequence_optimization.workflows import generate_candidates, select_sequence_panel, write_fasta

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs/promoter_run1")
raw = load_model("dream_legnet", variable_region_only=True, device=DEVICE, batch_size=512)
scorer = PromoterScorer(raw)
initial = random_seed_sequences(80, 100, seed=42)
print(validate_scorer(scorer, initial[:4]))
print(scorer.score(initial[:4]))
```

模型预测预期FACS bin，不能解释成表达倍数。输入包装拒绝非80 nt候选，避免误把93 nt区域再次补接头。

## 2. 11算法各100条

```python
records, times = generate_candidates(
    scorer, 80, initial, ["maximize"],
    output_dir=OUT, num_candidates=100, diversity=5, seed=42,
)
assert len(records) == 1100
assert records.sequence.str.len().eq(80).all()
print(times[["algorithm", "seconds", "queries", "stop_reason"]])
```

## 3. 高活性门槛与序列多样性

当前示例没有独立验证的酵母motif catalog，因此先提供**DNA多样性**筛选。对唯一DNA设定活性前10%门槛，然后从最高活性候选开始，逐次最大化与已选候选的最小Hamming距离；距离相近1 nt内优先更高预测值。

```python
from sequence_optimization.workflows import deduplicate_candidates
unique, provenance = deduplicate_candidates(records)
gate = float(unique.score.quantile(0.9))
selected, distances = select_sequence_panel(records, top_n=10, min_score=gate, distance_tie_nt=1)
print(selected[["selection_rank", "sequence_id", "score", "algorithms"]])
print("Minimum pairwise DNA difference:", distances[np.triu_indices(10, 1)].min())
```

这个门槛是预测分数分位数，与优化器90%增益档不是同一个含义。若合格唯一DNA不足10条，会报错；需检查分布后修改门槛，不能补重复序列。

## 4. 输出插入片段与实验构建

```python
panel = Path("outputs/promoter_top10_run1")
panel.mkdir(parents=True, exist_ok=False)
selected["construct_110nt"] = PROMOTER_LEFT + selected.sequence + PROMOTER_RIGHT
assert selected.construct_110nt.str.len().eq(110).all()
selected.to_csv(panel / "selected_sequences.csv", index=False)
write_fasta(panel / "inserts_80nt.fa", selected.sequence_id, selected.sequence)
write_fasta(panel / "constructs_110nt.fa", selected.sequence_id, selected.construct_110nt)
np.savetxt(panel / "hamming_distances.csv", distances, delimiter=",", fmt="%d")
```

## 5. 用ISM解释Top10

```python
from sequence_optimization import ism_contributions
contribution = ism_contributions(
    selected.sequence.to_numpy(), lambda seq: scorer.score(seq)[:, 0], batch_size=512,
)
np.savez_compressed(panel / "top10_ism.npz", observed=contribution.observed,
                    logo=contribution.logo, mutants=contribution.mutant_predictions)
print(contribution.observed.shape)  # (10,80)
```

每条需要80×3次替换，10条共2,400次突变预测，加原序列评分。正贡献表示该位点实际碱基相对四种碱基平均的局部优势，不是加和分解。若继续做motif diversity，应对整个候选池提取seqlets、独立验证酵母catalog，再接[motif发现](../motif-discovery.md)和[分层选择](../motif-diversity.md)；不要使用人类enhancer catalog替代。

## 6. 画局部ISM贡献logo

```python
import matplotlib.pyplot as plt
import logomaker
fig, ax = plt.subplots(figsize=(14, 2.5))
logomaker.Logo(pd.DataFrame(contribution.logo[0], columns=list("ACGT")), ax=ax)
ax.set_xlabel("Position in 80-nt insert (0-based)")
ax.set_ylabel("Local ISM effect")
fig.tight_layout()
fig.savefig(panel / "sequence01_ism_logo.png", dpi=180)
```
