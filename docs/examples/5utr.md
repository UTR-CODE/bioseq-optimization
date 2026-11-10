# 5′UTR：Optimus-5′ Prime高MRL设计

[返回主文档](../../README.md) · [Notebook](../../examples/5utr.ipynb) · [候选生成](../candidate-generation.md)

优化50 nt的5′UTR，提高模型预测的mean ribosome load（MRL）。模型输出不是mRNA稳定性、蛋白产量的绝对单位，也不是组织特异性。

## 1. 公开接口加载与输入验证

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bioseq_models import load_model
from sequence_optimization import random_seed_sequences
from sequence_optimization.models import validate_scorer
from sequence_optimization.workflows import (
    generate_candidates, deduplicate_candidates, select_sequence_panel, write_fasta,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs/utr_run1")
scorer = load_model("optimus5prime", device=DEVICE, batch_size=512)
initial = random_seed_sequences(50, 100, seed=42)
print(scorer.task_names)  # ('mrl',)
print(validate_scorer(scorer, initial[:4]))
print(scorer.score(initial[:4]))
```

Optimus adapter本身支持ACGT概率输入，不需要额外重写网络。公开adapter负责恢复checkpoint中的标签尺度。

## 2. 收集1,100条候选与运行时间

```python
records, times = generate_candidates(
    scorer, 50, initial, ["maximize"], output_dir=OUT,
    num_candidates=100, diversity=5, seed=42,
)
assert len(records) == 1100
assert records.sequence.str.fullmatch("[ACGT]{50}").all()
print(times[["algorithm", "seconds", "queries", "score_median", "stop_reason"]])
```

所有11算法共享种子，分别到平台期停止，保留90%增益档。Random Search单独执行，见[随机参考](../candidate-generation.md#5-可选random-search参考)。

## 3. 选择10条高MRL、DNA差异较大的序列

```python
unique, provenance = deduplicate_candidates(records)
gate = float(unique.score.quantile(0.9))
selected, distances = select_sequence_panel(records, top_n=10, min_score=gate)
print(selected[["selection_rank", "sequence_id", "score", "algorithms"]])
print("MRL gate:", gate)
print("Minimum DNA difference:", distances[np.triu_indices(10, 1)].min())
```

这里采用序列Hamming Max-Min作为没有验证catalog时的基础方案，不能把它标成motif diversity。RNA结构、uAUG、uORF及短序列上下文可能共同影响模型输出，不能只用TF结合motif概括UTR机制。

## 4. 检查upstream AUG并导出

```python
panel = Path("outputs/utr_top10_run1")
panel.mkdir(parents=True, exist_ok=False)
selected["upstream_ATG_positions0"] = selected.sequence.map(
    lambda s: ";".join(str(i) for i in range(len(s)-2) if s[i:i+3] == "ATG")
)
selected["has_upstream_ATG"] = selected.upstream_ATG_positions0.ne("")
selected.to_csv(panel / "selected_sequences.csv", index=False)
write_fasta(panel / "selected_sequences.fa", selected.sequence_id, selected.sequence)
np.savetxt(panel / "hamming_distances.csv", distances, delimiter=",", fmt="%d")
print(selected[["sequence_id", "score", "has_upstream_ATG", "upstream_ATG_positions0"]])
```

这是注释检查，默认不会悄悄删除uAUG。如果实验要求禁止uAUG，应先明确设计约束再重新搜索/筛选；是否形成完整uORF还需要主ORF和完整转录本上下文，不能从这50 nt单独断言。

## 5. ISM解释与单条sequence logo

```python
from sequence_optimization import ism_contributions
import matplotlib.pyplot as plt
import logomaker

contribution = ism_contributions(
    selected.sequence.to_numpy(), lambda seq: scorer.score(seq)[:, 0], batch_size=512,
)
np.savez_compressed(panel / "top10_ism.npz", observed=contribution.observed, logo=contribution.logo)
fig, ax = plt.subplots(figsize=(12, 2.5))
logomaker.Logo(pd.DataFrame(contribution.logo[0], columns=list("ACGT")), ax=ax)
ax.set_xlabel("Position in 50-nt UTR (0-based)")
ax.set_ylabel("Local ISM effect")
fig.tight_layout()
fig.savefig(panel / "sequence01_ism_logo.png", dpi=180)
```

这是带局部贡献的单序列logo，不是由多条序列保守性生成的PWM。要发现UTR特征，可以对全部候选使用ISM/DeepLIFTShap和seqlets，进行独立支持与扰动验证后构建任务特异catalog；参考[motif发现](../motif-discovery.md)。
