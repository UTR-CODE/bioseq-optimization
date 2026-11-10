# Enhancer：设计K562特异的200 nt序列

[返回主文档](../../README.md) · [候选生成原理](../candidate-generation.md) · [Notebook](../../examples/enhancer.ipynb)

目标是让Malinois预测的K562活性高于HepG2和SKNSH。采用

`S = activity(K562) - max(activity(HepG2), activity(SKNSH))`。

三个细胞的原始输出仍保存，避免只看到高差值却不知道目标本身是否活跃。原始输出是模型活性尺度，不能解释为分类概率。

## 1. 加载模型与任务包装

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bioseq_models import load_model
from sequence_optimization import MultiTaskObjective, random_seed_sequences
from sequence_optimization.models import MalinoisScorer, validate_scorer
from sequence_optimization.workflows import generate_candidates, DESIGN_ALGORITHMS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs/enhancer_k562_run1")
raw = load_model("malinois", device=DEVICE, batch_size=512)
scorer = MalinoisScorer(raw, target_cell="K562")
print(scorer.task_names)
# K562, HepG2, SKNSH, K562_minus_HepG2, K562_minus_SKNSH
objective = MultiTaskObjective(high=[3, 4], aggregation="worst")
```

对两个差值取最小值，等价于上述S。不能用`K562-HepG2-SKNSH`代替；后者是另一个目标。`MalinoisScorer`在hard和soft输入中都加入官方MPRA左右上下文，实际优化区仍只有中间200 nt。

## 2. 检查预测与梯度一致

```python
initial = random_seed_sequences(200, 100, seed=42)
validation = validate_scorer(scorer, initial[:4], atol=1e-3)
print(validation)
p = scorer.score(initial[:4])
np.testing.assert_allclose(p[:, 3:].min(axis=1), p[:, 0]-p[:, 1:3].max(axis=1))
```

输入梯度必须非零且有限；one-hot soft预测应与字符串路径一致。如果这里失败，先修正输入编码或模型版本，不应继续比较梯度算法和黑盒算法。

## 3. 生成11×100候选并查看耗时

```python
records, times = generate_candidates(
    scorer, 200, initial, objective,
    output_dir=OUT, algorithms=DESIGN_ALGORITHMS,
    num_candidates=100, diversity=5, seed=42,
)
assert len(records) == 1100
assert records.groupby("algorithm").size().eq(100).all()
assert times.stop_reason.eq("plateau").all()
print(times[["algorithm", "seconds", "queries", "score_median"]])
```

这是完整生产规模，不设置固定budget。一次完整搜索可能需要较长时间；要同时运行11算法，替换为[多进程示例](../parallel-scoring.md)，后续表结构不变。

## 4. 从原始预测再次核查特异性

```python
np.testing.assert_allclose(
    records.score,
    records.activity_K562-records[["activity_HepG2", "activity_SKNSH"]].max(axis=1),
    atol=1e-3,
)
print(records.groupby("algorithm").agg(
    specificity_median=("score", "median"),
    K562_median=("activity_K562", "median"),
))
```

## 5. Motif组成与位置多样性选择

先按[motif发现文档](../motif-discovery.md)从本次候选构建`catalog.meme`，并保存支持证据。catalog应来自同一目标的贡献分或独立的生物学数据库；不要把K562的motif catalog直接当作HepG2、SKNSH的完整机制列表。

该步骤需要发现流程已经产生`catalog.meme`，执行前请完成链接文档。以下目录、文件名是两个步骤之间的明确接口。

```python
from sequence_optimization.motifs import read_meme, scan_fimo, select_motif_panel

catalog = read_meme("outputs/enhancer_k562_motifs/catalog.meme")
hits = scan_fimo(
    "outputs/enhancer_k562_motifs/catalog.meme",
    OUT / "unique_candidates.fa",
    "outputs/enhancer_k562_fimo_run1",
)
selected, trace = select_motif_panel(
    records, hits, catalog,
    output_dir="outputs/enhancer_k562_top10_run1",
    top_n=10, min_score=20.0, min_sites=30,
    match_similarity=0.5, composition_tie=0.02, position_tie_bp=2.0,
)
print(selected[["selection_rank", "sequence_id", "score", "present_motifs", "algorithms"]])
```

门槛20来自当前Malinois示例，需要检查候选分布再设定，不是所有模型的固定阈值。选择顺序为：最高S作为起点；优先增加motif composition多样性；组合差异接近时增加位置差异；仍接近时选S更高者。少于10条符合条件会明确报错，不会偷偷降低门槛。

如果只想先验证选择代码，可直接运行[免GPU的真实参考数据示例](../motif-diversity.md)，不必重新优化或发现motif。

## 6. 画Top10跨细胞预测热图

```python
import matplotlib.pyplot as plt
values = selected[["activity_K562", "activity_HepG2", "activity_SKNSH"]].to_numpy()
fig, ax = plt.subplots(figsize=(5, 6))
im = ax.imshow(values, aspect="auto", cmap="viridis")
ax.set_xticks(range(3), ["K562", "HepG2", "SKNSH"])
ax.set_yticks(range(len(selected)), selected.sequence_id, fontsize=7)
fig.colorbar(im, ax=ax, label="Predicted activity (model scale)")
fig.tight_layout()
fig.savefig("outputs/enhancer_k562_top10_run1/activity_heatmap.png", dpi=180)
```

## 更换目标细胞

把`MalinoisScorer(raw, target_cell="HepG2")`或`"SKNSH"`作为scorer，其最后两列仍是目标减另外两细胞的差值，因此`high=[3,4]`保持适用。使用新输出目录，并针对新目标重新计算贡献分和catalog；质量门槛需要重新检查。

## 扩展到小鼠组织

小鼠组织模型也通过同一公开接口加载：

```python
heart = load_model("deepstarr_mouse_heart_activity", device=DEVICE, batch_size=32)
probe = random_seed_sequences(1001, 4, seed=42)
print(heart.score(probe), heart.ensemble_size)
```

六组织模型分别为heart、limb、midbrain、forebrain、hindbrain、neural_tube；每组织官方ensemble含6个成员。组织特异性必须联合预测六组织，再计算目标概率减最高非目标概率；单独最大化heart不等于心脏特异性。对应E11.5胚胎阶段，见[官方模型来源](https://github.com/Shenzhi-Chen/DeepSTARR-Mouse)。这里仅演示公开加载与评分；本次11算法的完整教程使用已验证hard/soft上下文的Malinois，不能把缺少soft编码的组织adapter直接用于梯度算法。

## 7. 展示Top10的motif类型与位置

```python
from matplotlib.patches import Rectangle
sites = pd.read_csv("outputs/enhancer_k562_top10_run1/motif_sites.csv")
motif_ids = sorted(sites.motif_id.unique())
colors = {m: plt.get_cmap("tab20")(i % 20) for i, m in enumerate(motif_ids)}
fig, ax = plt.subplots(figsize=(12, 5))
for y, row in enumerate(selected.itertuples()):
    ax.plot([0, 200], [y, y], color="lightgray", lw=2, zorder=0)
    for hit in sites[sites.sequence_name.eq(row.sequence_id)].itertuples():
        ax.add_patch(Rectangle((hit.start0, y-.25), hit.end0-hit.start0, .5,
                               facecolor=colors[hit.motif_id], alpha=.75))
        ax.text(hit.center0, y+.3, hit.motif_id, ha="center", fontsize=6, rotation=45)
ax.set_xlim(0, 200); ax.set_ylim(-.5, len(selected))
ax.set_yticks(range(len(selected)), selected.sequence_id, fontsize=7)
ax.set_xlabel("Enhancer position (0-based bp)")
fig.tight_layout()
fig.savefig("outputs/enhancer_k562_top10_run1/motif_positions.png", dpi=180)
```

条形是FIMO命中footprint，标签为catalog motif ID；多个命中和相似PWM可能重叠，图形并不表示已验证TF占据。
