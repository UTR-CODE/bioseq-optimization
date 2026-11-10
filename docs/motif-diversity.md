# Motif组成与位置多样性：选择Top N

[返回主文档](../README.md) · [先构建catalog](motif-discovery.md) · [Enhancer示例](examples/enhancer.md)

输入为候选CSV、DNA PWM catalog和FIMO位点。先设定模型质量门槛，再最大化motif composition多样性，随后比较motif位置，最后用预测分数解决近似平局。这是全局候选选择，不要求先把序列分成固定数量的cluster。

## 1. 从PWM到motif相似性

```python
from sequence_optimization.motifs import read_meme
from sequence_optimization.motif_diversity import pwm_similarity_matrix

catalog = read_meme("outputs/enhancer_k562_motifs/catalog.meme")
motif_ids = list(catalog)
similarity, alignments = pwm_similarity_matrix(
    [catalog[m] for m in motif_ids], min_overlap=6,
)
```

每个PWM使用信息量加权的`P-0.25`对比，考察正反链、无gap平移后的cosine相似性，所有信息侧翼都保留在范数中。相似性不是TF身份或相同生物学功能的证明。高度冗余的模型可先做catalog去冗余，但相似核心、不同侧翼的motif不必强制合并。

## 2. FIMO扫描与低支持motif过滤

```python
from sequence_optimization.motifs import scan_fimo
hits = scan_fimo(
    "outputs/enhancer_k562_motifs/catalog.meme",
    "outputs/enhancer_k562_run1/unique_candidates.fa",
    "outputs/enhancer_k562_fimo_run1",
    p_threshold=0.001,
)
```

使用均匀背景、双链扫描；保留`p≤0.001, q≤0.05`。同一序列、同一motif的位点，如果重叠至少较短footprint的70%，只保留p最好的一个，消除反向链重复等现象。不同ID的motif命中不跨ID去重，相似性由后续soft距离处理。

过滤**少于30个物理位置**的motif：先在唯一DNA上NMS，再按该DNA在1,100来源记录中的出现次数加权。另存唯一DNA支持数，避免把来自同一个祖先的复制误当独立证据。这里30是扫描支持门槛，不替代发现阶段至少10条seqlets和独立家族验证。过滤后不重新扫描或重算q值，防止改变检验集合。

## 3. Soft-Jaccard组合距离

Sequence×Motif为0/1矩阵，只关心是否出现，不按位点数或贡献值加权。

对序列i，把presence沿PWM相似性传播：`F[i,k] = max(similarity[m,k] for m present in i)`。再用`1 - sum(min(Fi,Fj))/sum(max(Fi,Fj))`得到soft组合距离。identity相似矩阵时退化为普通binary Jaccard。重复实例不提高presence值。空/空距离为0，空/非空为1；最终实验选择默认排除无catalog命中者，它们不代表已经证实的新motif机制。

## 4. Hungarian位置距离

每条序列保留所有实例`(motif_index, center_bp)`。FIMO原始1-based闭区间转换为0-based半开区间，`center=(start0+end0)/2`。相似性≥0.5的motif允许匹配：优先最大化合法匹配数，再最小化总位移。距离是匹配位点的平均绝对位移，单位bp。

同一motif多个实例保留；未匹配实例不惩罚；不加入片段编辑、共现、贡献权重或全序列对齐。若两条序列没有可匹配motif，距离数值保存0且`matched_counts=0`，表示缺少位置证据。选择时不把这种pair当作真正的零位移，也不给无证据者额外多样性优势。

## 5. 分层Max-Min

```python
import pandas as pd
from sequence_optimization.motifs import select_motif_panel

records = pd.read_csv("outputs/enhancer_k562_run1/all_candidates.csv")
selected, trace = select_motif_panel(
    records, hits, catalog,
    output_dir="outputs/enhancer_k562_top10_run1",
    top_n=10, min_score=20.0, min_sites=30,
    composition_tie=0.02, position_tie_bp=2.0,
)
```

最高质量候选为第一条。每一步：

1. 对剩余候选求到**全部已选序列**的最小composition距离，最大者优先。
2. 距离在最优值0.02内的候选进入位置比较；对有匹配证据的已选序列取最小位置距离，再最大化它。
3. 位置差异在最优值2 bp内，选择score最高者；再平局按稳定sequence ID排序。

这里不把不同尺度的质量、组合和位置相加。`top_n`可修改；不足N条满足门槛会报错。`min_score`是优化目标的分数下限；特异性目标使用specificity，单输出目标则是该模型活性，不应在不同任务间直接照搬。

## 6. 无GPU、无需FIMO安装的真实参考数据

仓库提供已有K562实验运行的紧凑输入：11算法×100条来源记录、对应的11 PWM和已通过阈值/NMS的FIMO位点。已经保存位点，因此可以独立重算整个选择过程：

```python
from pathlib import Path
import pandas as pd
from sequence_optimization.motifs import read_meme, select_motif_panel

DATA = Path("examples/data")
records = pd.read_csv(DATA / "k562_candidates_1100.csv")
hits = pd.read_csv(DATA / "k562_fimo_sites.csv")
catalog = read_meme(DATA / "k562_catalog.meme")
assert len(records) == 1100
selected, trace = select_motif_panel(
    records, hits, catalog,
    output_dir="outputs/reference_k562_top10_run1",
    top_n=10, min_score=20., min_sites=30,
)
print(selected[["selection_rank", "sequence_id", "score", "present_motifs"]])
```

这些PWM仅用于复现K562参考流程；不能据此宣称promoter、UTR或小鼠也有相同motif。位点表的q值来自原始完整catalog扫描，未在打包时重算。fixture重建了稳定SHA256序列ID，旧结果的`U_...`ID通过DNA重新映射，DNA和预测数值保持不变。

输出包括：`selected_sequences.fa/csv`、唯一及来源记录表、motif位点与支持数、唯一DNA和全部记录presence matrix、PWM相似矩阵、composition/position/count距离NPZ、逐步选择trace和参数receipt。NPZ始终保存行顺序ID，不依赖CSV默认行号。
