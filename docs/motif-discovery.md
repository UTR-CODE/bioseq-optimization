# 贡献分、seqlets与motif catalog

[返回主文档](../README.md) · 下一步：[Motif diversity](motif-diversity.md)

本教程把模型认为重要的局部片段作为motif发现的输入，类似从ChIP-seq peaks发现富集模式，但seqlets不是实验结合peaks。可靠性需要独立序列支持、合适的背景和保留集验证，不能用模型贡献分本身证明TF结合。

下面接[enhancer示例](examples/enhancer.md)第3步得到的`scorer`、`records`。命令从仓库根目录执行。该流程会创建`outputs/enhancer_k562_motifs/catalog.meme`，供enhancer第5步使用。

## 1. 对唯一DNA计算ISM贡献

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sequence_optimization import ism_contributions
from sequence_optimization.workflows import deduplicate_candidates, write_fasta

WORK = Path("outputs/enhancer_k562_motifs")
WORK.mkdir(parents=True, exist_ok=False)
unique, provenance = deduplicate_candidates(records)
unique.to_csv(WORK / "unique_candidates.csv", index=False)
write_fasta(WORK / "unique_candidates.fa", unique.sequence_id, unique.sequence)

def specificity(sequences):
    x = scorer.score(sequences)
    return x[:, 3:].min(axis=1)   # 两条target-minus-offtarget margin

ism = ism_contributions(unique.sequence.to_numpy(), specificity, batch_size=512)
np.savez_compressed(WORK / "ism.npz", observed=ism.observed, logo=ism.logo,
                    sequence_ids=unique.sequence_id.to_numpy(dtype=str))
```

200 nt序列每条需要600条突变预测；约1,100条时约66万次，另加原序列预测。贡献是实际碱基预测减去四种碱基预测的平均，保持位置坐标不变，不是可加和的输出分解。不要误把图形库的sequence logo当作贡献估计。

DeepLIFTShap可以减少重复推理，需使用相同目标和上下文、dinucleotide背景、多个reference，并检查completeness与ISM关键位点一致性。它对网络非线性算子和版本更敏感；本教程默认用精确ISM，未将本地修改过的tangermeme实现复制到发布包。可参考[官方DeepLIFTShap文档](https://tangermeme.readthedocs.io/en/latest/api/deep_lift_shap.html)。

## 2. Tangermeme提取正贡献seqlets

```python
import torch
from tangermeme.seqlet import recursive_seqlets

seqlets = recursive_seqlets(
    torch.tensor(ism.observed, dtype=torch.float32),
    threshold=0.01, min_seqlet_len=6, max_seqlet_len=25, additional_flanks=2,
)
seqlets["example_idx"] = seqlets.example_idx.astype(int)
seqlets["start"] = seqlets.start.astype(int).clip(lower=0)
seqlets["end"] = seqlets.end.astype(int).clip(upper=200)
seqlets["sequence_id"] = seqlets.example_idx.map(unique.sequence_id.to_dict())
seqlets["dna"] = [unique.sequence.iloc[r.example_idx][r.start:r.end]
                   for r in seqlets.itertuples()]
seqlets = seqlets[seqlets.dna.str.len().ge(6) & seqlets.attribution.gt(0)].copy()
seqlets["seqlet_id"] = [f"Q{i+1:06}" for i in range(len(seqlets))]
seqlets.to_csv(WORK / "seqlets.csv", index=False)
write_fasta(WORK / "seqlets.fa", seqlets.seqlet_id, seqlets.dna)
```

传入每个位置实际碱基的贡献`[N,L]`，而不是`[N,4,L]`的未投影hypothetical分数。[recursive_seqlets](https://tangermeme.readthedocs.io/en/latest/api/seqlet.html)只调用正seqlets；`additional_flanks`补充侧翼，原始位置仍保存。重复DNA此前已去重，不能把11个算法得到的同一片段重复计作11条独立支持。

## 3. 按sequence family留出验证集

高度相似的候选必须在同一fold。使用CD-HIT-EST建立90%identity家族：

```bash
cd-hit-est -i outputs/enhancer_k562_motifs/unique_candidates.fa \
  -o outputs/enhancer_k562_motifs/families90.fa -c 0.9 -n 8 -d 0 -T 4 -M 0
```

```python
import re
family = {}
current = None
for line in (WORK / "families90.fa.clstr").read_text().splitlines():
    if line.startswith(">Cluster "):
        current = line.split()[-1]
    else:
        match = re.search(r">(.+?)\.\.\.", line)
        if match:
            family[match.group(1)] = current
assert set(unique.sequence_id) == set(family)
seqlets["family"] = seqlets.sequence_id.map(family)
rng = np.random.default_rng(42)
families = rng.permutation(sorted(set(family.values())))
if len(families) < 4:
    raise ValueError("Too few independent families for discovery and holdout validation")
training = set(families[:max(1, int(len(families)*0.7))])
seqlets["fold"] = np.where(seqlets.family.isin(training), "discovery", "holdout")
seqlets.to_csv(WORK / "seqlets.csv", index=False)
```

family是序列相似性分组，用于防止验证泄漏；不是最终motif类型或序列功能类别。如果知道seed lineage，最好同时检查相同lineage不会跨fold。

## 4. 生成匹配背景与STREME发现候选PWM

不同seqlet长度分开shuffle，保持双核苷酸组成。下面每条生成一个配对背景：

```python
from tangermeme.ersatz import dinucleotide_shuffle
from sequence_optimization import onehot
controls = {}
for length, group in seqlets.groupby(seqlets.dna.str.len()):
    x = torch.tensor(onehot(group.dna.to_numpy(), channels_first=True))
    shuffled = dinucleotide_shuffle(x, n=1, random_state=42)
    codes = shuffled[:, 0].argmax(dim=1).cpu().numpy()
    for sid, row in zip(group.seqlet_id, codes):
        controls[sid] = "".join(np.array(list("ACGT"))[row])
for fold in ["discovery", "holdout"]:
    group = seqlets[seqlets.fold.eq(fold)]
    write_fasta(WORK / f"{fold}.fa", group.seqlet_id, group.dna)
    write_fasta(WORK / f"{fold}_background.fa", group.seqlet_id,
                [controls[sid] for sid in group.seqlet_id])
```

```bash
streme --dna --p outputs/enhancer_k562_motifs/discovery.fa \
  --n outputs/enhancer_k562_motifs/discovery_background.fa \
  --oc outputs/enhancer_k562_motifs/streme --minw 6 --maxw 15 --nmotifs 30 --seed 42
```

`--nmotifs 30`是最多发现的模型数，不是预设最终要有30种motif。可以在相同discovery数据上另外用HOMER作稳定性比较，合并候选PWM时加方法前缀避免ID冲突。不要在holdout数据上调到得到指定数量的motif。

## 5. 至少10条seqlets支持，并在保留家族验证富集

将STREME候选重新扫描到全部seqlets及holdout背景。这里用**序列出现/不出现**统计，不让同一seqlet多个命中重复计数。discovery支持≥10条seqlets、≥3个亲本、≥3个family；再用holdout相对配对背景的富集作为保留条件。数字可以改变，但应事前确定并保存。

```python
from scipy.stats import binomtest
from sequence_optimization.motifs import read_meme, write_meme, scan_fimo

candidates = read_meme(WORK / "streme/streme.txt")
positive = scan_fimo(WORK / "streme/streme.txt", WORK / "seqlets.fa", WORK / "seqlet_scan")
negative = scan_fimo(WORK / "streme/streme.txt", WORK / "holdout_background.fa", WORK / "holdout_bg_scan")
rows = []
held = seqlets[seqlets.fold.eq("holdout")]
for motif_id in candidates:
    hit_ids = set(positive.loc[positive.motif_id.eq(motif_id), "sequence_name"])
    support = seqlets[seqlets.seqlet_id.isin(hit_ids) & seqlets.fold.eq("discovery")]
    # Family-level presence is the independent unit for validation.
    pos_families = set(held.loc[held.seqlet_id.isin(hit_ids), "family"])
    bg_ids = set(negative.loc[negative.motif_id.eq(motif_id), "sequence_name"])
    bg_families = set(held.loc[held.seqlet_id.isin(bg_ids), "family"])
    n = held.family.nunique()
    pos_only = len(pos_families-bg_families)
    bg_only = len(bg_families-pos_families)
    discordant = pos_only+bg_only
    pvalue = binomtest(pos_only, discordant, .5, alternative="greater").pvalue if discordant else 1.0
    rows.append(dict(motif_id=motif_id, seqlets=len(support),
                     parents=support.sequence_id.nunique(), families=support.family.nunique(),
                     holdout_hit_families=len(pos_families), holdout_bg_families=len(bg_families),
                     holdout_p=pvalue))
evidence = pd.DataFrame(rows)
pvalues = evidence.holdout_p.to_numpy()
order = np.argsort(pvalues)
q_sorted = np.minimum.accumulate((pvalues[order]*len(pvalues)/np.arange(1,len(pvalues)+1))[::-1])[::-1]
q = np.empty(len(pvalues)); q[order] = np.clip(q_sorted, 0, 1)
evidence["holdout_q"] = q
evidence["retained"] = ((evidence.seqlets >= 10) & (evidence.parents >= 3)
                        & (evidence.families >= 3) & (evidence.holdout_q <= .05))
evidence.to_csv(WORK / "catalog_evidence.csv", index=False)
retained = evidence.loc[evidence.retained, "motif_id"]
if len(retained) == 0:
    raise ValueError("No motifs passed independent support; inspect evidence instead of relaxing silently")
write_meme(WORK / "catalog.meme", {m:candidates[m] for m in retained})
```

这里用配对family的discordant presence做单侧精确二项检验，再按全部候选motif做BH校正；多个seqlets来自同一个family仍只计一次。可进一步做多背景重复、family级置换并报告置信区间。该示例不声称已重新完成所有任务的motif发现：它提供可复现步骤，真实参考catalog的来源见[验证文档](validation.md)。

## 6. 去冗余与数据库注释

可使用用户指定的[Almeida motif-clustering](https://github.com/bernardo-de-almeida/motif-clustering)。它用Tomtom E-value profile的相关性和complete linkage建立树，不能称作按PWM cosine直接合并。本项目提供[适配R入口](../tools/run_almeida_motif_dedup.R)，调用下载的**上游实际核心**，不把它替换成不同算法：

```bash
git clone https://github.com/bernardo-de-almeida/motif-clustering.git external/motif-clustering
tomtom -oc outputs/enhancer_k562_motifs/tomtom -dist kullback \
  -motif-pseudo 0.1 -min-overlap 1 -thresh 0.5 \
  outputs/enhancer_k562_motifs/catalog.meme outputs/enhancer_k562_motifs/catalog.meme
```

```python
tomtom = pd.read_csv(WORK / "tomtom/tomtom.tsv", sep="\t", comment="#").dropna(subset=["Query_ID", "Target_ID"])
ids = list(read_meme(WORK / "catalog.meme"))
evalues = pd.DataFrame(100., index=ids, columns=ids)
for _, row in tomtom.iterrows():
    evalues.loc[row["Query_ID"], row["Target_ID"]] = float(row["E-value"])
evalues.to_csv(WORK / "tomtom_evalues.csv")
```

```bash
Rscript tools/run_almeida_motif_dedup.R outputs/enhancer_k562_motifs/tomtom_evalues.csv \
  outputs/enhancer_k562_motifs 0.8 external/motif-clustering/Motif_clustering.R
```

inspect dendrogram、阈值敏感性和PWM侧翼，再决定代表模型；优先独立family、亲本、seqlets支持与holdout证据。保留去冗余映射，不把所有相似核心强制合并。常数profile/极少motif会使相关性无定义，R入口会拒绝，应检查原始证据而非填成有效距离。

最后可用Tomtom对同物种的JASPAR等数据库注释，保存版本、匹配统计和完整PWM。数据库命名是相似性注释，不是已证明TF在设计序列上结合。扰动motif后，目标活性与非目标活性都应单独查看；仅S下降不能说明该motif只驱动目标细胞。
