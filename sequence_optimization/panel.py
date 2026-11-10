"""Cluster candidate pools, highlight representatives, and recommend a small panel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .api import SequenceSpace, normalize_nucleotide_sequences


def _integer(name, value, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _sequence_features(sequences):
    rows = []
    for seq in sequences:
        fractions = np.array([seq.count(b) / len(seq) for b in "ACGT"])
        nonzero = fractions[fractions > 0]
        longest, run = 1, 1
        for a, b in zip(seq, seq[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        rows.append(
            [
                fractions[1] + fractions[2],
                float(-(nonzero * np.log2(nonzero)).sum()),
                seq.count("CG") / max(1, len(seq) - 1),
                longest / len(seq),
            ]
        )
    return np.asarray(rows), [
        "GC",
        "base_entropy",
        "CpG_fraction",
        "longest_run_fraction",
    ]


@dataclass
class ClusteredDesignPanel:
    candidates: pd.DataFrame
    representatives: pd.DataFrame
    recommendations: pd.DataFrame
    family_summary: pd.DataFrame
    diagnostics: pd.DataFrame
    feature_profiles: pd.DataFrame
    input_assignments: pd.DataFrame
    metadata: dict

    def save(self, directory):
        """Write tables, final FASTA and a portable report; plot with matplotlib if installed."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name in (
            "candidates",
            "representatives",
            "recommendations",
            "family_summary",
            "diagnostics",
            "feature_profiles",
            "input_assignments",
        ):
            getattr(self, name).to_csv(root / f"{name}.csv", index=False)
        (root / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )
        with (root / "recommendations.fa").open("w") as out:
            for _, row in self.recommendations.iterrows():
                out.write(
                    f">{row.candidate_id}|{row.family}|quality={row[self.metadata['quality_column']]:.6g}\n{row.sequence}\n"
                )
        plot = ""
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:
            pass
        else:
            fig = Figure(figsize=(9, 6), layout="constrained")
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            for family, group in self.candidates.groupby("family", sort=True):
                ax.scatter(group.PC1, group.PC2, s=24, alpha=0.55, label=family)
            r = self.representatives
            ax.scatter(
                r.PC1,
                r.PC2,
                s=115,
                facecolors="none",
                edgecolors="black",
                linewidths=1,
                label="Representative",
            )
            f = self.recommendations
            ax.scatter(
                f.PC1,
                f.PC2,
                s=190,
                marker="*",
                color="#e8a317",
                edgecolors="black",
                linewidths=0.6,
                label="Final candidate",
                zorder=5,
            )
            for row in f.itertuples():
                ax.annotate(
                    row.candidate_id,
                    (row.PC1, row.PC2),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            ratios = self.metadata["display_variance"]
            ax.set(
                xlabel=f"PC1 ({ratios[0]:.1%})",
                ylabel=f"PC2 ({ratios[1]:.1%})",
                title="Candidate groups and highlighted representatives",
            )
            ax.legend(fontsize=8)
            fig.savefig(root / "clusters.svg")
            fig.savefig(root / "clusters.pdf")
            fig.savefig(root / "clusters.png", dpi=180)
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg")
            plot = buffer.getvalue()
            plot = plot[plot.index("<svg") :]
            fig.clear()
        q = self.metadata["quality_column"]
        columns = list(
            dict.fromkeys(
                [
                    "candidate_id",
                    "family",
                    "representative_role",
                    "final_rank",
                    q,
                    *[c for c in self.candidates if c.startswith("activity_")],
                    "sequence",
                ]
            )
        )
        buttons = '<button data-family="all">全部</button>' + "".join(
            f'<button data-family="{escape(str(f), quote=True)}">{escape(str(f))}</button>'
            for f in self.family_summary.family
        )
        tables = ""
        for family, group in self.representatives.groupby("family", sort=True):
            tables += (
                f'<section data-family="{escape(str(family), quote=True)}"><h2>{escape(str(family))} 代表序列</h2>'
                + group[columns].to_html(
                    index=False, escape=True, classes="representatives"
                )
                + "</section>"
            )
        report = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>Sequence shortlist</title>
<style>body{{font:15px system-ui;margin:28px;color:#172b3a}}table{{border-collapse:collapse;width:100%;margin:16px 0}}
td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}td:last-child{{font:12px monospace;overflow-wrap:anywhere;max-width:500px}}
svg{{width:100%;max-width:1000px;height:auto}}button{{padding:8px 15px;margin:5px;cursor:pointer}}h2{{margin-top:32px}}</style>
<h1>候选序列聚类与推荐</h1><p>{len(self.candidates)} 条唯一序列；{self.metadata["n_clusters"]} 类；
{len(self.representatives)} 条代表；最终推荐 {len(self.recommendations)} 条。
最终候选用星形标记，代表序列用圆环标记。聚类使用保留的全部主成分，二维图仅用于展示。</p>
<p>这是模型预测候选的描述性分组，不代表独立 TF 家族或实验验证。候选缺额：{self.metadata["shortfall"]}。</p>
<p>质量下限：{escape(str(self.metadata["min_quality"]))}；最小全局 Hamming 距离：{self.metadata["min_hamming"]}。
Silhouette：{escape(str(self.metadata["silhouette"]))}；重采样稳定性 ARI：{escape(str(self.metadata["stability_ari"]))}。
稳定性条件于当前特征表示，不能替代生物学验证。</p>
{plot}<h2>分组诊断</h2>{self.diagnostics.to_html(index=False, escape=True)}
<h2>最终候选</h2>{self.recommendations[columns].to_html(index=False, escape=True)}
<h2>各类概览</h2>{self.family_summary.to_html(index=False, escape=True)}<nav>{buttons}</nav>{tables}
<script>document.querySelectorAll('button[data-family]').forEach(b=>b.onclick=()=>{{
document.querySelectorAll('section[data-family]').forEach(s=>s.hidden=b.dataset.family!=='all'&&s.dataset.family!==b.dataset.family);
}});</script></html>"""
        (root / "report.html").write_text(report, encoding="utf-8")


def cluster_designs(
    candidates: pd.DataFrame,
    *,
    descriptor,
    feature_blocks: Mapping | None = None,
    block_weights: Mapping | None = None,
    include_sequence_features=True,
    quality_column="quality",
    n_clusters=None,
    cluster_range=(2, 3, 4, 5, 6),
    method="ward",
    per_family=3,
    final_count=6,
    quality_quantile=0.5,
    min_quality=None,
    min_hamming=10,
    require_motif=True,
    pca_variance=0.9,
    stability_repeats=10,
    seed=42,
) -> ClusteredDesignPanel:
    """Group motif/position + optional signed numeric feature blocks, then shortlist.

    Motif counts use a Hellinger transform. Other blocks are standardized, clipped
    at five standard deviations, and normalized by sqrt(block width). Each block is
    multiplied by sqrt(weight) before PCA, so feature quantity/units do not silently
    dominate. Quality and source labels are excluded unless explicitly supplied as
    features. DataFrame blocks must have exactly the input index in the same order.

    Representatives are quality-gated central, high-quality and feature-diverse
    members. The final panel selects from these highlights, prioritizing one per
    family before adding more, with GLOBAL Hamming spacing. Unfilled requests are
    reported without relaxing filters. Clusters are descriptive sequence groups.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError as exc:
        raise ImportError(
            "Install bioseq-optimization[analysis] for clustering"
        ) from exc
    for name, value in (("per_family", per_family), ("final_count", final_count)):
        _integer(name, value)
    _integer("min_hamming", min_hamming, 0)
    _integer("stability_repeats", stability_repeats, 0)
    if n_clusters is not None:
        _integer("n_clusters", n_clusters)
    if method not in {"ward", "kmeans"}:
        raise ValueError("method must be ward or kmeans")
    if not 0 <= quality_quantile <= 1 or not 0 < pca_variance < 1:
        raise ValueError("quality_quantile must be in [0,1]; pca_variance in (0,1)")
    if min_quality is not None and not np.isfinite(min_quality):
        raise ValueError("min_quality must be finite")
    if not {"sequence", quality_column} <= set(candidates) or candidates.empty:
        raise ValueError(
            "candidates must contain sequence, finite quality and at least one row"
        )
    data = candidates.copy().reset_index(drop=True)
    if data.sequence.isna().any():
        raise ValueError("sequences cannot be missing")
    data["sequence"] = normalize_nucleotide_sequences(data.sequence.to_numpy())
    SequenceSpace(len(data.sequence.iloc[0])).validate(data.sequence)
    quality = data[quality_column].to_numpy(dtype=float)
    if not np.isfinite(quality).all():
        raise ValueError("quality must be finite")
    data[quality_column] = quality
    blocks = dict(feature_blocks or {})
    if {"motif", "sequence"}.intersection(blocks):
        raise ValueError("motif and sequence are reserved feature block names")
    arrays, names = {}, {}
    for name, block in blocks.items():
        if not isinstance(name, str) or not name:
            raise ValueError("feature block names must be nonempty strings")
        if isinstance(block, pd.DataFrame) and not block.index.equals(candidates.index):
            raise ValueError(
                "DataFrame feature block index must match candidates exactly"
            )
        array = np.asarray(block, dtype=float)
        if (
            array.ndim != 2
            or array.shape[0] != len(data)
            or not array.shape[1]
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "feature blocks must be finite numeric [input rows, columns] matrices"
            )
        arrays[name] = array
        names[name] = (
            list(map(str, block.columns))
            if isinstance(block, pd.DataFrame)
            else [str(i) for i in range(array.shape[1])]
        )
        if len(set(names[name])) != len(names[name]):
            raise ValueError("feature names within a block must be unique")
    # Stable sequence ordering makes results independent of row order and source labels.
    indices = []
    provenance = {}
    for seq, group in data.groupby("sequence", sort=True):
        ii = group.index.to_numpy()
        for array in arrays.values():
            if not np.allclose(array[ii], array[ii[0]], atol=1e-10, rtol=1e-8):
                raise ValueError("duplicate sequences have inconsistent extra features")
        indices.append(int(ii[np.argmax(quality[ii])]))
        provenance[seq] = (
            json.dumps(sorted(set(group["algorithm"].astype(str))))
            if "algorithm" in group
            else "[]"
        )
    unique = data.iloc[indices].reset_index(drop=True)
    unique["source_algorithms"] = unique.sequence.map(provenance)
    unique["candidate_id"] = [f"SEQ{i + 1:05d}" for i in range(len(unique))]
    motif = np.asarray(descriptor(unique.sequence), dtype=float)
    if (
        motif.ndim != 2
        or motif.shape[0] != len(unique)
        or not motif.shape[1]
        or not np.isfinite(motif).all()
        or (motif < 0).any()
    ):
        raise ValueError("descriptor must return finite nonnegative [N,features]")
    motif_names = list(
        getattr(descriptor, "feature_names", [str(i) for i in range(motif.shape[1])])
    )
    if len(motif_names) != motif.shape[1]:
        raise ValueError("descriptor feature_names must match its width")
    arrays = {name: array[indices] for name, array in arrays.items()}
    arrays["motif"], names["motif"] = motif, motif_names
    if include_sequence_features:
        arrays["sequence"], names["sequence"] = _sequence_features(unique.sequence)
    weights = {"motif": 1.0, "sequence": 0.25}
    if block_weights is not None:
        if set(block_weights) - set(arrays):
            raise ValueError("block_weights contains an unknown block")
        weights.update(block_weights)
    processed, raw_parts, raw_names, transforms = [], [], [], {}
    for name, array in arrays.items():
        weight = weights.get(name, 1.0)
        if isinstance(weight, bool) or not np.isfinite(weight) or weight < 0:
            raise ValueError("block weights must be finite and nonnegative")
        keep = array.std(axis=0) > 1e-12
        if weight == 0 or not keep.any():
            transforms[name] = dict(weight=float(weight), retained_columns=0)
            continue
        values = array[:, keep]
        if name == "motif":
            total = values.sum(axis=1, keepdims=True)
            normalized = np.sqrt(values / np.maximum(total, 1))
        else:
            normalized = np.clip(
                (values - values.mean(axis=0)) / values.std(axis=0), -5, 5
            ) / np.sqrt(values.shape[1])
        processed.append(normalized * np.sqrt(weight))
        raw_parts.append(values)
        raw_names.extend(
            f"{name}:{feature}" for feature, retain in zip(names[name], keep) if retain
        )
        transforms[name] = dict(weight=float(weight), retained_columns=int(keep.sum()))
    matrix = (
        np.concatenate(processed, axis=1) if processed else np.zeros((len(unique), 1))
    )
    raw = np.concatenate(raw_parts, axis=1) if raw_parts else np.zeros((len(unique), 0))
    if len(unique) > 1 and matrix.var(axis=0).sum() > 1e-12:
        pca = PCA(n_components=pca_variance, svd_solver="full")
        embedding = pca.fit_transform(matrix)
        explained = pca.explained_variance_ratio_.tolist()
    else:
        embedding, explained = np.zeros((len(unique), 1)), [0.0]
    distinct = len(np.unique(embedding.round(12), axis=0))

    def fit(values, count, local_seed):
        if count == 1:
            return np.zeros(len(values), dtype=int)
        if method == "ward":
            return AgglomerativeClustering(
                n_clusters=count, linkage="ward"
            ).fit_predict(values)
        return KMeans(n_clusters=count, n_init=20, random_state=local_seed).fit_predict(
            values
        )

    if n_clusters is None:
        counts = sorted(set(cluster_range))
        for count in counts:
            _integer("cluster_range entries", count)
        counts = [k for k in counts if 1 < k <= min(distinct, len(unique) - 1)] or [1]
    else:
        if n_clusters > min(distinct, len(unique)):
            raise ValueError(
                "n_clusters exceeds the number of distinct informative sequences"
            )
        counts = [n_clusters]
    rng = np.random.default_rng(seed)
    trials, fitted = [], {}
    for count in counts:
        labels = fit(embedding, count, seed)
        fitted[count] = labels
        silhouette = (
            float(silhouette_score(embedding, labels))
            if 1 < count < len(unique)
            else None
        )
        sizes = np.bincount(labels)
        stability = []
        sample_size = int(np.ceil(0.8 * len(unique)))
        if 1 < count < sample_size:
            for _ in range(stability_repeats):
                subset = np.sort(rng.choice(len(unique), sample_size, replace=False))
                if len(np.unique(embedding[subset].round(12), axis=0)) < count:
                    continue
                alternate = fit(embedding[subset], count, int(rng.integers(2**31 - 1)))
                stability.append(adjusted_rand_score(labels[subset], alternate))
        trials.append(
            dict(
                n_clusters=count,
                silhouette=silhouette,
                stability_ari=float(np.mean(stability)) if stability else None,
                smallest=int(sizes.min()),
                largest=int(sizes.max()),
            )
        )
    eligible = [
        r for r in trials if r["smallest"] >= max(2, int(np.ceil(0.05 * len(unique))))
    ]
    best = max(
        eligible or trials,
        key=lambda r: (
            -1 if r["silhouette"] is None else r["silhouette"],
            -1 if r["stability_ari"] is None else r["stability_ari"],
            -r["n_clusters"],
        ),
    )
    labels = fitted[best["n_clusters"]]
    ordering = sorted(
        set(labels),
        key=lambda k: (-int((labels == k).sum()), int(np.flatnonzero(labels == k)[0])),
    )
    mapping = {label: f"F{i + 1}" for i, label in enumerate(ordering)}
    unique["family"] = [mapping[label] for label in labels]
    unique["PC1"] = embedding[:, 0]
    unique["PC2"] = embedding[:, 1] if embedding.shape[1] > 1 else 0.0
    unique["motif_detected"] = motif.any(axis=1)
    unique["representative_role"] = ""
    unique["representative_rank"] = 0
    unique["final_rank"] = 0
    codes = np.array([list(seq.encode()) for seq in unique.sequence], dtype=np.uint8)
    q = unique[quality_column].to_numpy()
    pools, reps, profiles, summaries = {}, [], [], []
    families = list(mapping.values())
    centroids = {}
    for family in families:
        ids = np.flatnonzero(unique.family.eq(family))
        centroid = embedding[ids].mean(axis=0)
        centroids[family] = centroid
        threshold = max(
            float(np.quantile(q[ids], quality_quantile)),
            -np.inf if min_quality is None else min_quality,
        )
        valid = [
            i
            for i in ids
            if q[i] >= threshold
            and (not require_motif or unique.motif_detected.iloc[i])
        ]
        distance_to_center = ((embedding - centroid) ** 2).sum(axis=1)
        chosen = []
        for rank in range(per_family):
            possible = [
                i
                for i in valid
                if i not in chosen
                and all(
                    np.count_nonzero(codes[i] != codes[j]) >= min_hamming
                    for j in chosen
                )
            ]
            if not possible:
                break
            if rank == 0:
                selected = min(
                    possible, key=lambda i: (distance_to_center[i], -q[i], i)
                )
                role = "central"
            elif rank == 1:
                selected = min(
                    possible, key=lambda i: (-q[i], distance_to_center[i], i)
                )
                role = "quality_lead"
            else:
                selected = max(
                    possible,
                    key=lambda i: (
                        min(
                            float(np.linalg.norm(embedding[i] - embedding[j]))
                            for j in chosen
                        ),
                        q[i],
                        -i,
                    ),
                )
                role = "diverse"
            chosen.append(selected)
            unique.loc[selected, ["representative_role", "representative_rank"]] = [
                role,
                rank + 1,
            ]
        reps.extend(chosen)
        pools[family] = chosen
        contrast = (
            (
                (raw[ids].mean(axis=0) - raw.mean(axis=0))
                / np.maximum(raw.std(axis=0), 1e-12)
            )
            if raw.shape[1]
            else np.array([])
        )
        top = np.argsort(-contrast)[:5]
        top = [j for j in top if contrast[j] > 1e-10]
        for j in range(raw.shape[1]):
            profiles.append(
                dict(
                    family=family,
                    feature=raw_names[j],
                    family_mean=float(raw[ids, j].mean()),
                    global_mean=float(raw[:, j].mean()),
                    standardized_contrast=float(contrast[j]),
                )
            )
        summaries.append(
            dict(
                family=family,
                n=len(ids),
                quality_mean=float(q[ids].mean()),
                quality_threshold=float(threshold),
                eligible=len(valid),
                requested_representatives=per_family,
                representatives=len(chosen),
                representative_shortfall=per_family - len(chosen),
                top_features="; ".join(raw_names[j] for j in top),
            )
        )
    # First maximize family coverage, then balance allocation across represented families.
    final, allocation = [], {family: 0 for family in families}
    while len(final) < final_count:
        choices = [
            i
            for i in reps
            if i not in final
            and all(
                np.count_nonzero(codes[i] != codes[j]) >= min_hamming for j in final
            )
        ]
        if not choices:
            break

        def priority(i):
            family = unique.family.iloc[i]
            covered = [f for f in families if allocation[f]]
            novelty = min(
                (
                    float(np.linalg.norm(centroids[family] - centroids[f]))
                    for f in covered
                ),
                default=0,
            )
            return (-allocation[family], novelty, q[i], -i)

        selected = max(choices, key=priority)
        final.append(selected)
        allocation[unique.family.iloc[selected]] += 1
        unique.loc[selected, "final_rank"] = len(final)
    representatives = unique.iloc[reps].copy().reset_index(drop=True)
    recommendations = unique.iloc[final].copy().reset_index(drop=True)
    summary = pd.DataFrame(summaries)
    summary["final_selected"] = summary.family.map(allocation)
    assignments = candidates.copy().reset_index(drop=True)
    assignments["sequence"] = data.sequence
    lookup = unique.set_index("sequence")
    for column in (
        "candidate_id",
        "family",
        "representative_role",
        "representative_rank",
        "final_rank",
    ):
        assignments[column] = assignments.sequence.map(lookup[column])
    diagnostics = pd.DataFrame(trials)
    diagnostics["selected"] = diagnostics.n_clusters.eq(best["n_clusters"])
    metadata = dict(
        input_rows=len(data),
        unique_sequences=len(unique),
        quality_column=quality_column,
        n_clusters=int(best["n_clusters"]),
        method=method,
        seed=int(seed),
        per_family=int(per_family),
        requested_final=int(final_count),
        returned_final=len(final),
        shortfall=int(final_count - len(final)),
        covered_families=int(recommendations.family.nunique()),
        quality_quantile=float(quality_quantile),
        min_quality=min_quality,
        min_hamming=int(min_hamming),
        require_motif=bool(require_motif),
        feature_blocks=transforms,
        pca_components=embedding.shape[1],
        pca_explained_variance=float(sum(explained)),
        display_variance=(explained + [0, 0])[:2],
        silhouette=best["silhouette"],
        stability_ari=best["stability_ari"],
        stability_note="80% row subsamples; conditional on fixed pooled feature representation",
        selection_note="Quality-gated central/quality/diverse representatives, then family-balanced final panel with global Hamming spacing",
        clustering_note="Descriptive groups; not TF identities or biological validation. Quality and source labels are not automatically clustering features.",
    )
    return ClusteredDesignPanel(
        unique,
        representatives,
        recommendations,
        summary,
        diagnostics,
        pd.DataFrame(
            profiles,
            columns=[
                "family",
                "feature",
                "family_mean",
                "global_mean",
                "standardized_contrast",
            ],
        ),
        assignments,
        metadata,
    )
