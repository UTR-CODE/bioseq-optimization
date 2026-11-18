"""Label-blind sequence clustering and per-family candidate recommendation.

This is postprocessing: it neither reruns nor changes any optimizer.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


def kmer_position_features(sequences, kmers, *, n_bins=20):
    """Binary canonical k-mer x midpoint-position features, aligned to input rows.

    Reverse complements are merged as motif identity; positions remain measured
    in the supplied sequence orientation. All overlapping occurrences are scanned.
    Enrichment testing and k-mer vocabulary selection are caller responsibilities.
    """
    if (
        isinstance(n_bins, bool)
        or not isinstance(n_bins, (int, np.integer))
        or n_bins < 1
    ):
        raise ValueError("n_bins must be a positive integer")
    if isinstance(sequences, str) or isinstance(kmers, str):
        raise ValueError("sequences and kmers must be collections of strings")
    sequences = [str(s).upper().replace("U", "T") for s in sequences]
    kmers = [str(s).upper().replace("U", "T") for s in kmers]
    if (
        not sequences
        or not kmers
        or any(not s or set(s) - set("ACGT") for s in sequences + kmers)
    ):
        raise ValueError("sequences and kmers must contain nonempty ACGT strings")
    if len(set(map(len, sequences))) != 1:
        raise ValueError("sequences must have equal length")
    complement = str.maketrans("ACGT", "TGCA")

    def canonical(word):
        return min(word, word.translate(complement)[::-1])

    words = sorted(set(map(canonical, kmers)))
    lookup = {word: i for i, word in enumerate(words)}
    lengths = sorted(set(map(len, words)))
    if max(lengths) > len(sequences[0]):
        raise ValueError("k-mers cannot exceed sequence length")
    values = np.zeros((len(sequences), len(words) * n_bins), dtype=np.uint8)
    for i, sequence in enumerate(sequences):
        for k in lengths:
            for start in range(len(sequence) - k + 1):
                j = lookup.get(canonical(sequence[start : start + k]))
                if j is not None:
                    b = min(n_bins - 1, (start + k // 2) * n_bins // len(sequence))
                    values[i, j * n_bins + b] = 1
    return pd.DataFrame(
        values,
        columns=[f"{word}_bin{b + 1:02d}" for word in words for b in range(n_bins)],
    )


@dataclass
class FamilySelectionResult:
    candidates: pd.DataFrame
    recommendations: pd.DataFrame
    input_assignments: pd.DataFrame
    family_summary: pd.DataFrame
    algorithm_counts: pd.DataFrame
    diagnostics: pd.DataFrame
    metadata: dict

    def save(self, directory):
        """Export auditable tables, metadata, and candidate FASTA."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name in (
            "candidates",
            "recommendations",
            "input_assignments",
            "family_summary",
            "diagnostics",
        ):
            getattr(self, name).to_csv(root / f"{name}.csv", index=False)
        self.algorithm_counts.to_csv(root / "algorithm_family_counts.csv")
        (root / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        with (root / "recommendations.fa").open("w") as out:
            for row in self.recommendations.itertuples():
                out.write(f">{row.candidate_id}|{row.family}\n{row.sequence}\n")


def select_diverse_families(
    candidates: pd.DataFrame,
    features: np.ndarray,
    *,
    quality_column: str,
    n_clusters: int = 4,
    method: str = "ward",
    per_family: int = 5,
    pca_variance: float = 0.9,
    quality_quantile: float = 0.5,
    min_quality: float | None = None,
    min_hamming: int = 5,
    seed: int = 42,
) -> FamilySelectionResult:
    """Cluster pooled sequences in PCA space and recommend across families.

    ``features`` must be a nonnegative finite sequence-feature matrix, in exactly
    the same row order as candidates (e.g. k-mer x position-bin counts).
    No algorithm labels or activities are used in clustering. Duplicate sequences
    are clustered once and retain every source algorithm in input_assignments.
    Zero-feature sequences are marked Unresolved and are not recommended.

    ``quality_column`` must be a comparable, higher-is-better scalar computed
    from raw predictions with the SAME definition for every algorithm. Dynamic
    optimizer scores may not be comparable across runs.

    Within each family, filter by min_quality and the requested quality quantile;
    select the highest-quality candidate first, then greedy max-min Hamming
    representatives. Round-robin family allocation enforces a GLOBAL minimum
    Hamming distance. Return fewer than per_family if quality, available unique
    sequences, or spacing prevent filling the quota; never pad or relax filters.
    These recommendations are an assay shortlist, not validated functionality.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError(
            "Install MotifSift[analysis] for family selection"
        ) from exc

    for name, val, minimum in [
        ("n_clusters", n_clusters, 2),
        ("per_family", per_family, 1),
        ("min_hamming", min_hamming, 0),
    ]:
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, np.integer))
            or val < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if method not in {"ward", "kmeans"}:
        raise ValueError("method must be ward or kmeans")
    if not 0 < pca_variance < 1 or not 0 <= quality_quantile <= 1:
        raise ValueError("pca_variance must be in (0,1); quality_quantile in [0,1]")
    if min_quality is not None and not np.isfinite(min_quality):
        raise ValueError("min_quality must be finite or None")
    if not {"sequence", "algorithm", quality_column} <= set(candidates):
        raise ValueError(
            "candidates must include sequence, algorithm and quality_column"
        )
    d = candidates.copy().reset_index(drop=True)
    if d.empty or d.sequence.isna().any() or d.algorithm.isna().any():
        raise ValueError(
            "candidate sequences and algorithms must be nonempty and nonmissing"
        )
    d["sequence"] = d.sequence.astype(str).str.upper().str.replace("U", "T")
    d["algorithm"] = d.algorithm.astype(str)
    if (
        not d.sequence.map(lambda s: bool(s) and set(s) <= set("ACGT")).all()
        or d.sequence.str.len().nunique() != 1
    ):
        raise ValueError("sequences must be equal-length nonempty ACGT DNA")
    quality = d[quality_column].to_numpy(dtype=float)
    if not np.isfinite(quality).all():
        raise ValueError("quality values must be finite")
    d[quality_column] = quality
    x = np.asarray(features, dtype=float)
    if (
        x.ndim != 2
        or x.shape[0] != len(d)
        or x.shape[1] == 0
        or not np.isfinite(x).all()
        or (x < 0).any()
    ):
        raise ValueError(
            "features must be finite nonnegative [candidate rows, features]"
        )
    # Sequence-based ordering makes clustering independent of algorithm labels.
    indices = []
    origins = {}
    for sequence, group in d.groupby("sequence", sort=True):
        ii = group.index.to_numpy()
        if not np.allclose(x[ii], x[ii[0]], atol=1e-10, rtol=1e-8):
            raise ValueError("identical sequences must have identical feature vectors")
        indices.append(int(ii[np.argmax(quality[ii])]))
        origins[sequence] = json.dumps(sorted(group.algorithm.unique()))
    u = d.iloc[indices].reset_index(drop=True)
    u["source_algorithms"] = u.sequence.map(origins)
    x = x[indices]
    valid = x.sum(1) > 0
    if valid.sum() <= n_clusters or len(np.unique(x[valid], axis=0)) < n_clusters:
        raise ValueError("not enough distinct informative vectors for n_clusters")
    transformed = TfidfTransformer().fit_transform(x[valid]).toarray()
    pca = PCA(n_components=pca_variance, svd_solver="full")
    embedded = pca.fit_transform(transformed)
    if method == "ward":
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(embedded)
    else:
        labels = KMeans(
            n_clusters=n_clusters, n_init=30, random_state=seed
        ).fit_predict(embedded)
    # Stable size-ordered family names, independent of quality and algorithm.
    order = sorted(
        set(labels),
        key=lambda label: (
            -int((labels == label).sum()),
            int(np.flatnonzero(labels == label)[0]),
        ),
    )
    mapping = {label: f"F{i + 1}" for i, label in enumerate(order)}
    u["family"] = "Unresolved"
    u.loc[valid, "family"] = [mapping[label] for label in labels]
    u["candidate_id"] = [f"SEQ{i + 1:06d}" for i in range(len(u))]
    families = [f"F{i + 1}" for i in range(n_clusters)]
    pools = {}
    eligible_counts = {}
    for fam in families:
        group = u[u.family == fam]
        threshold = float(group[quality_column].quantile(quality_quantile))
        if min_quality is not None:
            threshold = max(threshold, min_quality)
        eligible = group[group[quality_column] >= threshold].sort_values(
            [quality_column, "sequence"], ascending=[False, True]
        )
        pools[fam] = eligible.index.tolist()
        eligible_counts[fam] = len(eligible)
    selected = []
    allocations = {fam: 0 for fam in families}
    codes = np.array([list(s.encode()) for s in u.sequence], dtype=np.uint8)
    while True:
        progress = False
        for fam in families:
            if allocations[fam] >= per_family:
                continue
            choices = [i for i in pools[fam] if i not in selected]
            if not choices:
                continue
            distances = np.array(
                [
                    min(
                        (int(np.count_nonzero(codes[i] != codes[j])) for j in selected),
                        default=codes.shape[1],
                    )
                    for i in choices
                ]
            )
            feasible = [
                i for i, distance in zip(choices, distances) if distance >= min_hamming
            ]
            if not feasible:
                continue
            if allocations[fam] == 0:
                chosen = feasible[0]
            else:
                lookup = dict(zip(choices, distances))
                chosen = max(
                    feasible,
                    key=lambda i: (lookup[i], float(u.loc[i, quality_column]), -i),
                )
            selected.append(chosen)
            allocations[fam] += 1
            progress = True
        if not progress:
            break
    recommendations = u.loc[selected].copy().reset_index(drop=True)
    recommendations["within_family_rank"] = (
        recommendations.groupby("family").cumcount() + 1
    )
    assignments = candidates.copy().reset_index(drop=True)
    assignments["sequence"] = d.sequence
    assignments["family"] = d.sequence.map(u.set_index("sequence").family)
    assignments["candidate_id"] = d.sequence.map(u.set_index("sequence").candidate_id)
    source_counts = pd.crosstab(
        assignments.drop_duplicates(["sequence", "algorithm"]).algorithm,
        assignments.drop_duplicates(["sequence", "algorithm"]).family,
    )
    summary = pd.DataFrame(
        [
            dict(
                family=fam,
                n=int((u.family == fam).sum()),
                quality_eligible=eligible_counts[fam],
                requested=per_family,
                selected=allocations[fam],
                shortfall=per_family - allocations[fam],
            )
            for fam in families
        ]
    )
    diagnostics = pd.DataFrame(
        [
            dict(
                method=method,
                n_clusters=n_clusters,
                original_cosine_silhouette=silhouette_score(
                    transformed, labels, metric="cosine"
                ),
                pca_silhouette=silhouette_score(embedded, labels),
            )
        ]
    )
    metadata = dict(
        quality_column=quality_column,
        quality_quantile=quality_quantile,
        min_quality=min_quality,
        min_hamming=min_hamming,
        seed=seed,
        method=method,
        n_clusters=n_clusters,
        per_family=per_family,
        pca_components=embedded.shape[1],
        pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
        input_rows=len(d),
        unique_sequences=len(u),
        unresolved=int((~valid).sum()),
        algorithm_labels_used_for_clustering=False,
        count_note="Shared sequences credited to every source algorithm; source counts can exceed unique candidates",
        duplicate_quality_rule="Keep row with highest supplied common quality; preserve all input rows in input_assignments",
    )
    return FamilySelectionResult(
        u, recommendations, assignments, summary, source_counts, diagnostics, metadata
    )
