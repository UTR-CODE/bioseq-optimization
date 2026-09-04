"""Loader for the exhaustive Pho4 10-mer binding-energy landscape.

The released ``TFBind10-Exact-v0`` archive stores integer-encoded DNA in
``x`` (A=0, C=1, G=2, T=3) and binding free-energy differences in ``y``.
The release has already converted the physical minimization objective to the
benchmark maximization convention: ``label = -ddG``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .base import DatasetLoader


class TFBind10Loader(DatasetLoader):
    name = "TFBind10"
    alphabet = "DNA"
    label_type = "negative_ddG_binding_affinity"
    organism = "Saccharomyces cerevisiae"
    source_reference = (
        "Le et al. PNAS 2018; doi:10.1073/pnas.1715888115; "
        "TFBind10-Exact-v0 corrected exhaustive lookup (DiBO)"
    )
    metadata_extras = {
        "tf_identity": "Pho4",
        "assay": "in vitro transcription-factor binding free-energy landscape",
        "score_definition": "label = -ddG; larger values indicate stronger binding.",
        "raw_encoding": {"0": "A", "1": "C", "2": "G", "3": "T"},
        "release_note": (
            "The DiBO TFBind10-Exact-v0 release provides a corrected, "
            "higher-is-better lookup over all 4^10 DNA sequences."
        ),
    }

    def load(self, raw_path: Path) -> pd.DataFrame:
        with np.load(raw_path, allow_pickle=False) as data:
            if not {"x", "y"}.issubset(data.files):
                raise ValueError(
                    f"{raw_path} must contain named arrays 'x' and 'y'; found {data.files}"
                )
            encoded = np.asarray(data["x"])
            labels = np.asarray(data["y"]).reshape(-1)
        if encoded.ndim != 2 or encoded.shape[1] != 10:
            raise ValueError(f"Expected TFBind10 encoded array of shape (N, 10), got {encoded.shape}")
        if len(encoded) != len(labels):
            raise ValueError(f"TFBind10 x/y row mismatch: {len(encoded)} != {len(labels)}")
        if not np.issubdtype(encoded.dtype, np.integer) or np.any((encoded < 0) | (encoded > 3)):
            raise ValueError("TFBind10 x must use integer base codes 0, 1, 2, 3")
        bases = np.asarray(list("ACGT"))
        sequence = np.fromiter(("".join(row) for row in bases[encoded]), dtype="U10", count=len(encoded))
        return pd.DataFrame(
            {
                "sequence": sequence,
                "label": labels,
                "tf_identity": "Pho4",
                "assay": "binding_free_energy",
            }
        )
