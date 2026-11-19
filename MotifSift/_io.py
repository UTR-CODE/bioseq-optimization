"""Read the outputs saved by the workflow without rerunning a step."""

import json
from pathlib import Path

import pandas as pd

from sequence_optimization.framework import MotifCatalog
from sequence_optimization.motifs import read_fimo, read_meme
from sequence_optimization.objectives import MultiTaskObjective, TargetObjective

from ._api import _objective_for


def load_designs(output_dir, *, objective=None):
    """Load designs and their metadata from a generate_designs output directory."""
    directory = Path(output_dir)
    designs = pd.read_csv(directory / "designs.csv")
    metadata = json.loads((directory / "config.json").read_text())
    saved = metadata.get("objective")
    if objective is not None:
        metadata["objective"] = _objective_for(objective, metadata.get("task_names"))
    elif isinstance(saved, dict):
        kind = saved.get("type")
        if kind == "custom":
            metadata["objective"] = None
            metadata["saved_custom_objective"] = saved["function"]
        elif kind == "MultiTaskObjective":
            metadata["objective"] = MultiTaskObjective(**saved["parameters"])
        elif kind == "TargetObjective":
            metadata["objective"] = TargetObjective(**saved["parameters"])
        else:
            raise ValueError(f"Unknown saved objective type {kind!r}; supply objective explicitly")
    designs.attrs.update(metadata)
    return designs


def load_motifs(output_dir):
    """Load a saved catalog, including strand rules and available evidence."""
    directory = Path(output_dir)
    config = json.loads((directory / "config.json").read_text())
    pwms = read_meme(directory / "catalog.meme")
    ids = tuple(pd.read_csv(directory / "sequences.csv", usecols=["sequence_id"]).sequence_id)
    support_path = directory / "support.csv"
    seqlets_path = directory / "seqlets.csv"
    hits_path = directory / "saved_hits.csv"
    old_scan = directory / "candidate_scan" / "fimo.tsv"
    hits = None
    if hits_path.exists():
        hits = pd.read_csv(hits_path)
    elif old_scan.exists():
        hits = read_fimo(old_scan)
        hits = hits[hits.motif_id.isin(pwms)].copy()
    return MotifCatalog(
        pwms,
        support=pd.read_csv(support_path) if support_path.exists() else pd.DataFrame(),
        seqlets=pd.read_csv(seqlets_path) if seqlets_path.exists() else pd.DataFrame(),
        hits=hits,
        sequence_ids=ids,
        output_dir=directory,
        p_threshold=config["fimo_p"],
        q_threshold=config["fimo_q"],
        reverse_complement=config["reverse_complement"],
    )
