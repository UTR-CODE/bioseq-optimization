"""Lazy registry for checkpoint-backed activity models shipped with the package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PretrainedModelInfo:
    """Stable metadata describing one bundled activity model."""

    name: str
    sequence_length: int
    task_names: tuple[str, ...]
    description: str
    differentiable: bool
    loader_path: str
    aliases: tuple[str, ...] = ()


_MODELS = (
    PretrainedModelInfo(
        name="optimus5prime",
        sequence_length=50,
        task_names=("mrl",),
        description="Optimus 5-Prime mean ribosome load model",
        differentiable=True,
        loader_path=("sequence_optimization.examples.models.utr5:Optimus5PrimeExample"),
        aliases=("optimus", "optimus_5prime"),
    ),
    PretrainedModelInfo(
        name="deepstarr",
        sequence_length=249,
        task_names=("developmental", "housekeeping"),
        description="DeepSTARR enhancer activity model",
        differentiable=True,
        loader_path=(
            "sequence_optimization.examples.models.deepstarr:DeepSTARRExample"
        ),
    ),
    PretrainedModelInfo(
        name="dream_legnet",
        sequence_length=110,
        task_names=("promoter_activity",),
        description="DREAM 2022 yeast promoter LegNet model",
        differentiable=False,
        loader_path=(
            "sequence_optimization.examples.models.dream_legnet:DreamLegNetExample"
        ),
        aliases=("dream", "autosome_legnet"),
    ),
    PretrainedModelInfo(
        name="zhao_ptrc",
        sequence_length=74,
        task_names=("log10_fluorescence_per_od600",),
        description="Zhao Ptrc promoter fluorescence CNN",
        differentiable=True,
        loader_path=("sequence_optimization.examples.models.zhao_ptrc:ZhaoPtrcExample"),
        aliases=("ptrc",),
    ),
    PretrainedModelInfo(
        name="zhou_l3",
        sequence_length=29,
        task_names=("promoter_activity",),
        description="Zhou L3 position-aware promoter CNN",
        differentiable=True,
        loader_path=("sequence_optimization.examples.models.zhou_l3:ZhouL3Example"),
        aliases=("l3",),
    ),
)


def _normalized_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


_LOOKUP = {
    _normalized_name(alias): info
    for info in _MODELS
    for alias in (info.name, *info.aliases)
}


def available_models() -> tuple[str, ...]:
    """Return canonical names accepted by :func:`load_model`."""
    return tuple(info.name for info in _MODELS)


def model_info(name: str) -> PretrainedModelInfo:
    """Return sequence length, outputs, and capability metadata."""
    key = _normalized_name(name)
    if key not in _LOOKUP:
        raise KeyError(
            f"unknown pretrained model {name!r}; choose from "
            f"{', '.join(available_models())}"
        )
    return _LOOKUP[key]


def load_model(
    name: str,
    *,
    checkpoint: str | Path | None = None,
    device: Any = "cpu",
    **kwargs: Any,
):
    """Load a bundled model as an optimization-ready sequence scorer."""
    info = model_info(name)
    module_name, class_name = info.loader_path.split(":", maxsplit=1)
    adapter = getattr(import_module(module_name), class_name)
    options = {"device": device, **kwargs}
    if checkpoint is not None:
        options["checkpoint"] = checkpoint
    scorer = adapter.from_checkpoint(**options)
    scorer.pretrained_model_info = info
    return scorer
