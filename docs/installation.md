# Installation

[Home](../README.md) · [PyTorch interface](pytorch-models.md)

Use Python 3.10 or later and a PyTorch installation matching your hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,motifs]'
```

The optimization package uses NumPy, pandas, PyTorch, and Numba. The `analysis` extra installs distance, plotting, and sequence-logo tools; `motifs` adds tangermeme for seqlet extraction. An optional C extension accelerates sequence construction.

## Motif discovery tools

The default built-in motif discovery uses Python, NumPy, Numba, SciPy, and tangermeme. Install MEME Suite for FIMO scanning during motif-based candidate selection, optional external STREME discovery, and Tomtom annotation:

```bash
conda install -c conda-forge -c bioconda meme
streme --version
fimo --version
tomtom --version
```

`call_motifs(..., motif_backend="streme")` checks STREME and FIMO before computing contributions or creating output files. Missing tools produce the installation command above. Activate the environment containing MEME Suite, or supply executable paths with `streme="/path/to/streme"` and `fimo="/path/to/fimo"`. The default `motif_backend="builtin"` and loading an existing catalog with `catalog=...` do not require these executables. Hamming candidate selection also works without MEME Suite.

The [motif discovery workflow](motif-discovery.md) also describes optional motif-catalog clustering with R.

## Development

```bash
python -m pip install -e '.[analysis,motifs,dev]'
python -m pytest
python tools/check_docs.py
```
