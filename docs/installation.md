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

Install MEME Suite for STREME, FIMO, and Tomtom:

```bash
conda install -c conda-forge -c bioconda meme
streme --version
fimo --version
tomtom --version
```

`call_motifs` checks for STREME and FIMO before computing contributions or creating output files. If either executable is missing, the error message provides the installation command above. Activate the environment containing MEME Suite, or supply executable paths with `streme="/path/to/streme"` and `fimo="/path/to/fimo"`. Loading an existing catalog with `catalog=...` does not require these tools.

The [motif discovery workflow](motif-discovery.md) also describes optional motif-catalog clustering with R.

## Development

```bash
python -m pip install -e '.[analysis,motifs,dev]'
python -m pytest
python tools/check_docs.py
```
