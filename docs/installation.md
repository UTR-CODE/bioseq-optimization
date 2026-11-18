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

Default motif discovery uses ISM contributions, tangermeme seqlets and official XSTREME with MEME disabled. Optional DeepLiftShap uses the same `motifs` Python dependencies and is enabled with `attribution_method="deepliftshap"`. Install MEME Suite for XSTREME discovery, FIMO scanning, and optional Tomtom annotation:

```bash
conda install -c conda-forge -c bioconda meme
xstreme --help
fimo --version
tomtom --version
```

`call_motifs` defaults to `motif_backend="xstreme"` and checks XSTREME and FIMO before computing contributions or creating output files. Missing tools produce the installation command above. Activate the environment containing MEME Suite, or supply executable paths with `xstreme="/path/to/xstreme"` and `fimo="/path/to/fimo"`. The optional `motif_backend="builtin"` replaces the discovery executable but still uses FIMO for candidate-wide filtering. Loading an existing catalog with `catalog=...` does not invoke these tools; subsequent motif-aware selection uses FIMO. Hamming candidate selection also works without MEME Suite.

The [motif discovery workflow](motif-discovery.md) also describes optional motif-catalog clustering with R.

## Development

```bash
python -m pip install -e '.[analysis,motifs,dev]'
python -m pytest
python tools/check_docs.py
```

## Pretrained-model notebooks

Install the model library and Jupyter in the environment used above. This revision includes the Saluki adapter used by the eGFP example.

```bash
python -m pip install 'bioseq-models @ git+https://github.com/UTR-CODE/bioseq-models.git@84a5075c3656127d99027c38705212a2de6e62c9' jupyterlab
```
