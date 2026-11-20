# Reproduce the Extended Data design figures

| Extended Data figure | Notebook | Designed region |
|---|---|---|
| 1 | [DREAM LegNet promoter](../examples/dream-legnet-promoter.ipynb) | 80-nt yeast promoter |
| 2 | [Optimus 5-Prime 5′ UTR](../examples/single-task.ipynb) | 50-nt 5′ UTR |
| 3 | [Saluki eGFP CDS](../examples/saluki-egfp.ipynb) | 720-nt CDS |
| 4 | [Saluki Gluc CDS](../examples/saluki-gluc-cds.ipynb) | 558-nt CDS |
| 5 | [Saluki Gluc 3′ UTR](../examples/saluki-gluc-3utr.ipynb) | 150-nt 3′ UTR |

Run each notebook in order from a clean output directory. The notebook outputs supply the prediction distributions, motif logos and selected panel in the figures. Both CDS notebooks return 50% and 90% gain candidates. The CDS figures use the 90% gain stage.

The reference environment used an RTX 4090, Python 3.10.15, PyTorch 2.5.1 with CUDA 11.8 and cuDNN 9.1.0, and MEME Suite 5.5.7. The notebook installation cell pins the model package and the [run manifest](../examples/data/extended_data_reproducibility.json) records the model-weight digests, other package versions and output digests. Set these environment variables before starting the Jupyter kernel.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
```

The manifest compares the full design table, motif steps and selected panel. The catalog digest normalizes the path printed on the MEME background header. The notebook code digest joins the code cells, with trailing whitespace removed from each cell, using the ASCII record separator.
