# Installation

Use Python 3.10 or later. From the repository directory, install MotifSift and MEME Suite:

```bash
python -m pip install '.[motifs]'
conda install -c conda-forge -c bioconda meme
```

To run the pretrained-model notebooks, install the model library and Jupyter in the same environment. Start the notebook kernel in `examples/` so it can read the accompanying files.

```bash
python -m pip install 'bioseq-models @ git+https://github.com/UTR-CODE/bioseq-models.git@84a5075c3656127d99027c38705212a2de6e62c9' jupyterlab
```
