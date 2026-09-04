# BioSeqBench

BioSeqBench is a PyTorch benchmark for sequence-to-function prediction and downstream biological sequence optimization. Canonical releases use `train.csv`, `val.csv`, `test.csv`, and `metadata.json` in each `dataset/<name>/` directory.

## Large releases

`yeast_gpra_glucose`, `yeast_gpra_sc_ura`, `parm_v1_phy3`, and `parm_v2_mh4` are streaming CSV datasets. Train them with the normal command; the trainer detects `streaming_csv` in metadata and does not load the full split with pandas.

```bash
python -m BioSeqBench.training.train --dataset parm_v1_phy3 --model parm_cnn --epochs 20 --batch-size 256 --device cuda --num-workers 4
python -m BioSeqBench.training.train --dataset yeast_gpra_glucose --model yeast_gpra_cnn --epochs 10 --batch-size 512 --device cuda --num-workers 4
```

## Added E. coli promoter datasets

- `zhao2022_ptrc_promoters`: Zhao et al. (ACS Synthetic Biology 2022, doi:10.1021/acssynbio.1c00117), experimentally measured Ptrc-derived promoter fluorescence/OD600.
- `zhou2026_l3_promoters`: Zhou et al. (ACS Synthetic Biology 2026, doi:10.1021/acssynbio.5c00765), FACS-seq L3 promoter strengths.
- `kosuri2013_composability`: Kosuri et al. (PNAS 2013, doi:10.1073/pnas.1301301110), paired promoter--RBS RNA and protein output.

Dataset provenance and raw-file locations are maintained in `configs/sources.json`. Large-data conversion is reproducible with `python -m BioSeqBench.preprocessing.prepare_large --help`.

## TFBind landscapes

- `tfbind8`: complete 8-mer SIX6_REF_R1 PBM landscape.
- `tfbind10`: complete 10-mer Pho4 binding-energy landscape. Its standardized target is `-ddG`, hence a larger score means stronger binding. The full lookup is retained deliberately: it can be used as an exact optimization oracle, rather than as an independent held-out-surrogate benchmark.

## Paper-model audit

[`docs/modeling_audit.md`](docs/modeling_audit.md) records, for every standardized
task, whether the *original paper* actually trained a deep-learning predictor, the
paper's data/split handling, and whether the current PyTorch model is an exact
reproduction or an explicitly labelled non-pretrained surrogate.  This distinction is
important when comparing sequence-optimization algorithms.
