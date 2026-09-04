# IGVF cCRE MPRA model

`IGVFLegNet` is a from-scratch MPRALegNet-style model: EfficientNetV2-like 1-D convolution blocks, squeeze-and-excitation, and a learned cell-type embedding. The conditioning implements `sequence + cell type → activity` without Enformer/Sei/pretrained genomic features. Reference: Agarwal et al., *Massively parallel characterization of transcriptional regulatory elements*, Nature 2025; Penzar et al., Bioinformatics 2023, doi:10.1093/bioinformatics/btad457.

