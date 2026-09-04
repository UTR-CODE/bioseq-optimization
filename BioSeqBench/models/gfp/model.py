from BioSeqBench.models.protein import ProteinTransformer


class GFPDesignBenchTransformer(ProteinTransformer):
    """Design-Bench GFP oracle architecture, randomly initialized (no pretraining)."""
    def __init__(self, output_dim: int = 1, max_length: int = 237):
        super().__init__(output_dim=output_dim, d_model=64, layers=4, heads=2, max_length=max_length)

