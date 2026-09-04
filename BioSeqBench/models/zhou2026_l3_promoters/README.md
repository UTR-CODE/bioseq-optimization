# Zhou 2026 L3 promoter model

The paper reports a non-pretrained CNN predictor but does not release a
layer-by-layer architecture or parameter file.  BioSeqBench therefore uses
`zhou_l3_conditional_cnn`, a position-aware CNN tailored to the fixed 29-nt L3
construct.  The raw workbook's `s1`--`s4` worksheets provide source-sublibrary
identity for every aggregate-table row; this is included as an assay context,
because it materially affects strength and cannot reliably be inferred from
the sequence alone.  The architecture has Conv1D(64×5), Conv1D(96×3), flatten,
Dense(256), then one small regression head per sublibrary.  Flattening (rather
than global pooling) preserves the absolute -35 and -10 box locations.
