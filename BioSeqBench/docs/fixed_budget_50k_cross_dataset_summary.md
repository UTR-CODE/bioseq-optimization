# Fixed 50k-query cross-dataset optimization benchmark

Three benchmark runs completed with eight optimizers, 12 weak-model thresholds,
five replicates, and exactly 50,000 weak queries per condition. Each condition
returns 100 candidates. The results are stored under
`results/fixed_budget_50k/{tfbind8,utr5,dream}`.

## Overall means

| Dataset | Best mean method | Mean score | Second | Mean score | Score definition |
|---|---|---:|---|---:|---|
| TFBind8 | Simulated annealing | 0.3153 | ISM | 0.3153 | exact landscape activity |
| 5′UTR MRL | AdaLead | 6.9903 | Fast SeqProp | 6.9660 | strong Optimus MRL |
| DREAM promoter | Fast SeqProp | 13.6299 | DEN | 13.6058 | strong Autosome/LegNet score |

Scores are not comparable across datasets because each task uses a different
oracle and activity scale.

## Threshold-dependent winners

- TFBind8: GA at 1%; Vanilla AM at 5–10%; Fast SeqProp at 20%; CbAS at
  30–70%; ISM at 80–95%.
- 5′UTR: AdaLead at 1–10%; Vanilla AM at 20%; SA at 30%; CbAS at 40–50%;
  Fast SeqProp at 60–70%; DEN at 80–90%; ISM at 95%.
- DREAM: CbAS at 1%; Fast SeqProp at 5%; Vanilla AM at 10%; GA at 20–30%
  and 50%; DEN at 40% and 60%; CbAS at 70%; AdaLead at 80%; Vanilla AM at
  90%; GA at 95%.

No method wins a majority of threshold/replicate conditions across the three
tasks. The leading method is dataset-dependent: SA/ISM for TFBind8, AdaLead
for 5′UTR, and Fast SeqProp for DREAM.

## Important protocol limitation

The existing TFBind8 and 5′UTR fixed-budget runners optimize a five/ten-member
weak ensemble for each threshold, whereas the DREAM runner binds each replicate
to one weak checkpoint. Thus these completed runs are valid fixed-budget
within-dataset comparisons, but not the final strict cross-dataset experiment
if “one weak model per condition” is required. A corrected follow-up should run
TFBind8 and 5′UTR member checkpoints independently, add Random and Train-Top
controls, and then compare paired optimizer effects.
