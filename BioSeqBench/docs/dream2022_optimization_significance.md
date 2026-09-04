# DREAM optimization statistical analysis

The unit of analysis is one paired `(weak-model threshold, weak-model seed)`
condition. Every algorithm in a condition received the same initialization
policy, and the response is the mean strong Autosome/LegNet score of its 100
selected candidates.

## Overall paired analysis (60 conditions)

The omnibus Friedman test rejects equality of the eight algorithms:

```text
chi-square = 54.78, df = 7, p = 1.65e-9
```

The corresponding Kendall W is about 0.13, a small-to-moderate overall rank
effect rather than a complete separation of methods.

Pairwise paired Wilcoxon tests with Holm correction identify these differences
at adjusted p < 0.05:

| Comparison | Mean difference in strong score | Holm p |
|---|---:|---:|
| Vanilla AM − CbAS | +0.720 | 0.0001 |
| DEN − CbAS | +0.706 | 0.0004 |
| Fast SeqProp − CbAS | +0.662 | 0.0006 |
| Genetic algorithm − CbAS | +0.690 | 0.0018 |
| Vanilla AM − ISM | +0.502 | 0.0044 |
| Genetic algorithm − ISM | +0.472 | 0.0314 |
| DEN − ISM | +0.487 | 0.0406 |

The leading group is not statistically separated: Vanilla AM, DEN, genetic
algorithm and Fast SeqProp have pairwise adjusted p-values above 0.05. Their
mean strong-score differences are only 0.015–0.058. Thus the data support a
leading group, not a single champion.

## Conservative threshold-level analysis (12 conditions)

To avoid treating correlated thresholds as 60 fully independent observations,
the five seeds were first averaged within each threshold and the paired tests
were repeated over only 12 thresholds. No pairwise comparison remained
significant after Holm correction. This is the more conservative inference and
is the appropriate basis for claims about generalization across threshold
levels. Several individual threshold Friedman tests have uncorrected p < 0.05,
but none survives correction across the 12 thresholds.

## Interpretation

The results show a real aggregate algorithm effect, but substantial condition
dependence. CbAS and ISM are consistently lower in the pooled paired analysis;
Vanilla AM, DEN, GA and Fast SeqProp form an overlapping top group. The absence
of threshold-level significance means the current five seeds do not establish
that any one of these top methods is universally superior across surrogate
strengths.

These tests use the current 480-condition benchmark. The `top_1pct_count` field
in the raw summaries is not used here because it is defined relative to the
99th percentile within each returned set of 100 candidates, rather than a fixed
global strong-score reference.
