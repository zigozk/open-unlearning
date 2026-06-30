# Baseline Metric Reliability Audit

## Configuration

- Summary rows: 16
- Forget collapse threshold: <= 0.05
- Forget collapse votes required: 2
- High FQ threshold: >= 0.5
- Low MU threshold: <= 0.2

## Main Checks

- Forget-side collapsed runs: 8 / 16
- Collapsed before reaching high FQ: 8 / 16
- Low model_utility runs: 4 / 16

Interpretation: `collapsed_before_high_fq` is the direct test for the concern that FQ can be too strict or too indirect: the forget-side scores already indicate broken outputs, but `forget_quality` has not reached the chosen high-FQ threshold.

## Correlations

- Pearson corr(forget_quality, forget_Q_A_Prob) = -0.0636
- Pearson corr(forget_quality, forget_Q_A_ROUGE) = 0.1133
- Pearson corr(forget_quality, forget_truth_ratio) = 0.4284
- Pearson corr(model_utility, forget_Q_A_Prob) = 0.5034

## Method Summary

| method | n_runs | collapsed_runs | collapsed_before_high_fq_runs | low_mu_runs | forget_quality_mean | model_utility_mean | forget_Q_A_Prob_mean | forget_Q_A_ROUGE_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradAscent | 4 | 4 | 4 | 4 | 1.064e-239 | 0.0000 | 8.911e-36 | 0.0010 |
| GradDiff | 4 | 4 | 4 | 0 | 2.155e-223 | 0.5298 | 1.960e-10 | 0.0025 |
| NPO | 4 | 0 | 0 | 0 | 0.1295 | 0.5491 | 0.1085 | 0.2616 |
| SimNPO | 4 | 0 | 0 | 0 | 1.855e-14 | 0.6053 | 0.5239 | 0.4855 |

## Top Collapsed-Before-High-FQ Cases

| model | method | forget_collapse_votes | forget_collapse_metrics | forget_quality | model_utility | forget_Q_A_Prob | forget_Q_A_ROUGE | forget_truth_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | GradAscent | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 1.064e-239 | 0.0000 | 0.0000 | 0.0000 | 3.232e-48 |
| Llama-3.2-1B-Instruct | GradDiff | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 8.509e-237 | 0.3551 | 7.168e-45 | 0.0000 | 1.419e-06 |
| Llama-2-7b-chat-hf | GradAscent | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 1.064e-239 | 0.0000 | 8.914e-43 | 0.0039 | 2.211e-32 |
| Llama-3.2-3B-Instruct | GradAscent | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 1.064e-239 | 0.0000 | 1.100e-40 | 0.0000 | 1.176e-30 |
| Llama-3.2-1B-Instruct | GradAscent | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 1.064e-239 | 0.0000 | 3.565e-35 | 0.0000 | 1.589e-25 |
| Llama-2-7b-chat-hf | GradDiff | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 1.802e-229 | 0.5808 | 3.703e-33 | 0.0054 | 1.458e-07 |
| Llama-3.2-3B-Instruct | GradDiff | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 4.311e-223 | 0.5738 | 9.361e-27 | 0.0017 | 0.0022 |
| Llama-3.1-8B-Instruct | GradDiff | 3 | forget_Q_A_Prob;forget_Q_A_ROUGE;forget_truth_ratio | 4.311e-223 | 0.6095 | 7.839e-10 | 0.0031 | 0.0003 |

## Retain Per-Index Sensitivity

No raw eval logs with retain `value_by_index` were found.
Run again with `--results-root /path/to/server_leftovers_20260624` after copying the raw eval JSONs to enable item-level retain sensitivity.
