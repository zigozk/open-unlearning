# Forget Quality Calculation Comparison

This note compares the TOFU `forget_quality` calculation in:

- `optml-group/unlearn-simple`
- `licong-lin/negative-preference-optimization`
- this OpenUnlearning workspace

The short version is: all three use the two-sample Kolmogorov-Smirnov test
p-value as Forget Quality. The main implementation difference is the direction
of the intermediate truth-ratio distribution.

## Summary

| Source | Forget Quality output | Truth-ratio distribution used before KS test |
|---|---:|---|
| `optml-group/unlearn-simple` | `ks_2samp(...).pvalue` | `exp(perturbed_loss - paraphrased_loss)` |
| `licong-lin/negative-preference-optimization` | `ks_2samp(...).pvalue` | `exp(perturbed_loss - paraphrased_loss)` |
| This repo | `ks_2samp(...).pvalue` | `wrong_prob / correct_prob = exp(correct_loss - wrong_loss)` |

The two external repositories use the same TOFU utility code. They compute a
ratio that is effectively `correct_prob / wrong_prob`. This repository computes
`wrong_prob / correct_prob`, which is the reciprocal.

Because Forget Quality compares two distributions with the same transform
applied to both sides, the KS p-value is usually expected to be the same or very
close when the underlying examples, losses, perturbed-answer aggregation, and
retain-reference logs are identical. Differences in those inputs can still
change the final value.

## External Repositories

Both external repositories implement TOFU Forget Quality in `TOFU/utils.py`.

Relevant upstream code:

- `optml-group/unlearn-simple`: https://github.com/optml-group/unlearn-simple/blob/main/TOFU/utils.py#L143-L160
- `licong-lin/negative-preference-optimization`: https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/utils.py#L143-L160

The implementation is:

```python
def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']

    unlearn_paraphrase_np_values = np.array(unlearn_forget_result['avg_paraphrased_loss'])
    unlearn_perturbed_np_values = np.array(unlearn_forget_result['average_perturb_loss'])
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(retain_forget_result['avg_paraphrased_loss'])
    retain_perturbed_np_values = np.array(retain_forget_result['average_perturb_loss'])
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio = np.exp(unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio = np.exp(retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return (
        {
            'Forget Quality': test_res.pvalue,
            'KS Test PVal Forget': test_res.pvalue,
            'KS Test Forget': test_res.statistic,
        },
        {
            'Unlearn Truth Ratio': unlearn_truth_ratio,
            'Retain Truth Ratio': retain_truth_ratio,
        },
    )
```

Their eval code first writes raw loss arrays:

- `avg_paraphrased_loss`: average token loss for the paraphrased/correct answer.
- `average_perturb_loss`: average token losses for perturbed/wrong answers.

Then `get_forget_quality()` averages the perturbed losses over the perturbed
answer axis and computes:

```text
truth_ratio_external = exp(perturbed_loss - paraphrased_loss)
                     = exp(-paraphrased_loss) / exp(-perturbed_loss)
                     = correct_prob / wrong_prob
```

The aggregate stat is written in their trainer dataloader path:

- `negative-preference-optimization`: https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/dataloader.py#L528-L541
- `unlearn-simple`: https://github.com/optml-group/unlearn-simple/blob/main/TOFU/dataloader.py#L564-L578

## This Repository

This workspace uses the OpenUnlearning metric-handler path.

Main local files:

- `configs/eval/tofu_metrics/forget_quality.yaml`
- `configs/eval/tofu_metrics/forget_Truth_Ratio.yaml`
- `configs/eval/tofu_metrics/retain_Truth_Ratio.yaml`
- `src/evals/metrics/memorization.py`
- `src/evals/metrics/privacy.py`
- `src/evals/metrics/utils.py`

The `forget_quality` config declares a dependency on `forget_Truth_Ratio` and
uses the `ks_test` handler:

```yaml
# configs/eval/tofu_metrics/forget_quality.yaml
defaults:
  - .@pre_compute.forget_truth_ratio: forget_Truth_Ratio

reference_logs:
  retain_model_logs:
    path: ${eval.tofu.retain_logs_path}
    include:
      forget_truth_ratio:
        access_key: retain

pre_compute:
  forget_truth_ratio:
    access_key: forget

handler: ks_test
```

The forget truth-ratio config maps paraphrased answers to `correct` and
perturbed answers to `wrong`:

```yaml
# configs/eval/tofu_metrics/forget_Truth_Ratio.yaml
pre_compute:
  forget_Q_A_PARA_Prob:
    access_key: correct
  forget_Q_A_PERT_Prob:
    access_key: wrong

handler: truth_ratio
aggregator: closer_to_1_better
```

The metric implementation computes:

```python
# src/evals/metrics/memorization.py
correct_prob = np.exp(-correct_avg_losses)
wrong_prob = np.exp(-wrong_avg_losses)
truth_ratios = wrong_prob / (correct_prob + 1e-10)
```

So the local intermediate distribution is:

```text
truth_ratio_local = wrong_prob / correct_prob
                  = exp(-wrong_loss) / exp(-correct_loss)
                  = exp(correct_loss - wrong_loss)
```

Finally, `forget_quality` compares the current model's forget truth-ratio
distribution with the retain-reference model's retain truth-ratio distribution:

```python
# src/evals/metrics/privacy.py
forget_tr_stats = np.array([
    evals["score"]
    for evals in kwargs["pre_compute"]["forget"]["value_by_index"].values()
])

retain_tr_stats = np.array([
    evals["score"]
    for evals in reference_logs["retain"]["value_by_index"].values()
])

fq = ks_2samp(forget_tr_stats, retain_tr_stats)
pvalue = fq.pvalue
```

If `retain_logs_path` is not provided, this repository sets
`forget_quality` to `None` rather than computing a fallback value.

## Code-Level Differences

| Detail | External repos | This repo |
|---|---|---|
| Final FQ statistic | `ks_2samp(unlearn_truth_ratio, retain_truth_ratio).pvalue` | `ks_2samp(forget_tr_stats, retain_tr_stats).pvalue` |
| Ratio direction | `correct_prob / wrong_prob` | `wrong_prob / correct_prob` |
| Loss formula | `exp(wrong_loss - correct_loss)` | `exp(correct_loss - wrong_loss)` |
| Perturbed-answer reduction | `average_perturb_loss.mean(axis=-1)` | `aggregate_to_1D()`, which means over non-leading axes |
| Reference source | `eval_cfg.retain_result` JSON | `${eval.tofu.retain_logs_path}` metric logs |
| Missing reference behavior | The external trainer expects the reference file to exist | Local handler returns `None` with a warning |
| Output key | `Forget Quality` | `forget_quality` / `agg_value` |

## Practical Interpretation For BRIDGE

For BRIDGE experiments, the important operational points are:

1. `forget_quality` being blank or `None` means the retain-reference logs were
   missing. It should not be interpreted as zero.
2. When comparing BRIDGE runs against NPO, compare methods at similar FQ rather
   than using Model Utility alone.
3. If local FQ differs from the external TOFU scripts, check the inputs first:
   same forget split, same retain-reference model, same sample count, same
   perturbed answers, and same loss aggregation.
4. The reciprocal truth-ratio convention is a notation/implementation
   difference. It is not expected to matter for the KS p-value when applied
   consistently to both compared distributions.

## Minimal Formula Reference

External repositories:

```text
FQ = KS_pvalue(
    exp(mean(loss_perturbed_unlearn) - loss_paraphrased_unlearn),
    exp(mean(loss_perturbed_retain_ref) - loss_paraphrased_retain_ref)
)
```

This repository:

```text
FQ = KS_pvalue(
    exp(loss_correct_forget - loss_wrong_forget),
    exp(loss_correct_retain_ref - loss_wrong_retain_ref)
)
```

where `wrong_loss` is averaged over perturbed answers before forming the
per-example ratio.
