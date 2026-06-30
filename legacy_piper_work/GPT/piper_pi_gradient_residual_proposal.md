# PIPER Revision Proposal: Finite-Step PI Beyond Gradient Similarity

## 1. Reviewer-Level Motivation

The current PIPER design uses a predictive interference probe to identify retain samples that are vulnerable to an unlearning update. This is a reasonable mechanism-level idea, but in its current form it risks being judged as a sample-selection heuristic:

```text
compute PI scores
select Top-K retain samples
apply local KL preservation
```

For a top-tier conference standard, the method needs a sharper position relative to gradient-similarity based unlearning methods such as OFMU-style approaches. The strongest revision is not to avoid gradient similarity, but to show that gradient similarity is only a first-order approximation of PIPER's finite-step interference estimate.

The revised claim should be:

```text
Gradient similarity estimates infinitesimal first-order conflict.
PIPER estimates finite-step predictive retain damage induced by an actual unlearning update.
The residual between PI and first-order gradient similarity captures nonlinear, optimizer-aware vulnerability missed by gradient-only methods.
```

This makes PIPER a principled extension rather than an alternative heuristic.

## 2. Core Theoretical Reframing

Let the forget objective be:

```latex
L_f(\theta)
```

and the retain loss of a candidate retain sample \(x_r\) be:

```latex
L_r(x_r; \theta).
```

A one-step virtual unlearning update is:

```latex
\theta' = \theta - \eta \nabla_\theta L_f(\theta).
```

The PIPER predictive interference score is:

```latex
\mathrm{PI}(x_r)
= L_r(x_r; \theta') - L_r(x_r; \theta).
```

Using first-order Taylor expansion:

```latex
L_r(x_r; \theta - \eta \nabla_\theta L_f)
\approx
L_r(x_r; \theta)
- \eta
\left\langle
\nabla_\theta L_r(x_r; \theta),
\nabla_\theta L_f(\theta)
\right\rangle.
```

Therefore:

```latex
\mathrm{PI}(x_r)
\approx
-\eta
\left\langle
\nabla_\theta L_r(x_r; \theta),
\nabla_\theta L_f(\theta)
\right\rangle.
```

This shows that gradient similarity is a first-order approximation to PI. The method can therefore define:

```latex
\mathrm{GS}(x_r)
=
-\eta
\left\langle
\nabla_\theta L_r(x_r; \theta),
\nabla_\theta L_f(\theta)
\right\rangle
```

and:

```latex
\mathrm{ResidualPI}(x_r)
=
\mathrm{PI}(x_r) - \mathrm{GS}(x_r).
```

The residual is the part of finite-step predictive damage not explained by first-order gradient similarity.

## 3. Revised Method Variants

### 3.1 Gradient-Similarity Local KL Baseline

This is the mandatory baseline. Without it, reviewers can argue that PI is only an expensive version of gradient similarity.

Score:

```latex
s_{\mathrm{GS}}(x_r)
=
-\left\langle
\nabla_\theta L_r(x_r),
\nabla_\theta L_f
\right\rangle.
```

Then select Top-K retain samples by \(s_{\mathrm{GS}}\) and apply the same local KL objective:

```latex
L
=
L_{\mathrm{unlearn}}

\lambda
\mathbb{E}_{x_r \in V_{\mathrm{GS}}}
\left[
\mathrm{KL}
\left(
p_{\theta_0}(\cdot|x_r)
\Vert
p_{\theta}(\cdot|x_r)
\right)
\right].
```

### 3.2 PI Top-K Local KL

This is the current PIPER method:

```latex
V_{\mathrm{PI}}
=
\mathrm{TopK}_{x_r}
\left(
\mathrm{PI}(x_r)
\right).
```

Objective:

```latex
L
=
L_{\mathrm{unlearn}}

\lambda
\mathbb{E}_{x_r \in V_{\mathrm{PI}}}
\left[
\mathrm{KL}
\left(
p_{\theta_0}(\cdot|x_r)
\Vert
p_{\theta}(\cdot|x_r)
\right)
\right].
```

### 3.3 Residual-PI Local KL

This is the main proposed extension.

```latex
V_{\mathrm{RPI}}
=
\mathrm{TopK}_{x_r}
\left(
\mathrm{ResidualPI}(x_r)
\right).
```

Interpretation:

```text
These retain samples are not merely gradient-conflicting.
They are vulnerable in ways that first-order gradient similarity fails to explain.
```

If this variant improves over gradient-similarity Top-K, PIPER has a stronger independent contribution.

### 3.4 Hybrid PI + Gradient Similarity

Define a normalized hybrid score:

```latex
s_{\mathrm{Hybrid}}(x_r)
=
\alpha \cdot \mathrm{Norm}(\mathrm{PI}(x_r))

(1-\alpha) \cdot \mathrm{Norm}(\mathrm{GS}(x_r)).
```

Test:

```latex
\alpha \in \{0.25, 0.5, 0.75\}.
```

This checks whether finite-step PI and first-order gradient similarity contain complementary information.

### 3.5 PI-Weighted KL Budget Allocation

Instead of selecting a hard Top-K set, allocate KL weight continuously:

```latex
\lambda_i
=
\lambda
\cdot
\frac{
\exp(\mathrm{PI}(x_i) / \tau)
}{
\sum_j \exp(\mathrm{PI}(x_j) / \tau)
}.
```

The objective becomes:

```latex
L
=
L_{\mathrm{unlearn}}

\sum_i
\lambda_i
\mathrm{KL}
\left(
p_{\theta_0}(\cdot|x_i)
\Vert
p_{\theta}(\cdot|x_i)
\right).
```

This turns PIPER from a Top-K sample selector into a preservation-budget allocation method.

## 4. Required Experiments

### 4.1 Minimal Main Comparison

Run the following methods under the same backbone, seed, lambda, train steps, Top-K size, and retain candidate pool:

| Method | Purpose |
|---|---|
| Backbone baseline | No preservation |
| Random Top-K local KL | Controls for adding local KL |
| TF-IDF semantic Top-K local KL | Controls for lexical semantic proximity |
| Gradient-similarity Top-K local KL | Controls for first-order gradient conflict |
| PI Top-K local KL | Current PIPER |
| Residual-PI Top-K local KL | Tests finite-step information beyond gradient similarity |
| PI-weighted local KL | Tests continuous budget allocation |

The key comparison is not PI versus random. It is:

```text
PI / Residual-PI versus gradient-similarity local KL.
```

### 4.2 Metrics

Report both proxy and official metrics.

Proxy metrics:

| Metric | Role |
|---|---|
| all retain damage | Global retain utility proxy |
| PI Top-K damage | Vulnerable retain protection |
| gradient-sim Top-K damage | Whether gradient-selected vulnerable samples are protected |
| forget damage | Proxy forget strength |
| forget answer probability delta | Proxy forget-side behavior |

Official TOFU metrics:

| Metric | Role |
|---|---|
| Forget Quality | Benchmark-level forgetting |
| Truth Ratio | Forget-side answer deviation |
| Q/A probability | Probability-level forgetting |
| Q/A ROUGE | Generation similarity |
| Model Utility | Retain and auxiliary utility |

### 4.3 Forget-Quality Matched Analysis

This is required if the paper wants to claim improved trade-off.

Do not compare methods only at the same lambda. Compare them at similar forget quality:

```text
Among runs with comparable forget-side metrics,
which method gives lower retain damage?
```

Without this step, the conclusion must remain weaker:

```text
PI improves local vulnerable-retain preservation under proxy metrics,
but strict Pareto improvement is not established.
```

## 5. Suggested Paper Positioning

Weak positioning:

```text
We propose PI to select retain samples for local KL.
```

Stronger positioning:

```text
We show that gradient similarity is a first-order approximation of finite-step retain interference. Building on this observation, we propose Predictive Interference probing to estimate update-induced retain vulnerability and use its residual over gradient similarity to allocate local preservation budget during unlearning.
```

This version makes the contribution methodological rather than heuristic.

## 6. Reviewer Risks

### Risk 1: PI is only expensive gradient similarity

Required response:

```text
Add gradient-similarity Top-K local KL and Residual-PI experiments.
```

### Risk 2: Retain improves only because forgetting weakens

Required response:

```text
Add forget-quality matched or Pareto-curve analysis.
```

### Risk 3: Semantic baseline is too weak

Required response:

```text
Clearly label current semantic baseline as TF-IDF semantic-neighbor baseline.
Optionally add embedding-similarity baseline as an appendix.
```

### Risk 4: PI Top-K is arbitrary

Required response:

```text
Add PI-weighted KL budget allocation and Top-K fraction sensitivity.
```

## 7. Recommended Next Implementation Step

The next coding step should not be another large benchmark sweep. It should be a focused mechanism experiment:

```text
NPO, forget10, seeds 0/1/2
lambda in {0.1, 0.2, 0.3, 0.5}
methods:
  baseline
  random local KL
  semantic local KL
  gradient-similarity local KL
  PI local KL
  residual-PI local KL
  PI-weighted local KL
```

If Residual-PI or PI-weighted KL does not beat gradient-similarity local KL, then the PI contribution should be narrowed to a diagnostic tool rather than a main unlearning method.

