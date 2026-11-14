# Does gamma predict SAE quality beyond L0, FVU, layer and width? (analysis plan, written before the analysis)

Written 2026-11-13, before any gamma-quality statistic was computed. At the time of writing I had built the
matched table (which SAEs have which public metrics, and their gamma values) but had not computed any
correlation, regression or plot of gamma against a quality measure, and the decoders of sample B had not been
scored.

## Question

Does the relative exponent gamma (decoder eigen-slope / Sigma eigen-slope in the same rank window) carry
information about established SAE quality measures that L0, FVU, layer and width do not already carry?
gamma falls with L0 by a known law (about 0.1 per doubling at a fixed layer), and quality measures also move
with L0, so raw gamma-quality correlations are not evidence. Only the association between the parts of gamma
and of quality that L0, layer, width (and FVU where stated) do not explain counts.

## Samples

A. Gemma Scope residual SAEs already scored in this project, matched by (model, layer, width, L0 name) to the
public SAEBench per-SAE results (Hugging Face dataset adamkarvonen/sae_bench_results, release
gemma-scope-2b-pt-res / gemma-scope-9b-pt-res). Gemma-2-2b: layers 5, 12, 19 at 16k and 65k (5 SAEs each)
plus layer 12 at 1M (2 SAEs); Gemma-2-9b: 16k at layers 9 (6), 20 (8), 31 (6). n = 52 SAEs in 6 layer
clusters. gamma as already computed: all decoder rows, unit norm, centred; Sigma from about 1M Pile tokens.
Gemma-2-2b windows ranks 4-50, 11-150, 31-450, 101-1000; Gemma-2-9b windows scaled by 3584/2304
(ranks 6-78, 17-233, 48-700, 157-1556). Windows are referred to as top, primary, bulk1, bulk2.

B. The SAEBench width-4096 suite on Gemma-2-2b layer 12 (adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108:
7 architectures x 6 sparsities, BatchTopK, Gated, JumpReLU, MatryoshkaBatchTopK, PAnneal, Standard (ReLU),
TopK), with its public SAEBench results (adamkarvonen/sae_bench_results_0125). Decoders are downloaded and
scored here exactly as the Gemma Scope SAEs (all rows, unit norm, centred covariance, 2b windows, layer-12
Sigma from 1M tokens). Expected n = 42 (fewer if any SAE lacks a result file or a decoder). One layer, one
width, so the controls are L0 (and FVU); architecture is the cluster.

C. Our own TopK SAEs on Gemma-2-2b (runs_real.jsonl, real condition, 57 SAEs in 18 (layer, k) cells, 3-4
seeds per cell). Only FVU and dead fraction exist for them, and no encoders or decoders are stored (only
decoder covariances), so nothing further can be computed locally. Within a cell L0 = k and layer are fixed,
so the only test is whether seed-to-seed variation in gamma (fixed-n scoring) tracks seed-to-seed variation in
held-out FVU and dead fraction. This is a small, secondary check. The real-minus-null gap is defined per
(layer, k) cell, and quality varies only between cells, where layer and k already explain it, so the gap
cannot be tested against quality with these data; I record that rather than test it.

## Quality measures (fixed now)

Primary, seven per SAE (SAEBench names):
1. ce_loss_score (fraction of the CE loss recovered; core)
2. explained_variance (core; FVU = 1 - explained_variance)
3. mean_absorption_score (absorption, first letter; lower is better)
4. sae_top_1_test_accuracy (sparse probing)
5. scr_metric_threshold_20 (spurious correlation removal, 20 latents)
6. tpp_threshold_20_total_metric (targeted probe perturbation, 20 latents)
7. autointerp_score

Reported but outside the decision family: delta CE (CE with SAE minus without), kl_div_score.
The unlearning metric is not used (it is meant for instruction-tuned models).

## Model

For each sample, metric m and window w:

Model 1 (all metrics):  m ~ layer fixed effects + log2 L0 + (log2 L0)^2 + log2 width
Model 2 (metrics 3-7):  Model 1 + FVU

L0 is SAEBench's measured L0 (core sparsity.l0). In sample A the layer fixed effects are per (model, layer),
which also absorbs the model and Sigma. In sample B there are no layer or width terms. For metrics 1-2 the
decision uses Model 1 (FVU is metric 2 itself); for metrics 3-7 it uses Model 2, the "beyond L0 and FVU"
question, with Model 1 also reported.

Effect size: the partial correlation of gamma with m given the covariates, computed as the correlation of the
residual of m and the residual of gamma, each regressed on the same covariates (Frisch-Waugh). Also reported:
the standardized coefficient (change in m per SD of residual gamma, in units of the SD of residual m), the
increase in R^2 from adding gamma, and the R^2 of gamma on the covariates (how much of gamma is left to work
with).

Intervals: 95% percentile intervals from 2,000 bootstrap replicates, two-stage, clustered: resample clusters
(layers in A, architectures in B, cells in C) with replacement, then SAEs within each drawn cluster with
replacement, refitting the whole model each time (a cluster drawn twice gets two fixed effects). With 6 to 7
clusters these intervals are rough and are stated as such.

Test: permutation p-value for the partial correlation, permuting the gamma residuals within clusters (10,000
permutations; within layer in A, within architecture in B, within cell in C), two-sided.

Robustness: leave-one-cluster-out (each layer in A, each architecture in B) refits, reporting the range of the
partial correlation; cubic in log2 L0 instead of quadratic; release-name L0 instead of measured L0 (A); in A,
a layer x log2 L0 interaction; in B, architecture fixed effects added (gamma within architecture).

## Multiplicity and decision rule

Family: 7 metrics x 4 windows = 28 tests per sample, Holm-adjusted at familywise 0.05, separately in A and B.
C is 2 measures x 4 windows = 8 tests, Holm-adjusted.

gamma "predicts" metric m in a sample if all of:
(a) the primary-window test is Holm-significant (decision model as above);
(b) the partial correlation has the same sign in at least two of the other three windows;
(c) the sign is unchanged in every leave-one-cluster-out refit;
(d) |partial r| >= 0.3 in the primary window.

The claim that gamma matters for practice needs a non-reconstruction metric (3-7) that predicts in A (or B)
and has the same sign with unadjusted p < 0.05 in the primary window in the other sample.

"No evidence": no metric passes (a) in a sample.

Clean null for a metric: the 95% bootstrap interval of the primary-window partial correlation lies inside
(-0.3, 0.3). Otherwise the result is inconclusive at that n. For orientation, with about 40 residual degrees
of freedom a two-sided 5% test has 80% power only for a partial correlation near 0.42, so before Holm
correction this design can find only fairly large effects, and a non-significant result alone is not a clean
null.

Raw (unadjusted) gamma-quality correlations are reported only for reference and are not interpreted.

## Outputs

Analysis script and table under the working scratch area; a numbers JSON and a figure under
results/r2/quality/ only if there is a positive result or a clean null.
