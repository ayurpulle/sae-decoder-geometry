# Pre-registration: the power-matched sparse null (job "pmatch")

Date: 2026-11-13. Written before any run of `experiments/r2_job_pmatch.py`. The rules below are also in that
file's docstring and are applied by `experiments/r2_summary.py` (section "pmatch"). If the code changes after
this note, the hashes at the end no longer match and the note must be redone with a new date.

## The question

Our TopK SAEs on Gemma-2-2b (width 16,384, k = 40) have flatter decoder spectra than the activation covariance Σ
in the bulk windows (ranks 31–450 and 101–1000). SAEs trained on the codable sparse null do not. That null
draws x = μ + √(1−ρ)(s − Es)A + √ρ g with M = 8,192 atoms, L = 16 active, half-normal values, a Gaussian share ρ
tuned to the real FVU, and atoms A = RΣ^{1/2}. All of its atoms carry the same power, and their directions are
Σ^{1/2}-shaped, so its slope near 1 in the bulk is partly built in. The gap is large and robust in sign
(γ_fixed, real vs null: layer 13 .595/.863 vs .930/1.156; layer 0 .375/.623 vs .906/1.030), but we cannot tell
what it means. The question here is narrow: if the null's atoms carry the unequal power we measured on the real
SAEs, does the bulk gap close?

## What changes in the null, and what does not

Only the dictionary A changes. Σ, μ, M, L, the value law, the dense Gaussian share and the way ρ is tuned all
stay as before.

The constraint is AᵀA = Σ, so the data covariance stays exactly Σ. Write P = AAᵀ. Its nonzero eigenvalues are
those of Σ, and its diagonal is the atom powers p_j = ‖a_j‖². By the Schur–Horn theorem, a power vector is
reachable exactly when it is majorized by Σ's eigenvalues: for every m, the m largest powers sum to no more than
the m largest eigenvalues. This bound has a direct consequence for the head. A small set of very strong atoms
can only exist if those atoms point along Σ's top eigenvectors. So asking for unequal power also changes
directions, and the covariance alone forces that change. We let it happen and do nothing more.

How a dictionary is built (`experiments/r2_pmatch_gen.py`):

1. The target is the real seed-0 SAE at the same layer: the power of each live latent on 1M tokens, from the
   robust job's step P. The atoms take the M evenly spaced power quantiles, scaled to sum to tr Σ.
2. Where the head breaks the Schur–Horn bound, it is capped at a little inside the bound (0.9 for the top atom,
   rising to 1 at rank d). The excess goes to the other atoms in proportion to their power. The share of power
   moved this way is recorded.
3. Directions start exactly as in the equal-power null (Σ^{1/2} z, independent of power).
4. Then repeat two steps. First apply the least linear change that makes AᵀA = Σ (the optimal-transport map
   T = S^{-1/2}(S^{1/2} Σ S^{1/2})^{1/2} S^{-1/2}, with S the current AᵀA; then T S T = Σ). Then rescale each
   row toward the target power that matches its current power rank.

Every dictionary we keep is the output of the first step, so AᵀA = Σ holds to rounding. In a local test the
relative error was 5e-13 in float64 and 3e-8 after the cast to float32. The job stops if the error exceeds
1e-6, and train_sae.py's 1M-token check sample measures the covariance a second time. No SAE decoder direction
enters anywhere; the only input taken from a real SAE is its list of latent powers.

I also tried matching the measured (power, Rayleigh quotient) pairs, i.e. how strongly each latent lines up with
Σ, together with its power. Locally it did not converge: power errors stayed at 2–8 in log and quotient errors
at 0.3–0.6 in log10. The constraint fixes the power-weighted second moment of the directions to Σ, and the real
latents do not decompose the covariance (their powers add up to well under the variance at middle layers). So
that variant is not run. The code stays in the generator for local use only.

## Settings (in this order; the job starts no SAE after 90 minutes)

All with SAE seed 0, width 16,384, k = 40, and 9.5M tokens × 3 epochs.

1. Layer 13, M 8192, L 16. Counts for the verdict.
2. Layer 0, M 8192, L 16. Counts for the verdict.
3. Layer 13, M 8192, L 8. Compared with the equal-power null at the same (M, L) from the robust job: .758/1.035.
4. Layer 13, M 8192, L 32. Compared the same way: 1.168/1.337.
5. Layer 13, M 8192, L 16, with a new dictionary draw and new token streams. A replicate of setting 1.

ρ is tuned to the real FVU with a tolerance of 0.03. The first run uses the equal-power null's ρ for that
setting. If needed, the second is a Newton step using that null's own dF/dρ, and the third a secant step.
Every run is scored.

## Decision rules (bulk windows, γ_fixed)

Notation for each setting: N is the power-matched null, R is real (mean of three seeds), Z is the equal-power
null at the same (layer, M, L), and f = (Z − N)/(Z − R) is the fraction of the gap that closes.

Each bulk window gets one class:

- closed: |N − R| ≤ 0.10.
- overshoot: N < R − 0.10.
- partial: f ≥ 0.30, and the window is neither closed nor overshoot.
- open: anything else.

Per setting:

- **Unequal power explains the bulk gap** if both bulk windows are closed.
- **Partly explains** if no window is open but the two are not both closed. If both overshoot, that is noted.
- **Does not explain** if at least one bulk window is open.

The headline answer comes from settings 1 and 2. If both layers give the same verdict, that is the answer. If
they disagree, the answer is "partly". Without an FVU match at both layers, the result is "insufficient".

For settings 3 and 4, the result is "same at other (M, L)" if both give setting 1's verdict, and "differs"
otherwise.

For setting 5, a difference above 0.05 from setting 1 in either bulk window means the 0.10 margin is too narrow
for the noise between dictionary draws. That gets flagged.

Why 0.10 and 0.30: the spread between seeds at a fixed generator is at most 0.011. The smallest gap to be
explained is 0.293. A 0.10 margin is a third of that gap, and 0.30 is three times what seed noise could fake.

A check that does not enter the verdict: we measure the trained null SAE's own latent-power distribution (top-10
and top-1% shares, effective-sample fraction) and set it beside the real SAE's. Even the equal-power null's SAE
has strongly unequal latent power (top 1% carry 48% at layer 13, from the dense share). So latent power and atom
power are different things, and this check shows how far the match carries through training.

## Prediction

Three things shape it, all fixed before any SAE is trained.

The pilot. One-parameter power reweightings could be fitted to the bulk at one layer, but never to the real
window profile. A dense share at the matched ρ also wiped out flattening that had been built into the atoms.
V1 has no free parameter (the powers are measured), so if it lands, it lands without fitting. But the dense share
at layer 13 (ρ ≈ .43) will probably pull the SAE back toward steeper slopes.

The generator. The covariance constraint forces strong atoms onto Σ's top eigenvectors. That leaves the rest of
the dictionary flatter than Σ, mostly in ranks 4–450. Atom-cloud γ for the four windows, with local stand-ins for
the real powers (equal-power atoms: .99–1.00 everywhere):

| Stand-in | Atom-cloud γ |
| --- | --- |
| Layer 13, Gemma Scope 65k layer-12 powers | .15 / .22 / .51 / .96 |
| Layer 13, the equal-power null SAE's own latent powers | .46 / .50 / .69 / 1.02 |
| Layer 0, Gemma Scope layer-0 powers | .17 / .28 / .48 / .84 |

A simple rule. Predict N as atom(V1) + [Z − atom(V0)]: the SAE moves the power-matched atoms by the same amount
it moves the equal-power ones. Each dictionary report stores both atom clouds before training, and the summary
prints this prediction beside every result. With the stand-ins, layer 13 comes out at about .44–.62 / 1.12–1.17
(31–450 closed or overshoot, 101–1000 open). Layer 0 comes out at about .39 / .87 (closed and partial).

My call:

- Layer 13: does not explain.
- Layer 0: partly explains.
- Headline: **partly explains**. My probabilities are partly .45, does not explain .45, explains .10.
- Settings 3 and 4: does not explain, the same as setting 1.
- Replicate: within 0.05.

If I am wrong, I expect the miss to be toward "does not": the SAE ends up steeper than the additive rule says,
because the dense share erases flattening.

What each outcome would mean:

- **Explains.** Unequal power, plus the alignment the covariance forces on it, accounts for the bulk gap. The gap
  then carries no evidence of structure beyond Σ and feature power.
- **Does not explain.** Something beyond power is needed, probably in how the directions of the many weak
  features are arranged.
- **Partly.** Power is part of the story. How large a part depends on the layer, and the paper would report that
  fraction.

## Files and hashes

- Job: `experiments/r2_job_pmatch.py`, sha256 `f9905b2932efc417bbf5976ef7f80d425664edb29304d7f75648d6fed14d2483`
- Generator: `experiments/r2_pmatch_gen.py`, sha256 `9794b80e6548e79bcee9181f9c8444d2058038f13e691d9585f0ddd43d80aa2a`
- Trainer: `train_sae.py`, sha256 `fb348efb95614cda8e1cdc8b33bfe0d9bf82a2ce2038a42973849afd4aae7548`
- Rules as applied: `experiments/r2_summary.py`, sha256 `bb67a086d0b6451495572a4537165a4fb5e43d3309df1ed5a71dabb700e3d043`
- Rows: `results/r2/runs_pmatch.jsonl`; dictionary reports: `results/r2/pmatch/`; null latent power:
  `results/r2/power_pmatch/`; calibration: `results/r2/pmatch_calibration.json`.
