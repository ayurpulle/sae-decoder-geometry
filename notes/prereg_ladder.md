# Rules fixed in advance: no-dense-share run, SAE seeds, and the random-model ladder

Date: 2026-11-19. Written before any run of `experiments/r2_job_pmatch2.py` or `experiments/r2_job_ladder.py`.
Hashes of both files are at the end; if either file changes after this note, the hashes no longer match and the
note must be redone with a new date. Windows, γ, fixed-n scoring (8,192 live latents, 4 draws) and the SAE recipe
(TopK, width 16,384, k = 40, 9.5M tokens × 3 epochs, 0.5M held out) are those of every earlier run. The bulk
windows are ranks 31–450 and 101–1000.

## Part 1. The power-matched null without a dense share (job `pmatch2`, part A)

Why. In the first run the SAE trained on the power-matched null had more concentrated latent power than the real
SAE (top 1% of latents: 53% against 35% at layer 13). A few latents may be absorbing the dense Gaussian share ρ.
At layer 13 the equal-power null with M = 32,768 atoms, L = 16 reaches the real FVU with ρ = 0 (FVU .309 against
.288), and has the smallest equal-power gap of any matched setting (γ .756 / 1.001 against real .595 / .863, gaps
.161 / .138). The power-matched null was not run there.

Run. Setting PM|V1|L13|M32768_L16|ds0, dictionary and calibration exactly as in `notes/prereg_pmatch.md`, first
run at ρ = 0. ρ cannot go below 0: if the first FVU is above .318 the run is kept and flagged unmatched, and the
setting gives no verdict. Three SAE seeds (0, 1, 2) at the accepted ρ, and seeds 1 and 2 of the equal-power
comparator, so R, Z and N are all three-seed means.

Rule. The per-setting rule of `prereg_pmatch.md`, unchanged, applied to the three-seed means: per bulk window,
closed if |N − R| ≤ 0.10, overshoot if N < R − 0.10, partial if the closed fraction (Z − N)/(Z − R) ≥ 0.30, open
otherwise; EXPLAINS if both closed, DOES NOT EXPLAIN if either is open, PARTLY otherwise. Because the gap here is
only 1.4–1.6 times the 0.10 margin, "closed" is easy; the closed fraction with its seed spread is reported beside
the verdict and is the number to read.

Manipulation check (reported, not part of the verdict): the null SAE's top-1% latent power share beside the real
SAE's .35.

Prediction. EXPLAINS (probability .55; PARTLY .35; DOES NOT .10). Closed fraction 40–80% in both windows. The
null SAE's top-1% share falls below .45, i.e. closer to real than the .53 of the reference setting; if it does
not, the dense share was not the cause of the extra concentration.

## Part 2. SAE seeds for the reference settings (job `pmatch2`, part B)

Run. Seeds 1 and 2 at the accepted ρ of PM|V1|L13|M8192_L16|ds0 (ρ .43) and PM|V1|L0|M8192_L16|ds0 (ρ .13). The
two dictionaries are rebuilt with the same generator, seed and stopping rule, because the files of the first run
were not kept; the rebuilt dictionaries' atom reports are compared with the first run's.

Reported. For each setting the three-seed mean and standard deviation of N in each bulk window, and of the closed
fraction (Z's two seeds from the first runs). The paper's ±5-point band, which borrowed the equal-power null's
seed spread, is replaced by the measured one.

Rule. A verdict of the first run is "confirmed" if the same rule applied to the three-seed mean gives the same
verdict (layer 13 PARTLY, layer 0 DOES NOT).

Prediction. Seed standard deviation of N at most 0.015 in both bulk windows at both layers; both verdicts
confirmed; the closed fractions move by less than 5 points.

## Part 3. The ladder (job `ladder`)

Question. The power-matched null leaves part of the real-versus-null bulk gap unproduced (N − R: layer 13
.155 / .13, layer 0 .36 / .29). Does that remainder need a trained model?

Datasets, each 10M Pile tokens, BOS dropped, same token stream as before:
- embT: the trained Gemma-2-2B input embedding of each token, no transformer blocks.
- embR: the same with a random embedding matrix (i.i.d. Gaussian, same overall RMS).
- randL0, randL13: resid_post of blocks 0 and 13 of a fully randomly initialised Gemma-2-2B (default initialisation
  of every weight, embeddings included; real tokenizer).
The trained counterparts of randL0 and randL13 are the existing layer-0 and layer-13 results.

Per dataset: its own Σ; three real SAEs (seeds 0–2); the equal-power null (M 8,192, L 16), ρ calibrated to the
mean real FVU within 0.03, two SAE seeds; the power-matched null built from the seed-0 real SAE's latent powers,
calibrated the same way, one SAE seed. R, Z, N are the means of γ at 8,192 live latents. Gap G = Z − R,
remainder Rem = N − R. Raw exponents are not compared across datasets.

Rule, per random dataset against its trained counterpart (embR against embT; randL0 and randL13 against trained
layers 0 and 13):
- uninformative, if the trained counterpart's Rem is below 0.10 in either bulk window, or a needed null is missing
  or not FVU-matched;
- absent, if Rem < 0.10 in both bulk windows;
- persists, if Rem ≥ 0.70 × the trained Rem in both bulk windows;
- partial, otherwise.
Reported separately: whether the random dataset's gap G is itself below 0.10 in both bulk windows ("no gap").

Stop rules and known weak points.
- If a real SAE has fewer than 8,192 live latents it is scored on all live latents and flagged; the dataset's
  verdict is then reported as provisional.
- Embedding datasets contain one vector per token type, so a k = 40 SAE may reconstruct them almost exactly; the
  FVU match is then met trivially at ρ ≈ 0 and the comparison is weak. Real SAEs at k = 4 and 16 (seed 0) are
  trained on both embedding datasets and reported descriptively.
- A randomly initialised model's Σ may be dominated by a few directions. Top-1 and top-10 eigenvalue shares and the
  participation ratio are logged for every dataset and reported with the result.
- The 0.70 line is a choice made for this note.

Predictions.
- embR: the real decoder cloud is close to isotropic (raw decoder slope in ranks 31–450 flatter than −0.15),
  because one latent per token in random directions has no preferred direction; γ is then small and the gap G is
  large by construction. Verdict against embT: uninformative or partial (probability .6 that embT's own remainder
  is below 0.10 in a window, making it uninformative).
- randL0 against trained layer 0: persists (.55; partial .30; absent .15). Layer 0 is mostly token identity in
  both models.
- randL13 against trained layer 13: partial or absent (.60 together; persists .40). This is the least certain.
- At least one of the four datasets fails a stop rule or comes out uninformative (.7).

## Hashes (sha256)

- `experiments/r2_job_pmatch2.py`: 8703fb0638d1e6b1199cd9df09bb8ea3a2233cf4501c6fed5124a834423e4b7f (recorded 2026-11-19, before its run).
- `experiments/r2_job_ladder.py`: df6dd2da349f31aca808c80f49159ca5b7d080eb1be352fd5fe08dd1cc2dcd72 (recorded 2026-11-19, before its run).
- `experiments/r2_ladder_summary.py`: e4affde34d59df204dab5aeca1736504069e6a2d797406e8b7793c8e625c2594 (recorded 2026-11-19, before the ladder's run).
- Unchanged since `prereg_pmatch.md`: `train_sae.py` fb348efb9561…, `experiments/r2_job_pmatch.py` f9905b2932ef…,
  `experiments/r2_pmatch_gen.py` 9794b80e6548…, `experiments/r2_summary.py` bb67a086d0b6….
