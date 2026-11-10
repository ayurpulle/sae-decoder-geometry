# Research log

## 2026-10-30 — Phase 0: reproduce galaxy-scale eigen slopes

**What the original computes** (ejmichaud/feature-geometry `galaxy/PointCloudStructure_Galaxy.ipynb`, cells 10, 15, 18):
- SAEs: `gemma-scope-2b-pt-res`, **width 65k** (not 16k), one L0 per layer (L0: 73, L12: 72, L24: 124; full table in cell 10).
- Cloud: raw `sae.W_dec` (65536 × 2304). No explicit row normalisation. Gemma Scope rows are already unit norm (we measured 1 ± 3e-7).
- Covariance: `np.cov(W_dec, rowvar=False)`, i.e. **centred**, ddof=1. Eigenvalues from `eigvalsh`, sorted descending.
- Fit: `linregress(log10 rank, log10 eig)` over 0-based slice `[10:150]`, i.e. **ranks 11–150** (only ~1.1 decades). The reported value is the slope; the "α" in their figure is −slope.
- Their null: N(0,1) of shape 65536×2304, scaled ×0.005 for the plot. Slope −0.0276 (Marchenko–Pastur spread at n/d≈28).
- No dead-latent filtering and no frequency weighting.

**Our implementation:** `geometry.eigen_slope` (defaults match the above), driven by `experiments/phase0_reproduce.py`.
Raw `params.npz` W_dec matches sae_lens `SAE.W_dec` for residual SAEs (the loader rescales only embedding SAEs).

**Results** (`results/phase0/phase0.json`, `spectra.png`):

| layer | paper | ours | rel diff | R² | uncentred | unit-norm | matched Gaussian |
|---|---|---|---|---|---|---|---|
| 0  | −0.2383 | −0.2383 | +0.02% | 0.999 | −0.2383 | −0.2383 | −0.0276 |
| 12 | −0.4684 | −0.4684 | −0.01% | 0.994 | −0.4692 | −0.4684 | −0.0276 |
| 24 | −0.2545 | −0.2545 | +0.01% | 0.999 | −0.2547 | −0.2545 | −0.0276 |

**GO**: reproduced to <0.1%.

**Surprises / flags for later phases**
- Centring is irrelevant here: the mean decoder direction is tiny, so top eigenvalues barely change.
- Fit-range sensitivity is large. The spectrum is curved, not a clean power law:

  | range (0-based) | L0 | L12 | L24 |
  |---|---|---|---|
  | [1,15)    | −0.25 | −0.37 | −0.18 |
  | [3,50)    | −0.22 | −0.48 | −0.25 |
  | [10,150)  | −0.24 | −0.47 | −0.25 |
  | [30,450)  | −0.26 | −0.42 | −0.24 |
  | [100,1500)| −0.37 | −0.57 | −0.34 |

  The ordering L12 steeper than L0 ≈ L24 holds in every window. Absolute values move by up to ~0.13 across a 1-decade shift, though, so any theory-vs-data comparison of the *value* (e.g. against (1+α)/(1+b)) has to fix the rank window and report this spread.
- The spectrum falls off sharply above rank ~1000 (bulk edge / rank limit at d = 2304). Fit windows must stay below that.
- The isotropic Gaussian null is very weak: at n/d = 28 it is nearly flat by construction. It says nothing about what an SAE trained on structureless data would produce (Phase 2).

**Available variants for Phase 1** (`results/available_saes.json`, from the sae_lens registry and cross-checked 1:1 against the HF repo tree, 316 SAEs):
- All 26 layers: 16k × 5–6 L0s and 65k × 5–6 L0s.
- Layer 12 only: 16k, 32k, 65k, 131k, 262k, 524k, 1m (6 L0s each). This is the only full width sweep.
- 1m also at layers 5 and 19.
- Each 65k npz is 1.2 GB. The 1m files are ~18 GB each, so pulling only W_dec matters for disk and time.

**Next decision:** Phase 1 scope. Recommend starting with the layer-12 width ladder (7 widths × 6 L0s), then 16k/65k across layers.

## 2026-10-30 — Pre-registration (written before Phase 1/2 results were analysed)

**Fit windows** (0-based [start, end) on descending eigenvalues; slope reported for all of them):
- Gemma (d=2304): primary (10,150) = paper; alternates (3,50), (30,450), (100,1000). Upper end kept below the bulk-edge fall-off (~rank 1000).
- Pythia-70m (d=512): primary (5,50); alternates (2,25), (10,100). The paper's (10,150) is reported for reference only (it reaches 30% of d).
- A conclusion counts only if it holds in the primary window and at least two of the alternates.

**Phase 1a (geometry only):** layer-12 width ladder. All L0s for 16k–131k; two L0s (~20, ~70) for 262k/524k/1m. Width effects are compared at matched L0 (log-linear interpolation in L0), because L0 differs across widths.

**Phase 2 (small model) design:** Pythia-70m `blocks.3.hook_resid_post`; 20M Pile-uncopyrighted tokens cached (ctx 256, BOS dropped), last 1M held out.
- SAEs: TopK k=32 with AuxK (k_aux=512, coef 1/32, dead window 2M tokens); widths 1024/4096/16384; 15M training tokens; batch 4096; Adam lr 2e-4·(16384/width)^½ with linear decay over the last 20%. Decoder rows unit-norm, with the parallel gradient component removed. Inputs scaled so E‖x‖² = d (the same constant for all conditions).
- Conditions: real; gaussian N(μ,Σ) with μ and Σ from all 20M tokens; scrambled = every coordinate independently permuted within 1M-token chunks.
- Seeds 0–2.
- Widths reduced from the planned 2k–32k because a 32k run takes ~35 min on the M2 Pro (MPS).

**Notes from setup**
- MPS vs CPU Pythia activations agree to 0.15% max relative error (below fp16 storage precision).
- Smoke test (2M tokens, w4096, seed 0, primary window): real −0.443, gaussian −0.454, scrambled −0.253. This was not a planned run and has only 1 seed; it is recorded so it isn't over-read later.
- Σ of the Pythia activations itself has slope −0.97 (5,50) and −0.84 (10,100); 36% of the trace is in the top eigenvalue.

## 2026-10-30 — Phase 1a results: layer-12 width ladder (geometry only)

Run: `experiments/phase1_width_ladder.py` → `results/phase1/width_ladder_L12.json`; plots `slope_vs_l0_L12.png` and `slope_vs_width_L12.png` (`phase1_plot.py`). 28 SAEs, 16k–1m. No dead-latent exclusion yet.

**Headline: L0, not width, sets the slope.** In the paper's window (ranks 11–150), slope is close to linear in log L0 and nearly width-independent:
- about −0.74 at L0≈12, −0.47 at L0≈72, −0.24 at L0≈445;
- roughly +0.10 per doubling of L0 (0.50 over 5.2 doublings).
- The paper's −0.47 (65k, L0=72) is one point on this line. Their across-layer comparison used different L0 per layer (L0 73 / 72 / 124 for layers 0 / 12 / 24). Part of the "middle layer is steepest" result could be an L0 effect: layer 24's L0 of 124 vs 72 (0.78 doublings) alone predicts a slope ~0.08 shallower, against an observed difference of 0.21. Must check with matched L0 across layers (Phase 1b).

**Width at matched L0** (log-interpolated; 262k/524k/1m have only 2 L0s, so their interpolation is coarse):

| window | L0 | 16k | 32k | 65k | 131k | 262k | 524k | 1m |
|---|---|---|---|---|---|---|---|---|
| 11–150 | 40 | −0.528 | −0.549 | −0.567 | −0.579 | −0.565 | −0.541 | −0.537 |
| 11–150 | 70 | −0.444 | −0.460 | −0.473 | −0.488 | – | – | −0.472 |
| 101–1000 | 40 | −0.591 | −0.558 | −0.534 | −0.510 | −0.491 | −0.476 | −0.447 |
| 31–450 | 40 | −0.495 | −0.482 | −0.471 | −0.462 | −0.456 | −0.457 | −0.450 |

- In the top windows (ranks 4–150), slope varies by at most ~0.05 over 6 octaves of width, non-monotonically: steepest at 131k, then slightly shallower. Within ±0.05 this counts as approximately converged.
- In the deeper windows (31–450, and especially 101–1000), slope flattens monotonically with width and has no plateau by 1m: −0.59 → −0.45 at L0=40.
- Under the pre-registered rule (primary window plus ≥2 alternates), "the slope converges with width" is **not** supported: only 2 of 4 windows show it. The top of the spectrum is width-stable; the bulk is width-dependent.

**Implications**
- Any theory has to explain the L0 dependence first. A latent-allocation account where L0 changes the effective b (how many latents tile each manifold) would be one route. Testing it needs frequencies.
- Phase 1b should compare layers at matched L0 (16k and 65k, all layers, interpolated to common L0s). That's cheap: 16k/65k × 26 layers × ~5 L0s, W_dec only, ~60 GB.

## 2026-10-30 — Phase 2 interim (seed 0 only; seeds 1–2 running)

Eigen slope, Pythia-70m L3, TopK k=32 (`results/phase2/phase2_small.json`, `.png`):

| window | real 1k / 4k / 16k | gaussian 1k / 4k / 16k | scrambled 1k / 4k / 16k |
|---|---|---|---|
| 3–25 | −0.29 / −0.28 / −0.26 | −0.18 / −0.24 / −0.28 | −0.21 / −0.17 / −0.20 |
| **6–50 (primary)** | **−0.32 / −0.32 / −0.32** | **−0.30 / −0.35 / −0.39** | **−0.24 / −0.22 / −0.24** |
| 11–100 | −0.36 / −0.35 / −0.36 | −0.47 / −0.51 / −0.54 | −0.30 / −0.33 / −0.33 |

FVU: real 0.13 / 0.11 / 0.11; gaussian ~0.23; scrambled ~0.38. Dead latents ≤1 everywhere.

- The real-data slope is width-stable (−0.32 ± 0.01 in the primary window). Neither null is.
- The Gaussian null with the real Σ is not a match. It crosses the real slope near 1k width, steepens with width, and is far more curved (much steeper deeper in the spectrum). The early smoke-test agreement (−0.45 vs −0.44 at 2M tokens) was a crossing point, not agreement.
- The scrambled null is consistently shallower than real by ~0.08–0.10 in the primary window.
- Provisional reading: the stop condition (b ≈ a within error) is **not** met at seed 0. Real data gives a width-stable slope that both nulls miss. The estimator still contributes a lot: an SAE trained on a Gaussian already produces slopes of −0.3 to −0.4.
- Needs seeds 1–2 for error bars before it counts.

## 2026-11-02 — Phase 2 final (3 seeds), frequencies, Test 2, Phase 1b

### Phase 2: estimator nulls (Pythia-70m L3, TopK k=32, 15M tokens, 3 seeds) — `results/phase2/`

Eigen slope, mean ± sd over seeds:

| window | cond | 1k | 4k | 16k |
|---|---|---|---|---|
| **6–50 (primary)** | real | −0.311±0.007 | −0.329±0.006 | −0.322±0.007 |
| | gaussian N(μ,Σ) | −0.299±0.001 | −0.350±0.004 | −0.391±0.004 |
| | scrambled | −0.248±0.014 | −0.220±0.013 | −0.249±0.006 |
| 3–25 | real / gauss / scr | −0.29 / −0.18 / −0.21 | −0.29 / −0.24 / −0.18 | −0.26 / −0.28 / −0.21 |
| 11–100 | real / gauss / scr | −0.36 / −0.47 / −0.31 | −0.35 / −0.51 / −0.33 | −0.36 / −0.54 / −0.33 |

FVU: real 0.11–0.14, gaussian 0.22–0.24, scrambled 0.37–0.41. Dead latents ≤2 in every run.

**Verdict: the stop condition is not met.** The Gaussian null does not reproduce the real slope in any pre-registered window at 4k or 16k; it only crosses the real curve near 1k in the primary window. Three robust facts (hold in all three windows, seed sd ≤0.014):
1. Real-data slope is width-stable (spread ≤0.03 across 16× width).
2. Gaussian-null slope steepens monotonically with width (−0.30 → −0.39 primary) and is far more curved (−0.18 at ranks 3–25 vs −0.54 at 11–100 for 16k).
3. Scrambled is shallower than real at every width (by 0.06–0.11 primary).
Reading: the SAE objective applied to Σ alone already generates most of the slope magnitude (vs −0.13 for the isotropic matched Gaussian), but what the model adds is not steepness — it is width stability and a straighter spectrum. At 4k/16k real is *shallower* than the Σ-matched null.

### Phase 1 step 2: Gemma Scope L12 firing frequencies (1M Pile tokens, ctx 1024) — `results/phase1/freqs_L12.json`, `freqs_L12.png`

22 SAEs, 16k–131k. Sanity check passes: measured mean L0 within 6% of the name on every SAE.
- **Dead latents** (zero firings in 1M tokens): ≤1.5% at 16k; grow with width and with lower L0 (65k L0=21: 6%; 131k L0=12: 14%). Excluding them changes the eigen slope by ≤0.002 on every SAE. Dead latents are not the cloud's geometry.
- **Rank-frequency exponent** (log freq vs log rank, ranks 11 to n_alive/2): 16k/32k −0.59 to −0.71; 65k −0.76 to −0.79; 131k −0.78 to −0.91; the highest-L0 SAE at each width is steeper (−0.93 to −1.01) because of a long tail of rare latents. Piecewise exponents range from −0.4 to −1.2 depending on the rank window — the curve is not a single power law, which accounts for the literature disagreement (−0.57..−0.74 vs −1.0..−1.2).
- **Key negative:** across L0 at fixed width the eigen slope moves by ~0.4–0.5 while the overall frequency exponent moves by ≤0.1 (excluding the highest-L0 tail effect). Latent-frequency exponent does not predict the eigen slope. For slope = (1+α)/(1+b) to hold, b would have to carry essentially all of the L0 dependence. Caveat: these are *latent* frequencies, not the theory's *feature* frequencies p_i.

### Test 2 (H2, feature splitting): near-duplicate latents vs L0 — `results/test2/neardup_L12.json`

28 SAEs, all layer-12 widths. Nearest-neighbour cosine for every latent (sampled 16k queries for ≥262k); greedy merge of near-duplicates at cos>τ and re-measured eigen slope for widths ≤131k.
- **(a) holds:** fraction of latents with a neighbour at cos>0.5 rises as L0 falls at every width (16k: 2%→20%; 131k: 5%→59%) and with width at fixed L0 (L0≈20: 16k 20%, 65k 35%, 131k 41%, 524k 61%, 1m 64%). Caveat: higher nn-cosine is partly a direct consequence of anisotropy, not independent evidence of splitting.
- **(b) fails decisively:** merging at cos>0.5 removes up to 34% of latents (131k L0=12: 45k latents) and changes the eigen slope by **≤0.010 in all 22 SAEs**, against an L0-driven range of ~0.5. Merging at 0.7 changes it by ≤0.002.
- **Conclusion:** the L0 dependence is not carried by tight split-feature clusters. The anisotropy is a cloud-wide concentration of decoder directions. This disfavours H2 and leaves H1 (covariance inheritance modulated by sparsity) and coarse-scale allocation (H3) as candidates. Test 1 (Σ spectra across Gemma layers) is the next discriminator.

### Phase 1b: slope vs layer at matched L0 — `results/phase1/layers.json`, `slope_vs_layer_matched_l0.png` (270/270 SAEs)

Ranks 11–150, interpolated in log L0:

| L0 | width | L0 | L5 | L9 | L13 | L15 | L17 | L20 | L23 | L25 |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 16k | −0.34 | −0.39 | −0.46 | −0.57 | −0.57 | −0.52 | −0.41 | −0.34 | −0.46 |
| 40 | 65k | −0.30 | −0.38 | −0.47 | −0.60 | −0.60 | −0.56 | −0.43 | −0.36 | −0.47 |
| 70 | 16k | −0.30 | −0.37 | −0.40 | −0.46 | −0.46 | −0.41 | −0.32 | −0.27 | −0.37 |
| 70 | 65k | −0.24 | −0.34 | −0.40 | −0.50 | −0.49 | −0.47 | −0.36 | −0.29 | −0.37 |
| 140 | 16k | −0.29 | −0.31 | −0.32 | −0.35 | −0.32 | −0.27 | −0.21 | −0.19 | −0.27 |

- **The middle-layer peak is real at the top of the spectrum and not an L0 artifact — but it is window-specific.** In the two top windows (ranks 4–50, 11–150) the peak sits at layers 13–15 (not 12), at every L0 and both widths. In the two deeper windows (31–450, 101–1000) there is no mid-stack peak: the spectrum is steepest at layers 2–7 and becomes monotonically shallower through the stack, at every L0. Under the pre-registered rule (primary + ≥2 alternates) "steepest in the middle" passes in only 2 of 4 windows, so it is **not** a robust property of the whole spectrum. Refined claim: the top ~150 eigendirections of the decoder cloud are most concentrated mid-stack; the bulk is most concentrated early. The paper's L0 choices only distort the edges of the top-window profile: their layer 24 (L0=124) reads −0.25, but at matched L0=70 layer 24 is −0.31, so the late-side contrast was overstated by ~0.06. Layer 0 (L0=73) was already matched.
- **Profile shape:** slow ramp from layer 0 to 13, faster decline from 17 to 23, then the final layer (25) steepens again at every L0 (unembedding-facing residual).
- **The peak amplitude shrinks with L0:** peak-minus-layer-0 is 0.23 at L0=40, 0.16 at L0=70, 0.06 at L0=140 (16k). High-L0 SAEs nearly erase the layer profile. Layer and sparsity interact; neither is a nuisance variable.
- 65k is slightly steeper than 16k mid-stack (−0.60 vs −0.57) and slightly shallower at layer 0 (−0.30 vs −0.34): width effects at matched L0 remain ≤0.04, consistent with Phase 1a.

### Where this leaves the hypotheses
- Paper's claim: survives in revised form — the spectrum is anisotropic and its top-window slope has a mid-stack peak that persists at matched L0. But the exponent is a joint property of (layer, L0, rank window), not of the layer alone, and the bulk of the spectrum has the opposite layer trend (steepest early). Any mechanism must explain both.
- Frequency-driven allocation (working hypothesis): not supported in its simple form. Latent frequencies are nearly L0-invariant while the slope is not; near-duplicate clusters carry none of the slope.
- Estimator artifact: not supported either. Real data is distinguishable from both nulls and, unlike them, width-stable.
- Open: whether the L0 and layer dependence are both inherited from Σ through the sparsity constraint (H1). Test 1 decides this cheaply.

## 2026-11-02 — Test 1 (H1, covariance inheritance): Σ spectra across all 26 Gemma-2-2B layers

Run: `experiments/test1_sigma_layers.py` (fp16 on MPS; validated against fp32 on layers 0–12: eigenvalue rel. error ≤3e-4, slope differences 0.0000), 200,508 Pile tokens, ctx 1024, BOS dropped, centred Σ per layer. Analysis `test1_sigma_plot.py` → `results/test1/sigma_slopes.json`, `sigma_vs_decoder_layers.png`.

**Σ eigen slope by layer:**

| window | L0 | L4 | L8 | L12 | L14 | L18 | L22 | L24 |
|---|---|---|---|---|---|---|---|---|
| 4–50 | −0.72 | −0.73 | −0.82 | −0.92 | −0.93 | −0.88 | −0.88 | −0.96 |
| 11–150 | −0.75 | −0.77 | −0.80 | −0.87 | −0.88 | −0.86 | −0.83 | −0.91 |
| 31–450 | −0.86 | −0.95 | −0.88 | −0.81 | −0.79 | −0.81 | −0.84 | −0.88 |
| 101–1000 | −0.99 | −1.15 | −1.05 | −0.84 | −0.83 | −0.84 | −0.89 | −0.91 |

**H1's central prediction holds: the window-dependent flip of the layer profile is already in Σ.** In the top windows Σ steepens into a mid-stack hump (layers 12–16) and rises again at the last layer; in the bulk window Σ is steepest at layers 2–6 and shallower after. That is the same qualitative pattern Phase 1b found in the decoder cloud, window for window.

**Decoder slope vs Σ slope across the 26 layers** (decoder at matched L0 from Phase 1b):

| window | 65k L0=40 r | 65k L0=70 r | 16k L0=40 r | γ (65k, L0=40) | γ (65k, L0=70) |
|---|---|---|---|---|---|
| 4–50 | +0.73 | +0.63 | +0.70 | 0.53±0.08 | 0.45±0.08 |
| 11–150 | +0.79 | +0.71 | +0.76 | 0.53±0.09 | 0.45±0.07 |
| 31–450 | +0.51 | +0.53 | +0.59 | 0.53±0.09 | 0.47±0.10 |
| 101–1000 | +0.71 | +0.68 | +0.73 | 0.56±0.11 | 0.51±0.11 |

- Σ explains roughly half of the across-layer variance of the decoder slope (r² ≈ 0.5–0.6 in the top and bulk windows, ≈0.3 at ranks 31–450).
- **γ = decoder slope / Σ slope ≈ 0.5 at L0=40 and ≈0.45 at L0=70, nearly the same in every window.** So to first order the decoder cloud's eigenvalues scale like Σ's eigenvalues to the power ~½: decoder spread along a direction tracks the activation *standard deviation* along it, not the variance. Pythia (Phase 2, k=32 of d=512): Σ slope −0.97, decoder −0.32 → γ ≈ 0.33, and the Σ-matched Gaussian null gives the same γ at 4k. γ falls as the SAE gets denser (L0 40 → 70; k/d 1.7% → 6%), so the L0 dependence from Phase 1a is a change in γ, not in Σ.
- **γ is not constant across layers: it has its own smooth profile** (bottom row of the plot), nearly identical in shape across all four windows and both widths. γ ≈ 0.35–0.45 at layers 0–3, rises to ≈0.65 mid-stack (peak at layers 13–15 in the top windows, layers 5–12 in the bulk), and falls back to ≈0.30–0.40 at layers 22–25. So the decoder cloud amplifies Σ's anisotropy in the middle of the stack by roughly 2× relative to the edges. The mid-stack decoder peak is therefore only partly inherited: Σ supplies a hump, and the SAE's response to Σ (γ) adds a second, co-located hump on top. The final-layer Σ steepening (layer 25) is largely *not* passed through (γ drops there).
- Candidate explanations for the γ profile, to test: (i) γ depends on how Gaussian the activations are at that layer (Phase 2 showed the Σ-matched Gaussian null gives a *different* γ from real data at larger widths); (ii) γ depends on Σ's own shape (e.g. top-eigenvalue fraction or effective dimension), which also varies by layer; (iii) γ tracks the fraction of near-duplicate latents (Test 2) — though merging them did not change the slope, so (iii) is unlikely.

**Verdict on H1:** supported as the baseline mechanism, with a structured residual. The reversal of the layer profile between the top and the bulk of the spectrum is inherited from Σ; the exponent is γ ≈ ½ at Gemma Scope sparsities on average. But γ itself varies ~2× across layers in a smooth mid-stack-peaked profile, so about half of the decoder's layer contrast comes from Σ and half from the SAE's layer-dependent response to it. That response is the part not explained by second moments and is where any feature-structure story (allocation, manifolds, non-Gaussianity) would have to live.

**Revised picture of the paper's finding.** The "galaxy-scale" power law in decoder directions is, to first order, the square root of the activation covariance spectrum. It is not an artifact (Phase 2) and not driven by latent frequencies or feature splitting (freqs, Test 2), but it is not independent evidence about feature structure either: anything that shapes Σ shapes it.

**Next (not started):**
1. Make γ(L0) quantitative: γ vs L0 at layer 12 across the 28 ladder SAEs (Σ_12 is in hand; one script, minutes). Check whether γ → 1 as L0 → 0 (one latent per direction ⇒ decoder cloud ∝ Σ?) and whether Phase 2 reproduces γ(k) on the Gaussian null alone.
2. Explain the γ layer profile: correlate per-layer γ with (a) Σ's top-eigenvalue fraction / effective dimension (in hand), (b) a per-layer non-Gaussianity measure of activations (kurtosis along top PCs; needs one more Gemma pass or the cached L12 acts for a start), (c) Test 2's near-duplicate fraction (in hand for L12 only).
3. Phase 3 (planted ground truth) is now a test of whether γ ≈ ½ falls out of TopK/JumpReLU on data with a *known* Σ, and whether manifold features perturb it.

## 2026-11-02 — Test 1b: γ(L0) at Gemma layer 12 — `results/test1/gamma_l0_L12.json`, `gamma_l0_L12.png`

γ = decoder eigen slope / Σ₁₂ eigen slope for the 28 ladder SAEs (Phase 1a slopes; measured L0 for ≤131k, name L0 above). Σ₁₂ slopes: −0.917 (ranks 4–50), −0.872 (11–150), −0.814 (31–450), −0.836 (101–1000).

**γ is a function of L0 almost alone.** Primary window (ranks 11–150), 28 SAEs, seven widths 16k–1m:

    γ = 1.245 − 0.114·log₂(L0),  R² = 0.984        (power law γ = 1.98·L0^−0.31, R² = 0.980)

| L0 | 12.8 | ~21 | ~40 | ~70 | ~140 | ~300 | 444 |
|---|---|---|---|---|---|---|---|
| γ (11–150) | 0.86 | 0.71–0.78 | 0.60–0.68 | 0.48–0.57 | 0.37–0.43 | 0.30–0.33 | 0.28 |

- Width spread at fixed L0 is ≤0.07 in γ (L0≈21: 16k 0.73 … 131k 0.78 … 1m 0.71), against an L0 range of 0.58. Same for ranks 4–50 (R² 0.97, −0.108 per doubling).
- **Sparsity acts at the top of the spectrum.** The L0 dependence weakens with depth: −0.114 per doubling at ranks 11–150, −0.071 at 31–450, −0.046 at 101–1000 (where R² falls to 0.65 and width re-enters: bulk γ at L0≈21 runs 0.78 at 16k → 0.60 at 1m, the Phase 1a width drift seen again).
- **Extrapolation to γ = 1** (decoder cloud ∝ Σ): the log-linear fits reach it at L0 ≈ 3 (ranks 4–50) and 4.5 (11–150), i.e. near "one to a few latents per token", not at L0 → 0. Below L0 ≈ 3 the fit would give γ > 1, so either γ saturates or the relation changes; no Gemma Scope SAE is that sparse. Testable in Phase 3 with k = 1, 2, 4.

**Pythia-70m L3 (Phase 2, k = 32, k/d = 6.25%), γ by window (3–25 / 6–50 / 11–100):**

| cond | 1k | 4k | 16k |
|---|---|---|---|
| real | 0.23 / 0.32 / 0.43 | 0.23 / 0.34 / 0.42 | 0.20 / 0.33 / 0.43 |
| gaussian N(μ,Σ) | 0.14 / 0.31 / 0.56 | 0.19 / 0.36 / 0.60 | 0.22 / 0.40 / 0.65 |
| scrambled | 0.16 / 0.26 / 0.37 | 0.14 / 0.23 / 0.39 | 0.16 / 0.26 / 0.40 |

- Real γ is width-invariant (0.32–0.34 primary) and in the same range as Gemma at the equivalent density (k/d = 6.25% ↔ L0 ≈ 144 at d = 2304 → γ ≈ 0.43); the windows are not strictly comparable across d, so this is order-of-magnitude agreement, not a test.
- The Σ-matched Gaussian null gives a γ that *rises* with width (0.31 → 0.40) and is far more window-dependent (0.14 → 0.65 across windows at 16k). Real data's γ is the width-stable, window-stable one. This is Phase 2's result restated in γ: the sparse-coding objective on Σ alone sets the scale of γ but not its stability.

**What a theory now has to produce.** A model-free statement of the empirical law: for a sparse autoencoder with unit-norm decoder trained on activations with covariance Σ, the decoder-direction covariance has eigenvalues λ_i ∝ (Σ eigenvalues)_i^γ with γ ≈ 1.25 − 0.11·log₂(L0) over the top ~150 eigendirections, nearly independent of width, and the mechanism also has to deliver (a) real-data width stability that a Gaussian with the same Σ lacks and (b) the mid-stack-peaked layer profile of γ at fixed L0 (Test 1). Mechanism sketch to test in Phase 3: at L0 → 1 each token is reconstructed by one latent, so decoder directions sample the activation directions themselves and the cloud inherits Σ (γ → 1 — consistent with the extrapolation hitting 1 near L0 ≈ 3–5); at larger L0 each latent explains a residual after the others, which pushes decoders toward the lower-variance directions and flattens the cloud (γ ↓). Frequency-driven allocation (2509.02565) is not needed to get this far; it would enter, if at all, through the residual γ profile across layers.

## 2026-11-02 — Test 1c: explaining the layer profile of γ (interim; Gaussian-null SAE sweep running)

γ (65k, L0=40, ranks 11–150) by layer: 0.40 (L0) → 0.38 (L2) → 0.51 (L6) → 0.60 (L10) → 0.65–0.66 (L12–16) → 0.59 (L18) → 0.50 (L21) → 0.43 (L23–24) → 0.49 (L25). Three candidates, correlated with γ across the 26 layers (`experiments/test1c_gamma_layers.py` → `results/test1/gamma_layers_explained.{json,png}`):

**(a) Σ's shape — no.** Top-1 fraction r = −0.01, top-10 fraction +0.08, participation ratio −0.33 (primary window). Σ's own slopes correlate moderately (bulk slope r = +0.65, primary slope −0.63), but γ is a ratio containing the Σ slope, so that is partly built in. Nothing about Σ's spectrum shape predicts how strongly the SAE amplifies it.

**(b) Non-Gaussianity — yes, and specifically massive-activation dimensions.** `experiments/test1c_nongauss_pass.py` (102k tokens, a different Pile shard from Test 1; `results/test1/nongauss.npz`):

| layer | 0 | 3 | 6 | 8 | 9 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max per-coordinate excess kurtosis | 7 | 7 | 4 | 95 | 471 | 574 | 650 | 758 | 683 | 397 | 208 | 82 | 31 | 20 |
| kurtosis along PC1 | −0.9 | −0.5 | −0.3 | −0.3 | 0.3 | 5.2 | 37.8 | −0.5 | −1.0 | −1.0 | −1.1 | −0.6 | −0.9 | −0.2 |
| median kurtosis, top-50 PCs | 1.2 | 0.3 | 0.2 | 0.1 | 0.2 | 0.2 | 0.2 | 0.2 | 0.1 | 0.1 | 0.2 | 0.2 | 0.2 | 0.4 |

- **Max per-coordinate kurtosis vs γ: r = +0.93** (ranks 11–150, 65k L0=40); +0.84 to +0.88 in the other top-window/L0 combinations; ~+0.3 in the bulk windows. It is the only candidate whose layer profile matches γ's in shape: flat and tiny through layers 0–6, a steep rise at 7–9, a plateau at 11–16, a decay through 24. The token-norm tail (q99/median) also correlates (r = +0.66) and is the same phenomenon.
- Along most top principal directions the activations are nearly Gaussian (median excess kurtosis of the top-50 PCs ≈ 0.1–0.2 at every layer). The heavy tails are confined to a few coordinates (and, at layers 10–13, to PC1 itself, kurtosis 5–38). This is why the Σ-matched Gaussian reproduces γ's *scale*: the bulk of the distribution is close to Gaussian.
- Caveat on the correlation: both curves are smooth single bumps over 26 layers, so the effective number of independent points is small; any mid-stack-peaked quantity would correlate. The shape match (sharp onset at layers 7–9, plateau, decay) is more persuasive than the r value, but it is still correlational.

**(c) Direct test — running.** Σ-matched Gaussian-null TopK SAEs (k = 40, width 4096, 3M tokens) at layers 0, 3, 7, 13, 17, 21, 24, plus a real-vs-null pair at layer 12 on the 1M cached tokens (`experiments/test1c_null_saes.py`). A Gaussian has kurtosis 0, so: if null γ is flat across layers while real γ humps, the hump comes from the non-Gaussian structure (plausibly the massive-activation dimensions); if null γ humps too, Σ is sufficient after all despite (a).

Mechanism to consider if (c) comes out "flat": heavy-tailed directions have a very large dynamic range of activation magnitudes; a thresholded sparse code (TopK/JumpReLU) can tile magnitude along one direction with several latents at different scales, placing a broad cone of decoders (not just near-parallel pairs — Test 2 only merged cos > 0.5) around Σ's heavy-tailed top directions and steepening the cloud there. Testable in Phase 3 by planting a heavy-tailed magnitude distribution on a few directions in otherwise Gaussian data.

**Addendum (b): it is one coordinate.** Residual dimension **1170** is the top heavy-tailed coordinate at every layer from 7 through 24 (kurtosis 24 → 95 → 471 → 574 → 661 → 650 → 642 → 758 (L14) → 729 → 683 → 558 → 397 → 310 → 208 → 127 → 82 → 57 → 31); no coordinate exceeds ≈7 before layer 7, and at layer 25 the maximum is 20 (dims 650, 2271). A second dimension, 334, is heavy-tailed only at layers 10–13 (kurtosis 12–63), which is where PC1 itself turns heavy-tailed. Dimension 1170's **variance share is ≤1% at every layer** (0.2–1.0%), so it is almost invisible in Σ — explaining why no Σ-shape statistic tracks γ while per-coordinate kurtosis does. This is the standard "massive activations" phenomenon (a single residual dimension taking rare, enormous values on specific tokens); the question for (c) is whether the SAE's response to it is what amplifies the decoder cloud's anisotropy mid-stack.

**(d) Direct mechanism ruled out: latents devoted to the massive dimension do not carry the slope.** `experiments/test1c_massive_latents.py` → `results/test1/massive_latents_L12.json`. Every layer-12 Gemma Scope SAE (16k, 65k, all L0) has a few latents aligned with dim 1170 (3–7 with |cos| > 0.5; max |cos| 0.81–0.95) and with dim 334 (2–7; max |cos| ≈ 0.97); a control dimension has none (max |cos| ≈ 0.1). Removing all latents with |cos| > 0.2 to either dimension (≤18 latents) changes the eigen slope by **0.000** in all 10 SAEs. So the SAE does allocate dedicated latents to the massive-activation axis, but the mid-stack amplification of γ is not those latents. If the heavy tails matter, it is through how the massive-activation *tokens* shape the rest of the dictionary (or the tail is merely a marker of a layer regime), not through decoders pointing at dim 1170.

**(c) first data point — layer 12, our TopK k=40, width 4096, 800k tokens each (short runs: FVU 0.49 real / 0.62 Gaussian, so absolute γ is not comparable with Gemma Scope's; only real-vs-null at equal budget is):**

| | γ ranks 4–50 | 11–150 | 31–450 | 101–1000 |
|---|---|---|---|---|
| real activations | 0.45 | 0.57 | 0.65 | 0.86 |
| N(μ, Σ₁₂) | 0.43 | 0.82 | 0.98 | 1.01 |
| Gemma Scope 65k L0=40 (reference) | 0.64 | 0.65 | 0.59 | 0.61 |

The Gaussian null gives a *higher* γ than real data from the primary window down (0.82 vs 0.57), and a far more window-dependent one (0.43 → 1.01 vs 0.45 → 0.86). Same direction as Pythia at 16k width (Phase 2). Σ alone over-concentrates the cloud relative to what real activations produce; real data's non-Gaussian structure *reduces* γ below the Gaussian value, and makes the spectrum straighter. The seven-layer null sweep (same budget at every layer) will show whether the Gaussian γ is flat across layers — in which case the whole mid-stack hump is a property of how real activations depart from Gaussian — or whether Σ already carries part of it.

**(c) result — Σ-matched Gaussian nulls across layers (`results/test1/null_saes.json`; TopK k=40, width 4096, 3M tokens each, same budget at every layer):**

| layer | 0 | 3 | 7 | 13 | 17 | 21 | 24 |
|---|---|---|---|---|---|---|---|
| γ null, ranks 11–150 | 0.74 | 0.74 | 0.69 | **0.60** | 0.64 | 0.70 | 0.65 |
| γ real (Gemma Scope 65k, L0=40) | 0.40 | 0.41 | 0.52 | **0.65** | 0.62 | 0.50 | 0.43 |
| γ null, ranks 4–50 | 0.41 | 0.43 | 0.34 | 0.28 | 0.32 | 0.34 | 0.30 |
| γ null, ranks 101–1000 | 2.22 | 2.09 | 2.08 | 2.18 | 2.21 | 2.17 | 1.96 |
| max per-coordinate kurtosis | 7 | 7 | 24 | 642 | 558 | 127 | 31 |

- **The Gaussian null does not reproduce the hump; it runs the other way.** Across layers, null γ correlates *negatively* with real γ (r = −0.81 primary, −0.70 ranks 4–50): it dips mid-stack exactly where real γ peaks. So Σ's layer-to-layer changes in shape, passed through the SAE objective, cannot produce the mid-stack amplification — consistent with (a). The hump comes from structure a Gaussian with the same Σ lacks.
- **The real-minus-null gap tracks the massive-activation kurtosis:** r = +0.93 (ranks 11–150), +0.85 (4–50). It is not explained by latents pointing at dim 1170 (d), so if the heavy-tailed tokens are causal they act on the dictionary as a whole.
- **Deep windows: the Gaussian null over-concentrates everywhere** (γ ≈ 2 at ranks 101–1000, ≈1.6 at 31–450, flat across layers), far above real (0.35–0.71). A short-trained TopK on Gaussian data piles its decoders into Σ's top subspace; real data spreads them out. This is the same direction as Phase 2 (Pythia) and the L12 real/null pair, now at every layer.
- **Caveat that the GPU queue fixes:** this compares Gemma Scope's real SAEs (JumpReLU, ~billions of tokens) with short TopK nulls (3M tokens, FVU ≈ 0.5). The *layer profiles* are comparable (every null has the same budget), but the absolute gap is not. The GPU queue's step (A) trains real, Gaussian and scrambled SAEs with identical architecture and budget at layers 3, 13 and 23 — the clean version of this table — and step (C) removes the massive-activation tokens at layer 13 to test causation directly.

**Test 1c verdict (provisional, pending GPU step A/C):** γ's mid-stack hump is a non-Gaussian effect, not a Σ effect, and is co-located (r ≈ 0.9) with the massive-activation dimension's heavy tail; latents aligned with that dimension do not carry it.

## 2026-11-02 — GPU queue results (Runpod H100, 88 runs) — `results/gpu/`

Setup: Gemma-2-2B `resid_post` at layers 3, 13, 23; 10M Pile tokens cached per layer (last 0.5M held out); our TopK SAE (`train_sae.py`, TF32 on CUDA), 3 epochs = 28.5M training tokens per run. γ = decoder slope / covariance slope. Pod `9885r7ihg6gcre`, 12:05–20:22 UTC. Scripts: `experiments/gpu_queue.py`, `gpu_summarize.py`, `gpu_analyze.py` → `results/gpu/{runs,sigma,analysis}.json`, `gpu_results.png`. Seed sd ≤ 0.015 on every γ in the top three windows, so every difference below of ≥0.03 is far outside noise.

**Validation.** Our real-data TopK SAEs reach eval FVU 0.18–0.29 (Gemma Scope-like quality) and give γ = 0.43 at L3 (16k, k=40), against 0.41 for Gemma Scope at L3 and L0≈40. A different architecture and recipe reproduces Gemma Scope's γ at that layer, so γ is a property of the activations there, not of one recipe. At L13 ours is lower (0.51 vs 0.65): the recipe changes the size of the mid-stack effect.

### A. Real vs Σ-matched Gaussian vs scrambled, equal budget, 3 seeds

γ (mean ± sd), ranks 11–150 / ranks 4–50:

| | L3 | L13 | L23 |
|---|---|---|---|
| real 16k | 0.430±0.002 / 0.313±0.008 | **0.508±0.003 / 0.484±0.009** | 0.389±0.003 / 0.421±0.008 |
| gaussian 16k | 0.627±0.002 / 0.357±0.005 | **0.533±0.006 / 0.281±0.008** | 0.677±0.003 / 0.355±0.008 |
| real 65k | 0.364±0.003 / 0.305±0.002 | 0.387±0.004 / 0.291±0.005 | 0.315±0.002 / 0.375±0.008 |
| gaussian 65k | 0.562±0.002 / 0.305±0.004 | 0.485±0.003 / 0.232±0.007 | 0.663±0.007 / 0.347±0.008 |
| scrambled 16k (vs own diag cov) | 0.640 / 0.443 | 0.557 / 0.412 | 1.077 / 0.722 |

- **Gaussian γ dips at L13 in every window/width combination above.** Real γ peaks at L13 at 16k in both top windows, and in the primary window at 65k (weakly: 0.36 → 0.39 → 0.32); at 65k ranks 4–50 real γ does not peak (0.31 → 0.29 → 0.38).
- **The robust quantity is the real-minus-Gaussian gap, which is largest at L13 in all four width×window cells** (16k primary −0.20 / −0.03 / −0.29; 16k top +0.04 / +0.20 / +0.07; 65k primary −0.20 / −0.10 / −0.35; 65k top 0.00 / +0.06 / +0.03). Restated: relative to what Σ alone produces, real activations concentrate the decoder cloud most at the middle layer.
- **Deep windows:** Gaussian and scrambled SAEs over-concentrate everywhere (γ 1.1–2.7), real stays at 0.29–0.81. Real activations spread decoders across far more directions than a Gaussian with the same covariance would induce — the largest real-vs-null effect in the whole project, and it is width-stable and layer-stable.
- Scrambled (vs its own diagonal covariance) behaves like the Gaussian at L3 and L13 and over-concentrates sharply at L23; FVU is worst for scrambled everywhere (0.61–0.77). Not interpreted further.

### B. Small-k sweep, L13, 16k, 2 seeds — the γ(L0) law, measured directly

| k (= measured L0) | 1 | 2 | 4 | 8 | 16 | 40 | 128 |
|---|---|---|---|---|---|---|---|
| real γ, ranks 11–150 | 1.034±0.006 | 0.893 | 0.824 | 0.732 | 0.665 | 0.509 | 0.354±0.010 |
| real γ, ranks 4–50 | 1.152±0.009 | 0.979 | 0.854 | 0.729 | 0.481 | 0.488 | 0.324 |
| gaussian γ, ranks 11–150 | 1.248 | 1.292 | 1.231 | 0.826 | 0.448 | 0.536 | 0.382 |
| dead latents, real (of 16384) | 15712 | 14494 | 12652 | 9915 | 1912 | 122 | 0 |

- **Real: γ = 1.016 − 0.094·log₂k, R² = 0.994 (ranks 11–150).** γ = 1.03 at k = 1: with one latent per token the decoder cloud's spectrum is Σ's (the decoders sample the activation directions); each doubling of k flattens it by ≈0.09. Gemma Scope's L12 law had −0.114 per doubling with a higher intercept (1.245); same functional form across recipes, intercept recipe-dependent. The extrapolated "γ = 1 at L0 ≈ 3–5" from Test 1b was wrong; measured directly, γ reaches 1 at L0 = 1.
- **Caveat:** at k ≤ 8 most latents die even with AuxK (k = 1: 672 alive of 16384), so small-k spectra come from a few hundred to a few thousand directions. The top windows (ranks ≤ 150) are well-defined with ≥ 672 points; the deep windows at k ≤ 4 are not (rank deficiency — γ = 30 at ranks 101–1000 for k = 1 is that artifact, not a finding). The ranks 11–150 law above uses only well-defined points.
- **The Gaussian null does not follow the law:** γ ≈ 1.25 for k ≤ 4, then drops abruptly between k = 4 and k = 16 and is non-monotone after. The smooth log-linear law is a property of real activations.

### C. De-massive, L13, 16k, k=40, 3 seeds — **null result**

Dropping tokens with the top 0.1% of |x₁₁₇₀| (threshold 256) or the top 1% (threshold 8.8) from training and eval changes γ by **≤ 0.005 in every window** (ranks 11–150: 0.508 → 0.507 → 0.508; ranks 4–50: 0.484 → 0.479 → 0.480) and FVU by ≤ 0.004. **The massive-activation tokens do not cause the mid-stack effect.** Together with Test 1c (d) (latents aligned with dim 1170 carry none of the slope), the r = 0.93 layer correlation between γ and dim 1170's kurtosis is a co-occurring marker of the mid-stack regime, not the mechanism.

### Where the project stands after the GPU queue

1. **Mechanism of the paper's finding (supported, now measured directly):** the decoder-direction covariance spectrum is Σ's spectrum raised to a power γ that is set by sparsity: γ ≈ 1 at L0 = 1 and falls ≈0.09–0.11 per doubling of L0, for two different SAE recipes. The paper's −0.47 is Σ's slope (−0.87 at L12) times γ(L0 = 72) ≈ 0.54.
2. **Not an estimator artifact:** real activations give a smooth γ(L0) law, width-stable γ, and far less over-concentration in the bulk than Σ-matched Gaussian or scrambled data; the nulls do none of these.
3. **Not frequency-driven allocation as written, not feature splitting, not massive activations.**
4. **Open:** what about mid-stack activations (beyond Σ, beyond dim 1170) makes the SAE concentrate its cloud relative to the Gaussian expectation, and why the size of that effect depends on the recipe (JumpReLU/Gemma Scope 0.65 vs TopK 0.51 at L13). Phase 3 (planted ground truth with known Σ and controllable non-Gaussian structure — sparse features, manifolds, heavy tails) is now the right tool: it can test which structure reproduces (i) γ = 1 at k = 1, (ii) the −0.1/doubling law, (iii) real-vs-Gaussian bulk spreading.

## 2026-11-03 — External review and first checks (`experiments/review_checks.py` → `results/review/checks.json`)

A review (pasted into the session) argued: (1) our layer-13 TopK γ(k) fits a zero-parameter law γ = 1 − ln k / ln d (max gap 0.023 by its reading); (2) "decoder slope ≈ γ × Σ slope" is an identity, and C_dec ∝ Σ^γ must be tested as a matrix statement (rotate into Σ's eigenbasis; principal angles); (3) "γ depends on L0 almost alone" contradicts the layer effect; check Gemma Scope's measured L0 per layer; (4) null confounds: alive/duplicate counts, scrambled destroys Σ, width drift could be estimation (subsample 1m); (5) wording: r = 0.5–0.8 is moderate, frequency test is "underdetermined", not "not supported". Points 2, 3 and 5 are accepted as stated; the write-up needs reframing.

**1. Law γ = 1 − ln k / ln d (rank-window γ, ranks 11–150 Gemma):**
- Our TopK, Gemma L13 (d = 2304): gaps +0.034, −0.018, +0.003, +0.001, +0.023, −0.014, −0.020 at k = 1…128. Fits.
- Gemma Scope L12 (JumpReLU): above the law at every L0, by +0.04 to +0.13 (larger at low L0). Same shape, recipe-dependent offset.
- Pythia L3 (d = 512, k = 32, law 0.444): 0.21–0.23 (ranks 3–25), 0.32–0.34 (6–50), 0.42–0.43 (11–100). Matches only in the deepest window; the primary misses by −0.11. Rank windows are not comparable across d (11–150 is 0.4–6.5% of d = 2304; 6–50 is 1–10% of d = 512), so this is not yet a clean test of the ln d denominator. The PCA d-sweep and the d = 256 synthetic run are.

**2. Matrix test of C_dec ∝ Σ^γ (`geometry.basis_gamma`) — partly fails:**
- Gemma Scope L12, ranks 101–1000 (bulk): basis-matched γ 0.25–0.64, R² 0.94–0.99 for all 28 SAEs. The bulk of the decoder covariance is co-diagonal with Σ and scales as a power of it.
- Ranks 2–150 (top): holds only at low L0. L0 ≤ ~85: basis γ 0.36–0.78, R² 0.91–0.97, close to the law (L0 = 41: 0.510 vs 0.514). L0 ≥ ~150: basis γ collapses to ≈ 0 (R² 0.01–0.8) while the rank-window γ stays 0.28–0.43. Top-10 subspace overlap falls from 0.57 (L0 = 12) to 0.09–0.13 (L0 ≥ 270); random ≈ 0.004.
- So at high L0 the decoder spectrum still decays like a power law, but its leading directions are not Σ's leading directions: "Σ raised to a power" is the wrong description of the top of the spectrum there. The rank-window ratio hid this, because it compares two spectra sorted independently.
- Pythia real: basis γ −0.04 to 0.03 (ranks 2–50), 0.21 (6–50), 0.36–0.41 (11–100); top-10 overlap 0.25. Σ's top direction carries 36% of Pythia's variance and the decoders do not follow it.
- Nulls: Pythia Gaussian has negative basis γ in the top ranges and top-10 overlap ≈ 0.00–0.09 (it avoids Σ's leading directions) but overlap50 0.60–0.78 (it piles into ranks ~10–50). Local Gemma Gaussian nulls: top-10 overlap 0.44–0.57 at layers 0–7, 0.13–0.17 at layers 13–24.

**3. Null confounds:**
- Alive: GPU-queue Gaussian and scrambled SAEs have 0–1 dead latents of 16k/65k; real data has 2–1,385. Pythia nulls: all alive. Local Gemma nulls: all alive. Near-duplicate fraction (cos > 0.5) ≤ 0.004 for every null and every real run. "Real spreads decoders out" is not "more latents survive"; if anything the nulls keep more.
- Subsampling: 1m-wide decoders cut to 16k random rows move toward the 16k SAEs in the deep window (1m L0 = 19, ranks 101–1000: full −0.503, subsampled −0.544, real 16k at L0 = 22 −0.650; 1m L0 = 73: −0.402 → −0.461 vs −0.519 at 16k L0 = 82). Estimation (rows per dimension) accounts for roughly 30–50% of the deep-window width drift; the rest is geometry. Top windows move ≤ 0.04.

**Revised claims (to carry into the write-up):**
- Replace "decoder slope ≈ γ × Σ slope" (identity) with the matrix statement and its limits: C_dec ∝ Σ^γ holds in the bulk at all L0 and in the top only at low L0.
- The candidate headline is γ ≈ log_d(d/k) for our TopK; Gemma Scope sits above it by a recipe-dependent offset. Not established until the d-sweep.
- "γ depends on L0 almost alone" → "at layer 12, across widths". The layer term stays open until measured L0 per layer is checked.
- Frequency-allocation test → "underdetermined"; the scrambled null destroys Σ (say so); r = 0.5–0.8 is "moderate".

## 2026-11-03 — Phase 3 stage 1: planted data, d = 256 (`results/phase3/saes_p3.json`)

Data `synth-base`: d = 256, 4,096 point features, p_i ∝ i^−1.5 (top 5 capped at 0.5), 32 active per token, direction anisotropy β = 1 (Σ slope ≈ −1.0), noise 0.05. TopK width 4,096, k = 1…64, 2 epochs over 5.5M tokens, seed 0, real vs Gaussian twin.

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| law 1 − ln k / ln 256 | 1.00 | 0.88 | 0.75 | 0.63 | 0.50 | 0.38 | 0.25 |
| planted, rank γ (ranks 6–50) | 1.55* | 1.45* | 0.88 | 0.86 | 0.54 | 0.37 | 0.20 |
| Gaussian twin, rank γ | 1.70* | 1.67 | 0.89 | 0.64 | 0.42 | 0.31 | 0.23 |
| planted, basis γ (ranks 6–50) | 1.81 | 1.52 | 0.88 | 0.85 | 0.56 | 0.13 | −0.09 |
| planted, top-10 overlap with Σ | 0.96 | 0.98 | 0.88 | 0.81 | 0.59 | 0.16 | 0.07 |
| planted, live latents | 453 | 973 | 3702 | 3763 | 4001 | 4096 | 4096 |
\* fewer than 1,000 live latents.

- Law: close at k = 16–64; the fit over k ≥ 4 is 1.30 − 0.184·log₂k (R² 0.955), steeper than 1/log₂256 = 0.125.
- **Top-subspace decoupling reproduces at the same k/d as Gemma Scope.** Top-10 overlap collapses between k = 16 and 32 (k/d 0.06–0.12) here, and between L0 ≈ 150 and 270 (k/d 0.065–0.12) for Gemma Scope L12. Basis γ in the top ranges goes to ≈ 0 at the same point. Candidate regularity: decoders track Σ's leading directions only while k/d ≲ 0.06.
- **No real-vs-Gaussian contrast:** the Gaussian twin behaves almost identically (it over-concentrates the bulk somewhat more at k = 8–16: γ(11–100) 1.41/0.97 vs 0.95/0.71). With 32 of 4,096 features active in 256 dims, the planted data is close to Gaussian from the SAE's view. Stage 2 must lower mean_active and M/d.
- **Feature matching:** ≤ 51% of live latents match a planted feature at cos > 0.7 (8–13% at k = 4–32), covering 134–520 of 4,096 features. Features are not recoverable in this regime (heavy superposition plus anisotropic directions), so the allocation-law test also needs the sparser regime.
- Engineering: the k = 1/2 runs grew the MPS process to 17 GB; `train_sae.py` now calls `torch.mps.empty_cache()` every 200 steps (no numerical effect), after which runs went from ~0.4 s/step to ~0.03 s/step.

## 2026-11-03 — Review check 1: PCA d-sweep of the law γ = 1 − ln k / ln d (`results/review/dsweep.json`)

Gemma-2-2b L12 activations projected onto their top-d′ principal components, d′ ∈ {256, 512, 1024, 2304} (2304 = the original space, rotated). TopK width 8,192, k ∈ {8, 32}, 4 epochs over 1M tokens (eval 100k), seed 0. Basis γ = slope of log diag(UᵀC_dec U) on log λ_Σ; the d′/8 window scales with d′.

| d′ | k | law | basis γ (ranks 1–d′/8) | basis γ (1–50) | rank γ (6–50) | top-10 overlap | live | FVU |
|---|---|---|---|---|---|---|---|---|
| 256 | 8 | 0.625 | 0.329 | 0.379 | 0.483 | 0.73 | 8082 | 0.315 |
| 256 | 32 | 0.375 | 0.315 | 0.358 | 0.467 | 0.73 | 8192 | 0.214 |
| 512 | 8 | 0.667 | 0.439 | 0.419 | 0.480 | 0.77 | 7263 | 0.372 |
| 512 | 32 | 0.444 | 0.382 | 0.336 | 0.473 | 0.57 | 8192 | 0.262 |
| 1024 | 8 | 0.700 | 0.585 | 0.537 | 0.584 | 0.81 | 5506 | 0.441 |
| 1024 | 32 | 0.500 | 0.460 | 0.343 | 0.459 | 0.61 | 8192 | 0.331 |
| 2304 | 8 | 0.731 | 0.697 | 0.606 | 0.658 | 0.82 | 4782 | 0.513 |
| 2304 | 32 | 0.552 | 0.519 | 0.400 | 0.492 | 0.70 | 8192 | 0.403 |

- **Budget control passes.** At d′ = 2304, basis γ(1–d′/8) is 0.697 (k = 8) and 0.519 (k = 32), against 0.73 and 0.51 from the 28.5M-token GPU runs at L13. The 1M-token budget reproduces γ at full dimension, so the smaller-d′ numbers aren't a training artefact (smaller d′ trains more easily: FVU falls with d′).
- **The d-law fails.** The law predicts a k-gap γ(8) − γ(32) = ln 4 / ln d′ that shrinks slowly from 0.25 (d′ = 256) to 0.18 (d′ = 2304). Measured, the gap grows from 0.01 to 0.18: 0.014, 0.057, 0.125, 0.178. At d′ = 256 the two k values give the same γ. The law and the data agree only at d′ = 2304, the dimension where the law was found. The fixed rank window (6–50) shows the same pattern (gap 0.02, 0.01, 0.13, 0.17).
- Level: k = 32 sits 0.03–0.06 under the law at every d′; k = 8 sits 0.03 under at 2304 but 0.30 under at 256. γ rises with d′ at fixed k, but in a way the law does not predict.
- Caveat: PCA truncation is not a model with smaller d. It keeps the same features, squeezed into fewer dimensions (more superposition per dimension, and the dropped variance becomes reconstruction error). A cross-model d-test (Pythia-70m d = 512: γ 0.33 primary vs law 0.444, close to the d′ = 512 k = 32 value 0.38) points the same way, but it confounds d with the model.
- **Consequence for the write-up:** do not restructure around log_d(d/k). Keep it as an empirical fit at d = 2304 for our TopK recipe (γ = 1.016 − 0.094·log₂k), and report that the dependence on k vanishes when the same activations live in fewer dimensions. Review point 6 (d_eff without dim 1170) is moot because the law fails.

## 2026-11-03 — Review check 3: measured L0 per layer (`results/review/l0_layers.json`, `l0_reinterp.json`)

All 135 Gemma Scope 16k residual SAEs (26 layers) measured on the same 32,736 tokens (32 Pile sequences, ctx 1024, BOS dropped). The layer-12 activations from one full forward pass equal the cached layer-12 activations exactly (relative error 0).

- **Measured/named L0 has its own mid-stack hump.** Median ratio by layer: 0.92–0.98 at layers 0–4, 1.00 at 6–7, rising to 1.08–1.11 at layers 10–15 (peak 1.11 at L13), back to 0.97–1.01 at layers 20–25. Within a layer the spread across L0 variants is ≤ ±0.1. Mid-stack SAEs fire ~10% more on our tokens than their names say.
- **Re-interpolating at measured L0 strengthens the mid-stack γ peak slightly; it does not explain it.** Since γ falls with L0, the named L0s understated mid-stack density, so at matched true L0 mid-stack γ is a little higher: peak-minus-edges in the primary window 0.189 → 0.208 (L0 = 40), 0.133 → 0.155 (L0 = 70), 0.083 → 0.097 (L0 = 140). Largest per-layer shift 0.023 in any window/L0; r(named, measured profile) ≥ 0.99 everywhere. Peak layers unchanged (14–15 in the top windows, 5–12 in the bulk), except ranks 4–50 at L0 = 140, where the profile is nearly flat and the argmax moves from layer 3 to 14.
- Claim update: "γ depends on L0 almost alone" stays qualified to "at layer 12, across widths". Across layers there is a real, L0-independent mid-stack γ excess of 0.10–0.21; an L0 mismatch is not the cause (review point 3 closed).

## 2026-11-03 — Phase 3 stage 2: sparse planted features (`results/phase3/saes_p3s2.json`, `stage2_alloc.json`)

Data (`experiments/phase3_stage2.py`): d = 256, M = 512 features, 4 active per token, p_i ∝ i^−1.5 (top capped at 0.5), isotropic random directions (β = 0), noise 0.05; Σ slope −1.18 to −1.30 in the three windows, set by the frequencies rather than by direction anisotropy. `s2-point`: point features; `s2-circle`: every feature a circle in span(u_i, v_i). TopK width 4,096, k = 1…32, 2 epochs over 5.5M tokens, seed 0, real vs Gaussian twin. With near-orthogonal directions, a cloud of n_i latents per feature has eigenvalues ∝ n_i, so γ should equal the allocation exponent c in n_i ∝ p_i^c. Prediction for circles at k = 1 (our derivation): tiling a circle with n directions leaves error ∝ n^−2, b = 2, c = 1/(1+b) = 1/3. For points: one latent per feature, c ≈ 0.

γ, ranks 5–50 (alive latents; matched = best feature score > 0.7):

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| circle, real | 0.32 | 0.21 | 0.19 | 0.12 | 0.08 | 0.08 |
| circle, Gaussian twin | 1.46* | 1.33 | 0.88 | 0.58 | 0.42 | 0.31 |
| circle, c (all covered features) | 0.33 | 0.29 | 0.26 | 0.21 | 0.28 | 0.35 |
| circle, c (top 50 features) | 0.30 | 0.32 | 0.23 | 0.11 | 0.07 | −0.01 |
| point, real | 0.30* | 0.20 | 0.16 | 0.06 | 0.04 | 0.06 |
| point, Gaussian twin | 1.25* | 1.20 | 0.88 | 0.64 | 0.45 | 0.33 |
| point, c (top 50 features) | 0.46* | 0.44 | 0.11 | −0.02 | −0.14 | −0.14 |
\* fewer than 1,000 alive latents (circle Gaussian k = 1: 399; point k = 1: 527 real, 339 Gaussian).

- **Real-vs-Gaussian contrast reproduced, and larger than in Gemma.** Planted sparse structure gives γ 0.04–0.32; the Gaussian twin with the same Σ gives 0.31–1.46. Fits over k with ≥ 1,000 alive: Gaussian γ = 1.35–1.45 − 0.22–0.25·log₂k (R² 0.93–0.97); real γ = 0.22–0.29 − 0.04–0.05·log₂k (R² 0.80–0.90). Gemma (TopK, L13: 1.016 − 0.094·log₂k) and stage 1 (1.30 − 0.184·log₂k) sit between the two. Real activations look intermediate between "Gaussian with Σ" and "sparse recoverable features".
- **The allocation law's prediction holds at k = 1 for circles, and γ equals c there.** c over all covered features is 0.334 (prediction 1/3); over the top 50 it is 0.30; γ is 0.35 / 0.32 / 0.32 in the three windows. This is the first measurement in the project where γ and the frequency-allocation exponent are the same number, and where that number was predicted beforehand.
- **As k grows, allocation to frequent features flattens, and γ falls with it.** Circles, top 50: c 0.30 → 0.32 → 0.23 → 0.11 → 0.07 → −0.01; γ 0.32 → 0.08. Top-5 circles get 11–25 latents at k = 1 and 6–9 at k ≥ 8. Candidate mechanism (interpretation): once several latents can be active per feature, two or three latents combine to reconstruct any point on a circle, so a circle no longer needs a tiling, b grows, c → 0. In this regime the γ(k) law comes out of allocation. Points: c ≈ 0 for k ≥ 8 as predicted; at k = 1–2 the top points get 4–7 latents (one latent must cover co-active pairs).
- **The Gaussian twin has a steep γ(k) law with no features at all**, so the Gemma k-law is not by itself evidence for allocation. In stage 2 the real k-slope (−0.04) is far shallower than the Gaussian one (−0.22).
- **Junk latents:** 330–2,550 alive latents score < 0.3 against every feature. Their median firing is ~20× below matched latents' and they hold little feature energy (pilot: median 7% in the best feature), so they are fitting the noise. They add a near-isotropic floor: γ over matched latents only is 0.12–0.24 for circles and 0.15–0.26 for points, and for points it stays at ~0.15 where c ≈ 0, so matched-only γ ≠ c for points at high k (unexplained; top points still have 1–2 latents, and with M/d = 2 the directions are not exactly orthogonal).
- Caveats: one seed; one width (8 latents per feature); circles occupy 1,024 dimensions' worth of planes in d = 256, so they overlap; matching threshold 0.7.

**Consequence for the frequency-allocation hypothesis:** no longer just "underdetermined". In a regime with recoverable features it produces γ, matches its zero-parameter prediction at k = 1, and gives a k-dependence through the flattening of allocation. But the same data shows that a Gaussian produces a larger γ and a steeper k-law without any features. Working hypothesis for Gemma: γ mixes a Gaussian-like contribution (high, steep in k) and a feature-allocation contribution (low, ≈ c); the relative weight would be set by how recoverable the features are. Testable next: interpolate stage 1 ↔ stage 2 (mean_active, M/d) and check whether γ and its k-slope move together between the two endpoints.

## Phase 3, interpolation: dense-to-sparse sweep (2026-11-04)

Script `experiments/phase3_interp.py`, summary `experiments/phase3_interp_summary.py` → `results/phase3/interp_summary.{json,png}`, per-run analysis `results/phase3/saes_p3i.json`. Point features, β = 0, d = 256, p_i ∝ i^-1.5, noise 0.05; (M, active) = (512, 4) [stage 2], (1024, 8), (2048, 16), (4096, 32). TopK width 4096, k ∈ {4, 8, 16, 32, 64}, real vs Gaussian twin, 2 epochs over 6M tokens, seed 0. γ in the primary window (ranks 6–50), fit on log2 k over runs with ≥ 1000 alive latents. (The Mac slept overnight on battery: data generation took 1.5–4 h instead of ~70 s; a `caffeinate -i -w` hold fixed it. Results are unaffected.)

| data (M, active) | real A | real per-doubling | R² | Gaussian A | Gaussian per-doubling | matched frac (k=8) |
|---|---|---|---|---|---|---|
| 512, 4 | 0.144 | −0.015 | 0.27 | 1.198 | −0.178 | 0.18 |
| 1024, 8 | 0.548 | −0.078 | 0.92 | 1.167 | −0.175 | 0.27 |
| 2048, 16 | 1.170 | −0.176 | 0.96 | 1.389 | −0.210 | 0.29 |
| 4096, 32 | 2.107 | −0.314 | 0.97 | 1.597 | −0.224 | 0.14 |
| stage 1 (4096, 32, β = 1) | 1.302 | −0.184 | 0.95 | 1.161 | −0.166 | 0.11 |
| Gemma L13 TopK | 1.016 | −0.094 | | | | |

- **Density moves real γ monotonically from the allocation regime to Gaussian-like and beyond.** Real intercept 0.14 → 0.55 → 1.17 → 2.11. Real reaches its twin near (2048, 16) and overshoots it at (4096, 32): 2.11 vs 1.60. With crowded directions at the same density (stage 1, β = 1), real sits back near its twin (1.30 vs 1.16), so direction crowding, not density, is what pins real to the Gaussian value.
- **Every planted fit lies on one ray: per-doubling ≈ −0.15 × intercept.** −s/A = 0.14–0.15 for all four real densities above the sparsest, all four twins, and both stage-1 conditions; the sparsest real fit (0.11) is noise-dominated (R² 0.27). Equivalently γ(k) ≈ A·(1 − log2 k / log2 k0) with k0 ≈ 2^6.7 ≈ 100 throughout: density, direction crowding and real-vs-Gaussian change the amplitude A only, never the k at which γ would reach 0.
- **The path does not pass near Gemma.** Gemma's −s/A = 0.093 (k0 ≈ 2^10.8 ≈ 1800) is off the planted ray. The planted run nearest in intercept, (2048, 16) at 1.17, falls about twice as fast with k (−0.176 vs −0.094). No density of this generator reproduces Gemma's (intercept, slope) pair.
- Blend hypothesis: half supported. The density-driven move from allocation (γ ≈ c) to Gaussian-like (γ ≈ 1+) is real and monotone. But γ's k-dependence is governed by a separate scale k0 that density does not touch; at d = 256, width 4096 it is ~100, in Gemma ~1800. Candidates for what sets k0: d, SAE width, or the training recipe. The PCA d-sweep argues against d alone, since Gemma truncated to d′ = 256 had an almost flat k-gap (0.014), not the planted one.
- Caveats: one seed, d = 256 only, one window. matched frac falls at (4096, 32) because features are no longer recoverable (M/d = 16).

## Phase 3, width sweep: does SAE width set k0? (2026-11-04)

Script `experiments/phase3_width.py` (runs in `cache/saes_p3w`, analysis `results/phase3/saes_p3w.json`), summary `experiments/phase3_width_summary.py` → `results/phase3/width_summary.{json,png}`. Data fixed at i-m2048-a16 (M = 2048, 16 active, β = 0, d = 256; the density whose intercept is nearest Gemma's). Widths 1024–16384 (4096 reused from saes_p3i), k ∈ {4, …, 64}, real vs Gaussian twin, seed 0. γ in ranks 6–50; fits use runs with ≥ 250 alive latents (all qualified). log2 k0 = A / −s.

| width | real A | real s | real log2 k0 | Gaussian A | Gaussian s | Gaussian log2 k0 |
|---|---|---|---|---|---|---|
| 1024 | 1.315 | −0.207 | 6.34 | 2.511 | −0.424 | 5.92 (R² 0.76) |
| 2048 | 1.324 | −0.208 | 6.37 | 1.736 | −0.275 | 6.31 |
| 4096 | 1.170 | −0.176 | 6.63 | 1.389 | −0.210 | 6.62 |
| 8192 | 1.018 | −0.148 | 6.87 | 1.212 | −0.176 | 6.89 |
| 16384 | 0.866 | −0.122 | 7.11 | 1.080 | −0.159 | 6.77 |
| Gemma L13 TopK, 16k | 1.016 | −0.094 | 10.8 | | | |

- **Width moves k0, but slowly.** Real log2 k0 rises ≈ 0.25 per doubling of width from 2048 to 16384 (6.37 → 7.11). At Gemma's width (16k) it is 7.1 against Gemma's 10.8; closing that gap at this rate would take ~15 more doublings. The Gaussian twin rises with the real data up to 8192, then levels off (6.77 at 16384).
- **Width tilts γ(k) instead of shifting it.** Wider SAEs lower γ at small k (real k = 4: 0.97 → 0.67) and barely change it at k ≥ 16 (0.35–0.45 at k = 16, 0.17–0.19 at k = 64). So intercept and slope both shrink: at 8192 the intercept matches Gemma's (1.02) but the slope is −0.148; at 16384 the slope is closer (−0.122) but the intercept has dropped to 0.87. No width reproduces Gemma's pair.
- Conclusion: width is a minor contributor to the k-scale. Most of the gap to Gemma has to come from d, the data, or the recipe. Next: planted d-sweep at fixed density and width.
- Caveats: one seed. Gemma's fit uses ranks 11–150 and k = 1–128 at d = 2304; the planted fits use ranks 6–50 and k = 4–64 at d = 256, so k0 values are not strictly comparable across the two.

## Phase 3, d-sweep: does d set k0? (2026-11-05)

Script `experiments/phase3_dsweep.py` (runs in `cache/saes_p3d`), summary `experiments/phase3_dsweep_summary.py` → `results/phase3/dsweep_summary.{json,png}`. Density fixed as ratios to d, matching i-m2048-a16: M = 8d point features, d/16 active, β = 0, p_i ∝ i^-1.5, noise 0.05 per coordinate. Two arms: width = 16d (d = 64…512) and width 4096 (d = 128…1024); d = 256 reused from saes_p3i. k ∈ {4, …, 64}, real vs Gaussian twin, seed 0; all runs ≥ 1000 alive. γ in a rank window scaled with d, ranks d/43 … d/5 (6–51 at d = 256). The fixed window ranks 6–50 is unusable away from d = 256: at d = 64 it spans nearly the whole spectrum, and at d = 1024 the real fit is flat at γ ≈ 2 (R² 0.77).

Real, scaled window (log2 k0 = A / −s):

| d | width | A | s | R² | log2 k0 | Gaussian log2 k0 |
|---|---|---|---|---|---|---|
| 64 | 1024 | 0.44 | −0.065 | 0.74 | 6.83 | 6.33 |
| 128 | 2048 | 0.74 | −0.114 | 0.89 | 6.47 | 6.45 |
| 128 | 4096 | 0.61 | −0.088 | 0.94 | 6.98 | 6.38 |
| 256 | 4096 | 1.12 | −0.168 | 0.96 | 6.65 | 6.57 |
| 512 | 4096 | 1.68 | −0.243 | 0.98 | 6.92 | 6.80 |
| 512 | 8192 | 1.50 | −0.215 | 0.96 | 6.98 | 7.67 |
| 1024 | 4096 | 2.08 | −0.258 | 0.99 | 8.05 | 7.37 |

- **At fixed density ratios, d sets the amplitude A, and k0 barely moves.** A rises steadily with d (0.44 → 2.08 from d = 64 to 1024). log2 k0 stays at 6.5–7.0 from d = 64 to 512 and reaches 8.05 only at d = 1024. Fit over all runs: +0.24 per doubling of d (fixed-width arm alone: +0.35), driven mostly by the d = 1024 point. So k0 is set in absolute k (≈ 100–250), not as a fraction of d.
- **Extrapolated to Gemma's d and width, the planted k-scale reaches log2 k0 ≈ 8.3–8.6.** That is 7.7–8.1 at d = 2304 from the d trend, plus ≈ 0.5 for width 4096 → 16k from the width sweep.
- **Gemma's own k0 depends on the window and the k range** (refit from `results/gpu/analysis.json` B, TopK L13 16k): ranks 4–50: 9.2 (k = 1–128) / 8.9 (k = 4–40); ranks 11–150: 10.8 / 10.9; ranks 31–450: 9.4 (R² 0.76) / 12.1. The window closest to the planted scaled one (ranks 54–461 at d = 2304) is ranks 31–450, where k0 is the least stable.
- **Synthesis.** Across all Phase 3 sweeps, γ(k) ≈ A · (1 − log2 k / log2 k0). Density and d set A. k0 is ≈ 2^6.6 at d = 256, width 4096 for every density, and rises slowly with width (+0.25 per doubling) and with d (+0.24–0.35 per doubling). Together, d and width carry the planted k0 to ≈ 8.5 at Gemma's settings, against Gemma's 9–12. They account for part of Gemma's slow decline with k, possibly most of it in the top window, but not robustly in the bulk window. Gemma's intercept 1.02 is below the planted A at d ≥ 512 (1.5–2.1), so in this family a Gemma-like intercept needs lower density than M = 8d, d/16 active. Earlier sweeps showed density moves A without moving k0.
- Caveats: one seed; one data family (point features, β = 0, α = 0.5); the d trend rests mostly on one point (d = 1024, width 4096 = 4d); Gaussian twins are noisier (log2 k0 6.3–7.7).

## Phase 3, Gemma-matched planted check on GPU (2026-11-05)

Scripts `experiments/phase3_gemma_match.py` (job), `experiments/phase3_match_analyze.py` (per-run γ and eigenvalues, run on the pod → `results/phase3/gemma_match.{json,log}`), `experiments/phase3_match_summary.py` → `results/phase3/match_summary.{json,png}`. Run on Runpod pod `ivtfrxem2uevro` (A100 SXM 80GB, community, $1.39/h, 09:09–11:12 UTC, ≈ 2 h, ≈ $2.80; pod terminated and confirmed gone). Per-run `train.json` and `fires.npy` copied to `cache/saes_p3g`; decoders stayed on the pod (the first 600 eigenvalues of each are in gemma_match.json).

Design: Gemma's d = 2304 and width 16384; point features, β = 0, α = 0.5, noise 0.05, 6M tokens, 2 epochs; three densities (M, active) = (2d, d/64), (4d, d/32), (8d, d/16); k ∈ {4, …, 128}; real vs Gaussian twin; seed 0, plus seeds 1–2 at (4d, d/32) (auto-picked as nearest Gemma's intercept in ranks 11–150; that fit had R² 0.05, so the pick was arbitrary). Code change: `synth.py` draws its randoms on the GPU under CUDA (CPU draws took ~25 min per dataset at this M); the MPS/CPU stream is byte-identical to before, but the GPU datasets are a different random stream from what a Mac would generate.

**The top rank windows are invalid for this generator at d = 2304.** `synth.py` caps feature probabilities at P_MAX = 0.5. Holding active/d fixed while d grows pushes more features into the cap: 11 at d = 256, 25 / 50 / 101 here. Capped features have equal variance, so Σ has a plateau over its top ~25–100 eigenvalues. Σ's slope in ranks 4–50 is −0.36 / −0.16 / −0.11, against Gemma's −0.98; in ranks 11–150 it is −1.06 / −0.64 / −0.31, against −0.95. γ = decoder slope / Σ slope is then a ratio over a near-zero denominator. That gives γ = 2.6–3.8 at k = 4, jumps between k values, and fits with R² down to 0.05 in those windows. It is a property of the data design, not of the SAEs. It also explains the d-sweep's failure of the fixed window at d = 1024 (Σ slope in ranks 6–50 there is −0.26). The d-sweep's scaled window kept Σ at −1.0 for every d, so those results stand.

**Clean window: ranks 55–461** (the d-sweep's d/43 … d/5). Σ slope there is −1.29 / −1.28 / −1.02. Gemma has no fit in exactly this window; its nearest is ranks 31–450 (Σ slope −0.75).

| real γ, ranks 55–461 | k = 4 | 8 | 16 | 32 | 64 | 128 | A | s | R² | log2 k0 |
|---|---|---|---|---|---|---|---|---|---|---|
| (2d, d/64) | 0.41 | 0.37 | 0.26 | 0.26 | 0.18 | 0.12 | 0.52 | −0.057 | 0.96 | 9.2 |
| (4d, d/32), 3 seeds | 0.45 | – | 0.47 | 0.52 | 0.47 | 0.32 | 0.53 | −0.018 | 0.2 | – |
| (8d, d/16) | 1.11 | 1.28 | 1.07 | 1.03 | 0.89 | 0.70 | 1.43 | −0.093 | 0.77 | 15.3 |
| Gemma L13 TopK, ranks 31–450 | 0.92 | 0.80 | 0.79 | 0.60 (k = 40) | | 0.51 | 1.08 | −0.083 | | 13.0 |

(Gemma fit over k = 4–128 for comparability. k = 8 at (4d, d/32) has 944 alive latents, below the 1,000 cutoff.)

- **At Gemma's d and width, the planted k-scale reaches Gemma's range.** In the clean window, log2 k0 = 9.2 at the sparsest density and 15.3 at the densest, bracketing Gemma's 13.0 (ranks 31–450, k = 4–128). The d = 256, width 4096 value was ≈ 6.6 and the d-sweep extrapolation ≈ 8.5. The densest planted curve runs parallel to Gemma's: slope −0.093 vs −0.083 per doubling, about +0.2–0.35 higher. So d and width together can produce Gemma's slow decline with k. Unlike at d ≤ 1024, k0 now depends on density.
- **But the planted γ(k) is not a clean log-linear law.** (4d, d/32) is flat at 0.45–0.52 up to k = 64 and then drops (R² 0.2). The densest curve rises from k = 4 to 8 before falling. Gemma's ranks 11–150 law has R² 0.994. Part of this is dead latents at small k: real runs at k ≤ 16 keep 950–9,000 of 16,384 latents alive, Gemma keeps 3,700–14,500. 6M tokens is thin for a 16k-wide SAE at d = 2304.
- **Seeds:** spread is ≤ 0.02 in almost every cell (≤ 0.22 at k = 4 in the invalid top window). The irregular shapes are systematic, not noise.
- **Real-vs-Gaussian:** twins over-concentrate in the bulk at every density (γ 0.8–3.1 vs real 0.12–1.28), as in Gemma's deep windows and every earlier planted run.
- **Verdict:** partial. In the one window where the comparison is valid, planted data at Gemma's d and width puts k0 in Gemma's range (9–15 vs 13), so the k-scale gap from the d = 256 sweeps is mostly a d-and-width effect. The comparison where Gemma's law is cleanest (ranks 11–150) cannot be made with this generator, because its Σ is not a power law there.
- **Next (if pursued):** remove the plateau so Σ is a power law from rank 1. That means p_1 < P_MAX: steeper α or lower density, or power-law magnitudes scale_i. Then rerun the three densities in ranks 11–150 with more tokens (e.g. 30M) so small-k SAEs are not mostly dead. Cost ≈ 3–5 GPU-hours on an A100.

## Phase 3, power-law-Σ planted check on GPU (2026-11-05)

Scripts `experiments/phase3_powerlaw.py` (job), `experiments/phase3_match_analyze.py --group saes_p3p --out powerlaw_match` (on the pod → `results/phase3/powerlaw_match.{json,log}`), `experiments/phase3_match_summary.py --inp powerlaw_match --out powerlaw_summary --prefix p2304` → `results/phase3/powerlaw_summary.{json,png}`. Run on Runpod pod `f4suh4m1j1o0jd` (A100 SXM 80GB, secure cloud US-MD-1, $1.79/h, 13:56–17:33 UTC, ≈ 3.6 h, ≈ $6.50; billing read $4.83 at termination and lags; pod terminated and confirmed gone). Per-run `train.json` and `fires.npy` copied to `cache/saes_p3p`; decoders stayed on the pod.

Design: as the matched check (d = 2304, width 16384, point features, β = 0, α = 0.5, noise 0.05; (M, active) = (2d, d/64), (4d, d/32), (8d, d/16); k ∈ {4, …, 128}), with two changes. (1) `synth.py --var_slope 1.0` sets per-feature magnitudes so feature variance ∝ i^-1, which removes the P_MAX plateau: Σ slopes are −0.95 / −0.90 / −0.82 / −0.80 in ranks 4–50 / 11–150 / 31–450 / 55–461 at every density (Gemma −0.98 / −0.95 / −0.75 / −0.71), so every window is valid. (2) 12M tokens × 3 epochs (vs 6M × 2). Seed 0 at all densities, Gaussian twin of (4d, d/32), seed 1 at (8d, d/16) (auto-picked: lowest RMS vs Gemma in ranks 11–150).

| real γ, ranks 11–150 | k = 4 | 8 | 16 | 32 | 64 | 128 | RMS vs Gemma |
|---|---|---|---|---|---|---|---|
| (2d, d/64) | 0.13 | 0.19 | 0.86 | 0.40 | 0.27 | 0.11 | 0.42 |
| (4d, d/32) | 0.17 | 0.08 | 0.15 | 0.26 | 0.23 | 0.09 | 0.50 |
| (8d, d/16), 2 seeds | 0.36 | 0.46 | 0.42 | 0.42 | 0.41 | 0.36 | 0.27 |
| Gaussian twin (4d, d/32) | 1.86 | 1.20 | 0.57 | 0.61 | 0.52 | 0.45 | 0.51 |
| Gemma L13 TopK | 0.82 | 0.73 | 0.67 | 0.51 (k = 40) | | 0.35 | |

| real γ, ranks 31–450 | k = 4 | 8 | 16 | 32 | 64 | 128 | RMS vs Gemma |
|---|---|---|---|---|---|---|---|
| (2d, d/64) | 0.25 | 0.25 | 0.33 | 0.28 | 0.24 | 0.21 | 0.49 |
| (4d, d/32) | 0.23 | 0.19 | 0.22 | 0.22 | 0.26 | 0.20 | 0.53 |
| (8d, d/16), 2 seeds | 0.44 | 0.88 ± 0.18 | 0.80 | 0.67 | 0.55 | 0.38 | 0.23 |
| Gaussian twin (4d, d/32) | 2.90 | 2.17 | 1.93 | 1.72 | 1.02 | 0.73 | 1.26 |
| Gemma L13 TopK | 0.92 | 0.80 | 0.79 | 0.60 (k = 40) | | 0.51 | |

Alive latents (of 16384), real, k = 4 / 8 / 16 / 32 / 64 / 128: (2d) 4729 / 4772 / 6000 / 11454 / 12674 / 14519; (4d) 9069 / 8422 / 7465 / 9853 / 12478 / 13216; (8d) 5070 / 4177 (seed 1: 1860) / 11580 / 11376 / 12098 / 13656. Gemma k = 4 / 8 / 16: 3732 / 6469 / 14472. FVU at k = 4: 0.53 / 0.60 / 0.69.

- **Gemma's cleanest law (ranks 11–150, γ = 1.016 − 0.094·log2 k, R² 0.994) is not reproduced.** With Σ matched to Gemma's, real planted γ in ranks 11–150 is flat in k at every density: 0.1–0.3 with no trend at (2d) and (4d) (one outlier at (2d), k = 16), and 0.36–0.46 at (8d) (2 seeds, sd ≤ 0.02). Best RMS against Gemma is 0.27.
- **Planted decoders are much flatter than Σ.** γ ≈ 0.1–0.3 at the two sparser densities in every window means the decoder spectrum carries only a fraction of Σ's anisotropy. Planted features are random directions, so a dictionary that recovers them is close to isotropic whatever their variances. Gemma's γ of 0.35–1.0 needs decoders that inherit much more of Σ.
- **Only the densest setting, in the wide windows, looks like Gemma.** (8d, d/16), ranks 31–450, falls from 0.88 at k = 8 to 0.38 at k = 128: fit over k ≥ 8 A = 1.28, s = −0.124 per doubling, R² 0.98, log2 k0 = 10.3, against Gemma's A = 1.08, s = −0.083, log2 k0 = 13.0 (k = 4–128). Its k = 4 point (0.44) breaks the trend, and k = 8 varies across seeds (alive latents 4177 vs 1860).
- **The Gaussian twin now matches Gemma's Gaussian twin in shape.** In ranks 31–450 it falls 2.90 → 0.73 (Gemma's Gaussian: 2.20 → 0.70), and in ranks 11–150 1.86 → 0.45 (Gemma: 1.23 → 0.38). With Σ a power law, the Gaussian null behaves like Gemma's null; what does not carry over is the real-data curve.
- **Dead latents remain a confound at small k.** Doubling tokens and adding an epoch helped FVU (k = 4 at (2d): 0.53 vs 0.64), but real runs at k ≤ 8 still keep only 1,900–9,100 latents alive.
- **Verdict: negative for this generator family.** At Gemma's d, width and Σ, planted point features with frequency-driven allocation do not produce Gemma's γ(k) in ranks 11–150, and produce it only partly (densest setting, wide windows, k ≥ 8) elsewhere. The k-scale result of the matched check (k0 in Gemma's range at Gemma's d and width) survives only in the wide windows. Gemma's higher γ in the top window points to structure the planted family lacks: feature directions that are not isotropic random (e.g. correlated or clustered), which is what Phase 4 tests.
- Caveats: one data family (point features, β = 0, α = 0.5, var ∝ i^-1); seed 1 only at (8d); one Gaussian twin; alive fractions at small k are low.

## Phase 4: cluster structure of Gemma Scope decoder clouds (2026-11-05)

Scripts `experiments/phase4_clusters.py` (stages `l12`, `layers`, `layers_null`) → `results/phase4/{l12,layers,layers_null}.{json,log}`; `experiments/phase4_summary.py` → `results/phase4/summary.{json,png}`. All local (M2 Pro); no GPU.

Gate: the brief makes Phase 4 conditional on Phases 2–3 supporting the theory. They did not in its simple form (Phase 2: the Gaussian null carries much of the slope; Phase 3: planted random directions do not give Gemma's γ(k)). Run on the user's go as a direct measurement of the structure Phase 3 pointed to, not as a confirmation step.

Method: unit decoder rows; spherical k-means (k-means++ init, cosine) unless stated. The centred covariance splits exactly into between-cluster B = Σ n_c (m_c − μ)(m_c − μ)ᵀ / (N − 1) and within-cluster W = C − B. Per window, "explained" = mean of v_jᵀ B v_j / λ_j over C's eigenvectors in the window. Intrinsic dimension = participation ratio of clusters subsampled to 16 latents (PR16, max 15). Covariance-matched null per layer: N(μ, C) of that layer's cloud, rows unit-normalised, same N (its slopes come within 0.02–0.05 of the real ones; row normalisation changes the spectrum slightly).

**A. Clusterings are not stable; the cloud is a continuum at these scales** (layer 12, 16k, L0 = 82). k-means against itself across seeds: ARI 0.08–0.17, NMI 0.40–0.65. Between methods: ARI 0.02–0.15, NMI 0.29–0.76. Ward and average linkage produce giant clusters (23% / 17% of latents in one cluster at K = 64); k-means and spectral give balanced ones. The quantitative statistics below are robust anyway: at K = 256 the fraction of top eigenvalues explained by between-cluster structure is 0.20 / 0.12 / 0.07 / 0.04 (ranks 4–50 / 11–150 / 31–450 / 101–1000) for Ward and spectral against 0.21 / 0.11 / 0.06 / 0.04 for k-means; average linkage gives less (0.12 / 0.07) because of its giant cluster.

**B. Between-cluster structure carries a minority of the top eigenvalues** (layer 12, K = 1024, 16 latents per cluster):

| explained by between-cluster | 4–50 | 11–150 | 31–450 | 101–1000 | trace share |
|---|---|---|---|---|---|
| Gemma 16k L0 82 | 0.27 | 0.19 | 0.14 | 0.12 | 0.138 |
| covariance-matched Gaussian | 0.14 | 0.10 | 0.08 | 0.07 | 0.087 |
| isotropic unit vectors | 0.09 | 0.08 | 0.08 | 0.08 | 0.074 |
| our TopK w4096 k40, real (K = 256) | 0.17 | 0.12 | 0.09 | 0.07 | 0.099 |
| our TopK w4096 k40, Gaussian twin (K = 256) | 0.15 | 0.10 | 0.07 | 0.07 | 0.097 |

- Gemma has about twice the matched cloud's between-cluster share in the top windows, but within-cluster spread still carries most of the slope: in ranks 11–150 C −0.42, W −0.33, B −0.64. The first-order picture (eigenvalues = cluster spikes ∝ n_c) accounts for ≤ 30% of any window; consistent with Test 2 (anisotropy is cloud-wide).
- Our own TopK SAEs show almost no difference between real and Gaussian-twin training data (0.12 vs 0.10 in ranks 11–150). Gemma Scope (JumpReLU, 16k) has more structure than either; architecture and width differ, so this is not a clean comparison.

**C. The allocation law does not hold per cluster.** Slope of log mean latent frequency on log cluster size (theory: n_c ∝ P_c^(1/(1+b)) ⇒ slope = b > 0), 1M-token frequencies, layer 12 at L0 ≈ 70, against a frequency-permutation null:

| width | 256 per cluster | 64 per cluster | 16 per cluster |
|---|---|---|---|
| 16k | −0.95 (z −14) | −0.14 (z −4) | +0.04 (z −2) |
| 32k | −0.57 (z −9) | −0.61 (z −19) | −0.01 (z −5) |
| 65k | −1.52 (z −27) | −0.18 (z −8) | +0.21 (z +3) |
| 131k | −0.83 (z −21) | +0.29 (z +7) | +0.28 (z +5) |

The sign flips with cluster scale. At coarse scales large clusters are made of rarer latents (the opposite of the theory); only at the finest scale and largest widths is the slope positive. Correlations are weak (|r| ≤ 0.5). No single b describes it.

**D/E. Layer profile: cluster structure beyond anisotropy tracks the top-window slope, not the bulk** (16k SAE closest to L0 = 70 per layer; real minus covariance-matched):

| layer | excess between-share (K = 1024) | excess cos to centroid | PR16 real / matched | slope 11–150 | slope 31–450 |
|---|---|---|---|---|---|
| 0 | +0.006 | −0.007 | 14.86 / 14.86 | −0.28 | −0.32 |
| 4 | +0.046 | +0.064 | 14.15 / 14.82 | −0.35 | −0.54 |
| 13 | +0.054 | +0.079 | 14.10 / 14.84 | −0.43 | −0.42 |
| 15 | +0.054 | +0.078 | 14.10 / 14.84 | −0.44 | −0.38 |
| 23 | +0.036 | +0.049 | 14.33 / 14.87 | −0.27 | −0.26 |

Spearman over 26 layers (K = 1024; K = 256 in brackets), slope vs:

| | 4–50 | 11–150 | 31–450 | 101–1000 |
|---|---|---|---|---|
| excess between-share | −0.87 (−0.78) | −0.89 (−0.90) | −0.14 (−0.42) | −0.14 (−0.45) |
| excess cos to centroid | −0.91 (−0.87) | −0.91 (−0.83) | −0.13 (+0.02) | −0.13 (+0.02) |
| excess PR16 | +0.81 (+0.89) | +0.71 (+0.72) | −0.09 (−0.15) | −0.07 (−0.16) |
| matched-cloud share | −0.23 (−0.33) | −0.52 (−0.59) | −0.94 (−0.88) | −0.96 (−0.92) |

- **Layer 0 has no cluster structure beyond its covariance.** The excess rises over layers 1–4, plateaus through 4–15, peaks at 13–15, falls to layer 23–24 and jumps again at layer 25, the same shape as the top-window slope.
- **Mid-stack clusters are lower-dimensional than anisotropy implies** (PR16 up to 0.75 below the matched cloud), the weak form of the brief's "more manifold-like clusters" prediction. At 16 latents they are still close to full-dimensional (14.1 of 15).
- **The deep-window layer profile is anisotropy alone:** the matched cloud's own share tracks it (ρ −0.88 to −0.96) and the excess does not.
- Together with Phase 1b's split (top windows peak mid-stack; the bulk is steepest early), this resolves the two layer profiles: the bulk follows Σ-like anisotropy, and the top-window mid-stack peak goes with genuine cluster structure.

**Verdict.**
- Allocation law per cluster: not supported (sign depends on scale; coarse clusters contradict it).
- Cluster → eigenvalue map: holds only weakly. Between-cluster structure explains ≤ 27% of the top eigenvalues, roughly twice a covariance-matched cloud's.
- Middle-layer steepening and clusters: supported for the top windows. Cluster structure beyond anisotropy tracks the top-window slope across layers (ρ ≈ −0.9), with lower intrinsic dimension mid-stack; the bulk windows follow anisotropy alone.
- This is the structure Phase 3's planted random directions lacked.

Caveats: correlational across layers (26 points along one smooth profile, so effective n is small); one SAE per layer at L0 ≈ 70 (layer 2 at L0 = 53); clusterings themselves are unstable, only their summary statistics are robust; the matched null is approximate (row normalisation); frequencies only at layer 12; our TopK and Gemma Scope differ in architecture and width.

Next (if pursued): plant clustered directions in `synth.py` (features drawn around shared centres) and check whether that restores Gemma's top-window γ(k) in the Phase 3 setup; repeat D/E at other L0s to see whether the excess shrinks with L0 the way the layer profile does (Phase 1b).

## Phase 3, clustered planted directions on GPU (2026-11-06)

Question: Phase 4 found Gemma's decoder clouds carry cluster structure beyond their covariance, tracking the top-window layer profile. Does planting that structure in the power-law-Σ generator restore Gemma's γ(k) = 1.02 − 0.095·log₂k in ranks 11–150?

Setup (`experiments/phase3_clustered.py`, commit 896435f): same generator as the power-law check (α 0.5, β 0, noise 0.05, `--var_slope 1.0`, 12M tokens × 3 epochs, TopK width 16384, k ∈ {4…128}). Added `--n_clusters M/16 --cluster_cos ρ`: features in clusters of 16, each at exact cosine ρ to a random centre. ρ ∈ {0.4, 0.6} × densities (4d features, d/32 active) and (8d, d/16). 24 SAEs at seed 0, plus seed 1 on the dataset nearest Gemma (8d, ρ 0.6). The ρ = 0 control is the power-law run. Runpod A100 SXM 80GB community pod `z9i10qsguj7upk`, $1.39/h, 10:02–14:06 UTC. Terminated (get-pod 404, list-pods empty). Billed so far $4.43; billing lags, and the runtime implies about $5.70.

Pre-registered pass: ranks 11–150 RMS vs Gemma < 0.15 and per-doubling slope within 0.04 of −0.094.

Real γ(k), ranks 11–150 (`results/phase3/clustered_summary.{json,png}`; Gemma 0.824 / 0.732 / 0.665 / 0.509@k40 / 0.354):

| data | ρ | k=4 | 8 | 16 | 32 | 64 | 128 | mean | RMS vs Gemma | per doubling |
|---|---|---|---|---|---|---|---|---|---|---|
| 4d, d/32 | 0 | 0.17 | 0.08 | 0.15 | 0.26 | 0.23 | 0.09 | 0.16 | 0.50 | +0.004 |
| 4d, d/32 | 0.4 | 0.16 | 0.12 | 0.15 | 0.30 | 0.29 | 0.13 | 0.19 | 0.49 | +0.014 |
| 4d, d/32 | 0.6 | 0.22 | 0.17 | 0.20 | 0.45 | 0.35 | 0.19 | 0.26 | 0.43 | +0.018 |
| 8d, d/16 (2 seeds) | 0 | 0.36 | 0.46 | 0.42 | 0.42 | 0.41 | 0.36 | 0.41 | 0.27 | −0.005 |
| 8d, d/16 | 0.4 | 0.37 | 0.51 | 0.42 | 0.43 | 0.42 | 0.41 | 0.43 | 0.26 | −0.001 |
| 8d, d/16 (2 seeds) | 0.6 | 0.37 | 0.50 | 0.51 | 0.54 | 0.47 | 0.48 | 0.48 | 0.24 | +0.013 |

- **Fails the pre-registered test on every dataset.** Best RMS is 0.24, against a threshold of 0.15. Per-doubling slopes run −0.016 to +0.026 in every window, against Gemma's −0.095 / −0.103 / −0.083.
- **Clusters raise γ's level but leave it flat in k**, the outcome named before the run. In ranks 11–150, ρ 0.6 adds +0.07 to +0.10 to the mean γ; seed sd is ≤ 0.035. In ranks 31–450 at 4d the rise is larger: mean 0.22 → 0.30 → 0.44 for ρ 0 / 0.4 / 0.6.
- Σ's slopes barely move with clustering, by ≤ 0.03 (ranks 11–150: −0.90 → −0.87), as the calibration predicted.
- **The decoders do carry the cluster structure at 4d, and γ(k) still stays flat.** At 4d, ρ 0.4, the excess between-cluster share over a matched cloud is +0.04 to +0.08 across k, against Gemma's +0.054. Excess cos runs +0.05 to +0.16, against Gemma's +0.079. Yet γ in ranks 11–150 is about 0.19 and flat. Matching Gemma's cluster excess is therefore not sufficient for Gemma's γ(k).
- At 8d, d/16 (more features than latents), the decoders lose the planted clusters. The excess is about 0 for k ≥ 16, and only k = 4 at ρ 0.6 keeps +0.06. This density was nevertheless closest to Gemma, as in the power-law run.

Verdict: cluster structure in feature directions (balanced clusters of 16, one tightness per dataset) is not the missing ingredient for Gemma's k-dependence. It shifts γ up modestly but does not produce the decline with k. Every planted variant so far (random, Gemma-matched Σ, power-law Σ, clustered) gives a flat γ(k). The k-law seems to need something none of them have: frequency–direction coupling, hierarchical or unbalanced clusters, or structure in how features co-occur.

Caveats:
- Clusters are balanced, same-tightness and independent of frequency, whereas real clusters vary in size and tightness.
- Only seed 0 except one dataset; the per-k seed sd reaches 0.08 in ranks 4–50.
- Ranks 4–50 sit on the P_MAX-saturated top of Σ.

## Phase 3, resynthesis dissection on GPU (2026-11-08)

Question: every planted generator so far gives a flat γ(k). Which property of real activations produces Gemma's γ(k) law: co-occurrence among features, the link between a feature's statistics and its direction, or neither?

Setup (`resynth.py`, `experiments/resynth_job.py`, commit b189bfe):
- **Source:** Gemma-2-2B layer 12, 12M Pile tokens (last 500k held out). Codes come from the Gemma Scope 65k L0-72 JumpReLU SAE (measured L0 76.7, 63,650 alive).
- **Variants:**
  - **R:** real activations.
  - **A:** resynthesis z·W_dec + b_dec (FVU vs real 0.215).
  - **C:** real codes with W_dec rows permuted across latents.
  - **B:** independent per-latent firing, each latent with its own frequency and its own 64-quantile value distribution (statistics from 4M tokens).
  - **D:** B and C together.
- **Training:** TopK width 16384, k ∈ {4…128}, 3 epochs. Seeds 0 and 1 for R and A; seed 0 for B, C and D. 42 SAEs.
- **Pod:** Runpod A100 SXM 80GB secure pod `idb0vuz4fi99nq`, US-MD-1, $1.79/h. The HF token came from a Runpod secret.
  - Runpod auto-stopped the pod at 18:24 on 11-07, out of balance. The first pod `pl4s7hkp3qtsc8` (6 Nov, idle, waiting on the HF login) was stopped the same way.
  - The run was slow because of CPU oversubscription: the pod shows 128 cores under a ~13.6-CPU cgroup quota. R's `--clusters` analysis ran >3.5 h; with threads capped in `gpu/env.sh` it took 2 min.
  - Restarted 11-08 11:12 and finished about 15:00. Terminated (get-pod 404, list-pods empty).
  - Billed so far: $15.39 for this pod (the 11-08 bucket lags; the runtime implies about $17) plus $4.79 for the first pod.
- **Pre-flight check:** the pod's first 200k tokens gave Σ slopes identical to the local L12 cache's first 200k (−0.917 / −0.872 / −0.814 / −0.837).

Pre-registered reading, ranks 11–150, per-doubling slope s from 6 k:
- Gate: s_R ≤ −0.05, RMS(A−R) ≤ 0.15 and |s_A − s_R| ≤ 0.04.
- Then retained = s_X/s_A: < 0.5 removes the law, > 0.75 keeps it, partial between.

γ(k), ranks 11–150 (`results/phase3/resynth_summary.{json,png}`):

| variant | k=4 | 8 | 16 | 32 | 64 | 128 | s | retained | s, k ≥ 8 (post hoc) |
|---|---|---|---|---|---|---|---|---|---|
| R real L12 (2 seeds) | 0.81 | 0.70 | 0.67 | 0.57 | 0.46 | 0.36 | −0.089 (seeds −0.089, −0.089) | | −0.090 |
| A resynthesis (2 seeds) | 0.78 | 0.70 | 0.67 | 0.54 | 0.44 | 0.34 | −0.089 (−0.088, −0.090) | 1.00 | −0.095 |
| C permuted directions | 0.67 | 0.87 | 0.59 | 0.50 | 0.42 | 0.36 | −0.085 | 0.95 keeps | −0.118 |
| B independent firing | 0.53 | 0.80 | 0.67 | 0.48 | 0.43 | 0.36 | −0.060 | 0.68 partial | −0.111 |
| D both | 0.55 | 0.85 | 0.66 | 0.52 | 0.45 | 0.35 | −0.067 | 0.76 keeps | −0.122 |

- **The gate passes.** Real L12 shows the law at −0.089 per doubling, close to our L13 result of −0.095. A reproduces it: RMS 0.022 and the same slope.
- **Pre-registered verdict: C keeps, D keeps (0.76, at the edge), B partial (0.68).**
- **B's "partial" rests on the k = 4 point.** For k ≥ 16, all five variants lie within about 0.03 of R at every k. The departures are at k = 4 (B, C and D low) and k = 8 (high), each on a single seed.
  - Leave-one-k-out ranges of s: B [−0.111, −0.041], C [−0.118, −0.063], D [−0.122, −0.044]; R [−0.092, −0.084].
  - Without k = 4, every variant keeps the law, and B, C and D are steeper than A.
  - So the data do not separate "partial" from "keeps" for B.
  - Low k = 4 alive counts are not special to B: R has 4,786 alive and B 4,753.
- **Read:** the k-law does not need co-occurrence between features, and it does not need the pairing between a feature's frequency or values and its direction. Independent firing with Gemma Scope's per-latent marginals, on Gemma Scope's set of dictionary directions, reproduces it from k = 16 up.
- Every planted generator lacked two things these variants have: (i) a real dictionary geometry (65k learned directions in 2304 d), and (ii) real heavy-tailed per-latent frequency and value distributions. The law lives in one or both. This run cannot split them. That would take Gemma marginals on random directions, and Gemma directions with uniform marginals.
- **Cluster excess:** A matches R (+0.03 to +0.07). C and D lose most of it (+0.01 to +0.03) yet keep γ(k), which again says clusters are not what drives the k-law. B sits between.
- **Ranks 31–450:** A reproduces R less well here (s −0.058 vs −0.082, RMS 0.061). B retains 0.23, C 0.93, D 0.84. B's curve is not monotone (0.59 / 0.92 / 0.69 / 0.64 / 0.71 / 0.63).
- **Σ slopes, ranks 11–150:** R −0.89, A −0.94, B −0.79, C −0.75, D −0.78. γ normalises by each dataset's own Σ.

Caveats:
- One seed for B, C and D. The k = 4 and k = 8 points look like optimisation noise but are untested.
- The Gemma Scope codes are themselves an SAE's view of the data (A's FVU is 0.215).
- Only layer 12, only width 16384.

## Phase 3, split test on GPU (2026-11-09)

Question: the resynthesis showed Gemma Scope's per-latent marginals on Gemma Scope's directions (B) keep the γ(k) law. Which half carries it?
- E: Gemma Scope marginals (independent firing) on random unit directions.
- F: Gemma Scope directions, every live latent given the mean frequency and pooled values.
- B with a second SAE seed.
Same pipeline as the resynthesis: 12M tokens, TopK w16384, k ∈ {4..128}, 3 epochs, SAE seeds 0, 1 (one data draw per variant). Pre-registered in experiments/resynth_split_job.py: primary fit k 8–128, retained vs A (s_A −0.095), > 0.75 keeps, < 0.5 removes.

Pod r15p06eyda02hl (secure A100 SXM 80GB, US-MD-1, $1.79/h), 30 SAEs in ~3.3 h, cost $4.62. Deleted (get-pod 404, list-pods empty). The aborted community pod dc69s4wix76ubu (no public SSH port) cost $0.32.

Ranks 11–150 (γ, mean of seeds 0, 1):

| variant | Σ slope | k=4 | 8 | 16 | 32 | 64 | 128 | s k8–128 | retained | s all-k | retained |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R real | −0.89 | 0.81 | 0.70 | 0.67 | 0.57 | 0.46 | 0.36 | −0.090 | | −0.089 | |
| A resynthesis | −0.94 | 0.78 | 0.70 | 0.67 | 0.54 | 0.44 | 0.34 | −0.095 | 1.00 | −0.089 | 1.00 |
| B marginals + dirs | −0.79 | 0.53 | 0.79 | 0.68 | 0.48 | 0.43 | 0.36 | −0.111 | 1.17 | −0.060 | 0.68 |
| E marginals + random dirs | −0.84 | 0.13 | 0.23 | 0.08 | 0.07 | 0.07 | 0.06 | −0.036 | 0.38 | −0.025 | 0.28 |
| F uniform marginals + dirs | −0.47 | 1.02 | 0.92 | 0.82 | 0.75 | 0.68 | 0.59 | −0.080 | 0.84 | −0.083 | 0.94 |

Verdict (pre-registered): E removes, F keeps → **the dictionary geometry carries the law**. Σ is well conditioned for all three (F's −0.47 is past the −0.3 flag).

Seeds agree closely: per-seed k8 slopes B −0.111/−0.111, E −0.037/−0.034, F −0.080/−0.079. B seed 1 reproduces seed 0 at k=4 (0.524 vs 0.527), so the k=4 dip under independent firing is a property of the data, not SAE optimisation noise (one data draw, so data-draw noise is still untested).

Other windows (not pre-registered as primary):
- Ranks 4–50: E retained 0.62 and F 0.81 at k8 (E partial).
- Ranks 31–450: E 0.67 and F 0.87 (E partial).
So E's removal is cleanest in the main window; in the others it is partial while F keeps everywhere.

What E and F look like:
- E: γ is near the floor at every k (0.06–0.23). With random directions, the SAE's decoder spectrum is almost flat against a steep Σ, so there is little γ to fall. The law goes with the level.
- F: every latent is alive at every k (16383–16384) and γ starts at ~1.0 at k=4. Uniform frequencies don't stop γ falling at roughly A's rate.
- Cluster excess share: E ≈ 0, F 0.036 → 0.001 (A 0.072 → 0.026).

Interpretation: what makes the decoder spectrum follow Σ, and then flatten with k at the observed rate, is the geometry of Gemma Scope's directions. Heterogeneous firing frequencies in the generating codes aren't needed: with them made uniform, the law stays. (The trained SAE's own latent frequencies can still differ; this tests the data's frequency structure, not the SAE's allocation.) This argues against data-frequency structure as the driver, at least at L12 w16384.

Caveats:
- One data draw per variant; two SAE seeds.
- E's low γ means "removes" partly reflects a floor effect, not only a lost slope.
- F changes Σ (−0.47 vs −0.94), so F's γ is measured against a flatter target.
- Layer 12 and width 16384 only; the Gemma Scope codes are themselves an SAE's view (A's FVU 0.215).

## Phase 3, matched-covariance control on GPU (2026-11-10)

Why: an external review of the write-up raised three problems with the split test.
- In raw decoder-slope units, F retains 0.42 (E 0.34), so "directions, not frequencies" flips.
- F is near-circular: with uniform firing, F's Σ is the dictionary's own covariance (F's Σ slope −0.469 against the Gemma Scope L12 decoder's −0.4684).
- Measured γ is biased by the live-latent count. Simulated: a true γ = 0.5 cloud measures 0.32 at n = 672, 0.41 at 4,786 and 0.47 at 16,384 (reproduced here).

Control (pre-registered in experiments/control_job.py, commit 11ae9f8):
- G: Gemma Scope marginals on unit directions drawn from a Gaussian with W_dec rows' mean and covariance (compare D).
- H: uniform marginals on the same Gaussian-matched directions (compare F).
- D and F were retrained as references.
- Seed 0, k 8–128, TopK w16384, 12M tokens, 3 epochs.
- Primary scoring: 8,192 random live latents per SAE (mean of 4 draws), in γ and raw units. A variant keeps the law if retained ≥ 0.75 in both units.
- Local Σ check (results/resynth/control_check.json), ranks 11–150: G −0.769 (D −0.780 on the pod), H −0.453 (F −0.464).

Pod wul64qzwixgr2g: secure A100 SXM 80GB, US-MD-1, $1.79/h, 10:46–13:25 UTC (about $4.70; billing showed $2.96 while still catching up). Deleted: get-pod 404, list-pods empty.

Ranks 11–150, fixed-n γ:

| variant | Σ slope | k=8 | 16 | 32 | 64 | 128 | s γ | s raw | FVU k8 → k128 |
|---|---|---|---|---|---|---|---|---|---|
| D Gemma marginals, Gemma rows permuted | −0.780 | 0.838 | 0.637 | 0.487 | 0.425 | 0.327 | −0.123 | +0.096 | 0.42 → 0.18 |
| G Gemma marginals, Gaussian-matched dirs | −0.770 | 0.814 | 0.545 | 0.436 | 0.453 | 0.392 | −0.094 | +0.072 | 0.40 → 0.19 |
| F uniform marginals, Gemma rows | −0.469 | 0.849 | 0.756 | 0.683 | 0.614 | 0.531 | −0.078 | +0.037 | 0.50 → 0.31 |
| H uniform marginals, Gaussian-matched dirs | −0.460 | 1.226 | 1.202 | 1.069 | 0.911 | 0.781 | −0.118 | +0.054 | 0.67 → 0.41 |

Retained (fixed-n; full-live in brackets):
- G/D: 0.76 γ, 0.75 raw (0.78, 0.77).
- H/F: 1.52 γ, 1.49 raw (1.49, 1.46).

Pre-registered verdict: both keep. Reading: the dictionary's covariance suffices; Gemma Scope's specific directions are not needed.

Checks:
- D and F reproduce their earlier full-live γ exactly (max diff 0.000).
- Live-count bias on the slopes is negligible here: fixed-n minus full-live s γ is −0.001 (D) and +0.002 (F). The bias lowers levels (by 0.02–0.07 in γ) but barely moves slopes, since every SAE here has ≥ 13,400 live latents at k ≥ 8. The real-data runs (R: 9,121 live at k = 8, 4,786 at k = 4) are more exposed and weren't rescored.

What this means:
- G only just keeps (raw retained 0.750, right at the cut), and its γ is not monotone (0.436 at k = 32, 0.453 at k = 64). A Gaussian dictionary with Gemma Scope's covariance gives most, not all, of D's decline.
- H is the telling case. Its γ sits above 1 at low k (1.23 at k = 8) and falls, the same shape as the Gaussian N(μ, Σ) null on real layer-13 activations (1.25 at small k, falling to 0.38). With 65k Gaussian-drawn directions and uniform firing, the data is close to Gaussian, and the SAE recovers little of the dictionary (FVU 0.67 at k = 8). So H "keeps the law" the way a Gaussian null does.
- Together: a decline of γ with k at roughly the observed rate is produced by data that only matches Σ, including near-Gaussian data. The γ(k) decline is not, by itself, evidence of structure in Gemma Scope's directions. The split test's "directions, not frequencies" is withdrawn. What is left: covariance, not specific directions, sets the decline in these resynthesised data, and its rate does not separate structured from near-Gaussian data.
- What still separates real from null data is the level (γ near or above 1 for nulls at small k) and the bulk windows, as before, not the k-slope in ranks 11–150.

Caveats: one seed, one layer, one width; G is at the threshold; H's SAEs fit poorly, so its γ reflects a Gaussian-like regime rather than dictionary recovery.

## Write-up revised after external review (2026-11-10)

The doc (claude.ai artifact 1sjdrStUPDhczw8sH2R7Xc, now rev 45) and paper/writeup.tex were revised to match the review and the matched-covariance control.
- Withdrawn: "directions, not frequencies". Narrowed: "not an artifact" (now scoped to bulk windows, k ≤ 16 and layers 3 and 23), "width-stable" (Pythia only), and "confirmed on planted data" (points give the same γ as circles; added the point-c and all-covered-c rows).
- Added: limits of the matrix test (diagonal share 0.60–0.76, no finite-sample benchmark, basis γ 0.08–0.12 below rank γ), the γ(k) fit's R² in all four windows (0.95, 0.99, 0.76, 0.36), live-count bias, the 200-document Σ caveat, a related-work section, and raw-unit retention for the resynthesis.
- Moved the planted sweeps to an appendix. Replaced the internal Phase labels, made "the paper" read "Li et al.", and added the arXiv:2505.00190 authors (Peter and Søgaard, 2025).
- doc2tex.py: merges split bold runs, keeps trailing full stops out of exponents, turns "Appendix:" headings into \appendix, and uses a smaller font for very wide tables.
