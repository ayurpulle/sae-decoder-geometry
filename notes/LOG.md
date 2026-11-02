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
