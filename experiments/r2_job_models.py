"""Round-2 GPU job 3: a second and a third model. M2 (Gemma-2-9b, needs HF_TOKEN) runs first
because it is cheap, then M1 (GPT-2 small, not gated). Unattended, restartable, same conventions as r2_job_real.py
(rows in results/r2/runs_models.jsonl, per-run decoder covariance and live mask in results/r2/dec/, per-dataset Sigma
in results/r2/sigma/). Launch with gpu/run_job.sh; it can follow r2_job_real.py on pod 1 or run on its own pod.

PRE-REGISTRATION (fixed before any run; experiments/r2_summary.py implements it unchanged)
==========================================================================================
Rank windows scale with d. Gemma-2-2b (d = 2304) windows are 0-based slices (3,50), (10,150), (30,450), (100,1000).
Each bound is multiplied by d / 2304 and rounded to the nearest integer:
  GPT-2 small, d = 768 (x 1/3):     (1,17), (3,50), (10,150), (33,333)    = ranks 2-17, 4-50, 11-150, 34-333.
  Gemma-2-9b, d = 3584 (x 1.5556):  (5,78), (16,233), (47,700), (156,1556) = ranks 6-78, 17-233, 48-700, 157-1556.
Primary window: the image of ranks 11-150, i.e. (3,50) for GPT-2 and (16,233) for Gemma-2-9b. Top windows are the first
two, bulk windows the last two. Window rule as before: a reading counts if it holds in the primary window and at least
two of the other three (statements that are about the top or the bulk windows only say so below).

M1  GPT-2 small (HF `gpt2`), resid_post (output of block L, before ln_f at L = 11), all 12 layers.
  Data: 10M Pile tokens per layer (monology/pile-uncopyrighted, streaming shuffle seed 0, ctx 1024, BOS = <|endoftext|>
  prepended and dropped, as collect_acts.py), fp32 forward, fp16 cache, last 500k held out; one layer at a time,
  cache deleted after its SAEs.
  SAEs: TopK width 16,384, k = 32, conds real / gaussian / sparse, SAE seeds 0, 1, 3 epochs, at all 12 layers; plus at
  layer 6 a k sweep k in {8, 16, 32, 64, 128} for the three conds, seeds 0, 1 (k = 32 shared; 96 SAEs).
  Sparse null (train_sae.py --cond sparse, as job 1's Q5): M = 65,536 sources and L = 48 expected active per token,
  data seed 0. Choice: the same M and SAE width as for Gemma (4 sources per latent), and L/k = 1.5, close to Gemma's
  64/40 = 1.6. Values N(0,1)/sqrt(p), p = L/M; Cov = Sigma exactly in expectation.
  Scoring as in job 1 (fixed-n 8,192 live latents, 4 draws; raw decoder slope and gamma against each training Sigma).
  Statements (each tested against both nulls; D_null(L) = null raw fixed-n decoder slope - real's, at k = 32):
  R1 "real is flatter than both nulls in the bulk": per bulk window and null, the (ii)(b) rule of job 1 over the 12
     layers (supports if -D > 2 se at every layer; contradicts if D > 2 se at any layer; partial otherwise).
     REPLICATES if it supports in both bulk windows for both nulls; FAILS if it contradicts in either bulk window for
     either null; partial otherwise.
  R2 "the real-minus-null top-window slope varies across layers": per top window and null, range = max_L D - min_L D;
     se = combined seed se of those two layers. The window shows the profile if range > max(0.05, 2 se), shows none if
     range < 0.02 or range <= 2 se, partial otherwise. REPLICATES if both top windows show it for both nulls; FAILS if
     either top window shows none for either null. Where the extremum falls (argmax and argmin of D, depth L/11) is
     reported, not predicted.
  R3 "gamma declines with k for real vs null as on Gemma": the (i) rule of job 1 at layer 6, k in {16, 32, 64, 128},
     per null, under the window rule. REPLICATES if GPT-2's verdict for a null equals Gemma's job-1 verdict for that
     null ((i) for the Gaussian, (v)'s i_sparse for the sparse null; both from runs_real.jsonl), FAILS if one says
     "supports" and the other "contradicts"; partial otherwise; "pending" if job 1's verdict is not available.
  Reported, not verdicts: FVU, L0, live count per cond and layer; the sparse ground-truth dictionary's gamma per layer.

M2  Gemma-2-9b (HF `google/gemma-2-9b`, gated; d = 3584, 42 layers) and Gemma Scope 9b residual SAEs.
  Sigma: one forward pass over the first 1M Pile tokens (same stream as above), hooks on every decoder layer's
  resid_post, BOS dropped, float32 Gram per batch accumulated in float64, exactly as test1_sigma_layers.py (which made
  the 2b 1M-token Sigma) except: bf16 compute (2b used fp16; Gemma-2 is unstable in fp16 at 9b), batch 4, ctx 1024,
  and on CUDA the float64 accumulators live on the GPU (the same arithmetic). Saved: eigenvalues at all 42 layers;
  the full covariance at layers 9, 20, 31.
  Decoders: google/gemma-scope-9b-pt-res, width 16k: every available L0 at layers 9, 20, 31, and at every layer the L0
  nearest 40 and nearest 70 (by |log(L0 / target)|, named L0). The repo tree is read first and saved, with exactly what
  was fetched (path, size, LFS sha256). Each file is deleted after its spectrum is computed. Spectrum: all 16,384
  W_dec rows (as for Gemma Scope 2b in phase 1, no dead-latent mask), centred covariance, ddof 1.
  Out-of-sample predictions, computed from the 2b results on disk (results/phase1 Gemma Scope 2b spectra and
  results/test1/sigma_eigenvalues_1m.npy) before this run:
  (a) Per-doubling change of the decoder slope with L0 at a fixed layer. 2b, layer 12, width 16k (5 SAEs), 2b windows:
      +0.094, +0.091 (primary), +0.056, +0.049 per doubling (all 28 layer-12 widths: +0.099, +0.099, +0.058, +0.038;
      the 26 layers at 16k span +0.021 to +0.129 in the primary window, median +0.083). Prediction for 9b: the mean
      over layers 9, 20, 31 of the per-doubling slope (fit of decoder slope on log2 L0 over every available L0) lies
      in [0.5, 1.5] x the 2b layer-12 16k value of the corresponding window, i.e. primary [+0.045, +0.136]; also
      [+0.047, +0.141], [+0.028, +0.084], [+0.024, +0.073]. AGREES in a window if the mean is in its interval and every
      layer's slope is > 0; DISAGREES if the mean is outside [0.25, 2] x the 2b value or any layer's slope is <= 0;
      partial otherwise. Window rule.
  (b) Decoder slope is steeper mid-stack at matched L0 in the top windows. Depth = L / (n_layers - 1); mid = depth in
      [0.35, 0.65], early = depth <= 0.12, late = depth >= 0.88; contrast = min(mean early, mean late) - mean mid of the
      decoder slope (positive: mid-stack steeper). 2b at 16k: L0 ~ 40: +0.120 (ranks 4-50), +0.146 (11-150);
      L0 ~ 70: +0.103, +0.100. Prediction for 9b: contrast >= +0.05 in both top windows at both L0 targets (SUPPORTED);
      CONTRADICTED if contrast <= 0 in both top windows at both targets; partial otherwise. Top windows only; the bulk
      contrast is reported.
  (c) Activations' covariance spectrum is steeper than the decoder spectrum (gamma = decoder slope / Sigma slope < 1)
      at every layer for SAEs with named L0 >= 20. 2b: 127 of 127 16k SAEs with L0 >= 20, in every window (largest
      gamma 0.68, 0.73, 0.79, 0.91). Prediction for 9b, per window over every fetched SAE with L0 >= 20: HOLDS if all
      have gamma < 1, FAILS if more than 10% have gamma >= 1, partial otherwise. Window rule.

Order and budget: M2 (about 0.6 h), then M1 layer 6 (k sweep), then the other 11 layers. --cut keeps seed 1 at layer 6
only (drops 33 SAEs, about 1.1 h) if the job must stay under about 4 GPU-hours.
Usage (pod): python experiments/r2_job_models.py [--cut]        smoke (Mac): python experiments/r2_job_models.py --smoke
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import r2_common as rc  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB = "models"
OUT, N_FIX = ROOT / "results" / "r2", rc.N_FIX
# M1: GPT-2 small
G_HF, G_MODEL, G_LAYERS, G_D = "gpt2", "gpt2", list(range(12)), 768
G_WINDOWS = [(1, 17), (3, 50), (10, 150), (33, 333)]
WIDTH, TOKENS, EVAL, EPOCHS, K = 16384, 10_000_000, 500_000, 3, 32
CONDS, SEEDS, KS_LAYER, KS = ["real", "gaussian", "sparse"], [0, 1], 6, [8, 16, 32, 64, 128]
SP_M, SP_L, SP_DS = 65536, 48, 0
GROUP = "saes_r2m"
# M2: Gemma-2-9b
N_HF, N_MODEL, N_SAE_REPO = "google/gemma-2-9b", "gemma-2-9b", "google/gemma-scope-9b-pt-res"
N_WINDOWS = [(5, 78), (16, 233), (47, 700), (156, 1556)]
N_TOKENS, N_FULL_COV, N_ALL_L0, N_TARGETS = 1_000_000, [9, 20, 31], [9, 20, 31], [40, 70]
SMOKE = None


def smoke_config():
    """GPT-2 stands in as pythia-70m layer-3 activations (file symlinks); Gemma-2-9b as the locally cached
    gemma-2-2b fed random token ids (offline, no downloads); Gemma Scope 9b as local 2b W_dec files (read-only)."""
    global SMOKE, G_MODEL, G_LAYERS, WIDTH, TOKENS, EVAL, EPOCHS, KS_LAYER, KS, SP_M, SP_L, GROUP, OUT, N_FIX
    global N_HF, N_MODEL, N_TOKENS, N_FULL_COV, N_ALL_L0
    SMOKE = {"acts": Path(os.environ.get("SMOKE_SRC", ROOT / "cache" / "acts" / "pythia-70m" / "blocks.3.hook_resid_post")),
             "gs_dir": Path(os.environ.get("SMOKE_GS", "/Users/ayurpulle/Desktop/SAE/cache/gemmascope_wdec"))}
    G_MODEL, G_LAYERS, WIDTH, TOKENS, EVAL, EPOCHS = "smoke-gpt2", [3, 6], 512, 300_000, 16_384, 1
    KS_LAYER, KS, SP_M, SP_L = 6, [8, 32], 4096, 16
    GROUP, OUT, N_FIX = "saes_r2m_smoke", ROOT / "results" / "r2_smoke", 256
    N_HF, N_MODEL, N_TOKENS, N_FULL_COV, N_ALL_L0 = "google/gemma-2-2b", "smoke-9b", 8_000, [3, 12], [3, 12]
    os.environ["HF_HUB_OFFLINE"] = "1"


def use_windows(w):
    """r2_common reads WINDOWS / PRIMARY at call time; this job switches them per model (it is its own process)."""
    rc.WINDOWS, rc.PRIMARY = list(w), w[1]


# ---------------------------------------------------------------------------------------------------------- M1 ---
def hook(layer):
    return f"blocks.{layer}.hook_resid_post"


def act_dir(layer):
    return ROOT / "cache" / "acts" / G_MODEL / hook(layer)


def run_name(layer, cond, k, seed):
    return f"{G_MODEL}__{hook(layer)}__{cond}__w{WIDTH}__k{k}__s{seed}" + (f"__ds{SP_DS}" if cond == "sparse" else "")


def gt_name(layer):
    return f"{G_MODEL}__{hook(layer)}__groundtruth__M{SP_M}__L{SP_L:g}__ds{SP_DS}"


def done(name):
    return rc.row_path(OUT, JOB, name).exists()


class _Stop(Exception):
    pass


def collect_gpt2(layer):
    """GPT-2 resid_post of block `layer` into collect_acts.py's cache format (acts.npy fp16, stats.npz, config.json)."""
    d = act_dir(layer)
    if (d / "acts.npy").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    if SMOKE:
        for f in ("acts.npy", "stats.npz"):
            if not (d / f).exists():
                os.symlink((SMOKE["acts"] / f).resolve(), d / f)
        log(f"smoke: linked {d}")
        return
    import torch
    from transformers import AutoModel, AutoTokenizer
    from collect_acts import token_batches
    log(f"collect GPT-2 layer {layer} ({rc.disk_free_gb():.0f} GB free)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(G_HF)
    model = AutoModel.from_pretrained(G_HF, torch_dtype=torch.float32).to(dev).eval()
    model.h = model.h[:layer + 1]
    box = {}

    def grab(module, inputs, output):
        box["h"] = output[0] if isinstance(output, tuple) else output
        raise _Stop

    model.h[layer].register_forward_hook(grab)
    acts = np.lib.format.open_memmap(d / "acts.npy.part", mode="w+", dtype=np.float16, shape=(TOKENS, G_D))
    s, G, n, t0 = np.zeros(G_D), np.zeros((G_D, G_D)), 0, time.time()
    with torch.no_grad():
        for toks in token_batches(tok, 1024, 16, 0):
            try:
                model(input_ids=toks.to(dev))
            except _Stop:
                pass
            a = box["h"][:, 1:].reshape(-1, G_D).float()[: TOKENS - n]
            acts[n:n + len(a)] = a.cpu().numpy().astype(np.float16)
            s += a.sum(0).cpu().double().numpy()
            G += (a.T @ a).cpu().double().numpy()
            n += len(a)
            if n >= TOKENS:
                break
    acts.flush()
    del acts
    (d / "acts.npy.part").rename(d / "acts.npy")
    mean = s / n
    np.savez(d / "stats.npz", mean=mean, cov=(G - n * np.outer(mean, mean)) / (n - 1), n=n)
    json.dump({"model": G_HF, "hook": hook(layer), "n_tokens": TOKENS, "ctx": 1024, "batch": 16, "seed": 0,
               "dtype": "float32", "backend": "hf-gpt2 (r2_job_models.py)", "d_model": G_D, "n_tokens_written": n,
               "dataset": "monology/pile-uncopyrighted", "bos_dropped": True, "seconds": time.time() - t0},
              open(d / "config.json", "w"), indent=1)
    log(f"collected layer {layer}: {n:,} tokens, {time.time() - t0:.0f}s")


def drop_cache(layer):
    p = act_dir(layer) / "acts.npy"
    if p.exists() or p.is_symlink():
        if SMOKE:
            assert p.is_symlink(), f"smoke mode only removes symlinks; {p} is not one"
        p.unlink()
        log(f"deleted {p} (stats.npz kept)")


def m1_layer(scorer, layer, runs):
    """runs: (cond, k, seed). Collect if a real run still needs tokens, train, score, ground truth, delete."""
    names = [run_name(layer, *r) for r in runs] + [gt_name(layer)]
    if all(done(n) for n in names):
        log(f"M1 layer {layer}: all done")
        drop_cache(layer)
        return
    need = [r for r in runs if not done(run_name(layer, *r))
            and not (ROOT / "cache" / GROUP / run_name(layer, *r) / "train.json").exists()]
    if any(c == "real" for c, _, _ in need) or not (act_dir(layer) / "stats.npz").exists():
        collect_gpt2(layer)
    sig = rc.Sigma(act_dir(layer) / "stats.npz", f"{G_MODEL}_L{layer}", OUT)
    for cond, k, seed in runs:
        name = run_name(layer, cond, k, seed)
        if done(name):
            continue
        rd = ROOT / "cache" / GROUP / name
        if not (rd / "train.json").exists():
            toks = TOKENS if (act_dir(layer) / "acts.npy").exists() else TOKENS - EVAL  # same n_train either way
            log(f"train M1 {name}")
            py(ROOT / "train_sae.py", "--model", G_MODEL, "--hook", hook(layer), "--cond", cond, "--width", WIDTH,
               "--k", k, "--tokens", toks, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", seed,
               "--out", f"cache/{GROUP}",
               *(["--sparse_M", SP_M, "--sparse_L", SP_L, "--data_seed", SP_DS] if cond == "sparse" else []))
        meta = {"step": "M1", "model": "gpt2", "layer": layer, "cond": cond, "dataset": f"{G_MODEL}_L{layer}",
                "whitened": False, "windows": [list(w) for w in G_WINDOWS], "n_layers": 12}
        scorer.submit(rc.score, rd, sig, OUT, JOB, meta, n_fix=N_FIX)
    if not done(gt_name(layer)):
        from train_sae import sparse_dictionary
        tj = sorted((ROOT / "cache" / GROUP).glob(f"{G_MODEL}__{hook(layer)}__sparse__*__ds{SP_DS}/train.json"))
        settings = {"M": SP_M, "L": SP_L, "p": SP_L / SP_M, "data_seed": SP_DS, "values": "N(0,1)/sqrt(p)"}
        if tj:
            settings["check_1M"] = rc.sparse_meta(json.load(open(tj[0]))["sparse"], sig)["check_1M"]
        meta = {"step": "M1GT", "model": "gpt2", "layer": layer, "cond": "sparse_dictionary",
                "dataset": f"{G_MODEL}_L{layer}", "whitened": False, "windows": [list(w) for w in G_WINDOWS]}
        scorer.submit(rc.score_dictionary, sparse_dictionary(sig.cov, SP_M, SP_DS), sig, OUT, JOB, gt_name(layer), meta,
                      settings, n_fix=N_FIX)
    scorer.finish()
    drop_cache(layer)


# ---------------------------------------------------------------------------------------------------------- M2 ---
def sigma_pass():
    """Sigma at every layer of the 9b model from one pass over N_TOKENS (see docstring). Returns (eig (L, d), n)."""
    mdir = OUT / "models"
    f = mdir / f"sigma_{N_MODEL}_eigenvalues.npy"
    if f.exists():
        return np.load(f), json.load(open(mdir / f"sigma_{N_MODEL}.json"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    adev = "cuda" if dev == "cuda" else "cpu"  # float64 accumulators (MPS has no float64)
    tok = AutoTokenizer.from_pretrained(N_HF)
    model = AutoModelForCausalLM.from_pretrained(N_HF, torch_dtype=torch.bfloat16)
    model.lm_head = torch.nn.Identity()
    embed = model.model.embed_tokens.eval()
    model.model.embed_tokens = torch.nn.Embedding(1, model.config.hidden_size)
    model = model.to(torch.bfloat16).to(dev).eval()
    nl, d = len(model.model.layers), model.config.hidden_size
    S = [torch.zeros(d, dtype=torch.float64, device=adev) for _ in range(nl)]
    G = [torch.zeros(d, d, dtype=torch.float64, device=adev) for _ in range(nl)]
    cnt = [0]

    def mk(i):
        def f(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            a = h[:, 1:].reshape(-1, h.shape[-1]).float()  # drop BOS
            S[i] += a.sum(0).to(adev).double()
            G[i] += (a.T @ a).to(adev).double()
            if i == 0:
                cnt[0] += a.shape[0]
        return f

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(mk(i))
    if SMOKE:  # random token ids: no dataset download
        gen = torch.Generator().manual_seed(0)
        batches = (torch.cat([torch.full((2, 1), tok.bos_token_id), torch.randint(1000, 30000, (2, 255), generator=gen)], 1)
                   for _ in range(10 ** 6))
    else:
        from collect_acts import token_batches
        batches = token_batches(tok, 1024, 4, 0)
    t0 = time.time()
    log(f"Sigma pass {N_HF}: {nl} layers, d {d}, {N_TOKENS:,} tokens")
    with torch.no_grad():
        for toks in batches:
            model.model(inputs_embeds=embed(toks).to(torch.bfloat16).to(dev))
            if cnt[0] >= N_TOKENS:
                break
    n = cnt[0]
    eig = np.zeros((nl, d))
    mdir.mkdir(parents=True, exist_ok=True)
    for i in range(nl):
        m = (S[i] / n).cpu().numpy()
        C = (G[i].cpu().numpy() - n * np.outer(m, m)) / (n - 1)
        eig[i] = np.sort(np.linalg.eigvalsh(C))[::-1]
        if i in N_FULL_COV:
            p = OUT / "sigma" / f"{N_MODEL}_L{i}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, eigenvalues=eig[i], cov=C, mean=m, n=n)
    meta = {"model": N_HF, "n_tokens_seen": n, "n_layers": nl, "d_model": d, "dtype": "bfloat16", "ctx": 1024 if not SMOKE else 256,
            "batch": 4, "seed": 0, "bos_dropped": True, "centered": True, "seconds": time.time() - t0,
            "dataset": "random token ids (smoke)" if SMOKE else "monology/pile-uncopyrighted",
            "full_cov_layers": N_FULL_COV}
    json.dump(meta, open(mdir / f"sigma_{N_MODEL}.json", "w"), indent=1)
    np.save(f, eig)
    log(f"Sigma pass done: {n:,} tokens, {time.time() - t0:.0f}s")
    del model, S, G
    if dev == "cuda":
        torch.cuda.empty_cache()
    return eig, meta


def gs_tree():
    """List the Gemma Scope 9b repo (or, in smoke mode, a local 2b W_dec tree) as [{path, size, sha256}]."""
    if SMOKE:
        out = []
        for p in sorted(SMOKE["gs_dir"].glob("layer_*/width_16k/average_l0_*/W_dec.npy")):
            out.append({"path": str(p.relative_to(SMOKE["gs_dir"]).parent / "params.npz"), "size": p.stat().st_size,
                        "sha256": None, "local": str(p)})
        return out
    from huggingface_hub import HfApi
    out = []
    for e in HfApi().list_repo_tree(N_SAE_REPO, recursive=True):
        if getattr(e, "size", None) is not None and e.path.endswith(".npz"):
            lfs = getattr(e, "lfs", None)
            out.append({"path": e.path, "size": e.size, "sha256": getattr(lfs, "sha256", None) if lfs else None})
    return out


def select(tree, n_layers):
    pat = re.compile(r"layer_(\d+)/width_16k/average_l0_(\d+)/params\.npz$")
    by_layer = {}
    for e in tree:
        m = pat.search(e["path"])
        if m:
            by_layer.setdefault(int(m[1]), []).append(e | {"layer": int(m[1]), "l0": int(m[2])})
    chosen = {}
    for L, es in by_layer.items():
        if L >= n_layers:
            continue
        reasons = {}
        if L in N_ALL_L0:
            for e in es:
                reasons.setdefault(e["path"], []).append("all_l0")
        for t in N_TARGETS:
            e = min(es, key=lambda e: abs(np.log(e["l0"] / t)))
            reasons.setdefault(e["path"], []).append(f"nearest_{t}")
        for e in es:
            if e["path"] in reasons:
                chosen[e["path"]] = e | {"reasons": reasons[e["path"]]}
    return sorted(chosen.values(), key=lambda e: (e["layer"], e["l0"])), sorted(by_layer)


def dec_spectrum(W):
    """Descending eigenvalues of the centred row covariance (ddof 1), float64; on the GPU when there is one."""
    try:
        import torch
        if torch.cuda.is_available():
            X = torch.tensor(W, device="cuda", dtype=torch.float64)
            X -= X.mean(0)
            C = X.T @ X / (len(X) - 1)
            return torch.linalg.eigvalsh(C).flip(0).cpu().numpy(), C.cpu().numpy()
    except ImportError:
        pass
    C = rc.cloud_cov(W)
    return np.sort(np.linalg.eigvalsh(C))[::-1], C


def m2(scorer):
    use_windows(N_WINDOWS)
    eig_sig, smeta = sigma_pass()
    nl = eig_sig.shape[0]
    mdir = OUT / "models"
    tree = gs_tree()
    json.dump({"repo": N_SAE_REPO if not SMOKE else str(SMOKE["gs_dir"]), "listed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "files": tree}, open(mdir / "gemmascope9b_tree.json", "w"), indent=1)
    chosen, layers_listed = select(tree, nl)
    log(f"Gemma Scope tree: {len(tree)} npz files, 16k layers {layers_listed[:3]}..{layers_listed[-3:]}; fetching {len(chosen)}")
    assert chosen, f"no width_16k params.npz found in {N_SAE_REPO}: check the tree in {mdir / 'gemmascope9b_tree.json'}"
    fetched = []
    tmp = ROOT / "cache" / "gs9b_tmp"
    for e in chosen:
        name = f"{N_MODEL}_gemmascope__layer_{e['layer']}__w16k__l0_{e['l0']}"
        if done(name):
            continue
        t0 = time.time()
        if SMOKE:
            W, local, delete = np.load(e["local"]), e["local"], False
        else:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(N_SAE_REPO, e["path"], local_dir=tmp)
            with np.load(local) as z:
                W = z["W_dec"].astype(np.float32)
            delete = True
        ev, C = dec_spectrum(W)
        sl_sig = rc.slopes(eig_sig[e["layer"]])
        dec = rc.slopes(ev)
        row = {"job": JOB, "run": name, "step": "M2", "model": "gemma-2-9b", "layer": e["layer"], "l0": e["l0"],
               "n_layers": nl, "depth": e["layer"] / (nl - 1), "reasons": e["reasons"], "file": e["path"],
               "file_size": e["size"], "file_sha256": e["sha256"], "n_rows": int(W.shape[0]), "d": int(W.shape[1]),
               "row_norm_mean": float(np.linalg.norm(W, axis=1).mean()), "windows": [list(w) for w in N_WINDOWS],
               "sigma_slope": sl_sig, "dec_slope": dec, "gamma": rc.ratio(dec, sl_sig), "eigenvalues": ev.tolist(),
               "sigma_n_tokens": smeta["n_tokens_seen"], "git_hash": rc.git_hash(), "code_sha256": rc.code_sha256(),
               "whitened": False, "k": None, "seed": None, "cond": "gemmascope"}
        sp = OUT / "sigma" / f"{N_MODEL}_L{e['layer']}.npz"
        if e["layer"] in N_FULL_COV and sp.exists():
            bg = rc.basis_gamma(C, np.load(sp)["cov"], rank_ranges=tuple(N_WINDOWS))
            row["basis_gamma"] = bg | {"subspace_overlap": {str(m): v for m, v in bg["subspace_overlap"].items()}}
        rc.write_row(OUT, JOB, row)
        if delete:
            os.remove(local)
        fetched.append({k: e[k] for k in ("path", "size", "sha256", "layer", "l0", "reasons")})
        log(f"M2 {name}: decoder slope {dec[str(N_WINDOWS[1])]:+.3f}, gamma {row['gamma'][str(N_WINDOWS[1])]:.3f} "
            f"[{time.time() - t0:.0f}s]")
    prev = json.load(open(mdir / "gemmascope9b_fetched.json")) if (mdir / "gemmascope9b_fetched.json").exists() else []
    json.dump(prev + fetched, open(mdir / "gemmascope9b_fetched.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cut", action="store_true", help="seed 1 only at layer 6 (keeps the job under ~4 GPU-hours)")
    ap.add_argument("--skip_m2", action="store_true")
    ap.add_argument("--skip_m1", action="store_true", help="M2 only (M1 ran as a separate process)")
    a = ap.parse_args()
    if a.smoke:
        smoke_config()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    log(f"job {JOB}: git {rc.git_hash()}, code sha256 {rc.code_sha256()[:12]}, out {OUT}, {rc.disk_free_gb():.0f} GB free")
    scorer = rc.Scorer()
    if not a.skip_m2:
        log("M2: Gemma-2-9b Sigma and Gemma Scope 9b decoders")
        m2(scorer)
    use_windows(G_WINDOWS)
    order = [] if a.skip_m1 else [KS_LAYER] + [L for L in G_LAYERS if L != KS_LAYER]
    for L in order:
        seeds = SEEDS if (L == KS_LAYER or not a.cut) else SEEDS[:1]
        runs = [(c, K, s) for s in seeds for c in CONDS]
        if L == KS_LAYER:
            runs += [(c, k, s) for s in SEEDS for k in KS if k != K for c in CONDS]
        log(f"M1 layer {L}: {len(runs)} SAEs")
        m1_layer(scorer, L, runs)
    scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB")
    try:
        py(ROOT / "experiments" / "r2_summary.py", "--dir", OUT)
    except Exception as e:
        log(f"r2_summary failed on the pod ({e}); rerun locally")
    log("JOB DONE models")


if __name__ == "__main__":
    main()
