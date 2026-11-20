"""Round-2 GPU job 7 ("ladder"): the real / equal-power-null / power-matched-null comparison of r2_job_pmatch.py, repeated
on four activation datasets that remove trained structure one rung at a time. Unattended and restartable like the other
jobs: a run whose row exists is skipped, a trained but unscored run is scored, calibration reads finished runs back,
caches are collected only while a run still needs them, dictionaries are built once and kept. Rows go to
results/r2/runs_ladder.jsonl (rows/ladder/), calibration to results/r2/ladder_calibration.json, dictionary reports to
results/r2/ladder_pmatch/, the real seed-0 SAEs' per-latent power to results/r2/power/<run>.npz (the format of
r2_job_robust.py step P), the accepted power-matched null SAEs' power to results/r2/power_ladder/, and the state of every
dataset and step to results/r2/ladder_status.json. experiments/r2_ladder_summary.py reads the rows.

Datasets (in this order; each is 10M tokens of the Pile stream collect_acts.py uses: monology/pile-uncopyrighted,
shuffle seed 0, ctx 1024, BOS + 1023 packed tokens per row, BOS dropped, batch 16, bf16 compute, fp16 cache, float64
mean and covariance; the Gemma-2-2B tokenizer throughout):
  embT     the trained Gemma-2-2B token embeddings as the model feeds them to block 0 (embedding row x sqrt(d) with
           the model's own scaling), no transformer block. Read with a forward pre-hook on block 0, so the scaling is
           the model's whichever transformers version applies it.
  embR     the same with the embedding matrix replaced by i.i.d. N(0, s^2) entries, s = the RMS of all entries of the
           trained matrix, torch generator seed EMBR_SEED; the model's scaling is applied as for embT.
  randL0   resid_post of block 0 of a randomly initialised Gemma-2-2B: config of google/gemma-2-2b, every weight
           (embeddings included) from transformers' default initialisation after torch.manual_seed(RAND_SEED), built in
           float32 on the CPU at full depth, then cut after the block and cast to bf16 like the trained model.
  randL13  resid_post of block 13 of the same random model (rebuilt with the same seed; config.json records weight
           hashes and the job checks that embedding and block 0 match randL0's).
Cache directories cache/acts/<model>/<hook>/ use the models "ladder-embT", "ladder-embR" and "ladder-rand"; the hook
strings stay in the blocks.N.hook_resid_post form because the other code reads the layer from them, so the embedding
datasets sit under blocks.0.hook_resid_post. What each dataset really is goes into its config.json and every row's meta.

Per dataset, in order (a step whose inputs failed is skipped and recorded; a failure never stops the next dataset):
  a  collect (subprocess, only if a run still needs tokens); Sigma = stats.npz; log the top-1 and top-10 eigenvalue
     shares and the participation ratio; a non-finite Sigma, one with top-1 share > 0.99 or with fewer positive
     eigenvalues than min(d, 1000) is recorded as degenerate and the dataset stops there.
  b  real TopK SAEs, width 16,384, k = 40, seeds 0, 1, 2, 9.5M training tokens x 3 epochs, last 500k held out (as
     r2_job_real.py); seed 0 with --save_params. Scored with r2_common.score against this dataset's Sigma.
  c  power pass of the real seed-0 SAE on the last 1M cached tokens -> results/r2/power/<run>.npz (r2_job_robust.py
     step P's arrays); the V1 dictionary is then built in a background thread.
  d  V0, the equal-power codable null (train_sae.py --cond sparse, M 8192, L 16, half-normal values, data seed 0):
     target T = mean eval FVU of the three real SAEs, tolerance 0.03, at most 4 SAEs, SAE seed 0. First rho = 0.3, or
     rho = 0 if T < 0.045; then a Newton step with dF/drho = 0.65 (V0's measured slopes on Gemma were 0.59-0.71); then
     secant steps through the bracketing (else nearest) pair, rounded to 0.01 and clipped to [0, 0.95]. At rho = 0 with
     F > T + 0.03 the null cannot be made more codable and calibration stops. The first run within tolerance is used,
     else the closest (unmatched). Then SAE seed 1 at the accepted rho. Rows "v0" (accepted rho, both seeds) and
     "v0_cal" (other tried rho).
  e  V1, the power-matched null: dictionary from this dataset's Sigma and the powers of (c) via r2_pmatch_gen (V1,
     as r2_job_pmatch.ensure_dict; the job stops the step if A^T A misses Sigma by more than 1e-6 in float64);
     calibration from V0's accepted rho, Newton step with V0's own slope (from its two tried points nearest the
     accepted rho, else 0.65), then secant, at most 3 SAEs. Power check on the accepted run as r2_job_pmatch.py does.
     Rows "v1" and "v1_cal".
  f  embT and embR only: real SAEs k = 4 and k = 16, seed 0 (descriptive; rows "real_k").
  g  delete acts.npy (stats.npz and config.json stay), so one ~46 GB cache is on disk at a time.
No new SAE starts after --budget_min minutes (default 480); scoring, power passes and deletes still finish.

Budget on one 32 GB card, at ~8 min per SAE (the documented null-SAE wall time; real SAEs were 4-6 min): per dataset
3 real + V0 (2-4 tried + seed 1) + V1 (1-3) [+ 2 for embT/embR]: expected 9.5 SAEs for embT/embR and 7.5 for the
random model, 34 in all, about 4.5 h; worst case 48 SAEs, 6.4 h. Collection 15-25 min per dataset (Pile streaming and
tokenising; the GPU is nearly idle meanwhile), power pass and dictionary about 5 min per dataset: expected about 6.2 h,
worst about 8.1 h, cut by the budget at 8 h plus the last SAE. Disk: one 46 GB cache, ~5 GB of model weights in the HF
cache, ~2.5 GB of SAEs per dataset (W_dec 151 MB each, plus W_enc for real seed 0 and every V1 run), ~0.5 GB of
results per dataset.

  bash gpu/run_job.sh experiments/r2_job_ladder.py 9 20      (pod, HF_TOKEN set for google/gemma-2-2b)
  python experiments/r2_job_ladder.py --datasets randL0 randL13 --budget_min 240
  python experiments/r2_job_ladder.py --smoke                 (laptop: every step at toy scale, no download)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import r2_common as rc  # noqa: E402
import r2_pmatch_gen as pg  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB = "ladder"
OUT, N_FIX = ROOT / "results" / "r2", rc.N_FIX
GROUP, DICT_DIR = "saes_ladder", ROOT / "cache" / "ladder_dicts"
HF_MODEL = "google/gemma-2-2b"
WIDTH, TOKENS, EVAL, EPOCHS, K = 16384, 10_000_000, 500_000, 3, 40
SEEDS, EXTRA_K = [0, 1, 2], [4, 16]
SP_M, SP_L, SP_DS, VALUES = 8192, 16, 0, "halfnormal"
TOL, RHO_MAX, V0_MAX, V1_MAX = 0.03, 0.95, 4, 3
V0_RHO1, LOW_FVU, DEFAULT_SLOPE = 0.3, 0.045, 0.65
N_POWER, POWER_SEED = 1_000_000, 9_500_000
DICT_SEED = 6_500_000  # + dataset index; r2_job_pmatch.py uses 6_000_000 + layer + 1000 ds
DICT_SECONDS, DICT_TOL, DICT_RQ_TOL, DICT_MAX_ITER = 240, 0.03, 0.02, 400
CTX, BATCH, DTYPE = 1024, 16, "bfloat16"
RAND_SEED, EMBR_SEED = 1234, 4321
BUDGET_MIN, MIN_PER_SAE, COLLECT_MIN, OVERHEAD_MIN = 480.0, 8.0, 25.0, 5.0
EXTRA = []  # extra train_sae.py arguments (smoke only)
SMOKE = None
T0 = time.time()

DATASETS = {
    "embT": {"model": "ladder-embT", "layer": 0, "site": "embed", "weights": "trained", "extra_k": True,
             "desc": "trained Gemma-2-2B token embeddings as fed to block 0 (row x sqrt(d), model scaling), no blocks"},
    "embR": {"model": "ladder-embR", "layer": 0, "site": "embed", "weights": "random_embedding", "extra_k": True,
             "desc": "i.i.d. Gaussian embedding matrix with the trained matrix's RMS, model scaling, no blocks"},
    "randL0": {"model": "ladder-rand", "layer": 0, "site": "resid_post", "weights": "random_init", "extra_k": False,
               "desc": "resid_post of block 0 of a randomly initialised Gemma-2-2B (HF default init, fixed seed)"},
    "randL13": {"model": "ladder-rand", "layer": 13, "site": "resid_post", "weights": "random_init", "extra_k": False,
                "desc": "resid_post of block 13 of the same randomly initialised Gemma-2-2B"},
}
ORDER = ["embT", "embR", "randL0", "randL13"]


def smoke_config():
    """Toy scale on a laptop: a 14-block Gemma2 config of width 128 built from scratch (seed 999 stands in for the
    trained weights), a synthetic Zipf token stream, width-1024 SAEs, M = 512, 200k tokens, one epoch."""
    global SMOKE, OUT, N_FIX, GROUP, DICT_DIR, WIDTH, TOKENS, EVAL, EPOCHS, SP_M, N_POWER, DICT_SECONDS, CTX, DTYPE
    global EXTRA
    SMOKE = {"vocab": 4096, "d": 128, "trained_seed": 999}
    WIDTH, TOKENS, EVAL, EPOCHS, SP_M = 1024, 200_000, 16_384, 1, 512
    N_POWER, DICT_SECONDS, CTX, DTYPE = 32_768, 30, 128, "float32"
    GROUP, DICT_DIR, OUT, N_FIX = "saes_ladder_smoke", ROOT / "cache" / "ladder_smoke_dicts", \
        ROOT / "cache" / "ladder_smoke_out", 256
    EXTRA = ["--batch", 2048]
    for ds in DATASETS.values():
        ds["model"] = ds["model"].replace("ladder-", "ladder-smoke-")


# ------------------------------------------------------------------------------------------------------- names ---
def hook(name):
    return f"blocks.{DATASETS[name]['layer']}.hook_resid_post"


def act_dir(name):
    return ROOT / "cache" / "acts" / DATASETS[name]["model"] / hook(name)


def label(name):
    ds = DATASETS[name]
    return ds["model"] if ds["site"] == "embed" else f"{ds['model']}_L{ds['layer']}"


def dict_tag(name):
    return f"lad{name}M{SP_M}ds{SP_DS}"


def rel(p):
    """ROOT-relative path string (absolute only for a path outside the checkout)."""
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def job_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def minutes():
    return (time.time() - T0) / 60


def can_start():
    return minutes() <= BUDGET_MIN


class BudgetSpent(Exception):
    pass


class Degenerate(Exception):
    pass


def done(run):
    return rc.row_path(OUT, JOB, run).exists()


def cli_real(name, k, seed, save=False):
    return ["--model", DATASETS[name]["model"], "--hook", hook(name), "--cond", "real", "--width", WIDTH, "--k", k,
            "--tokens", TOKENS, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", seed, "--out", f"cache/{GROUP}",
            *(["--save_params"] if save else []), *EXTRA]


def cli_sparse(name, seed, rho, dpath=None):
    # stats-only: n_train = --tokens, so 9.5M whether or not acts.npy is still there (as r2_job_robust.py)
    return ["--model", DATASETS[name]["model"], "--hook", hook(name), "--cond", "sparse", "--width", WIDTH, "--k", K,
            "--tokens", TOKENS - EVAL, "--eval_tokens", EVAL, "--epochs", EPOCHS, "--seed", seed,
            "--out", f"cache/{GROUP}", "--sparse_M", SP_M, "--sparse_L", SP_L, "--data_seed", SP_DS,
            "--sparse_values", VALUES, "--sparse_rho", rho,
            *(["--save_params", "--sparse_dict", rel(dpath), "--sparse_dict_tag", dict_tag(name)]
              if dpath is not None else []), *EXTRA]


def run_dir(cli):
    """The directory train_sae.py writes for these arguments (its own parser and naming)."""
    from train_sae import build_parser, run_name
    return ROOT / "cache" / GROUP / run_name(build_parser().parse_args([str(a) for a in cli]))


def train(cli):
    """Train if needed (never after the budget); returns (eval FVU, run dir)."""
    rd = run_dir(cli)
    if not (rd / "train.json").exists():
        if not can_start():
            raise BudgetSpent(f"budget of {BUDGET_MIN:.0f} min spent ({minutes():.0f} min); not starting {rd.name}")
        log(f"train {rd.name} ({minutes():.0f} min into the job)")
        py(ROOT / "train_sae.py", *cli)
    return json.load(open(rd / "train.json"))["fvu_eval"], rd


def trained(cli):
    return (run_dir(cli) / "train.json").exists()


# ------------------------------------------------------------------------------------------------------ status ---
_slock = threading.Lock()
STATUS = {}


def save_status():
    with _slock:
        f = OUT / "ladder_status.json"
        for _ in range(5):  # the other thread may be adding a key; retry the snapshot
            try:
                text = json.dumps(STATUS, indent=1, default=str)
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            return
        tmp = f.with_suffix(".tmp")
        tmp.write_text(text)
        os.replace(tmp, f)


def note(name, msg):
    log(f"{name}: {msg}")
    with _slock:
        STATUS[name].setdefault("notes", []).append(msg)
    save_status()


def save_cal(key, entry):
    f = OUT / "ladder_calibration.json"
    cal = json.load(open(f)) if f.exists() else {}
    cal[key] = entry
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=1))
    os.replace(tmp, f)


def load_cal(key):
    f = OUT / "ladder_calibration.json"
    return (json.load(open(f)) if f.exists() else {}).get(key)


# ------------------------------------------------------------------------------------------------- collection ---
class _Stop(Exception):
    pass


def smoke_model_config():
    from transformers import Gemma2Config
    return Gemma2Config(vocab_size=SMOKE["vocab"], hidden_size=SMOKE["d"], intermediate_size=2 * SMOKE["d"],
                        num_hidden_layers=14, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                        query_pre_attn_scalar=64, max_position_embeddings=CTX, sliding_window=64,
                        pad_token_id=0, eos_token_id=1, bos_token_id=2)


def thash(t):
    """sha256 of a tensor's bytes (bf16 viewed as int16), no copy beyond the CPU tensor."""
    import torch
    t = t.detach().cpu().contiguous()
    a = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy()
    return hashlib.sha256(a.data).hexdigest()


def module_hash(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode() + thash(v).encode())
    return h.hexdigest()


def load_model(name):
    """(model, tokenizer or None, info) for a dataset, before truncation; see the docstring for each weight source."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    ds = DATASETS[name]
    info = {"transformers_version": transformers.__version__, "torch_version": torch.__version__}
    tok = None if SMOKE else AutoTokenizer.from_pretrained(HF_MODEL)
    if ds["weights"] == "random_init":
        cfg = smoke_model_config() if SMOKE else AutoConfig.from_pretrained(HF_MODEL)
        torch.manual_seed(RAND_SEED)
        model = AutoModelForCausalLM.from_config(cfg)  # float32, transformers' _init_weights for every module
        info |= {"init": "AutoModelForCausalLM.from_config, torch.manual_seed(%d), float32 on the CPU" % RAND_SEED,
                 "config_source": "smoke Gemma2Config" if SMOKE else HF_MODEL, "rand_seed": RAND_SEED,
                 "n_layers_built": cfg.num_hidden_layers}
    elif SMOKE:  # stand-in for the trained weights: the same tiny config at another seed
        torch.manual_seed(SMOKE["trained_seed"])
        model = AutoModelForCausalLM.from_config(smoke_model_config())
        info |= {"init": f"smoke stand-in for trained weights, seed {SMOKE['trained_seed']}"}
    else:
        try:  # transformers 5.x names the argument dtype; 4.x torch_dtype
            model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.bfloat16)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.bfloat16)
        info |= {"weights_source": HF_MODEL}
    emb = model.model.embed_tokens
    if ds["weights"] == "random_embedding":
        w = emb.weight.detach()
        rms = float(torch.sqrt((w.double() ** 2).mean()))
        g = torch.Generator().manual_seed(EMBR_SEED)
        new = torch.randn(w.shape, generator=g) * rms
        emb.weight.data = new.to(w.dtype)
        info |= {"embr_seed": EMBR_SEED, "embr_rms": rms, "trained_embedding_sha256": thash(w),
                 "embedding_law": "i.i.d. N(0, rms^2), rms over all entries of the trained matrix"}
    w = emb.weight.detach()
    info |= {"embedding_sha256": thash(w), "embedding_std": float(w.double().std()),
             "embedding_rms": float(torch.sqrt((w.double() ** 2).mean())), "embedding_shape": list(w.shape),
             "pad_row_norm": float(w[0].double().norm()),
             "embed_scale_attr": float(emb.embed_scale) if hasattr(emb, "embed_scale") else None,
             "attn_implementation": getattr(model.config, "_attn_implementation", None)}
    return model, tok, info


class Resid:
    """collect_acts.HFResid for one site: block-0 input (site "embed", read by a forward pre-hook, so the model's own
    embedding scaling is included) or resid_post of block `layer`. The model is cut after that block and cast to the
    compute dtype; the (scaled) embedding stays on the CPU in that dtype and only looked-up rows go to the device."""

    def __init__(self, model, site, layer, dev, dtype):
        import torch
        model.model.layers = model.model.layers[:layer + 1]
        model.lm_head = torch.nn.Identity()
        self.embed = model.model.embed_tokens.to(dtype).eval()
        model.model.embed_tokens = torch.nn.Embedding(1, model.config.hidden_size)
        self.model = model.to(dtype).to(dev).eval()
        self.d_model, self.dev, self.dtype = model.config.hidden_size, dev, dtype
        blk = self.model.model.layers[layer]
        if site == "embed":
            blk.register_forward_pre_hook(self._pre, with_kwargs=True)
        else:
            blk.register_forward_hook(self._post)

    def _pre(self, module, args, kwargs):
        self._out = args[0] if args else kwargs["hidden_states"]
        raise _Stop

    def _post(self, module, inputs, output):
        self._out = output[0] if isinstance(output, tuple) else output
        raise _Stop

    def __call__(self, toks):
        try:
            self.model.model(inputs_embeds=self.embed(toks).to(self.dtype).to(self.dev))
        except _Stop:
            pass
        return self._out


def synthetic_batches(vocab, ctx, batch, seed):
    """Smoke token stream: BOS (2) + ctx-1 i.i.d. Zipf(1.1) tokens over ids 3..vocab-1."""
    import torch
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab - 2) ** 1.1
    p /= p.sum()
    while True:
        t = rng.choice(len(p), size=(batch, ctx - 1), p=p) + 3
        yield torch.tensor(np.concatenate([np.full((batch, 1), 2), t], 1))


def collect(name):
    """10M tokens of one dataset, as collect_acts.main does (fp16 memmap appearing only when complete, per-batch fp32
    sums accumulated in float64, BOS dropped), plus a config.json saying what the dataset is."""
    import torch
    ds = DATASETS[name]
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, DTYPE)
    out = act_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok, info = load_model(name)
    if ds["weights"] == "random_init":
        info["block0_sha256"] = module_hash(model.model.layers[0])
        info[f"block{ds['layer']}_sha256"] = module_hash(model.model.layers[ds["layer"]])
        info["input_layernorm0_max_abs"] = float(model.model.layers[0].input_layernorm.weight.abs().max())
        info["q_proj0_std"] = float(model.model.layers[0].self_attn.q_proj.weight.double().std())
        ref = act_dir("randL0") / "config.json"
        if name != "randL0" and ref.exists():
            r = json.load(open(ref)).get("weights", {})
            same = r.get("embedding_sha256") == info["embedding_sha256"] and r.get("block0_sha256") == info["block0_sha256"]
            info["same_init_as_randL0"] = same
            log(f"collect {name}: embedding and block 0 {'match' if same else 'DIFFER FROM'} randL0's")
    resid = Resid(model, ds["site"], ds["layer"], dev, dtype)
    d = resid.d_model
    if SMOKE:
        stream = synthetic_batches(SMOKE["vocab"], CTX, BATCH, 0)
    else:
        from collect_acts import token_batches  # the Pile stream of every earlier Gemma cache
        stream = token_batches(tok, CTX, BATCH, 0)
    acts = np.lib.format.open_memmap(out / "acts.npy.part", mode="w+", dtype=np.float16, shape=(TOKENS, d))
    s, G, n = np.zeros(d), np.zeros((d, d)), 0
    log(f"collect {name} ({ds['desc']}) on {dev}, {TOKENS:,} tokens, {rc.disk_free_gb():.0f} GB free")
    with torch.no_grad():
        for toks in stream:
            a = resid(toks)[:, 1:].reshape(-1, d).float()[: TOKENS - n]
            acts[n:n + len(a)] = a.cpu().numpy().astype(np.float16)
            s += a.sum(0).cpu().double().numpy()
            G += (a.T @ a).cpu().double().numpy()
            n += len(a)
            if n // 1_000_000 != (n - len(a)) // 1_000_000:
                log(f"collect {name}: {n:,} tokens, {time.time() - t0:.0f}s")
            if n >= TOKENS:
                break
    acts.flush()
    del acts
    mean = s / n
    cov = (G - n * np.outer(mean, mean)) / (n - 1)
    tmp = out / "stats.tmp.npz"
    np.savez(tmp, mean=mean, cov=cov, n=n)
    os.replace(tmp, out / "stats.npz")
    json.dump({"dataset": name, "label": label(name), "description": ds["desc"], "model_label": ds["model"],
               "hook": hook(name), "site": ds["site"], "layer": ds["layer"], "weights": info,
               "hook_note": ("hook string kept in the blocks.N.hook_resid_post form for path parsing; the data are "
                             + ds["desc"]),
               "tokenizer": None if SMOKE else HF_MODEL,
               "token_stream": ("synthetic Zipf(1.1), smoke" if SMOKE else
                                "monology/pile-uncopyrighted via collect_acts.token_batches, shuffle seed 0"),
               "n_tokens": TOKENS, "ctx": CTX, "batch": BATCH, "seed": 0, "dtype": DTYPE, "device": dev,
               "d_model": d, "n_tokens_written": n, "bos_dropped": True, "seconds": time.time() - t0,
               "job_code_sha256": job_sha()}, open(out / "config.json", "w"), indent=1)
    (out / "acts.npy.part").rename(out / "acts.npy")
    log(f"collect {name}: done, {n:,} tokens x {d}, {time.time() - t0:.0f}s")


def drop_acts(name):
    for f in ("acts.npy", "acts.npy.part"):
        p = act_dir(name) / f
        if p.exists() or p.is_symlink():
            p.unlink()
            log(f"deleted {p} (stats.npz kept); {rc.disk_free_gb():.0f} GB free")


# --------------------------------------------------------------------------------------------------- scoring ---
def score_safe(name, rd, sig, meta, arrays=None, power_dir="power_ladder"):
    """rc.score in the scorer thread; a failure is logged and recorded instead of surfacing at the next submit."""
    try:
        if arrays is not None:
            (OUT / power_dir).mkdir(parents=True, exist_ok=True)
            tmp = OUT / power_dir / f"{rd.name}.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, OUT / power_dir / f"{rd.name}.npz")
        row = rc.score(rd, sig, OUT, JOB, meta, n_fix=N_FIX)
        if row["fixed_n_short"]:
            note(name, f"{rd.name}: {row['n_live']} live latents < {N_FIX}; scored on all live (fixed_n_short)")
    except Exception as e:
        log(f"score {rd.name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        with _slock:
            STATUS[name].setdefault("score_errors", []).append(f"{rd.name}: {type(e).__name__}: {e}")
        save_status()


def base_meta(name, step, cond, **kw):
    ds = DATASETS[name]
    cfgp = act_dir(name) / "config.json"
    return {"step": step, "rung": name, "dataset": label(name), "dataset_description": ds["desc"],
            "model": ds["model"], "layer": ds["layer"], "site": ds["site"], "weights": ds["weights"], "cond": cond,
            "whitened": False, "job_code_sha256": job_sha(),
            "collection": json.load(open(cfgp)) if cfgp.exists() else None, **kw}


def submit(scorer, name, rd, sig, meta, arrays=None):
    if not done(rd.name):
        scorer.submit(score_safe, name, rd, sig, meta, arrays)


# ---------------------------------------------------------------------------------------------------- steps ---
def real_clis(name):
    c = [cli_real(name, K, s, save=(s == 0)) for s in SEEDS]
    if DATASETS[name]["extra_k"]:
        c += [cli_real(name, k, 0) for k in EXTRA_K]
    return c


def power_path(name):
    return OUT / "power" / f"{run_dir(cli_real(name, K, 0)).name}.npz"


def power_done(name):
    """The power file and its row (written after it) both exist."""
    return power_path(name).exists() and done(f"{run_dir(cli_real(name, K, 0)).name}__power")


def needs_acts(name):
    """True while a real run still has to be trained or the power file is missing."""
    return (any(not trained(c) and not done(run_dir(c).name) for c in real_clis(name))
            or not power_done(name))


def spectrum(sig):
    e = sig.eig
    pos = e[e > 0]
    return {"top1_share": float(e[0] / e.sum()), "top10_share": float(e[:10].sum() / e.sum()),
            "participation_ratio": float(e.sum() ** 2 / (e ** 2).sum()), "n_positive": int((e > e[0] * 1e-9).sum()),
            "d": len(e), "min_positive": float(pos.min()) if len(pos) else None, "slopes": sig.slopes}


def step_a(name):
    d = act_dir(name)
    if not (d / "acts.npy").exists() and (needs_acts(name) or not (d / "stats.npz").exists()):
        if not can_start():
            raise BudgetSpent(f"budget spent ({minutes():.0f} min); not collecting {name}")
        need = TOKENS * (SMOKE["d"] if SMOKE else 2304) * 2 / 1e9 + (0 if SMOKE else 5)
        if rc.disk_free_gb() < need:  # earlier jobs on the volume may hold caches of their own
            raise RuntimeError(f"{rc.disk_free_gb():.0f} GB free, collection needs {need:.0f} GB")
        py(Path(__file__).resolve(), "--collect", name, *(["--smoke"] if SMOKE else []))
    sig = rc.Sigma(d / "stats.npz", label(name), OUT)
    sp = spectrum(sig)
    with _slock:
        STATUS[name]["sigma"] = sp
    log(f"{name}: Sigma top-1 share {sp['top1_share']:.4f}, top-10 {sp['top10_share']:.4f}, participation ratio "
        f"{sp['participation_ratio']:.1f} of d = {sp['d']}, {sp['n_positive']} positive eigenvalues")
    if not np.all(np.isfinite(sig.eig)) or sp["top1_share"] > 0.99 or sp["n_positive"] < min(sp["d"], 1000):
        raise Degenerate(f"degenerate Sigma: {sp}")
    return sig


def step_b(name, scorer, sig):
    fv = []
    for s in SEEDS:
        c = cli_real(name, K, s, save=(s == 0))
        f, rd = train(c)
        fv.append(f)
        submit(scorer, name, rd, sig, base_meta(name, "real", "real"))
    T = float(np.mean(fv))
    log(f"{name}: real k = {K} eval FVU {', '.join(f'{x:.3f}' for x in fv)}; target {T:.3f}")
    with _slock:
        STATUS[name]["real_fvu"] = fv
    return T


def acts_batches(name, scale, batch=8192):
    """The last N_POWER cached tokens (500k held out + 500k training at full scale), as r2_job_robust.acts_batches."""
    import torch
    from r2_job_robust import device
    dev = device()
    a = np.load(act_dir(name) / "acts.npy", mmap_mode="r")
    for b in range(len(a) - N_POWER, len(a), batch):
        yield torch.from_numpy(np.asarray(a[b:min(b + batch, len(a))], dtype=np.float32)).to(dev) * scale


def empty_cache():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # the next SAE trains in another process


def step_c(name, sig):
    """Power pass of the real seed-0 SAE -> power/<run>.npz with r2_job_robust.power_row's arrays."""
    p = power_path(name)
    if power_done(name):
        return p
    from r2_job_robust import power_pass
    rd = run_dir(cli_real(name, K, 0, save=True))
    assert (rd / "W_enc.npy").exists(), f"{rd.name} has no W_enc.npy (trained without --save_params)"
    scale = json.load(open(rd / "train.json"))["input_scale"]
    pw = power_pass(rd, acts_batches(name, scale))
    empty_cache()
    W = pw["W_dec"].astype(np.float64)
    norm = np.linalg.norm(W, axis=1)
    D = W / norm[:, None]
    n = pw["n"]
    freq, mean_act, msq = pw["fires"] / n, pw["s1"] / n, pw["s2"] / n
    pi = msq * norm ** 2
    live = pw["fires"] > 0
    live_eval = np.load(rd / "fires.npy") > 0
    rq = np.concatenate([((D[i:i + 4096] @ sig.cov) * D[i:i + 4096]).sum(1) for i in range(0, len(D), 4096)])
    rq /= np.trace(sig.cov) / D.shape[1]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, fires=pw["fires"], freq=freq, mean_act=mean_act, mean_sq_act=msq, pi=pi, rayleigh=rq,
                        dec_norm=norm, live=live, live_eval=live_eval, n_tokens=n)
    os.replace(tmp, p)
    lz = pg.lorenz(pi[live])
    row = {"job": JOB, "run": f"{rd.name}__power", **base_meta(name, "power", "real"), "sae_run": rd.name,
           "sigma_label": sig.label, "git_hash": rc.git_hash(), "code_sha256": rc.code_sha256(),
           "power_file": rel(p), "power_n_tokens": n, "power_fvu": pw["fvu"], "power_l0": pw["l0"],
           "n_live": int(live.sum()), "n_live_eval": int(live_eval.sum()), "power": lz,
           "sum_pi_over_total_var": float(pi[live].sum() / pw["total_var"]),
           "rq_median_log10": float(np.median(np.log10(np.clip(rq[live], 1e-12, None)))),
           "spearman_pi_rq": pg.spearman(pi[live], rq[live]), "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rc.write_row(OUT, JOB, row)
    log(f"{name}: power pass of {rd.name}: {int(live.sum())} live, FVU {pw['fvu']:.3f}, top-10 share "
        f"{lz['share_top10']:.3f}, top-1% {lz['share_top1pct']:.3f}, ESS fraction {lz['ess_frac']:.4f}")
    return p


_DEV = []


def dict_device():
    if not _DEV:
        from r2_job_pmatch import build_device
        _DEV.append(build_device())
    return _DEV[0]


def ensure_dict(name, sig):
    """r2_job_pmatch.ensure_dict with this job's paths, dataset-indexed seed and power file: V1 dictionary (powers from
    step c, directions as V0's, A^T A = Sigma exactly), report with both atom clouds' gamma."""
    from train_sae import sparse_dictionary
    t = dict_tag(name)
    path, rep = DICT_DIR / f"{t}.npz", OUT / "ladder_pmatch" / f"{t}.json"
    if path.exists() and rep.exists():
        return path
    src = power_path(name)
    z = np.load(src)
    lv = z["live"].astype(bool)
    pi, rq = z["pi"][lv], z["rayleigh"][lv]
    seed = DICT_SEED + ORDER.index(name)
    p_share, rq_t = pg.targets(pi, rq, SP_M)
    A, info, p_ach, rq_ach = pg.build(sig.cov, "V1", p_share, rq_t, seed, max_iter=DICT_MAX_ITER, tol=DICT_TOL,
                                      rq_tol=DICT_RQ_TOL, dev=dict_device(), log=log, max_seconds=DICT_SECONDS)
    assert info["cov_rel_frob_err_float64"] < 1e-6, f"{t}: A^T A differs from Sigma ({info['cov_rel_frob_err_float64']:.2e})"
    W = (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float32)
    cs = rc._cloud_stats(W, np.ones(len(W), bool), 0, SP_DS, N_FIX, sig.slopes)
    A0 = sparse_dictionary(sig.cov, SP_M, SP_DS).astype(np.float64)  # the equal-power dictionary V0 trains on
    cs0 = rc._cloud_stats((A0 / np.linalg.norm(A0, axis=1, keepdims=True)).astype(np.float32), np.ones(SP_M, bool), 0,
                          SP_DS, N_FIX, sig.slopes)
    del A0
    info |= {"v0_atom_cloud_gamma_fixed": cs0["gamma_fixed"], "tag": t, "rung": name, "dataset": label(name),
             "dict_seed": seed, "real_power_file": rel(src),
             "real_power_file_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "real_n_live": int(len(pi)),
             "real_power": pg.lorenz(pi), "real_rq_log10_quantiles": {str(q): float(np.quantile(np.log10(rq), q))
                                                                      for q in (.01, .1, .5, .9, .99)},
             "real_spearman_power_rq": pg.spearman(pi, rq),
             "atom_cloud_gamma_fixed": cs["gamma_fixed"], "atom_cloud_gamma_full": cs["gamma_full"],
             "generator_sha256": hashlib.sha256((ROOT / "experiments" / "r2_pmatch_gen.py").read_bytes()).hexdigest(),
             "job_code_sha256": job_sha(), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    DICT_DIR.mkdir(parents=True, exist_ok=True)
    rep.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, A=A.astype(np.float32))
    os.replace(tmp, path)
    info["dict_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez_compressed(rep.parent / f"{t}_atoms.npz", power=p_ach, rq=rq_ach, target_share=p_share, target_rq=rq_t)
    tmp = rep.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=1, default=float))
    os.replace(tmp, rep)
    g = " / ".join("%.3f" % cs["gamma_fixed"][str(w)] for w in rc.WINDOWS)
    log(f"{t}: atom-cloud gamma {g}, top-10 power share {info['power']['share_top10']:.3f} "
        f"(real {info['real_power']['share_top10']:.3f}), moved {info['power_share_moved_by_projection']:.3f}")
    return path


class DictJob:
    """ensure_dict in a background thread while the V0 nulls train; wait() returns the path or raises its error."""

    def __init__(self, name, sig):
        self.res, self.err = None, None
        self.th = threading.Thread(target=self._run, args=(name, sig), daemon=True)
        self.th.start()

    def _run(self, name, sig):
        try:
            self.res = ensure_dict(name, sig)
        except Exception as e:
            self.err = e
            log(f"{name}: dictionary build failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    def wait(self):
        self.th.join()
        if self.err is not None:
            raise self.err
        return self.res


def next_rho(pts, T):
    """Secant step through the bracketing pair of tried (rho, F) points, else the two nearest to T."""
    pts = sorted(pts.items())
    below = [p for p in pts if p[1] < T]
    above = [p for p in pts if p[1] >= T]
    if below and above:
        a, b = max(below, key=lambda p: p[1]), min(above, key=lambda p: p[1])
    else:
        a, b = sorted(pts, key=lambda p: abs(p[1] - T))[:2]
    if abs(b[1] - a[1]) < 1e-6:
        return None
    return round(min(max(a[0] + (T - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), 0.0), RHO_MAX), 2)


def calibrate(trainf, is_trained, T, rho1, slope, max_runs):
    """The rule of steps d and e. trainf(rho) -> FVU (trains if needed). Returns (rho, {rho: FVU}, matched, complete);
    complete is False when the budget stopped it. Raises BudgetSpent if not even the first run could start."""
    tried, rho, complete = {}, round(rho1, 2), True
    while len(tried) < max_runs and rho is not None and rho not in tried:
        if not is_trained(rho) and not can_start():
            log(f"budget: not starting rho {rho:.2f} ({minutes():.0f} min)")
            complete = False
            break
        F = tried[rho] = trainf(rho)
        log(f"calibrate: rho {rho:.2f} -> FVU {F:.4f} (target {T:.4f})")
        if abs(F - T) <= TOL:
            return rho, tried, True, True
        if rho == 0.0 and F > T + TOL:
            break  # rho = 0 is the most codable this null gets
        if len(tried) == 1:
            nxt = round(min(max(rho + (T - F) / slope, 0.0), RHO_MAX), 2)
            rho = nxt if nxt != rho else round(min(max(rho + (0.01 if T > F else -0.01), 0.0), RHO_MAX), 2)
        else:
            rho = next_rho(tried, T)
    if not tried:
        raise BudgetSpent(f"budget spent before the first calibration run ({minutes():.0f} min)")
    best = min(tried, key=lambda r: abs(tried[r] - T))
    return best, tried, abs(tried[best] - T) <= TOL, complete


def step_d(name, scorer, sig, T):
    rho1 = 0.0 if T < LOW_FVU else V0_RHO1
    runs = {}

    def trainf(rho):
        runs[rho] = train(cli_sparse(name, 0, rho))
        return runs[rho][0]
    rho, tried, matched, complete = calibrate(trainf, lambda r: trained(cli_sparse(name, 0, r)), T, rho1,
                                              DEFAULT_SLOPE, V0_MAX)
    entry = {"rung": name, "dataset": label(name), "null": "V0", "rho": rho, "target": T, "matched": matched,
             "complete": complete, "rho1": rho1, "slope": DEFAULT_SLOPE,
             "tried": {f"{r:.2f}": f for r, f in sorted(tried.items())}}
    save_cal(f"{name}|V0", entry)
    log(f"{name} V0: rho {rho:.2f}, FVU {tried[rho]:.3f}, target {T:.3f}, {'matched' if matched else 'UNMATCHED'}"
        f"{'' if complete else ' (calibration cut by the budget)'}")
    if not matched:
        note(name, f"V0 calibration unmatched: best rho {rho:.2f}, FVU {tried[rho]:.3f}, target {T:.3f}")
    meta = dict(setting={"M": SP_M, "L": SP_L, "values": VALUES, "ds": SP_DS, "dictionary": "R Sigma^1/2"},
                fvu_target=T, matched=matched, calibration_complete=complete, null="V0")
    for r in sorted(tried):
        submit(scorer, name, runs[r][1], sig, base_meta(name, "v0" if r == rho else "v0_cal", "sparse2", rho=r,
                                                         accepted=r == rho, calibration=r != rho, **meta))
    _, rd1 = train(cli_sparse(name, 1, rho))
    submit(scorer, name, rd1, sig, base_meta(name, "v0", "sparse2", rho=rho, accepted=True, calibration=False, **meta))
    return entry


def v0_slope(v0):
    """dF/drho from V0's two tried points nearest its accepted rho (else the default)."""
    pts = {float(r): f for r, f in v0["tried"].items()}
    if len(pts) < 2:
        return DEFAULT_SLOPE
    acc = v0["rho"]
    a, b = sorted(pts, key=lambda r: abs(r - acc))[:2]
    s = (pts[b] - pts[a]) / (b - a)
    return float(s) if np.isfinite(s) and s > 0.05 else DEFAULT_SLOPE


def step_e(name, scorer, sig, T, v0, dj):
    dpath = dj.wait()
    rep = json.loads((OUT / "ladder_pmatch" / f"{dict_tag(name)}.json").read_text())
    slope, runs = v0_slope(v0), {}

    def trainf(rho):
        runs[rho] = train(cli_sparse(name, 0, rho, dpath))
        return runs[rho][0]
    rho, tried, matched, complete = calibrate(trainf, lambda r: trained(cli_sparse(name, 0, r, dpath)), T, v0["rho"],
                                              slope, V1_MAX)
    save_cal(f"{name}|V1", {"rung": name, "dataset": label(name), "null": "V1", "rho": rho, "target": T,
                            "matched": matched, "complete": complete, "v0_rho": v0["rho"], "v0_slope": slope,
                            "dict_tag": dict_tag(name), "tried": {f"{r:.2f}": f for r, f in sorted(tried.items())}})
    log(f"{name} V1: rho {rho:.2f}, FVU {tried[rho]:.3f}, target {T:.3f}, {'matched' if matched else 'UNMATCHED'}")
    if not matched:
        note(name, f"V1 calibration unmatched: best rho {rho:.2f}, FVU {tried[rho]:.3f}, target {T:.3f}")
    dic = {k: rep.get(k) for k in ("power", "power_target", "rq_log10_quantiles", "spearman_power_rq",
                                   "power_share_moved_by_projection", "majorization_gap_before_projection",
                                   "cov_rel_frob_err_float64", "cov_rel_frob_err_float32", "atom_cloud_gamma_fixed",
                                   "real_power", "real_power_file_sha256", "dict_file_sha256", "iterations",
                                   "converged", "final_power_max_abs_log_err", "dict_seed", "v0_atom_cloud_gamma_fixed")}
    meta = dict(setting={"M": SP_M, "L": SP_L, "values": VALUES, "ds": SP_DS, "dict_tag": dict_tag(name)},
                fvu_target=T, matched=matched, calibration_complete=complete, null="V1", variant="V1",
                variant_label=pg.VARIANTS["V1"], dictionary=dic, v0_rho=v0["rho"],
                generator_sha256=rep.get("generator_sha256"))
    for r in sorted(tried):
        rd = runs[r][1]
        if done(rd.name):
            continue
        m, arrays = base_meta(name, "v1" if r == rho else "v1_cal", "sparse_pmatch", rho=r, accepted=r == rho,
                              calibration=r != rho, **meta), None
        if r == rho:
            from r2_job_pmatch import power_check
            from r2_job_robust import power_pass
            z = np.load(power_path(name))
            scale = json.load(open(rd / "train.json"))["input_scale"]
            pw = power_pass(rd, null_batches(sig, r, dpath, scale))
            empty_cache()
            m["power_check"], arrays = power_check(pw, sig, z["pi"][z["live"].astype(bool)])
            pc = m["power_check"]
            log(f"{rd.name}: null SAE top-1% power share {pc['null']['share_top1pct']:.3f} (real "
                f"{pc['real']['share_top1pct']:.3f}), ESS fraction {pc['null']['ess_frac']:.4f} (real "
                f"{pc['real']['ess_frac']:.4f})")
        scorer.submit(score_safe, name, rd, sig, m, arrays)


def null_batches(sig, rho, dpath, scale):
    """N_POWER fresh tokens from the V1 generator (stream POWER_SEED + 1000 ds), as r2_job_pmatch.null_batches."""
    from r2_job_robust import device
    from train_sae import sparse_batches, sparse_setup
    dev = device()
    sp = sparse_setup(sig.cov, sig.mean, SP_M, SP_L, SP_DS, dev, VALUES, rho, dict_path=dpath)
    return sparse_batches(sp, N_POWER, 8192, POWER_SEED + 1000 * SP_DS, scale, dev)


def step_f(name, scorer, sig):
    for k in EXTRA_K:
        _, rd = train(cli_real(name, k, 0))
        submit(scorer, name, rd, sig, base_meta(name, "real_k", "real"))


# ---------------------------------------------------------------------------------------------------- driver ---
def estimate(name):
    """(expected, worst) SAEs still to train and minutes, from the documented ~8 min per SAE."""
    real = sum(not trained(c) and not done(run_dir(c).name) for c in real_clis(name))
    v0c, v1c = load_cal(f"{name}|V0"), load_cal(f"{name}|V1")
    v0 = (0, 0) if v0c and v0c.get("complete") else (3, V0_MAX + 1)
    v1 = (0, 0) if v1c and v1c.get("complete") else (1.5, V1_MAX)
    e, w = real + v0[0] + v1[0], real + v0[1] + v1[1]
    col = COLLECT_MIN if not (act_dir(name) / "acts.npy").exists() and needs_acts(name) else 0
    oh = OVERHEAD_MIN if w else 0
    return e, w, e * MIN_PER_SAE + col + oh, w * MIN_PER_SAE + col + oh


def do_dataset(name, scorer):
    st = STATUS.setdefault(name, {})
    st.update({"started_min": round(minutes(), 1), "steps": {}, "label": label(name), "description":
               DATASETS[name]["desc"], "act_dir": rel(act_dir(name))})
    st.pop("score_errors", None)
    st.pop("notes", None)
    save_status()
    e, w, me, mw = estimate(name)
    log(f"=== {name}: {DATASETS[name]['desc']}; about {e:g} SAEs (worst {w}) x {MIN_PER_SAE:.0f} min: "
        f"{me:.0f} min expected, {mw:.0f} worst ({minutes():.0f} min into the job, budget {BUDGET_MIN:.0f})")
    res = {}

    def run(key, fn, *needs):
        miss = [n for n in needs if n not in res]
        if miss:
            st["steps"][key] = f"skipped (needs {', '.join(miss)})"
            save_status()
            return
        t = time.time()
        try:
            res[key] = fn()
            st["steps"][key] = "done"
        except BudgetSpent as ex:
            st["steps"][key] = f"budget: {ex}"
            log(f"{name} step {key}: {ex}")
        except Exception as ex:
            st["steps"][key] = f"failed: {type(ex).__name__}: {ex}"
            log(f"{name} step {key} FAILED: {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
        st.setdefault("step_minutes", {})[key] = round((time.time() - t) / 60, 2)
        save_status()

    try:
        run("a_collect_sigma", lambda: step_a(name))
        sig = res.get("a_collect_sigma")
        run("b_real", lambda: step_b(name, scorer, sig), "a_collect_sigma")
        run("c_power", lambda: step_c(name, sig), "b_real")
        dj = DictJob(name, sig) if "c_power" in res else None
        run("d_v0", lambda: step_d(name, scorer, sig, res["b_real"]), "b_real")
        run("e_v1", lambda: step_e(name, scorer, sig, res["b_real"], res["d_v0"], dj), "c_power", "d_v0")
        if DATASETS[name]["extra_k"]:
            run("f_extra_k", lambda: step_f(name, scorer, sig), "a_collect_sigma")
        if dj is not None:
            dj.th.join()  # never leave a build running past the delete
    finally:
        scorer.finish()
        try:
            drop_acts(name)
            st["steps"]["g_delete_acts"] = "done"
        except Exception as ex:
            st["steps"]["g_delete_acts"] = f"failed: {type(ex).__name__}: {ex}"
        st["finished_min"] = round(minutes(), 1)
        st["ok"] = all(v == "done" for v in st["steps"].values()) and not st.get("score_errors")
        save_status()
        log(f"=== {name}: {'all steps done' if st['ok'] else 'NOT all steps done'} "
            f"({json.dumps(st['steps'])}), {minutes():.0f} min into the job")


def main():
    global BUDGET_MIN, STATUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None, choices=ORDER, help="subset, run in the given order")
    ap.add_argument("--budget_min", type=float, default=None, help="no SAE starts after this many minutes (480)")
    ap.add_argument("--collect", default=None, help=argparse.SUPPRESS)  # collection worker
    a = ap.parse_args()
    if a.smoke:
        smoke_config()
    else:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if a.budget_min is not None:
        BUDGET_MIN = a.budget_min
    OUT.mkdir(parents=True, exist_ok=True)
    if a.collect is not None:
        collect(a.collect)
        return
    names = a.datasets or ORDER
    log(f"job {JOB}: git {rc.git_hash()}, code sha256 {rc.code_sha256()[:12]}, job sha256 {job_sha()[:12]}, out {OUT}, "
        f"{rc.disk_free_gb():.0f} GB free, budget {BUDGET_MIN:.0f} min, datasets {names}")
    if not SMOKE:
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(HF_MODEL)
        except Exception as e:
            log(f"WARNING no access to {HF_MODEL} ({type(e).__name__}: {e}); collections will fail (set HF_TOKEN)")
    f = OUT / "ladder_status.json"
    STATUS = json.load(open(f)) if f.exists() else {}
    tot = [estimate(n) for n in names]
    log(f"estimate: {sum(t[0] for t in tot):g} SAEs expected (worst {sum(t[1] for t in tot)}), "
        f"{sum(t[2] for t in tot) / 60:.1f} h expected, {sum(t[3] for t in tot) / 60:.1f} h worst")
    scorer = rc.Scorer()
    for n in names:
        try:
            do_dataset(n, scorer)
        except Exception as e:  # anything do_dataset did not catch must not stop the next dataset
            log(f"{n}: FAILED outside a step: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            STATUS.setdefault(n, {})["error"] = f"{type(e).__name__}: {e}"
            save_status()
    scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB; status {json.dumps({n: STATUS.get(n, {}).get('ok') for n in names})}")
    try:
        py(ROOT / "experiments" / "r2_ladder_summary.py", "--dir", OUT)
    except Exception as e:  # rerun locally after the pull; never fail the job on it
        log(f"r2_ladder_summary failed ({e}); rerun locally")
    log(f"JOB DONE ladder ({minutes():.0f} min)")


if __name__ == "__main__":
    main()
