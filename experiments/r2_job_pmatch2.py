"""Round-2 GPU job 7 ("pmatch2"): two follow-ups to the power-matched null of r2_job_pmatch.py, which is imported and
not changed. Rules and predictions: notes/prereg_ladder.md (written before any run of this file).

A. The power-matched null with no dense Gaussian share. At layer 13 the equal-power null with M = 32,768 atoms, L = 16
   reaches the real FVU at rho = 0 (FVU .309 against .288, robust step G), so its power-matched version can be run with
   nothing for a few latents to absorb. Setting PM|V1|L13|M32768_L16|ds0, calibrated by r2_job_pmatch's rule starting
   at rho = 0 (rho cannot go below 0: if the first FVU is above target + 0.03 the run is kept and flagged unmatched).
B. SAE seeds. Every power-matched number so far is one SAE seed. Seeds 1 and 2 are added at the accepted rho of
     PM|V1|L13|M8192_L16|ds0 (rho .43),  PM|V1|L0|M8192_L16|ds0 (rho .13),  and the setting of A;
   and seeds 1 and 2 of the equal-power null at layer 13, M 32,768, L 16, rho 0 (the comparator of A, one seed so far).
   The dictionaries of the two reference settings are rebuilt on the pod with the same generator, seed and stopping
   rule (the files of the first run were not kept); each row records the rebuilt file's sha256 and its atom report.

Everything is written under results/r2_pm2/ (rows, dec/, power_pmatch/, pmatch/ dictionary reports, sigma/), so no
file of the first run is overwritten. Restartable like the other jobs.

  bash gpu/run_job.sh experiments/r2_job_pmatch2.py 3.0 10
  python experiments/r2_job_pmatch2.py --smoke
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import r2_common as rc  # noqa: E402
import r2_job_pmatch as pm  # noqa: E402
from r2_common import log, py  # noqa: E402

JOB = "pmatch2"
OUT = ROOT / "results" / "r2_pm2"
POWER_SRC = ROOT / "results" / "r2" / "power"
BUDGET_MIN = 150.0
SEEDS_EXTRA = (1, 2)
RHO0 = ("V1", 13, 32768, 16, 0)  # part A
REFS = [(("V1", 13, 8192, 16, 0), 0.43), (("V1", 0, 8192, 16, 0), 0.13)]  # part B: setting, accepted rho of run 1
V0_RHO0 = (13, 32768, 16, 0.0)  # equal-power comparator of part A: layer, M, L, rho
RHO0_START, RHO0_SLOPE = 0.0, 0.6


def job_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def configure(smoke):
    """Point the imported job at this job's name and output directory."""
    global OUT, RHO0, REFS, V0_RHO0
    if smoke:
        pm.smoke_config()
        OUT = ROOT / "cache" / "pmatch2_smoke_out"
        RHO0 = ("V1", 13, 2048, 16, 0)
        REFS = [((v, L, 1024, l, ds), r) for (v, L, _, l, ds), r in REFS]
        V0_RHO0 = (13, 2048, 16, 0.0)
        pm.DICT_DIR = ROOT / "cache" / "pmatch2_smoke_dicts"
    pm.JOB, pm.OUT = JOB, OUT
    OUT.mkdir(parents=True, exist_ok=True)
    if not smoke:  # the real SAEs' power files are the dictionary targets
        (OUT / "power").mkdir(exist_ok=True)
        for L in (13, 0):
            src = POWER_SRC / pm.real_power_path(L).name
            assert src.exists(), f"{src} missing"
            if not pm.real_power_path(L).exists():
                shutil.copy2(src, pm.real_power_path(L))


def pm_name(layer, M, L, ds, rho, t, seed):  # train_sae.py's naming
    return (f"{pm.MODEL}__{pm.hook(layer)}__sparse__w{pm.WIDTH}__k{pm.K}__s{seed}__ds{ds}__M{M}__L{L:g}__{pm.VALUES}"
            f"__rho{rho:.4f}" + (f"__D{t}" if t else ""))


def train(layer, M, L, ds, rho, seed, dpath=None, t=None):
    """One null SAE (power-matched when dpath is given, equal-power otherwise); returns (eval FVU, run dir)."""
    rd = ROOT / "cache" / pm.GROUP / pm_name(layer, M, L, ds, rho, t, seed)
    if not (rd / "train.json").exists():
        log(f"train {rd.name} ({pm.minutes():.0f} min into the job)")
        py(ROOT / "train_sae.py", "--model", pm.MODEL, "--hook", pm.hook(layer), "--cond", "sparse", "--width", pm.WIDTH,
           "--k", pm.K, "--tokens", pm.TOKENS - pm.EVAL, "--eval_tokens", pm.EVAL, "--epochs", pm.EPOCHS, "--seed", seed,
           "--out", f"cache/{pm.GROUP}", "--sparse_M", M, "--sparse_L", L, "--data_seed", ds,
           "--sparse_values", pm.VALUES, "--sparse_rho", rho, "--save_params",
           *(["--sparse_dict", dpath.relative_to(ROOT), "--sparse_dict_tag", t] if dpath else []), *pm.EXTRA)
    return json.load(open(rd / "train.json"))["fvu_eval"], rd


def base_meta(setting, rep, part):
    variant, layer, M, L, ds = setting
    keys = ("power", "power_target", "power_share_moved_by_projection", "cov_rel_frob_err_float64",
            "cov_rel_frob_err_float32", "atom_cloud_gamma_fixed", "v0_atom_cloud_gamma_fixed", "real_power",
            "real_power_file_sha256", "dict_file_sha256", "iterations", "converged", "final_power_max_abs_log_err",
            "dict_seed")
    return {"layer": layer, "cond": "sparse_pmatch", "variant": variant, "dataset": f"{pm.SIGMA_MODEL}_L{layer}",
            "whitened": False, "model": "gemma-2-2b", "setting_key": pm.setting_key(*setting), "part": part,
            "setting": {"M": M, "L": L, "values": pm.VALUES, "dict_tag": pm.tag(variant, layer, M, ds), "ds": ds},
            "fvu_target": pm.TARGET[layer], "dictionary": {k: rep.get(k) for k in keys},
            "job_code_sha256": job_sha(), "pmatch_job_sha256": pm.job_sha(), "generator_sha256": pm.gen_sha()}


def score_pm(scorer, sig, setting, rd, rho, fvu, dpath, meta):
    """Power check on 1M generator tokens, then the usual scoring (on the scorer thread)."""
    variant, layer, M, L, ds = setting
    if pm.done(rd.name):
        return
    meta = {**meta, "rho": rho, "matched": abs(fvu - pm.TARGET[layer]) <= pm.TOL}
    arrays = None
    if (rd / "W_enc.npy").exists():
        from r2_job_robust import power_pass
        scale = json.load(open(rd / "train.json"))["input_scale"]
        pw = power_pass(rd, pm.null_batches(sig, M, L, ds, rho, dpath, scale))
        meta["power_check"], arrays = pm.power_check(pw, sig, pm.real_power(layer)[0])
        log(f"{rd.name}: null SAE top-1% power share {meta['power_check']['null']['share_top1pct']:.3f} "
            f"(real {meta['power_check']['real']['share_top1pct']:.3f})")
    scorer.submit(pm.score_one, rd, sig, meta, arrays)


def dictionary(setting, sig):
    variant, layer, M, L, ds = setting
    t = pm.tag(variant, layer, M, ds)
    dpath = pm.ensure_dict(variant, layer, M, ds, sig, pm.build_device())
    assert dpath is not None, f"{t}: no real power file"
    return dpath, t, json.loads((OUT / "pmatch" / f"{t}.json").read_text())


def part_a(scorer, sigs, can_start):
    """rho = 0 setting, SAE seed 0, calibrated by r2_job_pmatch's rule. Returns (accepted rho, dpath, tag, report)."""
    variant, layer, M, L, ds = RHO0
    sig = sigs[layer]
    dpath, t, rep = dictionary(RHO0, sig)
    runs = {}

    def trainf(rho):
        runs[rho] = train(layer, M, L, ds, rho, 0, dpath, t)
        return runs[rho][0]
    rho, tried, matched, complete = pm.calibrate(trainf, pm.TARGET[layer], RHO0_START, RHO0_SLOPE, can_start)
    pm.save_cal(pm.setting_key(*RHO0), {"layer": layer, "M": M, "L": L, "ds": ds, "rho": rho, "target": pm.TARGET[layer],
                                        "matched": matched, "complete": complete,
                                        "tried": {f"{r:.2f}": f for r, f in sorted(tried.items())}})
    if rho is None:
        return None
    log(f"{pm.setting_key(*RHO0)}: rho {rho:.2f}, FVU {tried[rho]:.3f}, {'matched' if matched else 'UNMATCHED'}")
    meta = base_meta(RHO0, rep, "A")
    for r in sorted(tried):
        score_pm(scorer, sig, RHO0, runs[r][1], r, tried[r], dpath,
                 {**meta, "step": "PM" if r == rho else "PMCAL", "calibration_complete": complete})
    return rho, dpath, t, rep


def seeds(scorer, sig, setting, rho, dpath, t, rep, can_start, part):
    variant, layer, M, L, ds = setting
    for s in SEEDS_EXTRA:
        if not can_start():
            log(f"{pm.setting_key(*setting)} seed {s}: budget spent; not started")
            return
        fvu, rd = train(layer, M, L, ds, rho, s, dpath, t)
        score_pm(scorer, sig, setting, rd, rho, fvu, dpath, {**base_meta(setting, rep, part), "step": "PMSEED"})


def v0_seeds(scorer, sig, can_start):
    layer, M, L, rho = V0_RHO0
    for s in SEEDS_EXTRA:
        if not can_start():
            log(f"equal-power M{M} seed {s}: budget spent; not started")
            return
        fvu, rd = train(layer, M, L, 0, rho, s)
        if pm.done(rd.name):
            continue
        meta = {"layer": layer, "cond": "sparse", "dataset": f"{pm.SIGMA_MODEL}_L{layer}", "whitened": False,
                "model": "gemma-2-2b", "step": "V0SEED", "part": "B", "rho": rho,
                "setting": {"M": M, "L": L, "values": pm.VALUES, "ds": 0}, "fvu_target": pm.TARGET[layer],
                "matched": abs(fvu - pm.TARGET[layer]) <= pm.TOL, "job_code_sha256": job_sha()}
        scorer.submit(pm.score_one, rd, sig, meta, None)


def main():
    global BUDGET_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget_min", type=float, default=None)
    a = ap.parse_args()
    configure(a.smoke)
    if a.budget_min is not None:
        BUDGET_MIN = a.budget_min
    log(f"job {JOB}: git {rc.git_hash()}, job sha256 {job_sha()[:12]}, pmatch job {pm.job_sha()[:12]}, generator "
        f"{pm.gen_sha()[:12]}, out {OUT}, {rc.disk_free_gb():.0f} GB free, budget {BUDGET_MIN:.0f} min")
    sigs = {L: pm.sigma(L) for L in sorted({RHO0[1]} | {s[1] for s, _ in REFS})}
    scorer = rc.Scorer()

    def can_start():
        return pm.minutes() <= BUDGET_MIN
    a_out = part_a(scorer, sigs, can_start)
    for setting, rho in REFS:
        dpath, t, rep = dictionary(setting, sigs[setting[1]])
        seeds(scorer, sigs[setting[1]], setting, rho, dpath, t, rep, can_start, "B")
    if a_out is not None:
        rho, dpath, t, rep = a_out
        seeds(scorer, sigs[RHO0[1]], RHO0, rho, dpath, t, rep, can_start, "A")
    v0_seeds(scorer, sigs[V0_RHO0[0]], can_start)
    scorer.finish()
    log(f"results {OUT}: {rc.du_gb(OUT):.2f} GB")
    log(f"JOB DONE pmatch2 ({pm.minutes():.0f} min)")


if __name__ == "__main__":
    main()
