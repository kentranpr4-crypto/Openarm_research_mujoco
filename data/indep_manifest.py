#!/usr/bin/env python3
"""INDEPENDENT held-out evaluation: manifest of NEW initial-state groups (no simulation here).  50 groups x 25 registered action variants = 1 250 episodes (cap).
Poses are drawn from a separate RNG namespace (eval_seed_label 7001, rng [20261101, 7001]) per x-stratum with the SAME ranges as the development intervention manifest (data/wm1_intervene_full.py); the draw does not
look at any model, prediction or result.  A pose is rejected if its Chebyshev distance (x[m], y[m], yaw[rad]) to ANY previously used initial state is < MIN_DIST: all groups of outputs/wm2/dataset, all poses of the WM-1 pilot /
intervention manifests, and every box_xy/box_yaw of any ACT/sweep episodes.jsonl under outputs/ (xsweep, coverage, e0, ...).  Forbidden seeds 31-33 are never used.  Variants are the 25 development variants, unchanged.
Run (any python with numpy): python data/indep_manifest.py"""
import sys, json, hashlib, glob, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/indep"; MIN_DIST = 0.003; EVAL_SEED_LABEL = 7001; RNG_KEY = [20261101, 7001]
STRATA = [("test_id", "TRAIN", (0.38, 0.45), 18), ("test_low_heldout", "LOW_A", (0.36, 0.38), 10), ("test_ood_low_strict", "LOW_B_strict", (0.34, 0.36), 10), ("test_ood_high", "HIGH", (0.4501, 0.48), 12)]
Y_RANGE, YAW_RANGE = (0.07, 0.13), (-0.20, 0.20)
def used_states():
    S = {}
    def add(src, x, y, yaw): S.setdefault(src, []).append((float(x), float(y), float(yaw)))
    for e in json.loads((ROOT / "outputs/wm2/dataset/index.json").read_text()): add("wm2_dataset", *e["group"])
    for p in json.loads((ROOT / "outputs/wm1/intervention_full/manifest.json").read_text())["poses"]: add("wm1_intervention_manifest", p["x"], p["y"], p["yaw"])
    for f in sorted(glob.glob(str(ROOT / "outputs/**/episodes.jsonl"), recursive=True)):
        for ln in open(f):
            try: r = json.loads(ln)
            except Exception: continue
            if "box_xy" in r and "box_yaw" in r: add("act_sweep_episodes", r["box_xy"][0], r["box_xy"][1], r["box_yaw"])
    return S
def build():
    used = used_states(); allu = np.array([s for v in used.values() for s in v]); rng = np.random.default_rng(RNG_KEY); poses = []; pid = 1000; rejected = 0
    for split, region, xr, n in STRATA:
        got = 0
        while got < n:
            p = np.array([rng.uniform(*xr), rng.uniform(*Y_RANGE), rng.uniform(*YAW_RANGE)]); p = np.round(p, 6)
            cur = np.array([[q["x"], q["y"], q["yaw"]] for q in poses]) if poses else np.zeros((0, 3))
            if np.abs(allu - p).max(1).min() < MIN_DIST or (len(cur) and np.abs(cur - p).max(1).min() < MIN_DIST): rejected += 1; continue
            poses.append(dict(pid=pid, region=region, split=split, x=float(p[0]), y=float(p[1]), yaw=float(p[2]))); pid += 1; got += 1
    import importlib.util
    spec = importlib.util.spec_from_file_location("wm1_intervene_vars", ROOT / "data/wm1_intervene.py"); txt = (ROOT / "data/wm1_intervene.py").read_text()
    # variants are read from the development module source without importing simulator code
    ns = {}; start = txt.index("VARIANTS = [dict(vid=0"); end = txt.index("assert len(VARIANTS)"); exec("import numpy as np\n" + txt[start:end], ns); variants = ns["VARIANTS"]
    assert len(variants) == 25 and [v["vid"] for v in variants] == list(range(25))
    P = np.array([[q["x"], q["y"], q["yaw"]] for q in poses]); mind = {k: float(np.abs(np.array(v)[:, None, :] - P[None]).max(2).min()) for k, v in used.items()}
    return dict(name="Independent held-out evaluation manifest", status="INDEPENDENT held-out set (never used for training, validation, model selection, gate design or normalisation)", n_groups=len(poses), n_variants=25, n_episodes=len(poses) * 25, cap_episodes=1250,
                eval_seed_label=EVAL_SEED_LABEL, rng_key=RNG_KEY, forbidden_seeds=[31, 32, 33], min_chebyshev_distance=MIN_DIST, y_range=Y_RANGE, yaw_range=YAW_RANGE, strata=[dict(split=s, region=r, x_range=x, n=n) for s, r, x, n in STRATA], poses=poses, variants=variants,
                limits="control/safety_limits.SafetyLimits (ctrlrange; 2.0 rad/s arm; 0.2 m/s gripper), same clip_and_slew as the ACT runner and the development collection",
                checks=dict(rejected_candidates=rejected, previously_used_states={k: len(v) for k, v in used.items()}, min_chebyshev_to_each_source=mind, min_chebyshev_overall=min(mind.values()), exact_collisions=int(sum(v == 0.0 for v in mind.values())),
                            pose_selection_independent_of_models=True, split_counts={s: sum(1 for p in poses if p["split"] == s) for s, *_ in STRATA}))
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / "manifest.json"
    if f.exists(): sys.exit(f"{f} exists (frozen)")
    m = build(); f.write_text(json.dumps(m, indent=1)); print("manifest sha256", hashlib.sha256(f.read_bytes()).hexdigest()); print(json.dumps(m["checks"], indent=1))
