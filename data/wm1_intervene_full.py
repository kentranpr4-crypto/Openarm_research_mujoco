#!/usr/bin/env python3
"""WM-2 full intervention collection (simulation only): 100 initial poses x 25 registered action variants = 2 500 trajectories (cap).  The first 10 poses are the PILOT poses (their 250 episodes already exist in
outputs/wm1/pilot and are NOT regenerated or overwritten); 90 new poses are generated here into outputs/wm1/intervention_full.  Variants, safety limits and intervention ranges are exactly those of data/wm1_intervene.py.
Pose splits follow the dataset region rule, assigned to the POSE (all 25 variants of a pose share the split): TRAIN region 54 train/7 val/7 test_id; LOW_A 6/2/4 (train/val/test_low_heldout); LOW_B strict 8 (test only); HIGH 12 (test only).
Run (ACT venv): python data/wm1_intervene_full.py [--workers 4] [--manifest-only]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, argparse, time, numpy as np
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_intervene as WI
OUT = ROOT / "outputs/wm1/intervention_full"
QUOTA = {"TRAIN": dict(train=54, val=7, test_id=7), "LOW_A": dict(train=6, val=2, test_low_heldout=4), "LOW_B_strict": dict(test_ood_low_strict=8), "HIGH": dict(test_ood_high=12)}
XR = {"TRAIN": (0.38, 0.45), "LOW_A": (0.36, 0.38), "LOW_B_strict": (0.34, 0.36), "HIGH": (0.4501, 0.48)}

def build_manifest():
    existing = {tuple(round(v, 6) for v in (e["group"][0], e["group"][1], e["group"][2])) for e in json.loads((ROOT / "outputs/wm1/dataset/index.json").read_text())}
    ex = np.array(sorted(existing)); pilot = WI.POSES; rng = np.random.default_rng(60); poses = [dict(p) for p in pilot]; pid = len(pilot)
    pilot_counts = {r: {} for r in QUOTA}
    for p in pilot: pilot_counts[p["region"]][p["split"]] = pilot_counts[p["region"]].get(p["split"], 0) + 1
    for reg, q in QUOTA.items():
        need = {s: n - pilot_counts[reg].get(s, 0) for s, n in q.items()}; assert all(v >= 0 for v in need.values()), (reg, need)
        N = sum(need.values()); P = []
        while len(P) < N:
            x = float(rng.uniform(*XR[reg])); y = float(rng.uniform(0.07, 0.13)); yaw = float(rng.uniform(-0.20, 0.20)); key = (round(x, 6), round(y, 6), round(yaw, 6))
            if key in existing or any(abs(np.array(key) - np.array([p["x"], p["y"], p["yaw"]])).max() < 1e-4 for p in poses + P): continue
            P.append(dict(x=key[0], y=key[1], yaw=key[2]))
        P.sort(key=lambda p: hashlib.sha1(f"{p['x']}|{p['y']}|{p['yaw']}|wm2".encode()).hexdigest()); k = 0
        for s, n in need.items():
            for _ in range(n): poses.append(dict(pid=pid, region=reg, split=s, **P[k])); pid += 1; k += 1
    assert len(poses) == 100 and len(WI.VARIANTS) == 25
    man = dict(name="WM-2 full intervention manifest", n_poses=100, n_variants=25, n_trajectories=2500, pilot_poses=10, pilot_dir="outputs/wm1/pilot (reused, never overwritten)", new_dir="outputs/wm1/intervention_full", eval_seed_label=60, forbidden_seeds=[31, 32, 33],
               quota=QUOTA, poses=poses, variants=WI.VARIANTS, limits="control/safety_limits.SafetyLimits (ctrlrange; 2.0 rad/s arm; 0.2 m/s gripper), same clip_and_slew as the ACT runner",
               checks=dict(no_exact_collision_with_existing_groups=True, min_chebyshev_to_existing_groups=float(np.min(np.abs(ex[:, None, :] - np.array([[p["x"], p["y"], p["yaw"]] for p in poses])[None]).max(2))), splits_per_pose_unique=True,
                           split_counts={s: sum(1 for p in poses if p["split"] == s) for s in sorted({p["split"] for p in poses})}))
    return man

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--manifest-only", action="store_true"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    mp_ = OUT / "manifest.json"
    if mp_.exists(): man = json.loads(mp_.read_text())
    else: man = build_manifest(); mp_.write_text(json.dumps(man, indent=1))
    print("manifest sha256", hashlib.sha256(mp_.read_bytes()).hexdigest(), man["checks"]); 
    if a.manifest_only: return
    assert [v["vid"] for v in man["variants"]] == list(range(25)) and man["variants"] == json.loads(json.dumps(WI.VARIANTS))
    WI.OUT = OUT; jobs = [dict(pose=p, variant=v) for p in man["poses"] if p["pid"] >= 10 for v in man["variants"]]; assert len(jobs) == 2250; t0 = time.time()
    with Pool(a.workers) as pool: R = pool.map(WI.run_one, jobs, chunksize=4)
    old = json.loads((OUT / "index.json").read_text())["episodes"] if (OUT / "index.json").exists() else []; have = {r["name"]: r for r in old}; [have.__setitem__(r["name"], r) for r in R if r["status"] == "new"]
    (OUT / "index.json").write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(mp_.read_bytes()).hexdigest(), episodes=list(have.values())), indent=1, default=str)); print(f"{len(jobs)} jobs in {time.time()-t0:.0f}s; new {sum(r['status']=='new' for r in R)}")
if __name__ == "__main__": main()
