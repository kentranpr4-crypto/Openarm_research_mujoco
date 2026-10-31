#!/usr/bin/env python3
"""Coverage experiment: deterministic replay (eval/h23_replay.replay_one, unchanged) of the sweep rollouts of the NEW checkpoints (outputs/cov/eval/<branch>_seed<k>/sweep).
Expert replays are reused from outputs/xsweep/replay (same 60 poses).  Writes outputs/cov/replay/*.npz and replay_meta.json.  Run (ACT venv): python eval/cov_replay.py [--workers 4]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, numpy as np
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
import eval.h23_replay as H
OUT = ROOT / "outputs/cov/replay"; CFG = json.loads((ROOT / "outputs/xsweep/xsweep_config.json").read_text())
RUNS = [("ext_c0", 3), ("ext_s1", 3), ("vol_c0", 3), ("vol_s1", 3)]
def jobs(runs):
    J = []
    for br, k in runs:
        for p in CFG["poses"]:
            J.append(dict(agent=f"{br}_s{k}", arm=br, tseed=k, S="SW", idx=p["idx"], seed=21, box_xy=[p["x"], p["y"]], box_yaw=p["yaw"], roll=str(ROOT / f"outputs/cov/eval/{br}_seed{k}/sweep/act/sweep/rollouts/ep{p['idx']:04d}.npz")))
    return J
def run(j): H.OUT = OUT; r = H.replay_one(j); return dict(agent=j["agent"], idx=j["idx"], status=r[3], err=r[4])
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--runs", nargs="*"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    runs = [(r.rsplit("_seed", 1)[0], int(r.rsplit("_seed", 1)[1])) for r in a.runs] if a.runs else RUNS; J = jobs(runs)
    with Pool(a.workers) as p: R = p.map(run, J, chunksize=2)
    mp = OUT.parent / "replay_meta.json"; old = json.loads(mp.read_text()) if mp.exists() else dict(episodes=[], jobs=[])
    have = {(e["agent"], e["idx"]) for e in R}; old["episodes"] = [e for e in old["episodes"] if (e["agent"], e["idx"]) not in have] + R
    old["jobs"] = [j for j in old["jobs"] if (j["agent"], j["idx"]) not in have] + [{k: v for k, v in j.items() if k != "roll"} for j in J]; mp.write_text(json.dumps(old, default=float))
    e = [r["err"] for r in R if r["err"] is not None]; print(f"{len(J)} replays; box error vs live: median {np.median(e):.2e} m, max {max(e):.3f} m, >5 mm: {sum(x > 0.005 for x in e)}")
if __name__ == "__main__": main()
