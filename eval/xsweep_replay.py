#!/usr/bin/env python3
"""x-sweep step 2: deterministic replay (eval/h23_replay.replay_one, unchanged) of the 360 logged policy rollouts and of the scripted expert on the same 60 poses,
adding TCP-frame box pose and finger/palm contacts.  Writes outputs/xsweep/replay/*.npz and outputs/xsweep/replay_meta.json (per-episode replay-vs-live box error).
Run (ACT venv): python eval/xsweep_replay.py [--workers 4]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, numpy as np
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
import eval.h23_replay as H
OUT = ROOT / "outputs/xsweep/replay"; H.OUT = OUT; CFG = json.loads((ROOT / "outputs/xsweep/xsweep_config.json").read_text())
def jobs():
    J = []
    for arm in ("c0", "s1"):
        for s in (0, 1, 2):
            for p in CFG["poses"]:
                J.append(dict(agent=f"{arm}_s{s}", arm=arm, tseed=s, S="SW", idx=p["idx"], seed=21, box_xy=[p["x"], p["y"]], box_yaw=p["yaw"], roll=str(ROOT / f"outputs/xsweep/eval/{arm}_seed{s}/act/sweep/rollouts/ep{p['idx']:04d}.npz")))
    for p in CFG["poses"]: J.append(dict(agent="expert", arm="expert", tseed=-1, S="SW", idx=p["idx"], seed=21, box_xy=[p["x"], p["y"]], box_yaw=p["yaw"], roll=None))
    return J
def run(j):
    H.OUT = OUT; r = H.replay_one(j); return dict(agent=j["agent"], idx=j["idx"], status=r[3], err=r[4])
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); J = jobs()
    with Pool(a.workers) as p: R = p.map(run, J, chunksize=2)
    (ROOT / "outputs/xsweep/replay_meta.json").write_text(json.dumps(dict(n=len(J), episodes=R, jobs=[{k: v for k, v in j.items() if k != "roll"} for j in J]), default=float))
    e = [r["err"] for r in R if r["err"] is not None]; print(f"{len(J)} replays; box error vs live: median {np.median(e):.2e} m, max {max(e):.3f} m, >5 mm: {sum(x > 0.005 for x in e)}")
if __name__ == "__main__": main()
