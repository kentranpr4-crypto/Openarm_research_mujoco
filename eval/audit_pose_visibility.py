#!/usr/bin/env python3
"""Where does the blue box appear in the policy's cameras at t=0 (home pose)? Uses MuJoCo segmentation rendering (ground truth,
diagnostics only). For every train-demo pose and every evaluation pose: pixel centroid (u,v), pixel area and whether it is visible,
in `front` and `wrist_left`. Usage: MUJOCO_GL=osmesa sim/.venv/bin/python eval/audit_pose_visibility.py --out outputs/eval/audit_v1/pose_visibility.json"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, numpy as np, mujoco
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--train-demos", type=Path, default=ROOT / "outputs/datasets/demo_train_clipped")
ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9]); ap.add_argument("--n", type=int, default=10); a = ap.parse_args()
from lab_env import LabEnv
from cameras import CameraRig
from policies.baseline import sample_box_pose
env = LabEnv(cameras=()); m, d = env.m, env.d
rig = CameraRig(m, 160, 120); box_geoms = {m.geom(n).id for n in ("blue_box_collision", "blue_box_visual")}
def probe(x, y, yaw):
    env.reset((x, y), yaw); out = {}
    for cam in ("front", "wrist_left"):
        seg = rig.segmentation(d, cam); ids, types = seg[..., 0], seg[..., 1]
        mask = (types == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(ids, list(box_geoms))
        v, u = np.nonzero(mask); out[cam] = dict(area=int(mask.sum()), u=float(u.mean()) if len(u) else None, v=float(v.mean()) if len(v) else None)
    return out
rows = []
for e in map(json.loads, (a.train_demos / "episodes.jsonl").read_text().splitlines()):
    rows.append(dict(set="train", seed=None, idx=e["episode_index"], box_xy=e["box_xy"], yaw=e["box_yaw"], **probe(*e["box_xy"], e["box_yaw"])))
for s in a.seeds:
    for cond, region in (("nominal", "train"), ("unseen_pose", "unseen")):
        for idx in range(a.n):
            x, y, yaw = sample_box_pose(np.random.default_rng(s * 100000 + idx), region)
            rows.append(dict(set=cond, seed=s, idx=idx, box_xy=[x, y], yaw=yaw, **probe(x, y, yaw)))
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rows, indent=1)); print(len(rows), "poses ->", a.out)
