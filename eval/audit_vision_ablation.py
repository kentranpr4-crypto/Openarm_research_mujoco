#!/usr/bin/env python3
"""Inference-only vision ablation (no retraining, no rendering): does the ACT policy actually USE its camera images?
Same seeded configs as evaluate.py; images fed to the policy are replaced by
  zero  : black images
  mean  : per-pixel mean over all training-demo frames
  swap  : time-matched frames from a DIFFERENT training demo (natural images, wrong box pose)
  real  : control, true images are NOT available here (no renderer) -> use eval/evaluate.py results for the real-image baseline.
Run (ACT venv): MUJOCO_GL=osmesa python eval/audit_vision_ablation.py --ckpt CKPT --demos outputs/datasets/demo_train_clipped --out OUT.json"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--demos", type=Path, required=True)
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--n", type=int, default=10); ap.add_argument("--conditions", nargs="+", default=["nominal", "unseen_pose"])
ap.add_argument("--modes", nargs="+", default=["zero", "mean", "swap"]); ap.add_argument("--horizon", type=int, default=200); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
from lab_env import LabEnv
from policies.act_infer import ACTRunner
from policies.baseline import sample_box_pose
from eval.metrics import task_labels
import lab_scene
env = LabEnv(cameras=()); runner = ACTRunner(a.ckpt, env.adapter, threads=2); cams = runner.cams
demos = sorted(a.demos.glob("episode_*.npz")); meta = json.loads((a.demos / "meta.json").read_text()); H, W = meta["image_hw"]
imgs = {c: [] for c in cams}
for f in demos:
    with np.load(f) as z:
        for c in cams: imgs[c].append(z[f"observation.images.{c}"])
mean_img = {c: np.concatenate(imgs[c]).mean(0).astype(np.uint8) for c in cams}
REGION = {"nominal": "train", "unseen_pose": "unseen"}; out = {}
for cond in a.conditions:
    for mode in a.modes:
        recs = []
        for idx in range(a.n):
            rng = np.random.default_rng(a.seed * 100000 + idx); x, y, yaw = sample_box_pose(rng, REGION[cond])
            o = env.reset((x, y), yaw); runner.reset(env.prev_ctrl); other = (idx * 7 + 3) % len(demos); B, T, BS = [], [], []
            for t in range(a.horizon):
                i = env.info(); B.append(i["box_pos"]); T.append(i["tcp_pos"]); BS.append(i["box_speed"])
                if mode == "zero": im = {c: np.zeros((H, W, 3), np.uint8) for c in cams}
                elif mode == "mean": im = mean_img
                else: im = {c: imgs[c][other][min(t, len(imgs[c][other]) - 1)] for c in cams}
                o = env.step(runner.act(o["state"], im))
            i = env.info(); B.append(i["box_pos"]); T.append(i["tcp_pos"]); BS.append(i["box_speed"])
            lab = task_labels(B, T, BS, lab_scene.TRAY_C, lab_scene.TABLE_TOP_Z + lab_scene.TRAY_BASE_T)
            recs.append(dict(idx=idx, box_xy=[x, y], yaw=yaw, **lab)); print(cond, mode, idx, lab["task_success"], flush=True)
        out[f"{cond}/{mode}"] = dict(n=len(recs), task=sum(r["task_success"] for r in recs), pick=sum(r["pick_success"] for r in recs), episodes=recs)
        print("==", cond, mode, out[f"{cond}/{mode}"]["task"], "/", len(recs), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()}, results=out), indent=1))
