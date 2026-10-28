#!/usr/bin/env python3
"""Open-loop sanity check (ACT venv): predict action chunks from dataset frames and compare with the demonstrated chunks in
REAL units (rad / m). Teacher-forced - it says nothing about closed-loop success, only whether the net reproduces the data.
Usage: python policies/open_loop_check.py CKPT DATASET --episodes 0 1 2 [--stride 5]"""
import sys, json, argparse, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("ckpt", type=Path); ap.add_argument("dataset", type=Path); ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--stride", type=int, default=5); ap.add_argument("--out", type=Path); a = ap.parse_args()
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
torch.set_num_threads(2)
pol = ACTPolicy.from_pretrained(str(a.ckpt)).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=str(a.ckpt)); K = pol.config.chunk_size
fps = json.loads((a.dataset / "meta/info.json").read_text())["fps"]
ds = LeRobotDataset("local/openarm_lab_pickplace", root=a.dataset, episodes=a.episodes, delta_timestamps={"action": [i / fps for i in range(K)]})
errs, rel = [], []; names = None
with torch.no_grad():
    for i in range(0, len(ds), a.stride):
        it = ds[i]; valid = ~it["action_is_pad"]
        b = {k: (v[None] if torch.is_tensor(v) else v) for k, v in it.items() if k == "observation.state" or k.startswith("observation.images.")}
        b = {k: v for k, v in b.items() if k in pol.config.input_features}
        pred = post(pol.predict_action_chunk(pre(b))).squeeze(0).numpy(); gt = it["action"].numpy()
        e = np.abs(pred - gt)[valid.numpy()]; errs.append(e)
E=np.concatenate(errs); arm = [i for i in range(16) if i not in (7, 15)]
res = dict(n_chunks=len(errs), mae_arm_rad=float(E[:, arm].mean()), p95_arm_rad=float(np.percentile(E[:, arm], 95)), max_arm_rad=float(E[:, arm].max()),
           mae_gripper_m=float(E[:, [7, 15]].mean()), mae_left_arm_rad=float(E[:, :7].mean()), mae_right_arm_rad=float(E[:, 8:15].mean()), per_joint_mae=[round(float(x), 5) for x in E.mean(0)])
print(json.dumps(res, indent=1)); (a.out or a.ckpt / "open_loop_check.json").write_text(json.dumps(res, indent=1))
