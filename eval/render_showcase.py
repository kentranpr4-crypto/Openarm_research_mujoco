#!/usr/bin/env python3
"""Record a presentation video (overview camera, studio look: shadows + reflection) of ONE rollout by the expert or ACT.
The policy still sees 'fast'-profile images (what it was trained on); only the overview render flips shadows on.
Usage (ACT venv): python eval/render_showcase.py --agent act --ckpt CKPT --seed 7 --idx 0 --out outputs/showcase/act.mp4"""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import sys, argparse, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
ap = argparse.ArgumentParser(); ap.add_argument("--agent", choices=["expert", "act"], required=True); ap.add_argument("--ckpt", type=Path)
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--idx", type=int, default=0); ap.add_argument("--region", default="train")
ap.add_argument("--horizon", type=int, default=200); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
from lab_env import LabEnv
from cameras import CameraRig, OVERVIEW
from policies.baseline import ScriptedExpert, sample_box_pose
from eval.metrics import task_labels
import lab_scene, imageio.v2 as iio
env_cams = ("front", "wrist_left") if a.agent == "act" else ()
env = LabEnv(cameras=env_cams); m = env.m
rig = CameraRig(m, 640, 480)
shadow0, refl0 = m.light_castshadow.copy(), m.mat_reflectance.copy()
def studio_overview():
    m.light_castshadow[:] = [1, 0, 0]; m.mat_reflectance[:] = refl_studio
    img = rig.rgb(env.d, OVERVIEW); m.light_castshadow[:] = shadow0; m.mat_reflectance[:] = refl0; return img
refl_studio = lab_scene.compile_lab()[0].mat_reflectance.copy()
rng = np.random.default_rng(a.seed * 100000 + a.idx); x, y, yaw = sample_box_pose(rng, a.region)
o = env.reset((x, y), yaw); frames, B, T, S = [], [], [], []
if a.agent == "expert": plan = ScriptedExpert(m).plan((x, y), yaw); N = len(plan["ctrl"])
else:
    from policies.act_infer import ACTRunner
    runner = ACTRunner(a.ckpt, env.adapter); runner.reset(env.prev_ctrl); N = a.horizon
for t in range(N):
    i = env.info(); B.append(i["box_pos"]); T.append(i["tcp_pos"]); S.append(i["box_speed"]); frames.append(studio_overview())
    ctrl = plan["ctrl"][t] if a.agent == "expert" else runner.act(o["state"], o["images"]); o = env.step(ctrl)
i = env.info(); B.append(i["box_pos"]); T.append(i["tcp_pos"]); S.append(i["box_speed"])
lab = task_labels(B, T, S, lab_scene.TRAY_C, lab_scene.TABLE_TOP_Z + lab_scene.TRAY_BASE_T)
try: iio.mimsave(a.out, frames, fps=env.hz, macro_block_size=1)
except TypeError: iio.mimsave(a.out, frames, fps=env.hz)
print(a.agent, "box", (round(x, 3), round(y, 3), round(yaw, 2)), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in lab.items()})
