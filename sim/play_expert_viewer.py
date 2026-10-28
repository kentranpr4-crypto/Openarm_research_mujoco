#!/usr/bin/env python3
"""Watch the scripted expert pick the blue box and place it in the tray, live in the MuJoCo viewer (needs a display).
No policy / torch needed. Usage (from sim/):  python play_expert_viewer.py [--x 0.42 --y 0.12 --yaw 0.0] [--loop] [--realtime 1.0]
Viewer controls: drag = rotate camera, right-drag = pan, scroll = zoom, Space = pause."""
import os; os.environ.setdefault("MUJOCO_GL", "glfw")
import sys, time, argparse, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)
import mujoco, mujoco.viewer
from lab_env import LabEnv
from policies.baseline import ScriptedExpert, sample_box_pose
ap = argparse.ArgumentParser(); ap.add_argument("--x", type=float); ap.add_argument("--y", type=float); ap.add_argument("--yaw", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--loop", action="store_true"); ap.add_argument("--realtime", type=float, default=1.0)
a = ap.parse_args()
env = LabEnv(cameras=()); expert = ScriptedExpert(env.m); m, d = env.m, env.d
rng = np.random.default_rng(a.seed)
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat[:] = [0.42, 0.12, 0.42]; v.cam.distance = 1.25; v.cam.azimuth = 150; v.cam.elevation = -22
    ep = 0
    while v.is_running():
        x, y, yaw = (a.x, a.y, a.yaw) if (a.x is not None and ep == 0) else sample_box_pose(rng, "train")
        plan = expert.plan((x, y), yaw); env.reset((x, y), yaw); v.sync(); time.sleep(1.0)
        print(f"episode {ep}: box x={x:.3f} y={y:.3f} yaw={yaw:+.2f}", flush=True)
        for c in plan["ctrl"]:
            d.ctrl[:] = c
            for k in range(env.substeps):
                mujoco.mj_step(m, d)
                if k % 10 == 0: v.sync(); time.sleep(10 * m.opt.timestep / a.realtime)
            if not v.is_running(): break
        ep += 1
        if not a.loop: 
            while v.is_running(): v.sync(); time.sleep(0.05)
