#!/usr/bin/env python3
"""Smoke-simulate an OpenArm MJCF scene headless. Usage: run_sim.py [scene.xml] [--seconds 5] [--out out/]
Drives every position actuator with a small sinusoid inside its ctrlrange, starting from keyframe 'home' if present."""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("scene", nargs="?", default=str(HERE/"openarm_mujoco/v1/scene.xml"))
ap.add_argument("--seconds", type=float, default=5.0)
ap.add_argument("--out", default=str(HERE/"out"))
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(a.scene); d = mujoco.MjData(m)
print(f"scene={a.scene}\nmujoco={mujoco.__version__} nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} timestep={m.opt.timestep}")
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
print("actuators:", names)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if key >= 0: mujoco.mj_resetDataKeyframe(m, d, key)
mujoco.mj_forward(m, d)
base = d.ctrl.copy(); lo, hi = m.actuator_ctrlrange.T
amp = 0.15 * (hi - lo) * (m.actuator_ctrllimited > 0)
n = int(a.seconds / m.opt.timestep); fps = 30; frames = []
r = mujoco.Renderer(m, 480, 640)
bad = None; maxv = 0.0
for i in range(n):
    t = d.time
    d.ctrl[:] = np.clip(base + amp * np.sin(2*np.pi*0.5*t + np.arange(m.nu)), lo, hi)
    mujoco.mj_step(m, d)
    if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()): bad = i; break
    maxv = max(maxv, float(np.abs(d.qvel).max()))
    if len(frames) < d.time * fps:
        r.update_scene(d); frames.append(r.render().copy())
out = Path(a.out); out.mkdir(exist_ok=True)
tag = Path(a.scene).parent.name + "_" + Path(a.scene).stem
import imageio.v2 as iio
iio.imwrite(out/f"{tag}.png", frames[len(frames)//2])
iio.mimsave(out/f"{tag}.mp4", frames, fps=fps)
r.close()
print(f"steps={i+1}/{n} sim_time={d.time:.2f}s nan_at={bad} max|qvel|={maxv:.2f} ncon_last={d.ncon}")
print("RESULT:", "FAIL(NaN)" if bad is not None else "PASS(finite)")
sys.exit(1 if bad is not None else 0)
