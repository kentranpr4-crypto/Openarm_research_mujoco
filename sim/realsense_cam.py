#!/usr/bin/env python3
"""Attach a simulated Intel RealSense D435-like RGB-D camera to OpenArm in MuJoCo and save rgb/depth/pointcloud.
Intrinsics are derived from FOV (pinhole, no distortion). Spec values below are from memory, NOT checked against
the Intel datasheet: verify FOV/resolution/depth range (or use calibrated intrinsics of YOUR unit) before sim2real use."""
import os; os.environ.setdefault("MUJOCO_GL","egl")
import sys, numpy as np, mujoco, imageio.v2 as iio
from pathlib import Path
HERE=Path(__file__).resolve().parent
W,H,VFOV = 640,480,42.5          # color stream; fovy must be the vertical FOV
ZMIN,ZMAX = 0.28,3.0              # D435 min/max useful depth (approx., unverified)
spec = mujoco.MjSpec.from_file(str(HERE/"openarm_mujoco/v1/scene.xml"))
torso = spec.body("openarm_body_link0")
# mount: on the head block, looking +x (hands reach toward +x in v1), pitched down (MuJoCo cameras look along their -z axis)
PITCH = np.radians(35)            # tilt down toward the workspace (hands sit ~0.3 m below the head)
c,sn = np.cos(PITCH), np.sin(PITCH)
torso.add_camera(name="rs_d435", pos=[0.06,0,0.80], xyaxes=[0,-1,0, sn,0,c], fovy=VFOV)
m = spec.compile(); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,"home")); mujoco.mj_forward(m,d)
r = mujoco.Renderer(m,H,W)
r.update_scene(d,camera="rs_d435"); rgb = r.render().copy()
r.enable_depth_rendering(); r.update_scene(d,camera="rs_d435"); depth = r.render().copy()   # metres, inf/far = background
r.close()
depth[(depth<ZMIN)|(depth>ZMAX)] = 0.0                          # RealSense convention: 0 = invalid
depth_noisy = depth + np.random.default_rng(0).normal(0,0.002*depth**2,depth.shape)*(depth>0)  # ~z^2 error model, illustrative
fy = (H/2)/np.tan(np.radians(VFOV)/2); fx = fy; cx,cy = W/2,H/2
v,u = np.mgrid[0:H,0:W]; z = depth_noisy
pts = np.stack([(u-cx)*z/fx,(v-cy)*z/fy,z],-1)[z>0]            # camera optical frame: x right, y down, z forward
out=HERE/"out"; out.mkdir(exist_ok=True)
iio.imwrite(out/"rs_rgb.png",rgb)
dv = np.clip(depth/ZMAX*255,0,255).astype(np.uint8); iio.imwrite(out/"rs_depth.png",dv)
np.save(out/"rs_depth_m.npy",depth_noisy.astype(np.float32))
print(f"intrinsics fx=fy={fx:.1f} cx={cx} cy={cy} | valid depth px={int((depth>0).sum())}/{W*H} | cloud pts={len(pts)}")
print("depth range m:", (depth[depth>0].min().round(3), depth[depth>0].max().round(3)) if (depth>0).any() else "none")
