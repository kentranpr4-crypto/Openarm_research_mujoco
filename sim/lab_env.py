"""Thin environment around the lab scene: reset with a box pose, step at a fixed control rate, observations for
policies (images + proprioception ONLY) and a separate `info` channel with ground truth for evaluation/labels.
Policies must never read `info`."""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import sys, numpy as np, mujoco
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)
import lab_scene
from cameras import CameraRig, POLICY_CAMERAS
from control.action_adapter import ActionAdapter

class LabEnv:
    def __init__(self, image_hw=(120, 160), control_hz=20, render_profile="fast", cameras=POLICY_CAMERAS):
        """render_profile: 'studio' = shadows + floor reflection + MSAA (presentation); 'fast' = none of these (~5x faster on CPU
        software GL; used for datasets/training/inference/eval, which must all use the SAME profile)."""
        self.m, self.d = lab_scene.compile_lab(); self.render_profile = render_profile
        if render_profile == "fast":      # set before the Renderer exists
            self.m.light_castshadow[:] = 0; self.m.mat_reflectance[:] = 0.0; self.m.vis.quality.offsamples = 0
        elif render_profile != "studio": raise ValueError(render_profile)
        self.hz, self.cams = control_hz, tuple(cameras)
        self.substeps = int(round(1.0 / (control_hz * self.m.opt.timestep)))
        self.adapter = ActionAdapter(self.m, control_hz)
        self.rig = CameraRig(self.m, image_hw[1], image_hw[0]) if self.cams else None
        self.box_free = self.m.joint("blue_box_free"); self.box = self.m.body("blue_box").id
        self.tcp_id = self.m.body("openarm_left_hand_tcp").id
        self._base = dict(ld=self.m.light_diffuse.copy(), lp=self.m.light_pos.copy(), ldir=self.m.light_dir.copy(),
                          cp=self.m.cam_pos.copy(), cq=self.m.cam_quat.copy(), hl=self.m.vis.headlight.diffuse.copy())
        self.prev_ctrl = None; self.t = 0.0

    # ---- perturbations used by evaluation (generalisation tests); default = nominal scene -----------------------
    def set_perturbation(self, light_scale=1.0, light_shift=(0, 0, 0), cam_shift=(0, 0, 0), cam_tilt_deg=0.0, cam="front"):
        m, b = self.m, self._base
        m.light_diffuse[:] = b["ld"] * light_scale; m.vis.headlight.diffuse[:] = b["hl"] * light_scale
        m.light_pos[:] = b["lp"] + np.asarray(light_shift)
        i = m.camera(cam).id; m.cam_pos[i] = b["cp"][i] + np.asarray(cam_shift)
        q = np.zeros(4); ax = np.array([0, 1, 0.0]); mujoco.mju_axisAngle2Quat(q, ax, np.radians(cam_tilt_deg))
        out = np.zeros(4); mujoco.mju_mulQuat(out, b["cq"][i], q); m.cam_quat[i] = out

    def reset(self, box_xy=None, box_yaw=0.0):
        lab_scene.reset_home(self.m, self.d)
        if box_xy is not None:
            a = self.m.jnt_qposadr[self.box_free.id]; z = lab_scene.TABLE_TOP_Z + 0.020 + 0.0005
            self.d.qpos[a:a + 3] = [box_xy[0], box_xy[1], z]
            self.d.qpos[a + 3:a + 7] = [np.cos(box_yaw / 2), 0, 0, np.sin(box_yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        for _ in range(int(0.2 / self.m.opt.timestep)): mujoco.mj_step(self.m, self.d)   # settle contacts, home ctrl
        self.prev_ctrl = self.d.ctrl.copy(); self.t = 0.0
        return self.obs()

    def obs(self):
        o = dict(state=self.adapter.state(self.d), velocity=self.adapter.velocity(self.d), t=self.t)
        o["images"] = {c: self.rig.rgb(self.d, c) for c in self.cams}
        return o

    def info(self):
        """Ground truth for labels/metrics ONLY (never a policy input)."""
        return dict(box_pos=self.d.xpos[self.box].copy(), box_quat=self.d.xquat[self.box].copy(), tcp_pos=self.d.xpos[self.tcp_id].copy(),
                    box_speed=float(np.linalg.norm(self.d.cvel[self.box][3:])))

    def step(self, ctrl):
        self.d.ctrl[:] = ctrl; self.prev_ctrl = np.asarray(ctrl, float).copy()
        for _ in range(self.substeps): mujoco.mj_step(self.m, self.d)
        self.t += 1.0 / self.hz
        return self.obs()

    def close(self):
        if self.rig: self.rig.close()
