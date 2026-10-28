"""Scripted IK expert (pick the blue box with the LEFT arm, place it in the tray). Open-loop, waypoint based:
  settle -> pre-approach -> pregrasp -> approach -> close -> lift -> transport -> lower -> release -> retreat.
Joint targets come from damped-LS IK (hand orientation fixed: fingers +x, optionally yawed to the box); each ctrl is
gravity compensated (position servos have no gravity comp). It sees the box pose it is GIVEN - it is the data
generator / evaluation baseline, not a perception system. sim/grasp_blue_box.py is kept unchanged as the original demo."""
import sys, numpy as np, mujoco
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
import lab_scene
from ik import ArmIK

G_OPEN, G_CLOSE = 0.044, 0.012          # v1 finger ctrl [m]; 0.012 squeezes a 4 cm box (~50 N, see grasp report)
Z_LIFT = 0.47
GRASP_DZ = 0.025                        # TCP height above box centre (finger pads span +-3 cm vertically; keeps them above the tray rim)

def _smooth(s): return s * s * (3 - 2 * s)
def _rz(yaw): c, s = np.cos(yaw), np.sin(yaw); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

class ScriptedExpert:
    def __init__(self, model, control_hz=20, speed=0.18, min_dur=0.5):
        self.m, self.hz, self.speed, self.min_dur = model, control_hz, speed, min_dur
        self.ik, self.ikr = ArmIK(model, "left"), ArmIK(model, "right")
        dd = mujoco.MjData(model); lab_scene.reset_home(model, dd)
        self.R0 = dd.xmat[self.ik.tcp].reshape(3, 3).copy(); self.home_tcp = dd.xpos[self.ik.tcp].copy()
        self.q_home_r = self.ikr.q(dd); self.ctrl_right = self.ikr.gravity_comp(self.q_home_r)
        self.g_l = model.actuator("left_finger1_ctrl").id; self.g_r = model.actuator("right_finger1_ctrl").id
        self.g_r_home = float(model.key_ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")][self.g_r])

    def segments(self, box_xy, yaw):
        bx, by = box_xy; f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        zg = lab_scene.TABLE_TOP_Z + 0.020 + GRASP_DZ          # box centre z + GRASP_DZ: fingers grip the upper part of the box
        c = np.array([bx, by, zg]); h = self.home_tcp; tray = np.array([*lab_scene.TRAY_C, 0.0])
        z_place = lab_scene.TABLE_TOP_Z + lab_scene.TRAY_BASE_T + 0.020 + GRASP_DZ + 0.003
        pre_a = c - 0.09 * f; pre_a[1], pre_a[2] = h[1], h[2]
        grasp = c + 0.01 * f; above_tray = tray + 0.01 * f
        return [("settle", h, G_OPEN, 0.4), ("pre_a", pre_a, G_OPEN, None), ("pregrasp", c - 0.09 * f, G_OPEN, None),
                ("approach", grasp, G_OPEN, None), ("close", grasp, G_CLOSE, 0.7), ("grip_hold", grasp, G_CLOSE, 0.4),
                ("lift", np.array([*grasp[:2], Z_LIFT]), G_CLOSE, None), ("transport", np.array([*above_tray[:2], Z_LIFT]), G_CLOSE, None),
                ("lower", np.array([*above_tray[:2], z_place]), G_CLOSE, None), ("release", np.array([*above_tray[:2], z_place]), G_OPEN, 0.5),
                ("release_hold", np.array([*above_tray[:2], z_place]), G_OPEN, 0.3), ("retreat", np.array([*above_tray[:2], Z_LIFT]), G_OPEN, None),
                ("end_hold", np.array([*above_tray[:2], Z_LIFT]), G_OPEN, 0.3)]

    def plan(self, box_xy, box_yaw=0.0):
        """-> dict(ctrl (T,16), phase (T,), phase_names, tcp_target (T,3), ik_fail_steps, ik_pos_err/ik_rot_err (T,), ok)."""
        dd = mujoco.MjData(self.m); lab_scene.reset_home(self.m, dd); R = _rz(box_yaw) @ self.R0
        cur, g_cur = self.home_tcp.copy(), G_OPEN
        qs, gs, tg, ph, fails, names, pe, re = [], [], [], [], 0, [], [], []
        qt = np.zeros(4); mujoco.mju_mat2Quat(qt, R.ravel())
        for si, (name, pos, g, dur) in enumerate(self.segments(box_xy, box_yaw)):
            names.append(name); dist = np.linalg.norm(pos - cur)
            n = max(1, int(round((dur if dur else max(self.min_dur, dist / self.speed)) * self.hz)))
            for i in range(1, n + 1):
                s = _smooth(i / n); p = cur + (pos - cur) * s
                fails += 0 if self.ik.solve(dd, p, R) else 1
                qc = np.zeros(4); er = np.zeros(3); mujoco.mju_mat2Quat(qc, dd.xmat[self.ik.tcp]); mujoco.mju_subQuat(er, qt, qc)
                pe.append(float(np.linalg.norm(p - dd.xpos[self.ik.tcp]))); re.append(float(np.linalg.norm(er)))
                qs.append(self.ik.q(dd)); gs.append(g_cur + (g - g_cur) * s); tg.append(p); ph.append(si)
            cur, g_cur = pos.copy(), g
        ctrl = np.zeros((len(qs), 16)); ctrl[:, self.ik.act] = [self.ik.gravity_comp(q) for q in qs]
        ctrl[:, self.g_l] = gs; ctrl[:, self.ikr.act] = self.ctrl_right; ctrl[:, self.g_r] = self.g_r_home
        lo, hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        n_clip = int(np.count_nonzero((ctrl < lo - 1e-9) | (ctrl > hi + 1e-9)))    # e.g. gravity-comp offset pushes left_joint2 past its limit
        ctrl = np.clip(ctrl, lo, hi)      # MuJoCo clamps ctrl itself; clipping here makes the RECORDED action equal the EXECUTED command
        return dict(ctrl=ctrl, phase=np.array(ph), phase_names=names, tcp_target=np.array(tg), ctrl_clipped_values=n_clip, ik_fail_steps=fails, ik_pos_err=np.array(pe), ik_rot_err=np.array(re),
                    ok=bool(max(pe) < 2e-3 and max(re) < 0.03))     # looser than the solver's own tol; residuals are stored


# ---- box-pose distributions (metres / rad). TRAIN is where demos are collected; UNSEEN is a wider box minus the train box.
TRAIN_REGION = dict(x=(0.38, 0.45), y=(0.07, 0.13), yaw=(-0.20, 0.20))
WIDE_REGION = dict(x=(0.36, 0.47), y=(0.05, 0.15), yaw=(-0.30, 0.30))   # expert planning is clean only for |yaw|<~0.24 (see report)

def _inside(pose, r): return all(r[k][0] <= v <= r[k][1] for k, v in zip(("x", "y", "yaw"), pose))

def sample_box_pose(rng, region="train"):
    """-> (x, y, yaw). 'train' uniform in TRAIN_REGION; 'unseen' uniform in WIDE_REGION by rejection outside TRAIN_REGION."""
    r = TRAIN_REGION if region == "train" else WIDE_REGION
    while True:
        pose = tuple(float(rng.uniform(*r[k])) for k in ("x", "y", "yaw"))
        if region == "train" or not _inside(pose, TRAIN_REGION): return pose
