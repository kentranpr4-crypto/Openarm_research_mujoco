#!/usr/bin/env python3
"""Left-arm pick of the blue box on the desk scene (open-loop, IK waypoints -> position-servo ctrl), with checks + mp4.
Usage: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venv/bin/python grasp_blue_box.py
Simulation only: friction/mass/pad geometry are placeholders, a pass here says nothing about the real gripper."""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import json, sys, time, numpy as np, mujoco, imageio.v2 as iio
from pathlib import Path
from desk_scene import build_spec, reset, OUT, TABLE_TOP_Z

SIDE = "left"
GRIP_OPEN, GRIP_CLOSE = 0.044, 0.012      # ctrlrange -0.005..0.044 (v1, metres per finger); 0.012 -> ~50 N squeeze on a 4 cm box
VIEWER = "--viewer" in sys.argv          # live window instead of mp4 (needs a display)
RENDER = os.environ.get("GRASP_RENDER", "1") == "1" and not VIEWER
Z_GRASP, Z_LIFT = 0.385, 0.47
X_PRE, X_GRASP = 0.33, 0.43               # tcp ~ fingertip: fingers span tcp_x-0.12 .. tcp_x

spec = build_spec(with_camera=False)
spec.worldbody.add_camera(name="grasp_cam", pos=[0.95, 0.75, 0.75], xyaxes=[-0.62, 0.78, 0, -0.38, -0.30, 0.87])
m = spec.compile(); d = mujoco.MjData(m); reset(m, d)
dt = m.opt.timestep
bid = m.body(f"openarm_{SIDE}_hand_tcp").id
jids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        and m.body(m.jnt_bodyid[j]).name.startswith(f"openarm_{SIDE}_link")]
qadr = [m.jnt_qposadr[j] for j in jids]; dofs = [m.jnt_dofadr[j] for j in jids]
act = [m.actuator(f"{SIDE}_joint{i}_ctrl").id for i in range(1, 8)]
gact = m.actuator(f"{SIDE}_finger1_ctrl").id
R0 = d.body(bid).xmat.reshape(3, 3).copy()   # keep home hand orientation (fingers point +x)

def ik(dd, pos, iters=400):
    for _ in range(iters):
        mujoco.mj_forward(m, dd)
        ep = pos - dd.xpos[bid]
        er = np.zeros(3)
        qc = np.zeros(4); mujoco.mju_mat2Quat(qc, dd.xmat[bid]); qt = np.zeros(4); mujoco.mju_mat2Quat(qt, R0.ravel())
        mujoco.mju_subQuat(er, qt, qc)          # rotation taking current -> target, in current(local) frame
        er = dd.xmat[bid].reshape(3, 3) @ er    # to world frame
        e = np.r_[ep, er]
        if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3: return True
        Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, dd, Jp, Jr, bid)
        J = np.vstack([Jp, Jr])[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        for k, j in enumerate(jids):
            lo, hi = m.jnt_range[j]; dd.qpos[qadr[k]] = np.clip(dd.qpos[qadr[k]] + 0.5 * dq[k], lo, hi)
    return False

def plan(waypoints, step=0.01):
    """Cartesian line interpolation (<=1 cm per IK solve) -> list of joint vectors; IK seeded from previous solution."""
    dd = mujoco.MjData(m); reset(m, dd); mujoco.mj_forward(m, dd)
    cur = dd.xpos[bid].copy(); qs = [np.array([dd.qpos[a] for a in qadr])]; ok = True
    for w in waypoints:
        n = max(1, int(np.ceil(np.linalg.norm(w - cur) / step)))
        for i in range(1, n + 1):
            ok &= ik(dd, cur + (w - cur) * i / n); qs.append(np.array([dd.qpos[a] for a in qadr]))
        cur = w.copy()
    return qs, ok

reset(m, d); bx, by = d.body("blue_box").xpos[:2]
home = d.xpos[bid].copy()
wps = {"pregrasp": np.array([X_PRE, by, Z_GRASP]), "grasp": np.array([X_GRASP, by, Z_GRASP])}
q_pre, ok1 = plan([np.array([X_PRE, home[1], home[2]]), wps["pregrasp"]])       # approach; first move keeps height
q_grasp, ok2 = plan([np.array([X_PRE, home[1], home[2]]), wps["pregrasp"], wps["grasp"]])[0][len(q_pre):], True
q_lift, ok3 = plan([np.array([X_PRE, home[1], home[2]]), wps["pregrasp"], wps["grasp"], np.array([X_GRASP, by, Z_LIFT])])
q_lift = q_lift[len(q_pre) + len(q_grasp):]
ok_ik = ok1 and ok3
print("IK converged on all waypoints:", ok_ik, "| steps", len(q_pre), len(q_grasp), len(q_lift))

frames = []; log = {"phases": {}}
r = mujoco.Renderer(m, 480, 640) if RENDER else None; FPS = 30; every = int(round(1 / (FPS * dt)))
tick = [0]
if VIEWER:
    import mujoco.viewer
    view = mujoco.viewer.launch_passive(m, d)
    view.cam.lookat[:] = [0.42, 0.1, 0.4]; view.cam.distance = 1.1; view.cam.azimuth = 150; view.cam.elevation = -20
    time.sleep(1.5)
def step(n=1):
    for _ in range(n):
        mujoco.mj_step(m, d); tick[0] += 1
        if VIEWER and tick[0] % 16 == 0:
            if not view.is_running(): sys.exit(0)
            view.sync(); time.sleep(16 * dt)      # ~real time
        if RENDER and tick[0] % every == 0:
            fr = []
            for cam in ("grasp_cam", "overview"):
                r.update_scene(d, camera=cam); fr.append(r.render().copy())
            frames.append(np.concatenate(fr, 1))
kp = np.array([m.actuator_gainprm[a][0] for a in act])
def comp(q):
    """Position servos have no gravity compensation: static balance kp*(ctrl-q)=qfrc_bias -> ctrl = q + bias/kp."""
    dd = mujoco.MjData(m); mujoco.mj_resetData(m, dd)
    for a, v in zip(qadr, q): dd.qpos[a] = v
    mujoco.mj_forward(m, dd)
    return q + dd.qfrc_bias[dofs] / kp
def follow(qs, dur_per_pt, grip):
    for q in qs:
        d.ctrl[act] = comp(q); d.ctrl[gact] = grip; step(int(dur_per_pt / dt))
def finger_contacts():
    out = set()
    for c in d.contact[:d.ncon]:
        a, b = m.geom(c.geom1).name, m.geom(c.geom2).name
        if "blue_box" in a + b and "finger" in a + b: out.add(a if "finger" in a else b)
    return out

d.ctrl[gact] = GRIP_OPEN; step(int(0.5 / dt))
follow(q_pre, 0.03, GRIP_OPEN)
tcp_pre = d.xpos[bid].copy(); follow(q_grasp, 0.04, GRIP_OPEN); step(int(0.6 / dt))
tcp_grasp = d.xpos[bid].copy(); box_before = d.body("blue_box").xpos.copy()
log["phases"]["grasp_pose"] = {"tcp": tcp_grasp.round(4).tolist(), "tcp_target": wps["grasp"].tolist(), "box": box_before.round(4).tolist()}
# close gradually
for g in np.linspace(GRIP_OPEN, GRIP_CLOSE, 40):
    d.ctrl[gact] = g; step(int(0.02 / dt))
step(int(0.5 / dt)); contacts_closed = finger_contacts(); box_closed = d.body("blue_box").xpos.copy()
fingers_q = [d.qpos[m.jnt_qposadr[m.joint(f"openarm_{SIDE}_finger_joint{i}").id]] for i in (1, 2)]
follow(q_lift, 0.03, GRIP_CLOSE)
step(int(1.0 / dt))                                    # hold 1 s after the lift
box_end = d.body("blue_box").xpos.copy(); contacts_end = finger_contacts(); tcp_end = d.xpos[bid].copy()
rise = float(box_end[2] - box_before[2]); rel = float(np.linalg.norm(box_end - tcp_end))
if r: r.close()
if VIEWER:
    print("Done - close the viewer window to exit.")
    while view.is_running(): view.sync(); time.sleep(0.05)
if RENDER: iio.mimsave(OUT / "grasp_blue_box.mp4", frames, fps=FPS, macro_block_size=1)
if RENDER:
    iio.imwrite(OUT / "grasp_blue_box_last.png", frames[-1]); iio.imwrite(OUT / "grasp_blue_box_mid.png", frames[len(frames) // 2])
others = {n: float(np.linalg.norm(d.body(n).xpos[:2] - np.array(p0))) for n, p0 in
          {"red_cylinder": (0.40, -0.12), "green_cube": (0.52, 0.0)}.items()}
res = {"ik_converged": bool(ok_ik), "box_rise_m": round(rise, 4), "box_to_tcp_after_hold_m": round(rel, 4),
       "contacts_after_close": sorted(contacts_closed), "contacts_after_hold": sorted(contacts_end),
       "finger_q_after_close": [round(float(x), 4) for x in fingers_q], "box_end": box_end.round(4).tolist(),
       "tcp_end": tcp_end.round(4).tolist(), "box_xy_shift_during_close_m": round(float(np.linalg.norm(box_closed[:2] - box_before[:2])), 4),
       "other_objects_xy_shift_m": {k: round(v, 4) for k, v in others.items()}, "frames": len(frames), "nan": not np.isfinite(d.qpos).all()}
log.update(res); json.dump(log, open(OUT / "grasp_blue_box_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("AC-A rise>=0.05:", rise >= 0.05, "| AC-B still held:", rise >= 0.05 and rel < 0.12, "| AC-C two-finger contact:", len(contacts_end) >= 2)
