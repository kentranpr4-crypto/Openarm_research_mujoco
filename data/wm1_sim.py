#!/usr/bin/env python3
"""WM-1 simulation core: replay a 16-D executed-action sequence through the SAME LabEnv and record, per row t (observation BEFORE action t; T+1 rows incl. the final one):
joint state (16), box pose/orientation, box in TCP frame, TCP position, gripper command, finger/palm/table/tray contact forces and the finger contact position (TCP frame).
Same geometry/contact conventions as eval/h23_replay.py (which is left unchanged); this version additionally stores the joint state so scripted/intervention episodes have s_t.
All arrays are privileged-simulator quantities EXCEPT `state` and `action`."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)

def yaw_of(q): w, x, y, z = q; return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
_ENV = {}
def get_env():
    if "env" not in _ENV:
        from sim.lab_env import LabEnv
        env = LabEnv(cameras=()); m = env.m
        _ENV.update(env=env, gb=m.geom("blue_box_collision").id, gL=m.geom("openarm_left_left_finger_collision").id, gR=m.geom("openarm_left_right_finger_collision").id, gH=m.geom("openarm_left_hand_collision").id, jq=m.joint("openarm_left_finger_joint1").id)
    return _ENV

def simulate(A, box_xy, box_yaw):
    """A (T,16) executed targets (already clipped/slewed by the caller). Returns dict of float32 arrays; state/box/contact rows have T+1 entries, action T."""
    import mujoco
    E = get_env(); env, m, d = E["env"], E["env"].m, E["env"].d; box = env.box; gb, gL, gR, gH, jq = E["gb"], E["gL"], E["gR"], E["gH"], E["jq"]
    env.set_perturbation(); env.reset(tuple(box_xy), float(box_yaw)); T = len(A); f6 = np.zeros(6)
    keys = ["state", "box_pos", "box_yaw", "box_tilt", "box_in_tcp", "tcp_pos", "box_speed", "gripper_cmd", "gripper_q", "finger_force", "finger_contacts", "finger_contact_a", "hand_force", "table_force", "tray_force"]
    R = {k: [] for k in keys}
    for t in range(T + 1):
        Rt = d.xmat[env.tcp_id].reshape(3, 3); Tp = d.xpos[env.tcp_id].copy(); Bp = d.xpos[box].copy(); Rb = d.xmat[box].reshape(3, 3)
        R["state"].append(env.adapter.state(d)); R["box_pos"].append(Bp); R["box_yaw"].append(yaw_of(d.xquat[box])); R["box_tilt"].append(float(np.arccos(np.clip(Rb[2, 2], -1, 1))))
        R["box_in_tcp"].append(Rt.T @ (Bp - Tp)); R["tcp_pos"].append(Tp); R["box_speed"].append(float(np.linalg.norm(d.cvel[box][3:]))); R["gripper_cmd"].append(float(A[min(t, T - 1), 7])); R["gripper_q"].append(float(d.qpos[m.jnt_qposadr[jq]]))
        ff, nf, ca, hf, tf, rf = 0.0, 0, [], 0.0, 0.0, 0.0
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = c.geom1, c.geom2
            if gb not in (g1, g2): continue
            o = g2 if g1 == gb else g1; mujoco.mj_contactForce(m, d, i, f6); fn = abs(f6[0])
            if o in (gL, gR): ff += fn; nf += 1; ca.append((Rt.T @ (c.pos - Tp))[2])
            elif o == gH: hf += fn
            elif m.body(m.geom_bodyid[o]).name == "tray": rf += fn
            else: tf += fn
        R["finger_force"].append(ff); R["finger_contacts"].append(nf); R["finger_contact_a"].append(np.mean(ca) if ca else np.nan); R["hand_force"].append(hf); R["table_force"].append(tf); R["tray_force"].append(rf)
        if t < T: env.step(A[t])
    out = {k: np.asarray(v, np.float32) for k, v in R.items()}; out["action"] = np.asarray(A, np.float32); return out

def contact_features(R):
    """Contract-v1 privileged blocks from a simulate() dict, aligned to rows 0..T-1: b (T,7) = [box x,y,z, yaw, speed, tilt, a_offset]; c (T,7) = [finger_force/50, n_finger_contacts, a_contact(0 if none), a_valid, palm_force, table_force, tray_force]."""
    T = len(R["action"]); sl = slice(0, T); ca = R["finger_contact_a"][sl]; valid = (~np.isnan(ca)).astype(np.float32)
    b = np.concatenate([R["box_pos"][sl], R["box_yaw"][sl, None], R["box_speed"][sl, None], R["box_tilt"][sl, None], R["box_in_tcp"][sl, 2:3]], 1).astype(np.float32)
    c = np.stack([R["finger_force"][sl] / 50.0, R["finger_contacts"][sl], np.nan_to_num(ca), valid, R["hand_force"][sl], R["table_force"][sl], R["tray_force"][sl]], 1).astype(np.float32)
    return b, c
