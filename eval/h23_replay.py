#!/usr/bin/env python3
"""H2/H3 diagnostics, step 1: deterministic REPLAY of logged dev rollouts (no policy inference, no new evaluation protocol, no checkpoint use).
For every logged ACT episode (E0 seeds 1-2 in outputs/eval/e0, Stage-1 seed 0 in outputs/eval/e6b; dev seeds 21-22 only) the executed 16-D action sequence is replayed
through the same LabEnv (same box pose, same physics) and, in addition to what the rollout logged, we record per step: box pose/orientation expressed in the TCP frame,
TCP pose in the box frame, hand yaw, finger/hand/box contacts (normal force, contact position in the TCP frame).  The scripted expert is replayed on the same
dev configurations (paired).  The replay is checked against the logged box trajectory.  Writes outputs/h23/replay/<agent>_<set>_<idx>.npz.
Run (ACT venv, no GL needed):  python eval/h23_replay.py [--workers 4]
Frames: world x = fingertip approach direction (hand yaw ~ box yaw), TCP frame columns of data.xmat: x_tcp ~ world +z, y_tcp ~ world -y (closing axis), z_tcp ~ world +x (approach)."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, numpy as np, time
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
OUT = ROOT / "outputs/h23/replay"
SETS = {"NOM": ("nom21", "nominal"), "B21": ("bound21", "unseen_pose"), "B22": ("bound22", "unseen_pose")}
ARMS, SEEDS = ("c0", "s1"), (0, 1, 2)

def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def base(arm, seed): return (ROOT / "outputs/eval/e6b" / arm) if seed == 0 else (ROOT / "outputs/eval/e0" / f"{arm}_seed{seed}")
def yaw_of(q): w, x, y, z = q; return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

def jobs():
    J = []
    for S, (tag, cond) in SETS.items():
        for arm in ARMS:
            for s in SEEDS:
                d = base(arm, s) / tag / "act" / cond
                for e in jl(d / "episodes.jsonl"):
                    J.append(dict(agent=f"{arm}_s{s}", arm=arm, tseed=s, S=S, idx=e["episode_index"], seed=e["seed"], box_xy=e["box_xy"], box_yaw=e["box_yaw"], roll=str(d / "rollouts" / f"ep{e['episode_index']:04d}.npz")))
        d = base("c0", 0) / tag / "act" / cond                      # the expert gets the same dev configurations (paired)
        for e in jl(d / "episodes.jsonl"):
            J.append(dict(agent="expert", arm="expert", tseed=-1, S=S, idx=e["episode_index"], seed=e["seed"], box_xy=e["box_xy"], box_yaw=e["box_yaw"], roll=None))
    return J

_ENV = {}
def env_for():
    if "env" not in _ENV:
        from sim.lab_env import LabEnv
        from policies.baseline import ScriptedExpert
        env = LabEnv(cameras=()); _ENV.update(env=env, expert=ScriptedExpert(env.m, control_hz=20))
    return _ENV["env"], _ENV["expert"]

def replay_one(j):
    import mujoco
    f = OUT / f"{j['agent']}_{j['S']}_{j['idx']:03d}.npz"
    if f.exists(): return j["agent"], j["S"], j["idx"], "cached", None
    env, expert = env_for(); m, d = env.m, env.d; box = env.box
    gb = m.geom("blue_box_collision").id; gL = m.geom("openarm_left_left_finger_collision").id; gR = m.geom("openarm_left_right_finger_collision").id; gH = m.geom("openarm_left_hand_collision").id
    if j["roll"] is None:
        plan = expert.plan(tuple(j["box_xy"]), j["box_yaw"]); A = np.asarray(plan["ctrl"], np.float32); logged_B = None
    else:
        with np.load(j["roll"]) as z: A = z["action"]; logged_B = z["box_pos"]
    env.set_perturbation(); env.reset(tuple(j["box_xy"]), j["box_yaw"])
    T = len(A); keys = ["box_in_tcp", "tcp_in_box", "box_pos", "tcp_pos", "box_yaw", "hand_yaw", "box_tilt", "gripper_cmd", "gripper_q", "finger_force", "finger_contacts", "finger_contact_a", "finger_contact_c", "finger_contact_v", "hand_force", "table_force", "tray_force", "box_speed"]
    jq = m.joint("openarm_left_finger_joint1").id; R_ = {k: [] for k in keys}; f6 = np.zeros(6)
    for t in range(T + 1):
        Rt = d.xmat[env.tcp_id].reshape(3, 3); Tp = d.xpos[env.tcp_id].copy(); Bp = d.xpos[box].copy(); Rb = d.xmat[box].reshape(3, 3)
        R_["box_in_tcp"].append(Rt.T @ (Bp - Tp)); R_["tcp_in_box"].append(Rb.T @ (Tp - Bp)); R_["box_pos"].append(Bp); R_["tcp_pos"].append(Tp)
        R_["box_yaw"].append(yaw_of(d.xquat[box])); R_["hand_yaw"].append(float(np.arctan2(Rt[1, 2], Rt[0, 2]))); R_["box_tilt"].append(float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1)))))
        R_["gripper_cmd"].append(float(A[min(t, T - 1), 7])); R_["gripper_q"].append(float(d.qpos[m.jnt_qposadr[jq]]))
        ff, nf, ca, cc, cv, hf, tf, rf = 0.0, 0, [], [], [], 0.0, 0.0, 0.0
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = c.geom1, c.geom2
            if gb not in (g1, g2): continue
            o = g2 if g1 == gb else g1; mujoco.mj_contactForce(m, d, i, f6); fn = abs(f6[0])
            if o in (gL, gR): ff += fn; nf += 1; p = Rt.T @ (c.pos - Tp); ca.append(p[2]); cc.append(p[1]); cv.append(p[0])
            elif o == gH: hf += fn
            else:
                nm = m.body(m.geom_bodyid[o]).name
                if nm == "tray": rf += fn
                else: tf += fn
        R_["finger_force"].append(ff); R_["finger_contacts"].append(nf); R_["finger_contact_a"].append(np.mean(ca) if ca else np.nan); R_["finger_contact_c"].append(np.mean(cc) if cc else np.nan); R_["finger_contact_v"].append(np.mean(cv) if cv else np.nan)
        R_["hand_force"].append(hf); R_["table_force"].append(tf); R_["tray_force"].append(rf); R_["box_speed"].append(float(np.linalg.norm(d.cvel[box][3:])))
        if t < T: env.step(A[t])
    out = {k: np.array(v, np.float32) for k, v in R_.items()}; out["action"] = A
    err = None
    if logged_B is not None: err = float(np.abs(out["box_pos"][:len(logged_B)] - logged_B).max())
    OUT.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **out)
    return j["agent"], j["S"], j["idx"], "ok", err

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    J = jobs(); J = J[:a.limit] if a.limit else J; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); errs = []
    print(f"{len(J)} replays, {a.workers} workers", flush=True)
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap_unordered(replay_one, J, chunksize=2)):
            if r[4] is not None: errs.append(r[4])
            if k % 50 == 0: print(k, f"{time.time()-t0:.0f}s", flush=True)
    meta = dict(n=len(J), replay_vs_logged_box_max_abs_err_m=dict(max=float(max(errs)) if errs else None, median=float(np.median(errs)) if errs else None, n=len(errs)), jobs=[{k: v for k, v in j.items() if k != "roll"} for j in J])
    (ROOT / "outputs/h23").mkdir(exist_ok=True, parents=True); (ROOT / "outputs/h23/replay_meta.json").write_text(json.dumps(meta, indent=0, default=float)); print("replay error vs logged box trajectory:", meta["replay_vs_logged_box_max_abs_err_m"])

if __name__ == "__main__": main()
