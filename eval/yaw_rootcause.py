#!/usr/bin/env python3
"""YAW ROOT-CAUSE ANALYSIS (offline, no training, no checkpoint/normaliser change).  Stage A: event table of every per-step |delta yaw| > pi/2 in the 4 800 frozen wm2 episodes, with lineage and a units check of the tilt channel.
Stage B: deterministic simulator replay of the affected episodes (same reset/actions as the dataset builders) to read raw quaternion, angular velocity and the body-x-axis verticality (Euler-yaw singularity).  Stage C: per-window / per-regime contribution of
the yaw channel to the training target energy and to the frozen WM3-B's squared error (forward pass only).  Stage D: offline effect of alternative target scalings / representations on target-energy shares.  Output: outputs/yaw_audit/yaw_rootcause.json
Run (ACT venv): python eval/yaw_rootcause.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data"), str(ROOT / "eval")):
    if p not in sys.path: sys.path.insert(0, p)
OUT = ROOT / "outputs/yaw_audit"; wrap = lambda x: np.arctan2(np.sin(x), np.cos(x)); DEG = np.pi / 180
def tilt_unit(e):
    """Unit of b[:,5] (box tilt), determined by the replay script that produced the episode: outputs/wm1/replay (data/wm1_sim.py) and all interventional episodes -> radians; outputs/h23, outputs/xsweep, outputs/cov replays (eval/h23_replay.py: np.degrees) -> degrees."""
    if e["kind"] == "interventional" or (e.get("replay") or "").startswith("outputs/wm1/replay"): return "radians"
    return "degrees"
def stage_A():
    idx = json.loads((ROOT / "outputs/wm2/dataset/index.json").read_text()); ev = []; tilt_units = {}
    for e in idx:
        if not e["replay_ok"]: continue
        b = np.load(ROOT / f"outputs/wm2/dataset/episodes/{e['eid']}.npz")["b"]; unit = tilt_unit(e); tilt_units.setdefault(unit, []).append(float(b[:, 5].max())); d = np.diff(b[:, 3]); sc = DEG if unit == "degrees" else 1.0
        for t in np.nonzero(np.abs(d) > np.pi / 2)[0]:
            ev.append(dict(eid=e["eid"], kind=e["kind"], split=e["split"], source=e["source"].split("/")[0], agent=e["agent"], family=e["family"], t=int(t), T=int(e["T"]), raw=float(d[t]), wrapped=float(wrap(d[t])), yaw0=float(b[t, 3]), yaw1=float(b[t + 1, 3]), tilt0_rad=float(b[t, 5] * sc), tilt1_rad=float(b[t + 1, 5] * sc),
                           z0=float(b[t, 2]), speed0=float(b[t, 4]), speed1=float(b[t + 1, 4]), task_success=bool(e["task_success"]), pick=bool(e["pick"]), replay=e.get("replay"), prov=e.get("provenance")))
    units = {k: dict(n_episodes=len(v), max_of_episode_max=max(v), frac_episodes_max_gt_pi=float(np.mean(np.array(v) > np.pi + 1e-3))) for k, v in tilt_units.items()}
    return idx, ev, units
def yaw_of(q): w, x, y, z = q; return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
def stage_B(idx, ev):
    from sim.lab_env import LabEnv
    env = LabEnv(cameras=()); m, d = env.m, env.d; box = env.box; by = {e["eid"]: e for e in idx}; rows = {}
    for eid in sorted({x["eid"] for x in ev}):
        e = by[eid]
        if e["kind"] == "observational": A = np.load(ROOT / e["replay"])["action"]
        else: A = np.load(ROOT / e["provenance"]["source_episode"])["action"]
        env.set_perturbation(); env.reset(tuple(e["box_xy"]), float(e["box_yaw"])); R = dict(yaw=[], omega=[], xvert=[], tilt=[], speed=[], quat=[])
        for t in range(len(A) + 1):
            Rb = d.xmat[box].reshape(3, 3); R["yaw"].append(yaw_of(d.xquat[box])); R["omega"].append(d.cvel[box][:3].copy()); R["xvert"].append(abs(Rb[2, 0])); R["tilt"].append(float(np.arccos(np.clip(Rb[2, 2], -1, 1)))); R["speed"].append(float(np.linalg.norm(d.cvel[box][3:])))
            if t < len(A): env.step(A[t])
        rows[eid] = {k: np.array(v) for k, v in R.items()}
    return rows
def classify(ev, rows):
    out = []
    for x in ev:
        r = rows[x["eid"]]; t = x["t"]; om = r["omega"]; dt = 0.05; w_mag = float(np.linalg.norm(om[t:t + 2], axis=1).max()); wz_dt = float(om[t:t + 2, 2].mean() * dt); xv = float(max(r["xvert"][t], r["xvert"][t + 1])); wr = abs(x["wrapped"]); consistent = abs(x["wrapped"] - wz_dt) < 0.5 * max(wr, 0.05)
        yaw_replay_matches = abs((r["yaw"][t + 1] - r["yaw"][t]) - x["raw"]) < 1e-3
        if xv > 0.99 and w_mag < 0.5: cat = "S_singularity(body-x vertical, heading undefined, omega~0)"
        elif xv > 0.99: cat = "S2_singularity_while_moving(body-x vertical)"
        elif abs(x["raw"]) > np.pi and wr < np.pi / 4: cat = "W_wrap_crossing(+-pi discontinuity, small physical rotation)"
        elif wr >= np.pi / 4 and w_mag > 5: cat = "P_physical_fast_rotation(large omega)"
        else: cat = "O_other"
        out.append(dict(x, cat=cat, omega_max=w_mag, wz_dt=wz_dt, body_x_vertical=xv, consistent_with_omega_z=bool(consistent), replay_reproduces_dataset_delta=bool(yaw_replay_matches), replay=None, prov=None))
    return out
def main():
    idx, ev, units = stage_A(); rows = stage_B(idx, ev); cl = classify(ev, rows); cats = {}
    for x in cl: cats.setdefault(x["cat"], []).append(x)
    summ = {c: dict(n=len(v), episodes=len({q["eid"] for q in v}), by_kind={k: sum(1 for q in v if q["kind"] == k) for k in ("observational", "interventional")}, median_speed0=float(np.median([q["speed0"] for q in v])), median_omega_max=float(np.median([q["omega_max"] for q in v])), median_tilt_rad=float(np.median([q["tilt0_rad"] for q in v])),
                    median_body_x_vertical=float(np.median([q["body_x_vertical"] for q in v])), frac_replay_reproduces=float(np.mean([q["replay_reproduces_dataset_delta"] for q in v]))) for c, v in cats.items()}
    lin = {}
    for x in cl: lin.setdefault(x["source"] + "/" + x["kind"], 0); lin[x["source"] + "/" + x["kind"]] += 1
    res = dict(status="offline analysis; no training; frozen artifacts untouched", tilt_channel_units=units, n_events=len(cl), n_episodes_with_events=len({x["eid"] for x in cl}), categories=summ, lineage_counts=lin, events_in_train_windows_note="events counted at episode level (consecutive windows share steps)",
               first_events=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k not in ("replay", "prov")} for x in cl[:25]])
    (OUT / "yaw_rootcause_AB.json").write_text(json.dumps(res, indent=1, default=str)); print(json.dumps({k: res[k] for k in ("tilt_channel_units", "n_events", "n_episodes_with_events", "categories", "lineage_counts")}, indent=1, default=str)); json.dump([{k: v for k, v in x.items() if k not in ("replay", "prov")} for x in cl], open(OUT / "yaw_events.json", "w"))
if __name__ == "__main__": main()
