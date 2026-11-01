#!/usr/bin/env python3
"""DATA CONTRACT v2 — VALIDATION by simulator replay of a SAMPLE of the frozen episodes (all 34 episodes with yaw events + a seeded random sample of the others).  Nothing official is rebuilt; outputs go to outputs/contract_v2/.
Checks: quaternion / body-axis conventions, angular velocity, Euler-yaw singularity, geometric yaw-validity mask (threshold from geometry, never from predictions), physical rotation vs singular discontinuity, tilt unit repair, D2 symmetry.
Run (ACT venv): python eval/contract_v2_validate.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, numpy as np
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data"), str(ROOT / "eval")):
    if p not in sys.path: sys.path.insert(0, p)
import geometry_v2 as Gm
OUT = ROOT / "outputs/contract_v2"; _E = {}
def tilt_unit(e): return "radians" if (e["kind"] == "interventional" or (e.get("replay") or "").startswith("outputs/wm1/replay")) else "degrees"
def replay(e):
    if "env" not in _E:
        from sim.lab_env import LabEnv
        _E["env"] = LabEnv(cameras=())
    env = _E["env"]; m, d = env.m, env.d; box = env.box
    A = np.load(ROOT / e["replay"])["action"] if e["kind"] == "observational" else np.load(ROOT / e["provenance"]["source_episode"])["action"]
    env.set_perturbation(); env.reset(tuple(e["box_xy"]), float(e["box_yaw"])); R = dict(q=[], xmat=[], omega=[], pos=[], speed=[])
    for t in range(len(A) + 1):
        R["q"].append(d.xquat[box].copy()); R["xmat"].append(d.xmat[box].copy()); R["omega"].append(d.cvel[box][:3].copy()); R["pos"].append(d.xpos[box].copy()); R["speed"].append(float(np.linalg.norm(d.cvel[box][3:])))
        if t < len(A): env.step(A[t])
    return e["eid"], {k: np.array(v) for k, v in R.items()}
def main():
    idx = json.loads((ROOT / "outputs/wm2/dataset/index.json").read_text()); by = {e["eid"]: e for e in idx if e["replay_ok"]}; ev = json.load(open(ROOT / "outputs/yaw_audit/yaw_events.json")); ev_eids = sorted({x["eid"] for x in ev}); rng = np.random.default_rng(0)
    groups = {"obs_degrees": [], "obs_radians": [], "interventional": []}
    for eid, e in by.items():
        if eid in ev_eids: continue
        groups["interventional" if e["kind"] == "interventional" else ("obs_radians" if tilt_unit(e) == "radians" else "obs_degrees")].append(eid)
    sample = list(ev_eids) + [str(x) for g in groups.values() for x in rng.choice(g, 120, replace=False)]
    with Pool(4) as pool: rows = dict(pool.map(replay, [by[s] for s in sample], chunksize=4))
    np.savez_compressed(OUT / "replay_sample.npz", **{f"{eid}/{k}": v for eid, r in rows.items() for k, v in r.items()}); res = dict(n_episodes=len(rows), n_event_episodes=len(ev_eids), sample_groups={k: 120 for k in groups}, n_steps=int(sum(len(r["q"]) for r in rows.values())))
    # --- V1 conventions
    mx = dict(q_to_xmat=0.0, orth=0.0, det=0.0, yaw_vs_dataset=0.0, tilt_vs_acos=0.0)
    for eid, r in rows.items():
        R_ = Gm.quat_to_R(r["q"]); mx["q_to_xmat"] = max(mx["q_to_xmat"], float(np.abs(R_ - r["xmat"].reshape(-1, 3, 3)).max())); mx["orth"] = max(mx["orth"], float(np.abs(R_ @ R_.transpose(0, 2, 1) - np.eye(3)).max())); mx["det"] = max(mx["det"], float(np.abs(np.linalg.det(R_) - 1).max()))
        b = np.load(ROOT / f"outputs/wm2/dataset/episodes/{eid}.npz")["b"]; mx["yaw_vs_dataset"] = max(mx["yaw_vs_dataset"], float(np.abs(Gm.wrap(Gm.yaw_euler(r["q"])[:len(b)] - b[:, 3])).max()))
        sc = np.pi / 180 if tilt_unit(by[eid]) == "degrees" else 1.0; mx["tilt_vs_acos"] = max(mx["tilt_vs_acos"], float(np.abs(b[:, 5] * sc - Gm.tilt_rad(r["q"])[:len(b)]).max()))
    res["V1_conventions_max_abs_diff"] = mx
    # --- V2 angular velocity (world frame, MuJoCo cvel[:3]) vs finite-difference rotation vectors
    cs_w, cs_b, ratio = [], [], []
    for eid, r in rows.items():
        rv = Gm.step_rotvecs(r["q"]) / 0.05; om = 0.5 * (r["omega"][:-1] + r["omega"][1:]); big = np.linalg.norm(rv, axis=1) > 0.5
        if big.any():
            ob = np.einsum("tji,tj->ti", r["xmat"][:-1].reshape(-1, 3, 3), om)          # body-frame omega = R^T omega
            cs_w += list(np.sum(rv[big] * om[big], 1) / (np.linalg.norm(rv[big], axis=1) * np.linalg.norm(om[big], axis=1) + 1e-12)); cs_b += list(np.sum(rv[big] * ob[big], 1) / (np.linalg.norm(rv[big], axis=1) * np.linalg.norm(ob[big], axis=1) + 1e-12)); ratio += list(np.linalg.norm(rv[big], axis=1) / (np.linalg.norm(om[big], axis=1) + 1e-12))
    res["V2_angular_velocity"] = dict(n_steps_compared=len(cs_w), median_cos_world=float(np.median(cs_w)), median_cos_body=float(np.median(cs_b)), p10_cos_world=float(np.percentile(cs_w, 10)), median_norm_ratio_fd_over_sim=float(np.median(ratio)), note="rv/dt of q_{k+1} conj(q_k) (world frame) vs mean of cvel[:3] at rows k,k+1; the control step spans several physics sub-steps so exact equality is not expected")
    # --- V3 Euler singularity: yaw-jump inconsistency J = |wrap(dyaw)| - symmetric geodesic step vs min horizontal norm of the two ends
    bins = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.01]; J_all, hn_all = [], []; geo_all = []
    for eid, r in rows.items():
        yaw = Gm.yaw_euler(r["q"]); hn = Gm.hnorm(r["q"]); J = np.abs(Gm.wrap(np.diff(yaw))) - Gm.geodesic_sym(r["q"][:-1], r["q"][1:]); J_all.append(J); hn_all.append(np.minimum(hn[:-1], hn[1:])); geo_all.append(Gm.geodesic_sym(r["q"][:-1], r["q"][1:]))
    J = np.concatenate(J_all); hnm = np.concatenate(hn_all); geo = np.concatenate(geo_all); tab = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (hnm >= lo) & (hnm < hi); tab.append(dict(hnorm_min_range=[lo, hi], n_steps=int(mk.sum()), frac_J_gt_0p25=float((J[mk] > 0.25).mean()) if mk.any() else None, max_J=float(J[mk].max()) if mk.any() else None, median_geodesic_step=float(np.median(geo[mk])) if mk.any() else None))
    res["V3_yaw_inconsistency_by_hnorm_min"] = tab
    ok_tau = [t for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5) if (J[hnm >= t] <= 0.25).all()]; res["V3_smallest_tau_with_no_inconsistent_step"] = ok_tau[0] if ok_tau else None
    # --- V4 events vs mask
    ev_rows = []; evd = {}
    for x in ev:
        r = rows[x["eid"]]; t = x["t"]; hn = Gm.hnorm(r["q"][t:t + 2]); Jt = abs(float(Gm.wrap(Gm.yaw_euler(r["q"][t + 1]) - Gm.yaw_euler(r["q"][t])))) - float(Gm.geodesic_sym(r["q"][t], r["q"][t + 1])); grp = "singular" if x["cat"].startswith("S") else ("physical" if x["cat"].startswith("P") else "wrap_crossing")
        ev_rows.append(dict(group=grp, hnorm_min=float(hn.min()), J=Jt, geo_sym_step=float(Gm.geodesic_sym(r["q"][t], r["q"][t + 1])), valid_at={str(tau): bool(hn.min() >= tau) for tau in (0.1, 0.2, 0.3)}))
    for g in ("singular", "physical", "wrap_crossing"):
        rr = [x for x in ev_rows if x["group"] == g]; evd[g] = dict(n=len(rr), median_hnorm_min=float(np.median([x["hnorm_min"] for x in rr])) if rr else None, max_hnorm_min=float(max(x["hnorm_min"] for x in rr)) if rr else None, median_J=float(np.median([x["J"] for x in rr])) if rr else None, median_geo_sym_step=float(np.median([x["geo_sym_step"] for x in rr])) if rr else None,
                                                    flagged_invalid={tau: int(sum(not x["valid_at"][tau] for x in rr)) for tau in ("0.1", "0.2", "0.3")})
    res["V4_events_vs_mask"] = evd
    # --- V5 mask coverage and physical rotation lost under the mask
    inv_frac = {}; lost = {}
    for kind in ("obs_degrees", "obs_radians", "interventional"):
        eids = [e for e in rows if (("interventional" if by[e]["kind"] == "interventional" else ("obs_radians" if tilt_unit(by[e]) == "radians" else "obs_degrees")) == kind) and e not in ev_eids]; tot = inv = 0
        for e in eids: hn = Gm.hnorm(rows[e]["q"]); v = (hn[:-1] >= 0.2) & (hn[1:] >= 0.2); tot += len(v); inv += int((~v).sum())
        inv_frac[kind] = dict(steps=tot, invalid_steps=inv, frac=inv / max(tot, 1))
    big_geo = hnm < 0.2; phys_under_mask = int(((geo > 0.1) & big_geo).sum()); res["V5_mask_coverage_non_event_episodes_tau0.2"] = inv_frac; res["V5_steps_with_geodesic_gt_0.1rad_under_mask"] = dict(n=phys_under_mask, of_masked_steps=int(big_geo.sum()), note="physical rotation that the legacy yaw channel would lose; the 3-D rotation-vector target keeps it supervised")
    # --- V6 tilt repair exactness, V7 D2 symmetry at rest-singular events
    res["V6_tilt_repair"] = dict(max_abs_diff_rad_after_unit_conversion=mx["tilt_vs_acos"], rule="degrees episodes (h23/xsweep/cov replays) are divided by 180/pi; radians episodes unchanged")
    srest = [x for x in ev if x["cat"].startswith("S_") and x["omega_max"] < 1e-3]; d2 = [float(Gm.geodesic_sym(rows[x["eid"]]["q"][x["t"]], rows[x["eid"]]["q"][x["t"] + 1])) for x in srest]; raw = [float(Gm.geodesic(rows[x["eid"]]["q"][x["t"]], rows[x["eid"]]["q"][x["t"] + 1])) for x in srest]
    res["V7_singular_at_rest"] = dict(n=len(srest), max_geodesic_sym_step=float(max(d2)) if d2 else None, max_plain_geodesic_step=float(max(raw)) if raw else None, median_abs_yaw_jump=float(np.median([abs(x["wrapped"]) for x in srest])) if srest else None)
    (OUT / "validation.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
if __name__ == "__main__": main()
