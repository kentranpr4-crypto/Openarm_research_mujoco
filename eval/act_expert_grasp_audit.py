#!/usr/bin/env python3
"""READ-ONLY audit of ACT vs scripted-expert grasp geometry, contact sequence and servo tracking, from STORED data only:
  xsweep replays (outputs/xsweep/replay/*.npz: 6 ACT checkpoints x 60 poses + expert x 60 poses; deterministic replays of the logged executed actions), xsweep rollouts (state, executed action, RAW ACT output), xsweep episode outcomes,
  wm1 expert control episodes (state, executed and requested expert actions, 100 poses), Stage-1 trajectories (contact block).  No simulator, no policy inference, no raw action reconstructed from executed targets.
POST-HOC DIAGNOSTIC (development): associations are not causal.   Run: python3 eval/act_expert_grasp_audit.py -> outputs/closed_loop_design/act_expert_grasp_audit.json"""
import json, glob, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; XS = ROOT / "outputs/xsweep"; DT = 0.05
CK = {"c0_s0": "c0_seed0", "c0_s1": "c0_seed1", "c0_s2": "c0_seed2", "s1_s0": "s1_seed0", "s1_s1": "s1_seed1", "s1_s2": "s1_seed2"}
POSES = {p["idx"]: p for p in json.loads((XS / "xsweep_config.json").read_text())["poses"]}
G1 = dict(depth=0.010, lateral=0.005, yaw=0.05, aperture=(0.008, 0.016), close_dur_s=(0.4, 1.0))        # G1 frozen perturbation ranges (relative to the expert grasp)
G_OPEN, CLOSE_TRIG, PALM_THR = 0.044, 0.040, 0.01
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def q(v, ps=(5, 25, 50, 75, 95)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return dict(n=int(len(v)), **{f"p{p}": float(np.percentile(v, p)) for p in ps}, mean=float(v.mean())) if len(v) else dict(n=0)
def feat(R, act, state=None, raw=None):
    """All quantities at/around t_c = first control step whose executed left-gripper target is < 0.040 m (4 mm below open).  Row t = observation BEFORE action t."""
    cmd = act[:, 7]; idx = np.where(cmd < CLOSE_TRIG)[0]
    if len(idx) == 0: return None
    t = int(idx[0]); yb = float(R["box_yaw"][t]); f = np.array([np.cos(yb), np.sin(yb)]); n = np.array([-np.sin(yb), np.cos(yb)]); d = R["tcp_pos"][t] - R["box_pos"][t]
    y0 = float(R["box_yaw"][0]); f0 = np.array([np.cos(y0), np.sin(y0)]); n0 = np.array([-np.sin(y0), np.cos(y0)]); d0 = R["tcp_pos"][t] - R["box_pos"][0]
    w = cmd[t:t + 41]; mn = float(w.min()); dur = int(np.argmax(w <= mn + 0.001)) * DT; spd = float(np.max(-np.diff(w)) / DT) if len(w) > 1 else 0.0; tcp = R["tcp_pos"]
    out = dict(t_close=t, df_cm=float(d[:2] @ f * 100), dn_cm=float(d[:2] @ n * 100), dz_cm=float(d[2] * 100), df0_cm=float(d0[:2] @ f0 * 100), dn0_cm=float(d0[:2] @ n0 * 100), a_close_cm=float(R["box_in_tcp"][t, 2] * 100), yaw_err_rad=float(wrap(R["hand_yaw"][t] - yb)), box_tilt_rad=float(R["box_tilt"][t]),
               gripper_q_m=float(R["gripper_q"][t]), min_cmd_m=mn, close_dur_s=dur, close_speed_ms=spd, tcp_speed_ms=float(np.linalg.norm(tcp[min(t + 1, 200)] - tcp[max(t - 1, 0)]) / (2 * DT)), tcp_box_dist_cm=float(np.linalg.norm(d) * 100),
               box_disp_before_close_cm=float(np.linalg.norm(R["box_pos"][t, :2] - R["box_pos"][0, :2]) * 100), box_yaw_change_before_close=float(abs(wrap(yb - y0))), action_left7=act[t, :7].copy())
    hf = np.where(R["hand_force"] > PALM_THR)[0]; fc = np.where(R["finger_contacts"] > 0)[0]; tp = int(hf[0]) if len(hf) else None; tf = int(fc[0]) if len(fc) else None
    out.update(t_palm=tp, t_finger=tf, palm_before_finger=(None if tp is None else bool(tf is None or tp < tf)), palm_minus_finger_steps=(None if (tp is None or tf is None) else tp - tf), palm_minus_close_steps=(None if tp is None else tp - t), finger_before_close_cmd=(None if tf is None else bool(tf < t)), palm_steps=int((R["hand_force"] > PALM_THR).sum()), palm_force_max_N=float(R["hand_force"].max()))
    rise = R["box_pos"][:, 2] - R["box_pos"][0, 2]; up = np.where(rise >= 0.05)[0]; tl = int(up[0]) if len(up) else None; td = None
    if tl is not None:
        dn_ = np.where(rise[tl:] < 0.02)[0]; td = tl + int(dn_[0]) if len(dn_) else None
    sep = np.linalg.norm(R["box_pos"] - R["tcp_pos"], axis=1); out.update(t_lift=tl, t_drop=td, max_box_tcp_sep_after_lift_cm=(float(sep[tl:].max() * 100) if tl is not None else None), box_disp_final_cm=float(np.linalg.norm(R["box_pos"][-1, :2] - R["box_pos"][0, :2]) * 100))
    if state is not None:
        e = act[max(t - 1, 0), :8] - state[t, :8]; win = np.abs(act[max(t - 21, 0):t, :7] - state[max(t - 20, 1):t + 1, :7][:len(act[max(t - 21, 0):t])]) if t >= 21 else None
        out.update(track_err_arm_norm_rad=float(np.linalg.norm(e[:7])), track_err_arm_max_rad=float(np.abs(e[:7]).max()), track_err_gripper_m=float(e[7]), track_err_arm_win20_mean_rad=(float(win.mean()) if win is not None else None))
    if raw is not None:
        rr = raw[:, 7]; ri = np.where(rr < CLOSE_TRIG)[0]; out.update(raw_first_close_step=(int(ri[0]) if len(ri) else None), raw_min_cmd_m=float(rr[t:t + 41].min()), raw_close_speed_ms=float(np.max(-np.diff(rr[t:t + 41])) / DT) if len(rr[t:t + 41]) > 1 else None,
                                                                    raw_vs_exec_arm_win_max_rad=float(np.abs(raw[max(t - 20, 0):t + 21, :7] - act[max(t - 20, 0):t + 21, :7]).max()), raw_vs_exec_grip_win_max_m=float(np.abs(raw[max(t - 20, 0):t + 21, 7] - act[max(t - 20, 0):t + 21, 7]).max()))
    return out
# ---------------------------------------------------------------------------------------------------------------------------------- load
EXP = {}; ACT = {}
for i in POSES:
    R = dict(np.load(XS / f"replay/expert_SW_{i:03d}.npz")); EXP[i] = feat(R, R["action"])
for k, d in CK.items():
    eps = {json.loads(l)["episode_index"]: json.loads(l) for l in open(XS / f"eval/{d}/act/sweep/episodes.jsonl")}
    for i in POSES:
        R = dict(np.load(XS / f"replay/{k}_SW_{i:03d}.npz")); T = dict(np.load(XS / f"eval/{d}/act/sweep/rollouts/ep{i:04d}.npz")); f_ = feat(R, R["action"], T["state"], T["raw_action"])
        if f_ is not None: f_.update(ckpt=k, pose=i, x=POSES[i]["x"], block=POSES[i]["block"], success=bool(eps[i]["task_success"]), outcome=eps[i]["outcome"], stratum=("low(<=0.37)" if POSES[i]["x"] <= 0.375 else "in_support(0.38-0.45)" if POSES[i]["x"] <= 0.455 else "high(>=0.46)"))
        ACT[(k, i)] = f_
out = dict(label="READ-ONLY post-hoc audit of stored data; development; associations, not causal evidence", n_act_episodes=len(ACT), n_act_with_close_command=sum(v is not None for v in ACT.values()), n_expert_with_close_command=sum(v is not None for v in EXP.values()))
rows = [v for v in ACT.values() if v is not None]
out["definitions"] = dict(t_close="first control step with executed left-gripper target < 0.040 m; row = observation before that action", frames="box frame: origin = box centre at t_close, x_f = [cos(yaw_box), sin(yaw_box), 0] (finger direction of the expert), n = [-sin, cos, 0], z up; df/dn/dz = TCP minus box centre [cm]; *0 = same but relative to the INITIAL box pose",
                          a_close_cm="box_in_tcp z component (box centre minus TCP along the TCP z-axis) [cm] (xsweep definition, evaluated at t_close)", yaw_err_rad="hand_yaw - box_yaw wrapped [rad]", min_cmd_m="min executed gripper target within 40 steps after t_close [m]", close_dur_s="time from t_close until the target is within 1 mm of its minimum [s]",
                          palm="hand-collision force > 0.01 N; finger contact = >=1 finger-box contact point", servo_lag="action[t-1] - state[t] (left arm [rad], gripper [m])", success="task_success from xsweep episodes.jsonl")
sanity = [r for r in EXP.values() if r]; out["frame_sanity_expert_at_close"] = dict(df_cm=q([r["df_cm"] for r in sanity]), dz_cm=q([r["dz_cm"] for r in sanity]), dn_cm=q([r["dn_cm"] for r in sanity]), a_close_cm=q([r["a_close_cm"] for r in sanity]), note="scripted expert: TCP = box centre + 0.01 f at height +0.025 m -> expect df=+1 cm, dz=+2.5 cm, dn=0")
# ---------------------------------------------------------------------------------------------------------------------------------- distributions
KEYS = ["df_cm", "dn_cm", "dz_cm", "a_close_cm", "yaw_err_rad", "gripper_q_m", "min_cmd_m", "close_dur_s", "close_speed_ms", "tcp_speed_ms", "t_close", "tcp_box_dist_cm", "box_disp_before_close_cm", "track_err_arm_norm_rad", "track_err_gripper_m", "palm_minus_close_steps", "palm_force_max_N", "palm_steps", "raw_min_cmd_m", "raw_close_speed_ms", "raw_vs_exec_grip_win_max_m", "raw_vs_exec_arm_win_max_rad"]
out["distributions"] = dict(expert={k: q([r[k] for r in EXP.values() if r and k in r]) for k in KEYS if any(r and k in r for r in EXP.values())}, act_all={k: q([r.get(k) for r in rows]) for k in KEYS},
                            act_by_stratum={s: {k: q([r.get(k) for r in rows if r["stratum"] == s]) for k in ("df_cm", "dn_cm", "dz_cm", "a_close_cm", "yaw_err_rad", "min_cmd_m", "close_dur_s")} for s in ("low(<=0.37)", "in_support(0.38-0.45)", "high(>=0.46)")},
                            act_success={k: q([r.get(k) for r in rows if r["success"]]) for k in KEYS}, act_failure={k: q([r.get(k) for r in rows if not r["success"]]) for k in KEYS}, n_success=sum(r["success"] for r in rows), n_failure=sum(not r["success"] for r in rows))
# ---------------------------------------------------------------------------------------------------------------------------------- ACT - expert errors (same pose)
err = []
for (k, i), r in ACT.items():
    e = EXP.get(i)
    if r is None or e is None: continue
    err.append(dict(ckpt=k, pose=i, success=r["success"], stratum=r["stratum"], d_df_cm=r["df_cm"] - e["df_cm"], d_dn_cm=r["dn_cm"] - e["dn_cm"], d_dz_cm=r["dz_cm"] - e["dz_cm"], d_yaw_rad=float(wrap(r["yaw_err_rad"] - e["yaw_err_rad"])), d_a_close_cm=r["a_close_cm"] - e["a_close_cm"], d_min_cmd_m=r["min_cmd_m"] - e["min_cmd_m"], d_close_dur_s=r["close_dur_s"] - e["close_dur_s"], d_t_close=r["t_close"] - e["t_close"],
                    joint_target_l2_rad=float(np.linalg.norm(r["action_left7"] - e["action_left7"])), joint_target_max_rad=float(np.abs(r["action_left7"] - e["action_left7"]).max()), cmd_close_dur_s=r["close_dur_s"], min_cmd_m=r["min_cmd_m"], palm=r["palm_steps"] > 0))
EK = ["d_df_cm", "d_dn_cm", "d_dz_cm", "d_yaw_rad", "d_a_close_cm", "d_min_cmd_m", "d_close_dur_s", "d_t_close", "joint_target_l2_rad", "joint_target_max_rad"]
out["act_minus_expert_same_pose"] = dict(all={k: q([e[k] for e in err]) for k in EK}, success={k: q([e[k] for e in err if e["success"]]) for k in EK}, failure={k: q([e[k] for e in err if not e["success"]]) for k in EK}, by_stratum={s: {k: q([e[k] for e in err if e["stratum"] == s]) for k in ("d_df_cm", "d_dn_cm", "d_dz_cm", "d_yaw_rad", "d_a_close_cm")} for s in ("low(<=0.37)", "in_support(0.38-0.45)", "high(>=0.46)")})
# ---------------------------------------------------------------------------------------------------------------------------------- G1 coverage
def cov(sel):
    n = len(sel)
    if not n: return dict(n=0)
    dep = np.array([abs(e["d_df_cm"]) / 100 <= G1["depth"] for e in sel]); lat = np.array([abs(e["d_dn_cm"]) / 100 <= G1["lateral"] for e in sel]); yaw = np.array([abs(e["d_yaw_rad"]) <= G1["yaw"] for e in sel]); ap = np.array([G1["aperture"][0] <= e["min_cmd_m"] <= G1["aperture"][1] for e in sel]); du = np.array([G1["close_dur_s"][0] <= e["cmd_close_dur_s"] <= G1["close_dur_s"][1] for e in sel])
    return dict(n=n, depth_within=float(dep.mean()), lateral_within=float(lat.mean()), yaw_within=float(yaw.mean()), aperture_within=float(ap.mean()), close_duration_within=float(du.mean()), geometry_all_three_within=float((dep & lat & yaw).mean()), all_five_within=float((dep & lat & yaw & ap & du).mean()),
                depth_exceed_cm_p50_p95=[float(np.percentile([max(0.0, abs(e["d_df_cm"]) - 1.0) for e in sel], 50)), float(np.percentile([max(0.0, abs(e["d_df_cm"]) - 1.0) for e in sel], 95))], lateral_exceed_cm_p95=float(np.percentile([max(0.0, abs(e["d_dn_cm"]) - 0.5) for e in sel], 95)), yaw_exceed_rad_p95=float(np.percentile([max(0.0, abs(e["d_yaw_rad"]) - 0.05) for e in sel], 95)),
                dz_error_cm_abs_p50_p95_NOT_VARIED_IN_G1=[float(np.percentile([abs(e["d_dz_cm"]) for e in sel], 50)), float(np.percentile([abs(e["d_dz_cm"]) for e in sel], 95))])
out["G1_perturbation_coverage"] = dict(ranges=G1, all=cov(err), failures=cov([e for e in err if not e["success"]]), successes=cov([e for e in err if e["success"]]), low_x=cov([e for e in err if e["stratum"] == "low(<=0.37)"]), in_support=cov([e for e in err if e["stratum"] == "in_support(0.38-0.45)"]),
                                       not_evaluable=["pre-grasp distance (ACT has no pre-grasp waypoint)", "closing dwell (no waiting concept in ACT)", "vertical (dz) error is real but G1 did not vary height"])
# ---------------------------------------------------------------------------------------------------------------------------------- failure vs success (association) with pose-cluster bootstrap AUROC
def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
def boot_auroc(metric, n=2000, seed=0):
    ps = sorted({e["pose"] for e in err}); by = {p: [e for e in err if e["pose"] == p] for p in ps}; rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        smp = sum([by[ps[j]] for j in rng.integers(0, len(ps), len(ps))], []); a = auroc([not e["success"] for e in smp], [metric(e) for e in smp])
        if np.isfinite(a): vals.append(a)
    return [auroc([not e["success"] for e in err], [metric(e) for e in err]), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
out["failure_association_AUROC_pose_bootstrap_(0.5=none)"] = dict(abs_depth_err=boot_auroc(lambda e: abs(e["d_df_cm"])), signed_depth_err_deeper=boot_auroc(lambda e: e["d_df_cm"]), abs_lateral_err=boot_auroc(lambda e: abs(e["d_dn_cm"])), abs_yaw_err=boot_auroc(lambda e: abs(e["d_yaw_rad"])), abs_dz_err=boot_auroc(lambda e: abs(e["d_dz_cm"])), joint_target_l2=boot_auroc(lambda e: e["joint_target_l2_rad"]),
                                                                  close_dur=boot_auroc(lambda e: e["cmd_close_dur_s"]), palm_any=boot_auroc(lambda e: float(e["palm"])), note="associational; 6 checkpoints share each pose and x is confounded with both errors and failure")
# ---------------------------------------------------------------------------------------------------------------------------------- contact sequence
pr = [r for r in rows if r["t_palm"] is not None]
out["contact_sequence_act"] = dict(n_episodes=len(rows), n_with_palm=len(pr), palm_before_first_finger_contact_frac=float(np.mean([r["palm_before_finger"] for r in pr])) if pr else None, palm_minus_first_finger_steps=q([r["palm_minus_finger_steps"] for r in pr]), palm_minus_close_cmd_steps=q([r["palm_minus_close_steps"] for r in pr]),
                                   finger_before_close_cmd_frac=float(np.mean([r["finger_before_close_cmd"] for r in rows if r["finger_before_close_cmd"] is not None])), palm_steps=q([r["palm_steps"] for r in pr]), palm_force_max_N=q([r["palm_force_max_N"] for r in pr]), P_fail_given_palm=float(np.mean([not r["success"] for r in pr])) if pr else None,
                                   P_fail_given_no_palm=float(np.mean([not r["success"] for r in rows if r["t_palm"] is None])), expert_palm_episodes=sum(1 for r in EXP.values() if r and r["t_palm"] is not None), box_disp_before_close_cm_act=q([r["box_disp_before_close_cm"] for r in rows]), box_disp_before_close_cm_expert=q([r["box_disp_before_close_cm"] for r in EXP.values() if r]))
fl = [r for r in rows if not r["success"]]; out["grasp_loss_timing_act_failures"] = dict(n_failures=len(fl), t_lift=q([r["t_lift"] for r in fl]), t_drop=q([r["t_drop"] for r in fl]), n_never_lifted=sum(r["t_lift"] is None for r in fl), n_lifted_then_dropped=sum(r["t_drop"] is not None for r in fl), max_sep_after_lift_cm=q([r["max_box_tcp_sep_after_lift_cm"] for r in fl]), box_disp_final_cm=q([r["box_disp_final_cm"] for r in fl]))
# Stage-1 cross-check (contact block c_rows: col1 finger contacts, col4 hand force)
t1 = []
for f in glob.glob(str(ROOT / "outputs/closed_loop_pilot/results/traj/*.npz")):
    c = np.load(f)["c_rows"]; hp = np.where(c[:, 4] > PALM_THR)[0]; fp = np.where(c[:, 1] > 0)[0]
    if len(hp): t1.append((int(hp[0]), int(fp[0]) if len(fp) else None))
out["stage1_crosscheck_palm_vs_finger"] = dict(n_with_palm=len(t1), palm_before_finger_frac=float(np.mean([(b is None) or (a < b) for a, b in t1])) if t1 else None)
# ---------------------------------------------------------------------------------------------------------------------------------- servo lag / expert reference (wm1 expert control episodes) and ACT raw vs executed
tl = [];
for f in sorted(glob.glob(str(ROOT / "outputs/wm1/intervention_full/episodes/p*_v00.npz"))):
    Z = dict(np.load(f)); a = Z["action"]; idx = np.where(a[:, 7] < CLOSE_TRIG)[0]
    if len(idx) and idx[0] > 21: t = int(idx[0]); e = a[t - 1, :8] - Z["state"][t, :8]; tl.append(dict(arm_norm=float(np.linalg.norm(e[:7])), arm_max=float(np.abs(e[:7]).max()), grip=float(e[7]), req_vs_exec_arm_max=float(np.abs(Z["requested_action"][max(t - 20, 0):t + 21, :7] - a[max(t - 20, 0):t + 21, :7]).max())))
out["servo_tracking_at_close"] = dict(expert_control_episodes_wm1=dict(n=len(tl), arm_norm_rad=q([x["arm_norm"] for x in tl]), arm_max_rad=q([x["arm_max"] for x in tl]), gripper_m=q([x["grip"] for x in tl]), requested_vs_executed_arm_max_rad=q([x["req_vs_exec_arm_max"] for x in tl]), note="different poses than xsweep (wm1 intervention manifest), same scripted expert; distribution-level comparison only"),
                                       act_xsweep=dict(arm_norm_rad=q([r.get("track_err_arm_norm_rad") for r in rows]), arm_max_rad=q([r.get("track_err_arm_max_rad") for r in rows]), gripper_m=q([r.get("track_err_gripper_m") for r in rows])),
                                       act_raw_vs_executed=dict(arm_win_max_rad=q([r.get("raw_vs_exec_arm_win_max_rad") for r in rows]), gripper_win_max_m=q([r.get("raw_vs_exec_grip_win_max_m") for r in rows]), raw_close_speed_ms=q([r.get("raw_close_speed_ms") for r in rows]), exec_close_speed_ms=q([r["close_speed_ms"] for r in rows]), raw_min_cmd_m=q([r.get("raw_min_cmd_m") for r in rows]), exec_min_cmd_m=q([r["min_cmd_m"] for r in rows])),
                                       expert_xsweep_servo_lag="NOT_EVALUABLE (xsweep expert replay stores no joint state)")

# ---------------------------------------------------------------------------------------------------------------------------------- descriptive binned association + outcome-agnostic error quantiles for the design
def binrow(sel): return dict(n=len(sel), fail_rate=float(np.mean([not e["success"] for e in sel])) if sel else None, palm_rate=float(np.mean([e["palm"] for e in sel])) if sel else None)
edges = [-99, -2, -1, 0, 1, 2, 3, 99]; out["depth_error_bins_descriptive"] = {f"[{edges[i]},{edges[i + 1]})cm": binrow([e for e in err if edges[i] <= e["d_df_cm"] < edges[i + 1]]) for i in range(len(edges) - 1)}
sp_ = {r["pose"]: r for r in []}; rowsm = {(r["ckpt"], r["pose"]): r for r in rows}; bs = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 9)]
out["tcp_speed_at_close_bins_descriptive"] = {f"[{a},{b})m/s": binrow([e for e in err if a <= rowsm[(e["ckpt"], e["pose"])]["tcp_speed_ms"] < b]) for a, b in bs}
out["pooled_error_quantiles_outcome_agnostic_for_design"] = {k: {f"p{p}": float(np.percentile([e[k] for e in err], p)) for p in (5, 10, 25, 50, 75, 90, 95)} for k in ("d_df_cm", "d_dn_cm", "d_yaw_rad", "d_dz_cm")}
out["error_correlations"] = {f"{a}~{b}": float(np.corrcoef([e[a] for e in err], [e[b] for e in err])[0, 1]) for a, b in (("d_df_cm", "d_dn_cm"), ("d_df_cm", "d_yaw_rad"), ("d_dn_cm", "d_yaw_rad"))}
out["not_evaluable"] = ["raw (pre-clip) expert actions for the xsweep poses (replay stores executed only; wm1 'requested_action' exists for other poses)", "expert joint state / servo lag on the xsweep poses", "pre-grasp distance and closing dwell for ACT", "Stage-1 raw ACT outputs (not saved); Stage-1 pose-level geometry beyond the contact block", "ACT intent: whether a deeper grasp was commanded deliberately cannot be inferred from executed targets"]
(ROOT / "outputs/closed_loop_design/act_expert_grasp_audit.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in out if k not in ("distributions", "definitions")}, indent=1, default=float)[:14000])
