#!/usr/bin/env python3
"""G2a analysis (numpy; pre-registered in outputs/g2a_act_corrections/protocol_freeze.json).  DEVELOPMENT; oracle-assisted interventions on the ACT grasp; short-horizon (t=40..100) lift-and-hold only.  Run (ACT venv): python eval/g2a_analysis.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); import branching_metrics as BM
D = ROOT / "outputs/g2a_act_corrections"; R = json.loads((D / "results/g2a_results.json").read_text()); rep = json.loads((D / "integration_report.json").read_text()); poses = R["poses"]; IDS = [f"c{i}" for i in range(9)]
out = dict(label="DEVELOPMENT G2a: oracle-assisted ACT grasp corrections; short-horizon lift-and-hold (t=40..100); not independent, not Sim2Real, NOT end-to-end carry/place", run_status=R["status"], stop_reason=R.get("stop_reason"), cpu_s_run=R.get("cpu_s"))
app = [p for p in poses if p["applicable"]]; cand = lambda p, c: next(x for x in p["candidates"] if x["cand"] == c)
out["counts"] = dict(n_poses=len(poses), n_applicable_states=len(app), n_not_applicable=len(poses) - len(app), not_applicable_reasons={r: sum(p.get("not_applicable_reason") == r for p in poses) for r in {p.get("not_applicable_reason") for p in poses if not p["applicable"]}}, n_candidates=sum(len(p["candidates"]) for p in app), n_executed_branches=sum(len(p["candidates"]) for p in app), expected_max_branches=432, strata=dict(hard=sum(p["pose"]["stratum"] == "hard" for p in app), mid=sum(p["pose"]["stratum"] == "mid" for p in app)))
allc = [x for p in app for x in p["candidates"]]; out["validity"] = dict(valid_fraction=float(np.mean([x["valid"] for x in allc])) if allc else None, by_candidate={c: float(np.mean([cand(p, c)["valid"] for p in app])) for c in IDS} if app else None, invalid_ik=sum(x["ik_fail"] > 0 for x in allc), invalid_illegal=sum(not x["legal"] for x in allc), invalid_nonfinite=sum(not x["finite"] for x in allc), invalid_penetration=sum(x["outcome"] is not None and x["outcome"]["obstacle_penetration_max_m"] > 0.002 for x in allc),
                                                       ik_max_pos_err_m=max(x["ik_max_pos_err_m"] for x in allc) if allc else None, ik_max_rot_err_rad=max(x["ik_max_rot_err_rad"] for x in allc) if allc else None)
out["nominal_reproduction_gate"] = next(g["evidence"] for g in rep["gates"] if g["gate"] == 2)
q = lambda v: dict(n=len(v), p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95))) if len(v) else dict(n=0)
out["physical_diversity"] = {c: dict(distinct_fraction=float(np.mean([cand(p, c)["distinct_vs_nominal"] for p in app])), arm_linf_rad=q([cand(p, c)["arm_linf_vs_nominal_rad"] for p in app]), tcp_dev_m=q([cand(p, c)["tcp_dev_vs_nominal_m"] for p in app]), df_shift_cm_at_close=q([cand(p, c)["frame_at_close"]["df_cm"] - cand(p, "c0")["frame_at_close"]["df_cm"] for p in app if cand(p, c)["frame_at_close"] and cand(p, "c0")["frame_at_close"]]),
                                    applied_correction=dict(f_cm=q([100 * cand(p, c)["applied"].get("f", 0) for p in app]), n_cm=q([100 * cand(p, c)["applied"].get("n", 0) for p in app]), psi_rad=q([cand(p, c)["applied"].get("psi", 0) for p in app]))) for c in IDS} if app else None
out["oracle_error_at_decision_state"] = dict(e_f_cm=q([100 * p["oracle_error"]["e_f"] for p in app]), e_n_cm=q([100 * p["oracle_error"]["e_n"] for p in app]), e_psi_rad=q([p["oracle_error"]["e_psi"] for p in app]), j_close=q([p["raw_j_close"] for p in app]))
def S(p, c):
    x = cand(p, c); return float(x["outcome"]["success"]) if (x["valid"] and x["outcome"]) else None
bypose = {c: {p["pose"]["idx"]: S(p, c) for p in app} for c in IDS}
rate = {}
for c in IDS:
    v = [x for x in bypose[c].values() if x is not None]; rate[c] = dict(n=len(v), success=float(np.mean(v)) if v else None, bootstrap_ci_by_pose=BM.boot_mean_by_pose({i: [x] for i, x in bypose[c].items() if x is not None}) if v else None)
out["success_rate_by_candidate"] = rate
palm = {c: dict(palm_rate=float(np.mean([bool(cand(p, c)["outcome"]["palm"]) for p in app if cand(p, c)["outcome"]])), palm_free_success=float(np.mean([bool(cand(p, c)["outcome"]["success"] and not cand(p, c)["outcome"]["palm"]) for p in app if cand(p, c)["outcome"]])), palm_force_max_N=q([cand(p, c)["outcome"]["palm_force_max_N"] for p in app if cand(p, c)["outcome"] and cand(p, c)["outcome"]["palm"]]),
                palm_before_finger_frac=(lambda v: float(np.mean(v)) if v else None)([cand(p, c)["outcome"]["palm_before_finger"] for p in app if cand(p, c)["outcome"] and cand(p, c)["outcome"]["palm"]]), slip_m=q([cand(p, c)["outcome"]["slip_m"] for p in app if cand(p, c)["outcome"] and cand(p, c)["outcome"]["slip_m"] is not None]), finger_force_max_N=q([cand(p, c)["outcome"]["finger_force_max_N"] for p in app if cand(p, c)["outcome"]]),
                reasons={r: sum(1 for p in app if cand(p, c)["outcome"] and cand(p, c)["outcome"]["reason"] == r) for r in ("success", "not_closed", "not_lifted", "late_lift", "dropped", "slipped")}) for c in IDS} if app else None
out["contact_slip_by_candidate"] = palm
# headroom: nominal fails and >=1 valid candidate succeeds (binary)
ev = [p for p in app if S(p, "c0") is not None]; hr = {}; n_head = 0; n_nom_fail = 0
for p in ev:
    vs = [S(p, c) for c in IDS if S(p, c) is not None]; h = bool(S(p, "c0") == 0 and max(vs) == 1); n_head += h; n_nom_fail += S(p, "c0") == 0; hr[p["pose"]["idx"]] = [float(h)]
out["headroom"] = dict(definition="state has oracle binary headroom iff nominal ACT GRASP_SUCCESS = 0 and >= 1 valid candidate GRASP_SUCCESS = 1", n_states_evaluable=len(ev), n_nominal_fail=int(n_nom_fail), n_with_headroom=int(n_head), frac_with_headroom=n_head / max(len(ev), 1), frac_with_headroom_ci=BM.boot_mean_by_pose(hr) if hr else None, frac_nominal_succeeds=float(np.mean([S(p, "c0") == 1 for p in ev])) if ev else None,
                       oracle_best_of_9_success=BM.boot_mean_by_pose({p["pose"]["idx"]: [max(S(p, c) for c in IDS if S(p, c) is not None)] for p in ev}) if ev else None, nominal_success=BM.boot_mean_by_pose({p["pose"]["idx"]: [S(p, "c0")] for p in ev}) if ev else None,
                       by_stratum={s: dict(n=sum(p["pose"]["stratum"] == s for p in ev), nominal_fail=sum(p["pose"]["stratum"] == s and S(p, "c0") == 0 for p in ev), headroom=sum(p["pose"]["stratum"] == s and hr[p["pose"]["idx"]][0] == 1.0 for p in ev)) for s in ("hard", "mid")},
                       verdict="NO_HEADROOM" if (ev and n_nom_fail == 0) else None)
def paired(a, b):
    ks = [p["pose"]["idx"] for p in ev if S(p, a) is not None and S(p, b) is not None]
    if not ks: return dict(n=0, verdict="NOT_EVALUABLE")
    d = {k: [S(next(p for p in ev if p["pose"]["idx"] == k), a) - S(next(p for p in ev if p["pose"]["idx"] == k), b)] for k in ks}; v = np.array([x[0] for x in d.values()]); ci = BM.boot_mean_by_pose(d)
    return dict(n=len(ks), mean_diff=ci[0], ci95=ci[1:], improved=int((v > 0).sum()), harmed=int((v < 0).sum()), unchanged=int((v == 0).sum()), verdict="SUPPORTED_BETTER" if ci[1] > 0 else ("SUPPORTED_WORSE" if ci[2] < 0 else "INCONCLUSIVE"))
out["paired_vs_nominal"] = {c: paired(c, "c0") for c in IDS[1:]}; out["PRIMARY_depth_only_c1_vs_nominal"] = out["paired_vs_nominal"]["c1"]; out["timing_depth_only_early_c1_minus_late_c2"] = paired("c1", "c2")
rnd = {p["pose"]["idx"]: [float(np.mean([S(p, c) for c in IDS if S(p, c) is not None])) - S(p, "c0")] for p in ev}; out["random_candidate_minus_nominal"] = dict(note="expected over the 9 valid candidates incl. nominal", ci=BM.boot_mean_by_pose(rnd) if rnd else None)
out["by_stratum_success"] = {s: {c: (lambda v: float(np.mean(v)) if v else None)([S(p, c) for p in ev if p["pose"]["stratum"] == s and S(p, c) is not None]) for c in ("c0", "c1", "c3", "c5", "c6")} for s in ("hard", "mid")}
out["end_to_end_carry_place"] = "NOT_EVALUABLE (horizon ends at t=100, before the median ACT carry drop at ~105; no claim about carry/place)"
stop = []
if out["validity"]["valid_fraction"] is not None and out["validity"]["valid_fraction"] < 0.8: stop.append("candidate validity < 80 %")
if ev and n_nom_fail == 0: stop.append("NO_HEADROOM: all nominal ACT grasps succeed")
elif ev and 1 - n_head / len(ev) > 0.8: stop.append("more than 80 % of decision states without oracle binary headroom")
if not ev: stop.append("no evaluable states")
if R["status"] != "COMPLETE": stop.append(f"run status {R['status']}")
out["stop_rules"] = dict(triggered=stop, verdict="STOP" if any(("< 80" in s) or ("NO_HEADROOM" in s) or ("80 %" in s) for s in stop) else ("PARTIAL" if stop else "PASS"))
(D / "results/g2a_analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float)[:15000])
