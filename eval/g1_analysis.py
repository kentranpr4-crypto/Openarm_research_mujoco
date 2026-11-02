#!/usr/bin/env python3
"""G1 analysis (numpy only; pre-registered in outputs/g1_grasp/protocol_freeze.json).  DEVELOPMENT; oracle-pose parametric grasp candidates; outcomes = executed lift-and-hold in simulation.  Run: python3 eval/g1_analysis.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); import branching_metrics as BM, g1_grasp as G
R = json.loads((ROOT / "outputs/g1_grasp/results/g1_results.json").read_text()); poses = R["poses"]; STATES = G.STATES
out = dict(label="DEVELOPMENT G1: oracle-pose parametric grasp candidates; executed lift-and-hold outcomes; not independent, not Sim2Real", run_status=R["status"], stop_reason=R.get("stop_reason"), cpu_s_run=R.get("cpu_s"))
cands = [(p["pose"]["idx"], s["state"], c) for p in poses for s in p["states"] for c in s["candidates"]]
out["counts"] = dict(n_poses=len(poses), n_states=sum(len(p["states"]) for p in poses), n_candidates=len(cands), n_executed_branches=sum(c["outcome"] is not None for _, _, c in cands), expected=dict(poses=24, states=72, candidates=648), n_nominal_plan_not_ok=sum(not p["nominal_plan_ok"] for p in poses))
valid = [c for _, _, c in cands if c["valid"]]; out["validity"] = dict(valid_fraction=len(valid) / max(len(cands), 1), by_state={s: float(np.mean([c["valid"] for _, st, c in cands if st == s])) for s in STATES}, invalid_reasons={r: sum(c.get("invalid_reason") == r for _, _, c in cands) for r in ("ik_or_prefix", "illegal", "non_finite", "obstacle_collision")},
                                                         obstacle_contact_candidates=sum(bool(c["outcome"] and c["outcome"]["obstacle_contact"]) for _, _, c in cands))
nd = [c for _, _, c in cands if c.get("distinct_vs_nominal") is not None and c["cand"] != "c0"]; out["physical_diversity"] = dict(distinct_vs_nominal_fraction=float(np.mean([c["distinct_vs_nominal"] for c in nd])) if nd else None, arm_linf_rad_median=float(np.median([c["div_arm_linf"] for c in nd])) if nd else None, gripper_linf_m_median=float(np.median([c["div_grip_linf"] for c in nd])) if nd else None,
                                                  tcp_max_dev_m_median=float(np.median([c["div_tcp_max_m"] for c in nd])) if nd else None, distinct_clusters_per_state_mean=float(np.mean([s["n_distinct_clusters"] for p in poses for s in p["states"]])), distinct_clusters_by_state={s: float(np.mean([st["n_distinct_clusters"] for p in poses for st in p["states"] if st["state"] == s])) for s in STATES})
oc = [(i, st, c) for i, st, c in cands if c["valid"] and c["outcome"] is not None]; out["outcome_distribution"] = dict(overall={r: sum(c["outcome"]["reason"] == r for _, _, c in oc) for r in ("success", "not_closed", "not_lifted", "dropped", "slipped")},
                                                                                                                      success_rate_by_state={s: float(np.mean([c["outcome"]["success"] for _, st, c in oc if st == s])) for s in STATES}, palm_rate_by_state={s: float(np.mean([c["outcome"]["palm"] for _, st, c in oc if st == s])) for s in STATES},
                                                                                                                      palm_free_success_rate_by_state={s: float(np.mean([c["outcome"]["success"] and not c["outcome"]["palm"] for _, st, c in oc if st == s])) for s in STATES}, slip_m_median_when_closed=float(np.median([c["outcome"]["slip_m"] for _, _, c in oc if c["outcome"]["closed"]])) if any(c["outcome"]["closed"] for _, _, c in oc) else None)
fx = {}
for s in STATES:
    for f in G.FACTORS[s]:
        lo = [c["outcome"]["success"] for _, st, c in oc if st == s and c["levels"].get(f) == -1]; hi = [c["outcome"]["success"] for _, st, c in oc if st == s and c["levels"].get(f) == 1]
        fx[f"{s}:{f}"] = dict(success_low=float(np.mean(lo)) if lo else None, success_high=float(np.mean(hi)) if hi else None, n_low=len(lo), n_high=len(hi))
out["main_effects_success_by_factor_level"] = fx
byp = {}; pol = {k: {} for k in ("default", "random", "heuristic", "oracle")}; pol_pf = {k: {} for k in ("default", "random", "heuristic", "oracle")}; head = {}; head_q = {}; n_eval = 0; n_head = 0; n_ceiling = 0; n_none = 0
for p in poses:
    for st in p["states"]:
        cs = {c["cand"]: c for c in st["candidates"]}; c0 = cs.get("c0")
        if not (c0 and c0["valid"] and c0["outcome"]): continue
        vs = [c for c in st["candidates"] if c["valid"] and c["outcome"]]; succ = lambda c: float(c["outcome"]["success"]); pf = lambda c: float(c["outcome"]["success"] and not c["outcome"]["palm"]); pid = p["pose"]["idx"]; n_eval += 1
        key = "dd" if st["state"] in ("D1", "D2") else "close_dur"; hc = next((c for c in st["candidates"] if c["cand"] != "c0" and c["levels"].get(key) == (-1 if key == "dd" else 1)), None); hv = hc if (hc and hc["valid"] and hc["outcome"]) else c0
        for name, fn, store in (("succ", succ, pol), ("pf", pf, pol_pf)):
            store["default"].setdefault(pid, []).append(fn(c0)); store["random"].setdefault(pid, []).append(float(np.mean([fn(c) for c in vs]))); store["heuristic"].setdefault(pid, []).append(fn(hv)); store["oracle"].setdefault(pid, []).append(max(fn(c) for c in vs))
        d = succ(c0); b = max(succ(c) for c in vs); h = bool(d == 0 and b == 1); head.setdefault(pid, []).append(float(h)); n_head += h; n_ceiling += d == 1; n_none += (d == 0 and b == 0)
        q = [c0["outcome"]["slip_m"] - c["outcome"]["slip_m"] for c in vs if c["outcome"]["success"] and c0["outcome"]["success"]]; head_q.setdefault(pid, []).append(float(max(q)) if q else 0.0)
out["headroom"] = dict(definition="state has headroom iff DEFAULT (nominal c0) GRASP_SUCCESS = 0 and at least one valid candidate GRASP_SUCCESS = 1 (binary gap = 1 >= 0.5)", n_states_evaluable=n_eval, n_states_with_headroom=int(n_head), frac_states_with_headroom=n_head / max(n_eval, 1), frac_states_default_succeeds_ceiling=n_ceiling / max(n_eval, 1), frac_states_default_fails_and_all_candidates_fail=n_none / max(n_eval, 1),
                       frac_with_headroom_pose_bootstrap_ci=BM.boot_mean_by_pose(head) if head else None, secondary_slip_headroom_m_mean_pose_ci=BM.boot_mean_by_pose(head_q) if head_q else None)
def pols(store): return {k: BM.boot_mean_by_pose(v) for k, v in store.items()} if store["default"] else None
out["policy_comparison_grasp_success"] = pols(pol); out["policy_comparison_palm_free_success"] = pols(pol_pf)
out["paired_differences_grasp_success"] = {f"{a}_minus_default": BM.boot_paired_by_pose(pol[a], pol["default"]) for a in ("random", "heuristic", "oracle")} if pol["default"] else None
out["by_state_headroom"] = {s: dict(n=sum(1 for p in poses for st in p["states"] if st["state"] == s and st["candidates"] and st["candidates"][0]["valid"] and st["candidates"][0]["outcome"]), with_headroom=int(sum(1 for p in poses for st in p["states"] if st["state"] == s and (lambda cs: cs[0]["valid"] and cs[0]["outcome"] and (not cs[0]["outcome"]["success"]) and any(c["valid"] and c["outcome"] and c["outcome"]["success"] for c in cs))(st["candidates"])))) for s in STATES}
out["by_stratum_default_success"] = {}
for stratum in ("hard", "mid"):
    v = [float(st["candidates"][0]["outcome"]["success"]) for p in poses if p["pose"]["stratum"] == stratum for st in p["states"] if st["candidates"] and st["candidates"][0]["valid"] and st["candidates"][0]["outcome"]]; out["by_stratum_default_success"][stratum] = dict(n=len(v), mean=float(np.mean(v)) if v else None)
stop = []
if out["validity"]["valid_fraction"] < 0.8: stop.append("valid candidate fraction < 80 %")
if n_eval == 0: stop.append("no evaluable states (default invalid everywhere) -> NOT_EVALUABLE")
elif 1 - n_head / n_eval > 0.8: stop.append("more than 80 % of decision states lack oracle outcome headroom")
if R["status"] != "COMPLETE": stop.append(f"run status {R['status']}")
out["stop_rules"] = dict(triggered=stop, verdict="STOP" if any("<" in s or "80 %" in s for s in stop) else ("PARTIAL" if stop else "PASS"))
(ROOT / "outputs/g1_grasp/results/g1_analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float)[:9000])
