#!/usr/bin/env python3
"""Stage 1 pilot analysis (pure numpy; pre-registered in outputs/closed_loop_pilot/protocol_freeze.json).  EXPLORATORY SIMULATION PILOT; DEVELOPMENT evidence; privileged oracle observer; NOT Sim2Real.
Run: python3 eval/closed_loop_pilot_analysis.py"""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "outputs/closed_loop_pilot"
R = json.loads((D / "results/pilot_results.json").read_text()); eps = [e for e in R["episodes"] if e.get("status") == "OK"]; hist = {}
for ln in open(ROOT / "outputs/xsweep/eval/c0_seed0/act/sweep/episodes.jsonl"): r = json.loads(ln); hist[r["episode_index"]] = r
poses = sorted({e["pose_idx"] for e in eps}); out = dict(label="EXPLORATORY SIMULATION PILOT (development evidence, privileged oracle observer, not Sim2Real)", run_status=R["status"], stop_reason=R.get("stop_reason"), n_episodes_ok=len(eps), n_non_ok=sum(e.get("status") != "OK" for e in R["episodes"]))
def arm_succ(arm):
    d = {}
    for e in eps:
        if e["arm"] == arm: d.setdefault(e["pose_idx"], []).append(float(e["task_success"]))
    return d
a0, a1, a2 = arm_succ("A0"), arm_succ("A1"), arm_succ("A2")
out["episodes_per_arm"] = {"A0": sum(len(v) for v in a0.values()), "A1": sum(len(v) for v in a1.values()), "A2": sum(len(v) for v in a2.values())}
def boot(vals, n=10000, seed=0):
    v = np.asarray(vals, float); rng = np.random.default_rng(seed); idx = rng.integers(0, len(v), (n, len(v))); m = v[idx].mean(1); return float(v.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
def paired(x, y, name):
    common = sorted(set(x) & set(y))
    if not common: return dict(name=name, n_poses=0, verdict="NOT_EVALUABLE")
    dx = np.array([np.mean(x[p]) - np.mean(y[p]) for p in common]); mean, ci = boot(dx); v = "SUPPORTED_BETTER" if ci[0] > 0 else ("SUPPORTED_WORSE" if ci[1] < 0 else "INCONCLUSIVE")
    return dict(name=name, n_poses=len(common), mean_diff=mean, ci95=ci, verdict=v, per_pose={int(p): float(d) for p, d in zip(common, dx)}, poses_better=int((dx > 0).sum()), poses_worse=int((dx < 0).sum()), poses_equal=int((dx == 0).sum()))
out["primary_A2_minus_A1"] = paired(a2, a1, "A2 - mean(A1 repeats), paired by pose, bootstrap over poses (10000, seed 0)")
out["secondary_A2_minus_A0"] = paired(a2, a0, "A2 - A0"); out["secondary_A1_minus_A0"] = paired(a1, a0, "mean(A1) - A0")
out["success_counts"] = {"A0": [int(sum(np.mean(v) for v in a0.values())), len(a0)], "A2": [int(sum(np.mean(v) for v in a2.values())), len(a2)], "A1_mean_successes_per_pose_sum": float(sum(np.mean(v) for v in a1.values())), "A1_n_poses": len(a1)}
# discordant poses
com = sorted(set(a2) & set(a1)); out["discordant_poses_A2_vs_A1"] = {int(p): dict(A2=a2[p][0], A1=a1[p]) for p in com if (a2[p][0] == 1.0) != (np.mean(a1[p]) >= 0.5)}
# failure classification
fc = {}
for arm in ("A0", "A1", "A2"):
    E = [e for e in eps if e["arm"] == arm]; fc[arm] = dict(n=len(E), outcomes={k: sum(e["outcome"] == k for e in E) for k in ("success", "no_pick", "dropped_or_misplaced")}, palm_contact=sum(e["palm_contact"] for e in E), cluster_flag=sum(e["cluster_flag"] for e in E), action_invalid_chunks=sum(e["action_invalid_chunks"] for e in E), non_finite=sum(not e["finite"] for e in E),
                                                   clipped=int(sum(e["n_clipped"] for e in E)), slewed=int(sum(e["n_slewed"] for e in E)), median_final_box_to_tray_xy_m=float(np.median([e["final_box_to_tray_xy_m"] for e in E])) if E else None)
out["failure_classification"] = fc
# A0 vs historical
A0e = [e for e in eps if e["arm"] == "A0"]; out["A0_vs_historical_xsweep"] = dict(n=len(A0e), outcome_differs_poses=[e["pose_idx"] for e in A0e if e["outcome"] != hist[e["pose_idx"]]["outcome"]], max_abs_final_xy_diff_m=float(max([abs(e["final_box_to_tray_xy_m"] - hist[e["pose_idx"]]["final_box_to_tray_xy_m"]) for e in A0e] or [0])))
# phase-wise diagnostics of A2 decisions (ranking among candidates from chunk 2 on)
dec = [dict(d, pose_idx=e["pose_idx"]) for e in eps if e["arm"] == "A2" for d in e["decisions"] if "J" in d]; ph = {}
for d in dec:
    J = np.array(d["J"]); T = d["terms"]; cont = np.array([q["dxy"] + q["dz"] for q in T]); binary = np.array([0.05 * q["palm"] + 0.05 * (not q["lifted"]) for q in T]); s = np.sort(J)
    row = dict(spread=float(np.ptp(J)), margin=float(s[1] - s[0]), tie_1e9=bool(d["n_within_tie_tol"] > 1), tie_1e3=bool(s[1] - s[0] <= 1e-3), tie_5e3=bool(s[1] - s[0] <= 5e-3), sel_not_default=bool(d["selected"] != "c0_s0"), cont_only_argmin_differs=bool(np.argmin(cont) != np.argmin(J)), binary_varies=bool(np.ptp(binary) > 0), cont_spread=float(np.ptp(cont)), bin_spread=float(np.ptp(binary)))
    ph.setdefault(d["phase"], []).append(row)
ph_out = {}
for k, rows in ph.items():
    g = lambda f: [r[f] for r in rows]; ph_out[k] = dict(n_decisions=len(rows), J_spread_median=float(np.median(g("spread"))), margin_median=float(np.median(g("margin"))), tie_frac_tol_1e9=float(np.mean(g("tie_1e9"))), tie_frac_margin_le_1e3=float(np.mean(g("tie_1e3"))), tie_frac_margin_le_5e3=float(np.mean(g("tie_5e3"))), selected_not_default_frac=float(np.mean(g("sel_not_default"))),
                                                      continuous_only_argmin_differs_frac=float(np.mean(g("cont_only_argmin_differs"))), binary_penalty_varies_frac=float(np.mean(g("binary_varies"))), cont_spread_median=float(np.median(g("cont_spread"))), binary_spread_median=float(np.median(g("bin_spread"))))
out["A2_decisions_by_phase"] = ph_out; out["A2_decisions_total"] = len(dec)
a1d = [d for e in eps if e["arm"] == "A1" for d in e["decisions"] if d["chunk"] > 0]; out["A1_pick_counts"] = {k: sum(d["selected"] == k for d in a1d) for k in ("c0_s0", "c0_s1", "c0_s2")}; a2d = [d for e in eps if e["arm"] == "A2" for d in e["decisions"] if d["chunk"] > 0]; out["A2_pick_counts"] = {k: sum(d["selected"] == k for d in a2d) for k in ("c0_s0", "c0_s1", "c0_s2")}
out["candidate_level_regret_and_pairwise_ranking_vs_ground_truth"] = "NOT_EVALUABLE: only the selected chunk is executed; counterfactual outcomes of the unselected candidates do not exist (no oracle estimated)"
out["compute"] = dict(episode_wall_s_median={a: float(np.median([e["wall_s"] for e in eps if e["arm"] == a] or [np.nan])) for a in ("A0", "A1", "A2")}, act_ms_per_episode_median={a: float(np.median([e["act_ms"] for e in eps if e["arm"] == a] or [np.nan])) for a in ("A0", "A1", "A2")}, wm_ms_per_episode_median_A2=float(np.median([e["wm_ms"] for e in eps if e["arm"] == "A2"] or [np.nan])), run_cpu_s_estimate=R.get("cpu_s_estimate"))
(D / "results/pilot_analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in out if k not in ("A2_decisions_by_phase",)}, indent=1, default=float)[:6000]); print(json.dumps(out["A2_decisions_by_phase"], indent=1))
