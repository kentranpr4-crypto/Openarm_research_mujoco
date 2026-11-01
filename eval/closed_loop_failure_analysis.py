#!/usr/bin/env python3
"""READ-ONLY post-hoc failure analysis of the Stage-1 pilot (numpy only; uses existing pilot_results.json and saved trajectories; no simulator, no model, no new episode).  POST-HOC DIAGNOSTIC, not independent evidence.
Run: python3 eval/closed_loop_failure_analysis.py   -> outputs/closed_loop_pilot/failure_analysis.json (new file)."""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); D = ROOT / "outputs/closed_loop_pilot"
import decision_utility_lib as L
R = json.loads((D / "results/pilot_results.json").read_text()); eps = [e for e in R["episodes"] if e["status"] == "OK"]; TR = {}
for e in eps: TR[(e["arm"], e["pose_idx"], e["repeat"])] = dict(np.load(D / f"results/traj/{e['arm']}_p{e['pose_idx']:02d}_r{e['repeat']}.npz"))
E = {(e["arm"], e["pose_idx"], e["repeat"]): e for e in eps}; poses = sorted({e["pose_idx"] for e in eps}); out = dict(label="POST-HOC DIAGNOSTIC (read-only; Stage-1 pilot results; not independent evidence)")
def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or a.std() == 0 or b.std() == 0: return None
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float); return float(np.corrcoef(ra, rb)[0, 1])
# 1. reconciliation --------------------------------------------------------------------------------------------------------------------------------------------------------------------
rows = []
for p in poses:
    a0 = E[("A0", p, 0)]["task_success"]; a2 = E[("A2", p, 0)]["task_success"]; a1 = [E[("A1", p, r)]["task_success"] for r in range(3)]; d = a2 - np.mean(a1); maj = np.mean(a1) >= 0.5
    rows.append(dict(pose=p, x=E[("A0", p, 0)]["x"], A0=int(a0), A2=int(a2), A1=[int(v) for v in a1], A1_mean=float(np.mean(a1)), diff=float(d), sign="better" if d > 1e-12 else ("worse" if d < -1e-12 else "equal"), discordant=bool(a2 != maj)))
cross = {}
for r in rows: cross.setdefault((r["sign"], "discordant" if r["discordant"] else "concordant"), []).append(r["pose"])
out["reconciliation"] = dict(definitions="diff = A2 success (0/1) - mean of 3 A1 successes (per pose); better/worse/equal = sign(diff); discordant = A2 success differs from the A1 MAJORITY (mean>=0.5). 'better/worse' counts every non-zero fractional difference, 'discordant' only majority flips, so the two counts differ by construction.",
                             sign_counts={k: sum(r["sign"] == k for r in rows) for k in ("better", "equal", "worse")}, discordant_count=sum(r["discordant"] for r in rows), crosstab={f"{k[0]}|{k[1]}": v for k, v in sorted(cross.items())}, equal_split=dict(both_always_success=sum(1 for r in rows if r["sign"] == "equal" and r["A2"] == 1 and r["A1_mean"] == 1.0), both_always_fail=sum(1 for r in rows if r["sign"] == "equal" and r["A2"] == 0 and r["A1_mean"] == 0.0)), per_pose=rows)
# 2. switching ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def seq(e): return [d["selected"] for d in e["decisions"]]
sw = {}
for arm in ("A0", "A1", "A2"):
    ss = [seq(e) for e in eps if e["arm"] == arm]; n_sw = [sum(a != b for a, b in zip(s[:-1], s[1:])) for s in ss]
    sw[arm] = dict(n_episodes=len(ss), switches_per_episode_mean=float(np.mean(n_sw)), boundary_switch_fraction=float(np.sum(n_sw) / (9 * len(ss))), max_run_length_mean=float(np.mean([max(len(list(g)) for _, g in __import__("itertools").groupby(s)) for s in ss])))
    for lab, flt in (("success", True), ("fail", False)):
        v = [n for e, n in zip([e for e in eps if e["arm"] == arm], n_sw) if bool(e["task_success"]) == flt]; sw[arm][f"switches_mean_{lab}"] = float(np.mean(v)) if v else None
out["checkpoint_switching"] = sw; out["checkpoint_switching"]["note"] = "A1 expectation for uniform random pick: P(switch between consecutive chunks 2..10) = 2/3; first boundary (chunk1->chunk2) switches only if the pick differs from c0_s0."
# 3. discontinuities at chunk boundaries (executed targets only; raw ACT targets were not saved -> raw jump NOT_EVALUABLE) -----------------------------------------------------------
def steps(a): return np.abs(np.diff(a[:, :8], axis=0))      # (199, 8); index k = step from t=k to k+1 ; boundary = transition INTO t=20m  i.e. k = 20m-1
bd = {}
for arm in ("A0", "A1", "A2"):
    B, W, SL = [], [], 0; nb = 0; per_joint = np.zeros(8)
    for e in [e for e in eps if e["arm"] == arm]:
        a = TR[(arm, e["pose_idx"], e["repeat"])]["action"]; s = steps(a); bi = [20 * m - 1 for m in range(1, 10)]; wi = [k for k in range(199) if k not in bi]; B.append(s[bi]); W.append(s[wi])
    B = np.concatenate(B); W = np.concatenate(W)
    bd[arm] = dict(boundary_step_mean_arm=float(B[:, :7].mean()), within_step_mean_arm=float(W[:, :7].mean()), ratio=float(B[:, :7].mean() / W[:, :7].mean()), boundary_at_slew_limit_frac_arm=float((B[:, :7] >= 0.0999).mean()), within_at_slew_limit_frac_arm=float((W[:, :7] >= 0.0999).mean()),
                   boundary_step_mean_gripper_m=float(B[:, 7].mean()), within_step_mean_gripper_m=float(W[:, 7].mean()), max_boundary_step_arm_rad=float(B[:, :7].max()))
# boundary split by whether the checkpoint switched
for arm in ("A1", "A2"):
    sw_s, ns_s = [], []
    for e in [e for e in eps if e["arm"] == arm]:
        a = TR[(arm, e["pose_idx"], e["repeat"])]["action"]; s = steps(a); ss = seq(e)
        for m in range(1, 10):
            (sw_s if ss[m] != ss[m - 1] else ns_s).append(s[20 * m - 1, :7].mean())
    bd[arm].update(boundary_mean_arm_when_switched=float(np.mean(sw_s)) if sw_s else None, n_switched=len(sw_s), boundary_mean_arm_when_not_switched=float(np.mean(ns_s)) if ns_s else None, n_not_switched=len(ns_s))
out["chunk_boundary_discontinuity_executed"] = bd; out["chunk_boundary_discontinuity_executed"]["raw_target_jump"] = "NOT_EVALUABLE (raw ACT targets before clip/slew were not saved in Stage 1; executed targets are slew-limited to 0.1 rad/step)"
# 4. failure timing -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def failure_mode(e):
    t = TR[(e["arm"], e["pose_idx"], e["repeat"])]; z = t["box_pos"][:, 2]; rise = z - z[0]; xy = t["box_pos"][:, :2]; tray = np.asarray(L.TRAY)
    if e["task_success"]: return "success", None
    if rise.max() < 0.05: return "never_lifted_5cm", None
    tp = int(np.argmax(rise >= 0.05)); after = np.where(rise[tp:] < 0.02)[0]
    if len(after): td = tp + int(after[0]); return ("dropped_far_from_tray(carry)" if np.linalg.norm(xy[td] - tray) >= 0.03 else "dropped_at_tray(place)"), td
    return "held_to_end_but_misplaced", None
fm = {}
for arm in ("A0", "A1", "A2"):
    c = {}
    for e in [e for e in eps if e["arm"] == arm]: m, td = failure_mode(e); c[m] = c.get(m, 0) + 1
    fm[arm] = c
out["failure_timing_by_mode"] = fm; fm_tdrop = {}
for arm in ("A0", "A1", "A2"):
    v = [failure_mode(e)[1] for e in eps if e["arm"] == arm]; v = [x for x in v if x is not None]; fm_tdrop[arm] = dict(n=len(v), drop_step_median=float(np.median(v)) if v else None)
out["drop_step"] = fm_tdrop
# 5/6/7. J vs realised, near ties, picks ---------------------------------------------------------------------------------------------------------------------------------------------------
def realised(t, tstart):
    bp = t["box_pos"][tstart:tstart + 21]; cr = t["c_rows"][tstart:min(tstart + 21, len(t["c_rows"]))]; b = np.zeros((len(bp), 7)); b[:, :3] = bp; c = np.zeros((len(cr), 7)); c[:, 4] = cr[:, 4]; tm = L.terms(b, c); return tm, L.score(tm)
dec = []
for e in [e for e in eps if e["arm"] == "A2"]:
    t = TR[("A2", e["pose_idx"], 0)]
    for d in e["decisions"]:
        if "J" not in d: continue
        k = d["candidates"].index(d["selected"]); tm, Jr = realised(t, d["t"]); dec.append(dict(pose=e["pose_idx"], t=d["t"], phase=d["phase"], sel=d["selected"], k=k, J_pred=d["J"][k], J_def=d["default_J"], J_real=Jr, dxy_pred=d["terms"][k]["dxy"], dxy_real=tm["dxy"], palm_pred=d["terms"][k]["palm"], palm_real=tm["palm"], margin=d["margin_to_second"], spread=d["spread"], success=e["task_success"], last=d["t"] == 180))
res5 = {}
for ph in ("approach", "grasp", "carry", "place", "ALL"):
    S = [x for x in dec if (ph == "ALL" or x["phase"] == ph) and not x["last"]]
    if not S: continue
    res5[ph] = dict(n=len(S), spearman_Jpred_Jreal=sp([x["J_pred"] for x in S], [x["J_real"] for x in S]), dxy_pred_vs_real_mae_m=float(np.mean([abs(x["dxy_pred"] - x["dxy_real"]) for x in S])), dxy_spearman=sp([x["dxy_pred"] for x in S], [x["dxy_real"] for x in S]), palm_pred_rate=float(np.mean([x["palm_pred"] for x in S])), palm_real_rate=float(np.mean([x["palm_real"] for x in S])), palm_agreement=float(np.mean([x["palm_pred"] == x["palm_real"] for x in S])))
out["J_pred_vs_realised_selected_chunks_only"] = dict(note="only the EXECUTED (selected) chunk has a realised outcome; last chunk (t=180) excluded because row t+20 does not exist; realised J uses the same formula on executed rows t..t+20 (palm over rows t..t+19)", by_phase=res5)
nt = {}
for tol in (1e-3, 5e-3):
    S = [x for x in dec if x["margin"] <= tol]; nt[f"margin<={tol}"] = dict(n=len(S), of=len(dec), frac=float(len(S) / len(dec)), by_phase={ph: sum(x["phase"] == ph for x in S) for ph in ("approach", "grasp", "carry", "place")}, selected_not_default=sum(x["sel"] != "c0_s0" for x in S), mean_realised_J=float(np.mean([x["J_real"] for x in S])) if S else None)
    eps_nt = {}
    for x in S: eps_nt[x["pose"]] = eps_nt.get(x["pose"], 0) + 1
    nt[f"margin<={tol}"]["episodes_with_any"] = len(eps_nt)
frac_nt = {x["pose"]: [] for x in dec}
for x in dec: frac_nt[x["pose"]].append(x["margin"] <= 5e-3)
pf = [np.mean(v) for p, v in frac_nt.items() if E[("A2", p, 0)]["task_success"]]; pfail = [np.mean(v) for p, v in frac_nt.items() if not E[("A2", p, 0)]["task_success"]]
nt["near_tie_fraction_per_episode_(<=5e-3)_success_vs_fail"] = dict(success=float(np.mean(pf)), fail=float(np.mean(pfail)), n_success=len(pf), n_fail=len(pfail)); out["near_ties_A2"] = nt
nd = [x for x in dec if x["sel"] != "c0_s0"]; dfl = [x for x in dec if x["sel"] == "c0_s0"]
out["A2_non_default_picks"] = dict(n_non_default=len(nd), n_default=len(dfl), mean_J_real_nondefault=float(np.mean([x["J_real"] for x in nd])), mean_J_real_default=float(np.mean([x["J_real"] for x in dfl])), note="descriptive and CONFOUNDED (different states/phases); not a counterfactual",
                                 by_phase={ph: dict(n_nd=sum(1 for x in nd if x["phase"] == ph), n_def=sum(1 for x in dfl if x["phase"] == ph), J_real_nd=(float(np.mean([x["J_real"] for x in nd if x["phase"] == ph])) if any(x["phase"] == ph for x in nd) else None), J_real_def=(float(np.mean([x["J_real"] for x in dfl if x["phase"] == ph])) if any(x["phase"] == ph for x in dfl) else None)) for ph in ("approach", "grasp", "carry", "place")})
# 8. first divergence A2 vs A0 (identical history until the first non-default pick -> same state at t_div; a real, executed pair for ONE chunk) -----------------------------------------------------
pair = []
for p in poses:
    e2 = E[("A2", p, 0)]; first = next((d for d in e2["decisions"] if d["chunk"] > 0 and d["selected"] != "c0_s0"), None)
    a0, a2 = TR[("A0", p, 0)], TR[("A2", p, 0)]; same_until = int(np.argmax(np.abs(a0["action"] - a2["action"]).max(1) > 1e-9)) if (np.abs(a0["action"] - a2["action"]).max(1) > 1e-9).any() else 200
    if first is None: pair.append(dict(pose=p, t_div=None, A0=int(E[("A0", p, 0)]["task_success"]), A2=int(e2["task_success"]))); continue
    t = first["t"]; assert same_until >= t, (p, same_until, t); tm0, J0 = realised(a0, t); tm2, J2 = realised(a2, t); k = first["candidates"].index(first["selected"])
    pair.append(dict(pose=p, t_div=t, phase=first["phase"], sel=first["selected"], dJ_pred_sel_minus_def=float(first["J"][k] - first["default_J"]), dJ_real_sel_minus_def=float(J2 - J0), margin=first["margin_to_second"], A0=int(E[("A0", p, 0)]["task_success"]), A2=int(e2["task_success"]), same_state_check=bool(same_until >= t), last=t == 180))
valid = [q for q in pair if q.get("t_div") is not None and not q["last"] and abs(q["dJ_real_sel_minus_def"]) > 1e-12]
agree = [np.sign(q["dJ_pred_sel_minus_def"]) == np.sign(q["dJ_real_sel_minus_def"]) for q in valid]
big = [q for q in valid if abs(q["dJ_real_sel_minus_def"]) >= L.MEANINGFUL_PAIR]; agree_big = [np.sign(q["dJ_pred_sel_minus_def"]) == np.sign(q["dJ_real_sel_minus_def"]) for q in big]
mg = [x["margin"] for x in dec]; gate = {str(t): dict(n=int(sum(m >= t for m in mg)), of=len(mg), frac=float(np.mean([m >= t for m in mg]))) for t in (0.005, 0.01, 0.02)}
out["margin_gate_coverage_A2_decisions"] = gate
out["A2_vs_A0_first_divergence_meaningful"] = dict(threshold=L.MEANINGFUL_PAIR, n=len(big), n_agree=int(np.sum(agree_big)), poses=[q["pose"] for q in big], realised_worse_with_palm_sized_jump=[q["pose"] for q in big if q["dJ_real_sel_minus_def"] > 0.04])
out["A2_vs_A0_first_divergence_pairs"] = dict(note="A0 and A2 share the identical executed history up to t_div (asserted from saved actions), so for that one chunk we have two REAL executed continuations from the same state (default c0_s0 in A0, WM-selected in A2). Later chunks differ in both arms; no counterfactual is inferred for unexecuted candidates.",
                                              n_poses=len(pair), n_with_nondefault_pick=sum(q["t_div"] is not None for q in pair), n_pairs_usable=len(valid), sign_agreement_pred_vs_real=float(np.mean(agree)) if agree else None, n_agree=int(np.sum(agree)), pairs=pair)
def _diff(arm_a, arm_b):    # poses where A2 differs from the reference
    return [dict(pose=r["pose"], x=r["x"], A0=r["A0"], A1=r["A1"], A2=r["A2"], A2_mode=failure_mode(E[("A2", r["pose"], 0)])[0], A0_mode=failure_mode(E[("A0", r["pose"], 0)])[0]) for r in rows if r["A2"] == 0]
out["A2_failure_poses_vs_A0_A1"] = _diff(None, None)
# 9. sufficiency ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
out["data_sufficiency"] = dict(raw_ACT_targets_before_clip="NOT_SAVED -> boundary raw jump and per-candidate raw diversity in the loop NOT_EVALUABLE", unselected_candidates_executed="never executed -> counterfactual outcomes of unselected candidates NOT_EVALUABLE (no oracle estimated)",
                               A1_WM_predictions="not computed in A1 (only the selected chunk was generated) -> WM ranking accuracy on A1 picks NOT_EVALUABLE without re-simulation (b rows incl. yaw/tilt/speed not saved)", sample_size="24 poses, 216 A2 decisions: all numbers descriptive; no significance claims")
(D / "failure_analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: out[k] for k in out if k not in ("reconciliation", "A2_vs_A0_first_divergence_pairs", "A2_failure_poses_vs_A0_A1")}, indent=1, default=float)[:7000])
print(json.dumps({k: v for k, v in out["reconciliation"].items() if k != "per_pose"}, indent=1)); print(json.dumps({k: v for k, v in out["A2_vs_A0_first_divergence_pairs"].items() if k != "pairs"})); print(json.dumps(out["A2_failure_poses_vs_A0_A1"]))
