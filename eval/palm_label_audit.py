#!/usr/bin/env python3
"""READ-ONLY palm-contact calibration / label-horizon audit on STORED data only (branching-replay results + arrays, Stage-1 results + trajectories).  No simulator, no model inference, no new episode.  POST-HOC DIAGNOSTIC (development).
Run: python3 eval/palm_label_audit.py  -> outputs/branching_replay/palm_label_audit.json (new file)"""
import json, glob, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; BR = ROOT / "outputs/branching_replay"; CL = ROOT / "outputs/closed_loop_pilot"
S = [s for f in sorted(glob.glob(str(BR / "results/worker*.json"))) for s in json.loads(Path(f).read_text())["states"]]
THR = 0.01; out = dict(label="READ-ONLY post-hoc audit (stored data only; development; not independent evidence)")
rows = []     # one per branch
for s in S:
    Z = np.load(BR / f"results/arrays/{s['state_id']}.npz")
    for k, b in enumerate(s["branches"]):
        c = Z["c"][k]; hf = c[:, 4]; rows.append(dict(sid=s["state_id"], pose=s["pose_idx"], t=s["t"], phase=s["phase"], k=k, pred=bool(s["pred"]["terms"][k]["palm"]), real=bool(b["terms_real"]["palm"]), hf=hf, fc=c[:, 1], ff=c[:, 0] * 50.0, a_valid=c[:, 3],
                                                    pred2=(bool(s["pred"]["terms2"][k]["palm"]) if "terms2" in s["pred"] else None), real2=(bool(b["terms2_real"]["palm"]) if "terms2_real" in b else None), J_real=b["J20_real"], terms_real=b["terms_real"], terms_pred=s["pred"]["terms"][k], J_pred=s["pred"]["J20"][k]))
def conf(sel):
    tp = sum(r["pred"] and r["real"] for r in sel); fp = sum(r["pred"] and not r["real"] for r in sel); fn = sum((not r["pred"]) and r["real"] for r in sel); tn = sum((not r["pred"]) and (not r["real"]) for r in sel); n = max(len(sel), 1)
    return dict(n=len(sel), tp=tp, fp=fp, fn=fn, tn=tn, pred_rate=(tp + fp) / n, real_rate=(tp + fn) / n, precision=(tp / (tp + fp)) if tp + fp else None, recall=(tp / (tp + fn)) if tp + fn else None, fpr=(fp / (fp + tn)) if fp + tn else None)
out["branch_level_confusion_J20_window_t_to_t20_threshold_0p01N"] = conf(rows); out["by_phase"] = {ph: conf([r for r in rows if r["phase"] == ph]) for ph in ("approach", "grasp", "carry", "place")}; out["by_candidate"] = {k: conf([r for r in rows if r["k"] == k]) for k in range(3)}
# B realised label threshold sensitivity (predicted flag is fixed at the registered 0.01 N)
sens = {}
for thr in (1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0):
    rr = [dict(r, real=bool((r["hf"] > thr).any())) for r in rows]; sens[str(thr)] = {k: v for k, v in conf(rr).items() if k in ("real_rate", "precision", "recall", "fpr", "fp", "tp")}
out["realised_label_threshold_sensitivity_predicted_flag_fixed"] = sens
fp = [r for r in rows if r["pred"] and not r["real"]]; hfmax = np.array([r["hf"].max() for r in fp]) if fp else np.array([0.0]); out["false_positive_realised_hand_force_max_N"] = dict(n=len(fp), quantiles={q: float(np.percentile(hfmax, q)) for q in (50, 75, 90, 99, 100)}, frac_exactly_zero=float(np.mean(hfmax <= 1e-9)), frac_between_1e4_and_0p01=float(np.mean((hfmax > 1e-4) & (hfmax <= 0.01))))
tpr = [r for r in rows if r["pred"] and r["real"]]; out["true_positive_realised_hand_force_max_N"] = dict(n=len(tpr), quantiles={q: float(np.percentile([r["hf"].max() for r in tpr], q)) for q in (50, 90, 100)} if tpr else None)
# C label horizon: default branches (k=0) are the real A0 continuation -> look at palm contact AFTER the J20 window in the stored Stage-1 A0 trajectory
hor = []
for r in rows:
    if r["k"] != 0: continue
    T = np.load(CL / f"results/traj/A0_p{r['pose']:02d}_r0.npz")["c_rows"][:, 4]; t = r["t"]
    hor.append(dict(pred=r["pred"], real20=bool((T[t:t + 21] > THR).any()), real40=bool((T[t:t + 41] > THR).any()), real60=bool((T[t:t + 61] > THR).any()), real_after20_to40=bool((T[t + 21:t + 41] > THR).any()) if t + 41 <= len(T) else None, real_whole_rest=bool((T[t:] > THR).any()), phase=r["phase"]))
def hc(sel, key):
    sel = [h for h in sel if h[key] is not None]; tp = sum(h["pred"] and h[key] for h in sel); fp_ = sum(h["pred"] and not h[key] for h in sel); fn = sum((not h["pred"]) and h[key] for h in sel); tn = len(sel) - tp - fp_ - fn
    return dict(n=len(sel), tp=tp, fp=fp_, fn=fn, tn=tn, real_rate=(tp + fn) / max(len(sel), 1), precision=(tp / (tp + fp_)) if tp + fp_ else None, recall=(tp / (tp + fn)) if tp + fn else None)
out["label_horizon_default_branch_only_A0_continuation"] = {k: hc(hor, k) for k in ("real20", "real40", "real60", "real_whole_rest")}; out["label_horizon_default_branch_only_A0_continuation"]["note"] = "prediction = J20 palm flag (rows t..t+20) of the default candidate; longer labels use the stored Stage-1 A0 trajectory (default policy continuation, not the candidate-specific counterfactual)"
out["label_horizon_by_phase_real40"] = {ph: hc([h for h in hor if h["phase"] == ph], "real40") for ph in ("approach", "grasp", "carry", "place")}
sel2 = [r for r in rows if r["pred2"] is not None and r["real2"] is not None]; out["J2_40step_horizon_confusion_all_candidates"] = conf([dict(r, pred=r["pred2"], real=r["real2"]) for r in sel2])
# D ranking impact of the palm term
st = {}
for r in rows: st.setdefault(r["sid"], []).append(r)
n_pv = n_rv = n_dec = n_dec_corr = 0; chg = 0
for sid, rr in st.items():
    pp = [x["pred"] for x in rr]; rp = [x["real"] for x in rr]; n_pv += len(set(pp)) > 1; n_rv += len(set(rp)) > 1
    Jp = np.array([x["J_pred"] for x in rr]); Jnp = Jp - 0.05 * np.array(pp); a1 = int(np.argmin(Jp)); a2 = int(np.argmin(Jnp)); chg += a1 != a2
out["palm_term_in_ranking"] = dict(n_states=len(st), states_where_predicted_palm_flag_differs_across_candidates=n_pv, states_where_realised_palm_differs_across_candidates=n_rv, states_where_argmin_changes_if_palm_term_removed=chg)
# E headroom decomposition (states with headroom >= 0.01)
dec = {"palm": [], "lift": [], "cont": []}; ph_sh = {}
for sid, rr in st.items():
    J = np.array([x["J_real"] for x in rr]); k = int(np.argmin(J)); h = J[0] - J[k]
    if h < 0.01: continue
    d, b = rr[0]["terms_real"], rr[k]["terms_real"]; dp = 0.05 * (d["palm"] - b["palm"]); dl = 0.05 * ((not d["lifted"]) - (not b["lifted"])); dc = (d["dxy"] + d["dz"]) - (b["dxy"] + b["dz"]); dec["palm"].append(dp); dec["lift"].append(dl); dec["cont"].append(dc); ph_sh.setdefault(rr[0]["phase"], []).append((dp, dl, dc, h))
out["headroom_decomposition_states_with_headroom_ge_0p01"] = dict(n=len(dec["palm"]), mean_palm_term=float(np.mean(dec["palm"])) if dec["palm"] else None, mean_lift_term=float(np.mean(dec["lift"])) if dec["lift"] else None, mean_continuous_dxy_dz=float(np.mean(dec["cont"])) if dec["cont"] else None,
                                                              states_where_palm_term_is_majority_of_headroom=int(sum(abs(p) >= 0.5 * h for p, _, _, h in sum(ph_sh.values(), []))), by_phase={ph: dict(n=len(v), palm=float(np.mean([a for a, _, _, _ in v])), lift=float(np.mean([b for _, b, _, _ in v])), cont=float(np.mean([c for _, _, c, _ in v])), headroom=float(np.mean([h for *_, h in v]))) for ph, v in ph_sh.items()})
# F Stage 1 closed loop: palm contact vs outcome and timing
R = json.loads((CL / "results/pilot_results.json").read_text()); eps = [e for e in R["episodes"] if e["status"] == "OK"]; tab = {}
for arm in ("A0", "A1", "A2"):
    E = [e for e in eps if e["arm"] == arm]; a = sum(e["palm_contact"] and not e["task_success"] for e in E); b = sum(e["palm_contact"] and e["task_success"] for e in E); c = sum((not e["palm_contact"]) and not e["task_success"] for e in E); d = sum((not e["palm_contact"]) and e["task_success"] for e in E)
    tab[arm] = dict(palm_fail=a, palm_success=b, nopalm_fail=c, nopalm_success=d, P_fail_given_palm=(a / (a + b)) if a + b else None, P_fail_given_no_palm=(c / (c + d)) if c + d else None)
out["stage1_palm_contact_vs_failure"] = tab
first = []; hmax = []
for e in eps:
    T = np.load(CL / f"results/traj/{e['arm']}_p{e['pose_idx']:02d}_r{e['repeat']}.npz"); hf = T["c_rows"][:, 4]; fc = T["c_rows"][:, 1]
    if (hf > THR).any(): first.append(dict(arm=e["arm"], t_first_palm=int(np.argmax(hf > THR)), t_first_finger_contact=int(np.argmax(fc > 0)) if (fc > 0).any() else None, n_steps_palm=int((hf > THR).sum()), hf_max=float(hf.max()), success=bool(e["task_success"])))
out["stage1_palm_events"] = dict(n_episodes_with_palm=len(first), t_first_palm_median=float(np.median([f["t_first_palm"] for f in first])) if first else None, palm_before_first_finger_contact_frac=float(np.mean([f["t_first_finger_contact"] is None or f["t_first_palm"] < f["t_first_finger_contact"] for f in first])) if first else None,
                                 n_steps_with_palm_median=float(np.median([f["n_steps_palm"] for f in first])) if first else None, hf_max_quantiles={q: float(np.percentile([f["hf_max"] for f in first], q)) for q in (50, 90, 100)} if first else None, success_with_palm=int(sum(f["success"] for f in first)))
out["not_evaluable"] = dict(predicted_palm_force_value="only the thresholded boolean palm flag was stored (not the predicted force) -> continuous calibration (reliability curve, ECE, threshold sweep of the PREDICTION) NOT_EVALUABLE without re-running the WM on windows that were not stored", predicted_contact_timing="row of first predicted palm contact not stored", candidate_specific_long_horizon_labels="only the default branch continues beyond t+20 (A0); other candidates have no executed continuation beyond t+20 (J2 hold continuation exists only to t+40)")
(BR / "palm_label_audit.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float))
