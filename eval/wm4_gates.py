#!/usr/bin/env python3
"""Gate v3 — DRAFT SPECIFICATION (prospective).  NOT FROZEN, NOT APPLIED to the independent evaluation I-1 (2026-11-01) nor to any development result, no training started.
Changes vs Gate v2 (config sha256 f6c32391…0627, unchanged and inherited): G3 becomes headroom-aware (G3v3); a new object-orientation gate G7 (angle-wrap-aware yaw; tilt reported only).  G0, G1, G2, G4, G5, G6 are exactly v2.
Verdicts: PASS / FAIL / NOT_EVALUABLE (FAIL if any evaluable condition fails, else NOT_EVALUABLE if any required condition is not evaluable, else PASS) — the v2 logic.
Aggregation (v2): windows -> per initial-state GROUP mean -> macro mean over groups; percentile bootstrap over groups.  AUROC is computed on pooled windows with a cluster (group) bootstrap, as in v2; Brier / Brier skill and all angular errors use the per-group mean -> macro mean.
Draft config:  python eval/wm4_gates.py --draft   -> outputs/gates_v3/gates_v3_config.DRAFT.json (never the frozen path)."""
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm3_gates as V
V2 = V.CONFIG; wrap = lambda x: np.arctan2(np.sin(x), np.cos(x))
CONFIG = dict(
    version="Gate v3 (DRAFT, prospective)", status="DRAFT: specification + synthetic-fixture tests only; NOT FROZEN; NOT applied to I-1 or development; no training started",
    inherits=dict(gates_v2_config_sha256="f6c323914a99b4bc884ffc88c4c538615ad3a1ba10649fd95e38282acbaa0627", unchanged=["G0", "G1", "G2", "G4", "G5", "G6", "aggregation", "windows", "floors", "baseline_pool", "contact_scales"], note="v3 evaluates the inherited gates with the v2 functions in eval/wm3_gates.py"),
    not_applied_to=["independent evaluation I-1 (2026-11-01, G3 FAIL stays FAIL; yaw 0.0515 vs 0.0146 rad stays as reported)", "all development results"],
    min_sample=V2["min_sample"], aggregation=V2["aggregation"],
    G3=dict(
        rule="Per primary target (close on test_id; palm on the pooled low sets, as v2). Reference = frozen outcome-only logistic baseline. Let A_ref = AUROC of the reference on the same at-risk windows. "
             "If A_ref <= 0.90: margin branch (v2 rule): AUROC >= 0.80, bootstrap CI lower >= 0.70, AUROC - A_ref >= 0.05. If A_ref > 0.90: ceiling branch: AUROC >= 0.98 AND Brier skill >= 0.50. "
             "In BOTH branches the hard-subset condition applies: AUROC >= 0.90 on windows where the reference probability lies in [0.2, 0.8], evaluable only if the hard subset has >=10 groups with a positive window, >=10 groups with a negative window, >=30 positive and >=30 negative windows; otherwise that condition is NOT_EVALUABLE (never PASS, never skipped). "
             "A target whose pooled at-risk sample fails the v2 sufficiency rule is NOT_EVALUABLE. Gate = verdict over the conditions of both primary targets.",
        reference="outcome_only", branch_threshold_auroc_ref=0.90, margin_branch=dict(auroc_min=0.80, ci_lower_min=0.70, margin_min=0.05),
        ceiling_branch=dict(auroc_min=0.98, brier_skill_min=0.50), hard_subset=dict(ref_prob_range=[0.2, 0.8], auroc_min=0.90, support="same as g3_* in min_sample, applied to the hard subset"),
        brier_skill=dict(definition="BSS = 1 - B_model / max(B_ref, floor); B_* = macro mean over initial-state groups of the per-group mean squared error (p - y)^2 on the at-risk windows", floor=1e-3,
                         floor_rule="if B_ref < floor the denominator is the floor (the reference is essentially perfect: BSS >= 0.5 then requires B_model <= 0.5*floor); the floor use is flagged in the detail"),
        primary=V2["G3"]["primary"], secondary=V2["G3"]["secondary"], at_risk=V2["G3"]["at_risk"], bins=15, bootstrap_resamples_auroc=2000,
        reported_not_gating=["AUROC bootstrap CI (ceiling branch)", "BSS bootstrap CI over groups", "AUROC_ref", "branch", "prevalence", "lifted/placed secondary"]),
    G7=dict(
        rule="Object orientation. theta_k = wrap(sum_{j<=k} Yb[j, yaw]) is the relative box yaw displacement over k steps (the sum of per-step deltas of the wrapped angle is wrapped back to (-pi, pi]); angular error e_k = |wrap(theta_hat_k - theta_k)| in [0, pi]. "
             "Gated windows: active windows without an 'event' (event = any TRUE per-step |delta yaw| > pi/2 within the horizon: Euler-yaw singularity of a box standing on its end, +-pi discontinuity or physical tumbling; excluded from the gated means, reported separately, and the all-window ratio is reported next to the gated one). "
             "Primary yaw non-inferiority at K=20: R = E_model / max(E_best, floor), E = macro mean over groups, E_best = min(E_persistence, E_ridge) (point estimate fixed before bootstrapping): PASS iff R <= 1.10 AND the group-bootstrap upper bound of R <= 1.25. "
             "Static-window hallucination (input-defined subset S_in): mean over groups of e_20 <= 0.01 rad. NOT_EVALUABLE if the relevant gated sample has < 10 groups or < 200 windows (S_in: < 10 groups or < 200 windows). Horizons 1/5/10/20 are reported; only K=20 is gating. Tilt is reported only (no threshold).",
        horizons_reported=[1, 5, 10, 20], gating_horizon=20, ratio_max=1.10, ratio_upper_bound_max=1.25, baselines=["persistence", "ridge"], error_floor_rad=1e-4,
        event=dict(per_step_abs_delta_gt=float(np.pi / 2), note="derived from the root-cause analysis: 80 % of |delta|>pi/2 events are Euler-yaw singularities (box standing on its end), 15 % physical fast rotation, 4 % pure +-pi discontinuity; groups by TRUE future, so this is a DIAGNOSTIC stratification, not an inference-time selection"),
        static_input_subset=dict(rule="S_in (INPUT-ONLY; no future ground truth): box speed <= 0.005 m/s in all H history rows AND no finger contact (C_hist[:,:,1]==0) and palm force <= 0.01 N in all history rows AND the candidate future gripper command stays open (all A_fut[:,:,7] >= 0.028 m)",
                                 speed_max_mps=0.005, palm_force_max_N=0.01, gripper_open_min_m=0.028, metric="mean over groups of e_20 on S_in gated windows", max_mean_rad=0.01,
                                 diagnostic_subset="S_gt (true |theta_20| < 0.003 rad) is reported as a DIAGNOSTIC subset (uses the future truth, never used for selection)"),
        tilt=dict(reported_only=True, note="the tilt channel of data contract v1 mixes radians (3 650 episodes) and degrees (1 046 episodes: h23/xsweep/cov replays); no threshold until a single-unit representation (e.g. world-z of the body z-axis) is defined"),
        requires_contract=dict(note="windows must carry yaw as per-step wrapped deltas in channel 3 of Y_b and tilt in a single declared unit")))
def config_bytes(): return json.dumps(CONFIG, indent=1, sort_keys=True).encode()
def config_sha(): return hashlib.sha256(config_bytes()).hexdigest()
# ------------------------------------------------------------------------------------------------ G3 v3
def _bs_auroc_ci(y, p, g, n, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(g); mem = {x: np.nonzero(g == x)[0] for x in ug}; vs = []
    for _ in range(n):
        ii = np.concatenate([mem[x] for x in rng.choice(ug, len(ug))]); v = V.auroc(y[ii], p[ii])
        if not np.isnan(v): vs.append(v)
    return [float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))] if vs else [float("nan")] * 2
def brier_macro(y, p, g): e = (p - y) ** 2; return V.group_means(e, g)[1]
def brier_skill(y, p, po, g, floor):
    bm, br = brier_macro(y, p, g), brier_macro(y, po, g); den = max(float(br.mean()), floor); return 1 - float(bm.mean()) / den, dict(B_model=float(bm.mean()), B_ref=float(br.mean()), denominator=den, floor_used=bool(br.mean() < floor)), bm, br
def gate_G3_v3(cfg, preds, T, groups, masks, model="wm3b"):
    c = cfg["G3"]; conds, detail = [], {}
    for tgt in c["primary"]:
        y, p, po, g = V._gather(preds[model], preds[c["reference"]], T, groups, masks, tgt["label"], tgt["sets"]); ok, suff = V._suff(cfg, y, g); key = f"{tgt['label']}/{'+'.join(tgt['sets'])}"; d = dict(sufficiency=suff, evaluable=ok)
        if not ok: conds.append(dict(name=key, passed=False, evaluable=False)); detail[key] = d; continue
        a, ar = V.auroc(y, p), V.auroc(y, po); branch = "margin" if ar <= c["branch_threshold_auroc_ref"] else "ceiling"; ci = _bs_auroc_ci(y, p, g, c["bootstrap_resamples_auroc"]); bss, bd, bm, br = brier_skill(y, p, po, g, c["brier_skill"]["floor"])
        rng = np.random.default_rng(0); G = len(bm); idx = rng.integers(0, G, (2000, G)); bss_ci = [float(np.percentile(1 - bm[idx].mean(1) / np.maximum(br[idx].mean(1), c["brier_skill"]["floor"]), q)) for q in (2.5, 97.5)]
        d.update(branch=branch, auroc=a, auroc_ci=ci, auroc_ref=ar, margin=a - ar, brier_skill=bss, brier_detail=bd, brier_skill_ci_reported=bss_ci)
        if branch == "margin":
            m = c["margin_branch"]; conds += [dict(name=f"{key}/auroc>={m['auroc_min']}", value=a, passed=bool(a >= m["auroc_min"]), evaluable=True), dict(name=f"{key}/auroc_ci_lower>={m['ci_lower_min']}", value=ci[0], passed=bool(ci[0] >= m["ci_lower_min"]), evaluable=True),
                                              dict(name=f"{key}/margin_over_ref>={m['margin_min']}", value=a - ar, passed=bool(a - ar >= m["margin_min"]), evaluable=True)]
        else:
            m = c["ceiling_branch"]; conds += [dict(name=f"{key}/ceiling:auroc>={m['auroc_min']}", value=a, passed=bool(a >= m["auroc_min"]), evaluable=True), dict(name=f"{key}/ceiling:brier_skill>={m['brier_skill_min']}", value=bss, passed=bool(bss >= m["brier_skill_min"]), evaluable=True)]
        lo, hi = c["hard_subset"]["ref_prob_range"]; hm = (po >= lo) & (po <= hi); hok, hsuff = V._suff(cfg, y[hm], g[hm]) if hm.any() else (False, dict(pos_groups=0, neg_groups=0, pos_windows=0, neg_windows=0)); d["hard_subset"] = dict(n_windows=int(hm.sum()), sufficiency=hsuff, evaluable=hok)
        if hok: ah = V.auroc(y[hm], p[hm]); d["hard_subset"]["auroc"] = ah; conds.append(dict(name=f"{key}/hard_subset_auroc>={c['hard_subset']['auroc_min']}", value=ah, passed=bool(ah >= c["hard_subset"]["auroc_min"]), evaluable=True))
        else: conds.append(dict(name=f"{key}/hard_subset_auroc>={c['hard_subset']['auroc_min']}", passed=False, evaluable=False))
        detail[key] = d
    sec = {}
    for lab in c["secondary"]:
        sets = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; y, p, po, g = V._gather(preds[model], preds[c["reference"]], T, groups, masks, lab, sets); bss, bd, _, _ = brier_skill(y, p, po, g, c["brier_skill"]["floor"]); sec[lab] = dict(auroc=V.auroc(y, p), auroc_ref=V.auroc(y, po), brier_skill=bss, prevalence=float(y.mean()))
    return dict(verdict=V.verdict(conds), conditions=conds, detail=detail, secondary=sec)
# ------------------------------------------------------------------------------------------------ G7
def theta(Yb, k): return wrap(np.cumsum(Yb[:, :k, 3], 1)[:, k - 1])
def ang_err(P, Tt, k): return np.abs(wrap(theta(P["Y_b"], k) - theta(Tt["Y_b"], k)))
def event_mask(Tt, thr=np.pi / 2): return (np.abs(Tt["Y_b"][:, :, 3]) > thr).any(1)
def static_input_mask(W, cfg=CONFIG):
    s = cfg["G7"]["static_input_subset"]; B, C, Af = W["B_hist"], W["C_hist"], W["A_fut"]
    return (B[:, :, 4].max(1) <= s["speed_max_mps"]) & (C[:, :, 1].max(1) == 0) & (C[:, :, 4].max(1) <= s["palm_force_max_N"]) & (Af[:, :, 7].min(1) >= s["gripper_open_min_m"])
def _macro(vals, g): return V.group_means(vals, g)
def gate_G7(cfg, preds, T, groups, active, model="wm3b", sets=("test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"), n_boot=10000):
    c = cfg["G7"]; ms = cfg["min_sample"]; K = c["gating_horizon"]; sets = list(sets)
    ev = {sp: event_mask(T[sp]) for sp in sets}; gated = {sp: active[sp] & ~ev[sp] for sp in sets}
    def per_group(name, k, mask_by_sp):
        ids, vals = [], []
        for sp in sets:
            mk = mask_by_sp[sp]
            if mk.sum() == 0: continue
            e = ang_err(V.sub(preds[name][sp], mk), V.sub(T[sp], mk), k); i, v = _macro(e, groups[sp][mk]); ids.append(i); vals.append(v)
        if not ids: return np.array([]), np.array([])
        ids, vals = np.concatenate(ids), np.concatenate(vals); o = np.argsort(ids); return ids[o], vals[o]
    nw = int(sum(gated[sp].sum() for sp in sets)); ng = len({int(x) for sp in sets for x in np.unique(groups[sp][gated[sp]])}); ev_ok = ng >= ms["groups_error_gates"] and nw >= ms["active_windows"]; conds, detail = [], {}
    _, em = per_group(model, K, gated); base = {b: per_group(b, K, gated)[1] for b in c["baselines"]}
    if ev_ok and len(em):
        best = min(base, key=lambda b: base[b].mean()); fl = c["error_floor_rad"]; R = float(em.mean() / max(base[best].mean(), fl)); rng = np.random.default_rng(0); G = len(em); idx = rng.integers(0, G, (n_boot, G)); Rb = em[idx].mean(1) / np.maximum(base[best][idx].mean(1), fl); hi = float(np.percentile(Rb, 97.5)); lo = float(np.percentile(Rb, 2.5))
        detail["primary"] = dict(E_model=float(em.mean()), E_best=float(base[best].mean()), best_baseline=best, E_by_baseline={b: float(v.mean()) for b, v in base.items()}, ratio=R, ratio_ci=[lo, hi], n_groups=int(G), n_windows=nw)
        conds += [dict(name="yaw_ratio<=1.10", value=R, passed=bool(R <= c["ratio_max"]), evaluable=True), dict(name="yaw_ratio_upper_bound<=1.25", value=hi, passed=bool(hi <= c["ratio_upper_bound_max"]), evaluable=True)]
    else:
        detail["primary"] = dict(n_groups=ng, n_windows=nw, evaluable=False); conds += [dict(name="yaw_ratio<=1.10", passed=False, evaluable=False), dict(name="yaw_ratio_upper_bound<=1.25", passed=False, evaluable=False)]
    # static-window hallucination (input-only subset, event windows excluded as in the primary)
    st = {sp: static_input_mask(T[sp], cfg) & ~ev[sp] for sp in sets}; sw = int(sum(st[sp].sum() for sp in sets)); sg = len({int(x) for sp in sets for x in np.unique(groups[sp][st[sp]])})
    if sg >= ms["groups_error_gates"] and sw >= ms["active_windows"]:
        _, es = per_group(model, K, st); detail["static_input_subset"] = dict(E_model=float(es.mean()), n_groups=sg, n_windows=sw, E_by_baseline={b: float(per_group(b, K, st)[1].mean()) for b in c["baselines"]}); conds.append(dict(name="static_hallucination_error<=0.01rad", value=float(es.mean()), passed=bool(es.mean() <= c["static_input_subset"]["max_mean_rad"]), evaluable=True))
    else: detail["static_input_subset"] = dict(n_groups=sg, n_windows=sw, evaluable=False); conds.append(dict(name="static_hallucination_error<=0.01rad", passed=False, evaluable=False))
    # reported (not gating): all-window ratio, per-horizon, event windows, diagnostic static-by-truth subset
    allw = {sp: active[sp] for sp in sets}; ea = per_group(model, K, allw)[1]; ba = {b: per_group(b, K, allw)[1] for b in c["baselines"]}
    detail["reported_all_windows_K20"] = dict(E_model=float(ea.mean()) if len(ea) else None, E_by_baseline={b: float(v.mean()) for b, v in ba.items() if len(v)}, ratio=float(ea.mean() / max(min(v.mean() for v in ba.values()), c["error_floor_rad"])) if len(ea) else None)
    detail["reported_horizons"] = {str(k): {n: float(per_group(n, k, gated)[1].mean()) for n in [model] + c["baselines"] if len(per_group(n, k, gated)[1])} for k in c["horizons_reported"]}
    evw = {sp: active[sp] & ev[sp] for sp in sets}; detail["reported_event_windows"] = dict(n_windows=int(sum(evw[sp].sum() for sp in sets)), n_groups=len({int(x) for sp in sets for x in np.unique(groups[sp][evw[sp]])}), E_K20={n: (float(per_group(n, K, evw)[1].mean()) if len(per_group(n, K, evw)[1]) else None) for n in [model] + c["baselines"]})
    sgt = {sp: static_input_mask(T[sp], cfg) & ~ev[sp] & (np.abs(theta(T[sp]["Y_b"], K)) < 0.003) for sp in sets}; detail["diagnostic_S_gt"] = dict(uses_future_truth=True, n_windows=int(sum(sgt[sp].sum() for sp in sets)), E_K20={n: (float(per_group(n, K, sgt)[1].mean()) if len(per_group(n, K, sgt)[1]) else None) for n in [model] + c["baselines"]})
    return dict(verdict=V.verdict(conds), conditions=conds, detail=detail)
def tilt_report(preds, T, groups, active, model, k=20, scale_to_rad=1.0, sets=("test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high")):
    """REPORTED ONLY: |cumulative tilt error| (channel units * scale_to_rad); no threshold."""
    out = {}
    for n in (model, "persistence", "ridge"):
        v = []
        for sp in sets:
            a = active[sp]; e = np.abs(np.cumsum(preds[n][sp]["Y_b"][a][:, :k, 5], 1)[:, k - 1] - np.cumsum(T[sp]["Y_b"][a][:, :k, 5], 1)[:, k - 1]) * scale_to_rad; v.append(V.group_means(e, groups[sp][a])[1])
        out[n] = float(np.concatenate(v).mean())
    return out
if __name__ == "__main__":
    if "--draft" in sys.argv:
        out = ROOT / "outputs/gates_v3"; out.mkdir(parents=True, exist_ok=True); p = out / "gates_v3_config.DRAFT.json"; p.write_bytes(config_bytes()); print("DRAFT written (not frozen):", p, "sha256", config_sha())
    else: print("sha256 of current DRAFT CONFIG:", config_sha())
