#!/usr/bin/env python3
"""Gate v3 — FINAL DRAFT SPECIFICATION for review (prospective).  NOT FROZEN, NOT APPLIED to the independent evaluation I-1 (2026-11-01) nor to any development result; no training started.
Inherited unchanged from Gate v2 (config sha256 f6c32391…0627): G0, G1, G2, G4, G5, G6, aggregation, floors, baseline pool, contact scales.   New: G3 (headroom-aware + rare-label branch) and G7 (object orientation).
Verdict logic (all gates, v2 logic): every gate is a LIST OF COMPONENTS (conditions).  A component is PASS, FAIL or NOT_EVALUABLE, or NOT_APPLICABLE (recorded with its reason, excluded from the verdict, never counted as a pass).
  gate verdict = FAIL if any evaluable component FAILS; else NOT_EVALUABLE if any applicable component is NOT_EVALUABLE; else PASS.   FAIL therefore always outranks NOT_EVALUABLE.
Aggregation: windows -> per initial-state GROUP mean -> macro mean over groups; percentile bootstrap over groups (10 000, seed 0).  AUROC / operating-point statistics use pooled windows with a group-cluster bootstrap.
G7 needs data contract v2 fields (eval/geometry_v2.py, data/contract_v2.py): T[sp]["Q_t"] (n,4) = quaternion of the last history row, T[sp]["Y_rv"] (n,K,3) = true per-step world-frame rotation vectors, preds[m][sp]["Y_rv"].
Draft config:  python eval/wm4_gates.py --draft   -> outputs/gates_v3/gates_v3_config.DRAFT.json (never the frozen path)."""
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm3_gates as V, geometry_v2 as Gm
V2 = V.CONFIG; wrap = Gm.wrap
CONFIG = dict(
    version="Gate v3 (FINAL DRAFT for review, prospective)", status="DRAFT: specification + synthetic-fixture tests only; NOT FROZEN; NOT applied to I-1 or development; no training started",
    inherits=dict(gates_v2_config_sha256="f6c323914a99b4bc884ffc88c4c538615ad3a1ba10649fd95e38282acbaa0627", unchanged=["G0", "G1", "G2", "G4", "G5", "G6", "aggregation", "windows", "floors", "baseline_pool", "contact_scales"], note="v3 evaluates the inherited gates with the v2 functions in eval/wm3_gates.py"),
    not_applied_to=["independent evaluation I-1 (2026-11-01; G3 FAIL stays FAIL; yaw 0.0515 vs 0.0146 rad stays as reported)", "all development results"],
    min_sample=V2["min_sample"], aggregation=V2["aggregation"],
    verdict_logic="component list per gate; FAIL if any evaluable component fails, else NOT_EVALUABLE if any applicable component is not evaluable, else PASS; NOT_APPLICABLE components are recorded with a reason and excluded; FAIL outranks NOT_EVALUABLE",
    G3=dict(
        rule="Per primary target (close on test_id; palm on the pooled low sets, as v2). Reference = frozen outcome-only logistic. A_ref = reference AUROC on the same at-risk windows; prevalence = event rate on those pooled windows. "
             "Branch by headroom: A_ref <= 0.90 -> margin branch (AUROC >= 0.80, bootstrap CI lower >= 0.70, AUROC - A_ref >= 0.05); A_ref > 0.90 -> ceiling branch (AUROC >= 0.98 AND Brier skill >= 0.50). "
             "Hard subset (reference probability in [0.2, 0.8], AUROC >= 0.90) applies ONLY if prevalence >= 5 %; it is evaluable only with >= 10 groups with a positive and >= 10 groups with a negative window and >= 30 positive and >= 30 negative windows, otherwise that component is NOT_EVALUABLE. "
             "RARE labels (prevalence < 5 %): the hard-subset component is NOT_APPLICABLE and the label must additionally meet Brier skill >= 0.50 (in both branches) and the pre-registered operating-point metric TPR at FPR 5 % >= 0.70. "
             "A target whose pooled at-risk sample fails the v2 sufficiency rule is NOT_EVALUABLE.",
        reference="outcome_only", branch_threshold_auroc_ref=0.90, margin_branch=dict(auroc_min=0.80, ci_lower_min=0.70, margin_min=0.05), ceiling_branch=dict(auroc_min=0.98, brier_skill_min=0.50),
        hard_subset=dict(ref_prob_range=[0.2, 0.8], auroc_min=0.90, min_prevalence=0.05, support="same as g3_* in min_sample, applied to the hard subset"),
        rare_label=dict(prevalence_below=0.05, brier_skill_min=0.50, operating_point=dict(fpr=0.05, tpr_min=0.70, rationale="a contact/failure detector must catch >= 70 % of events with <= 5 % false alarms; first proposal, not tuned on any result")),
        brier_skill=dict(definition="BSS = 1 - B_model / max(B_ref, floor); B_* = macro mean over initial-state groups of the per-group mean squared error (p - y)^2 on the at-risk windows", floor=1e-3,
                         floor_rule="if B_ref < floor the denominator is the floor (the reference is essentially perfect: BSS >= 0.5 then requires B_model <= 0.5*floor); flagged in the detail"),
        primary=V2["G3"]["primary"], secondary=V2["G3"]["secondary"], at_risk=V2["G3"]["at_risk"], bins=15, bootstrap_resamples_auroc=2000, reported_not_gating=["AUROC bootstrap CI (ceiling branch)", "BSS CI over groups", "AUROC_ref", "branch", "prevalence", "TPR/precision at the operating point for non-rare labels", "lifted/placed secondary"]),
    G7=dict(
        rule="Object orientation. Everything is computed from QUATERNIONS (data contract v2). Regimes (fixed GEOMETRIC rules on the TRUE trajectory q_t..q_{t+K}; independent of predictions; a diagnostic stratification, never an inference-time selection), precedence singular > tumbling > rotating > stable: "
             "singular = min over rows of hnorm (horizontal norm of the body-x axis) < 0.20 [Euler yaw undefined]; tumbling = not singular and some per-step box-symmetric geodesic rotation > 0.30 rad; rotating = neither and total box-symmetric geodesic rotation over the horizon >= 0.02 rad; stable = otherwise. "
             "Yaw error e_k = |wrap(yaw_hat_k - yaw_k)|, yaw_k = wrap(yaw_euler(q_{t+k}) - yaw_euler(q_t)) from composed quaternions, ONLY on heading-valid windows (stable + rotating); Euler yaw is never used as ground truth at a singularity. "
             "Primary yaw non-inferiority (K=20): R = E_model / max(E_best, 1e-4) with E_best = min(E_persistence, E_ridge) (fixed on the point estimate): R <= 1.10 AND group-bootstrap upper bound <= 1.25. "
             "Static hallucination on the INPUT-defined subset S_in (heading-valid windows): (a) mean e_20 <= 0.01 rad; (b) group-bootstrap upper bound of (E_model - E_persistence) <= 0.01 rad; (c) E_model / max(E_persistence, 0.003) <= 3.0. "
             "Singular + tumbling windows: orientation error = box-symmetric geodesic distance between the composed predicted and true quaternion at K=20; ratio to the best of {persistence (zero rotation), ridge if available} <= 1.10 and bootstrap upper bound <= 1.25; each regime also reported separately. "
             "Support: a component is NOT_EVALUABLE if its sample has < 10 groups or < 200 windows, or if the model lacks the required output (yaw components accept orientation outputs Y_rv OR legacy per-step yaw deltas Y_yaw; geodesic components need Y_rv). Horizons 1/5/10/20 reported; gating horizon 20. Tilt: separate metric, unit-checked, reported only.",
        gating_horizon=20, horizons_reported=[1, 5, 10, 20], baselines=["persistence", "ridge"], error_floor_rad=1e-4, ratio_max=1.10, ratio_upper_bound_max=1.25,
        regimes=dict(tau_hnorm_singular=0.20, tumble_step_geodesic_rad=0.30, rotating_total_geodesic_min_rad=0.02, precedence=["singular", "tumbling", "rotating", "stable"], note="thresholds are geometric/physical scales chosen from the replay validation of the frozen episodes (outputs/contract_v2/validation.json), not from model predictions"),
        static_input_subset=dict(rule="S_in (INPUT-ONLY): box speed <= 0.005 m/s in all history rows AND no finger contact and palm force <= 0.01 N in all history rows AND the candidate future gripper command stays open (>= 0.028 m); gated on heading-valid windows (regime stable/rotating)", speed_max_mps=0.005, palm_force_max_N=0.01, gripper_open_min_m=0.028,
                                 abs_max_rad=0.01, diff_vs_persistence_upper_max_rad=0.01, relative=dict(max_ratio=3.0, denominator_floor_rad=0.003, floor_rationale="0.003 rad is the true-static tolerance (S_gt) and the numerical-noise scale; below it the persistence error is not a meaningful denominator"),
                                 diagnostic_subset="S_gt (true geodesic rotation < 0.003 rad) is reported as a DIAGNOSTIC subset (uses the future truth)"),
        orientation_component=dict(regimes=["singular", "tumbling"], metric="box-symmetric quaternion geodesic distance at K=20 (D2 invariance: identity and 180-degree rotations about the three body axes)", ratio_max=1.10, ratio_upper_bound_max=1.25),
        tilt=dict(reported_only=True, metric="|sym_tilt_hat - sym_tilt| in radians, sym_tilt = min(tilt, pi - tilt), tilt = arccos(R22) from quaternions (unit-safe)", unit_check="if a stored tilt channel is supplied it must match arccos(R22) within 1e-3 rad, else the stored-channel metric is NOT_EVALUABLE (data contract v1 mixed radians and degrees)"),
        requires_contract="T[sp]['Q_t'], T[sp]['Y_rv'], preds[m][sp]['Y_rv']"))
def config_bytes(): return json.dumps(CONFIG, indent=1, sort_keys=True).encode()
def config_sha(): return hashlib.sha256(config_bytes()).hexdigest()
def comp(name, passed=None, evaluable=True, applicable=True, **kw): return dict(name=name, passed=bool(passed) if passed is not None else False, evaluable=bool(evaluable), applicable=bool(applicable), **kw)
def verdict(components):
    c = [x for x in components if x.get("applicable", True)]; ev = [x for x in c if x["evaluable"]]
    if any(not x["passed"] for x in ev): return "FAIL"
    if len(ev) < len(c): return "NOT_EVALUABLE"
    return "PASS"
# ------------------------------------------------------------------------------------------------ G3 v3
def _bs_auroc_ci(y, p, g, n, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(g); mem = {x: np.nonzero(g == x)[0] for x in ug}; vs = []
    for _ in range(n):
        ii = np.concatenate([mem[x] for x in rng.choice(ug, len(ug))]); v = V.auroc(y[ii], p[ii])
        if not np.isnan(v): vs.append(v)
    return [float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))] if vs else [float("nan")] * 2
def brier_macro(y, p, g): return V.group_means((p - y) ** 2, g)[1]
def brier_skill(y, p, po, g, floor):
    bm, br = brier_macro(y, p, g), brier_macro(y, po, g); den = max(float(br.mean()), floor); return 1 - float(bm.mean()) / den, dict(B_model=float(bm.mean()), B_ref=float(br.mean()), denominator=den, floor_used=bool(br.mean() < floor)), bm, br
def tpr_at_fpr(y, p, fpr):
    neg = p[y == 0]; thr = np.quantile(neg, 1 - fpr); sel = p > thr; return float(sel[y == 1].mean()), float(y[sel].mean()) if sel.any() else float("nan"), float(thr)
def gate_G3_v3(cfg, preds, T, groups, masks, model="wm3b"):
    c = cfg["G3"]; comps, detail = [], {}
    for tgt in c["primary"]:
        y, p, po, g = V._gather(preds[model], preds[c["reference"]], T, groups, masks, tgt["label"], tgt["sets"]); ok, suff = V._suff(cfg, y, g); key = f"{tgt['label']}/{'+'.join(tgt['sets'])}"; d = dict(sufficiency=suff, evaluable=ok)
        if not ok: comps.append(comp(key, evaluable=False)); detail[key] = d; continue
        prev = float(y.mean()); rare = prev < c["rare_label"]["prevalence_below"]; a, ar = V.auroc(y, p), V.auroc(y, po); branch = "margin" if ar <= c["branch_threshold_auroc_ref"] else "ceiling"; ci = _bs_auroc_ci(y, p, g, c["bootstrap_resamples_auroc"]); bss, bd, bm, br = brier_skill(y, p, po, g, c["brier_skill"]["floor"])
        rng = np.random.default_rng(0); G = len(bm); idx = rng.integers(0, G, (2000, G)); bss_ci = [float(np.percentile(1 - bm[idx].mean(1) / np.maximum(br[idx].mean(1), c["brier_skill"]["floor"]), q)) for q in (2.5, 97.5)]
        tpr, prec, thr = tpr_at_fpr(y, p, c["rare_label"]["operating_point"]["fpr"]); d.update(prevalence=prev, rare_label=rare, branch=branch, auroc=a, auroc_ci=ci, auroc_ref=ar, margin=a - ar, brier_skill=bss, brier_detail=bd, brier_skill_ci_reported=bss_ci, operating_point=dict(fpr=c["rare_label"]["operating_point"]["fpr"], tpr=tpr, precision=prec))
        if branch == "margin":
            m = c["margin_branch"]; comps += [comp(f"{key}/auroc>={m['auroc_min']}", a >= m["auroc_min"], value=a), comp(f"{key}/auroc_ci_lower>={m['ci_lower_min']}", ci[0] >= m["ci_lower_min"], value=ci[0]), comp(f"{key}/margin_over_ref>={m['margin_min']}", a - ar >= m["margin_min"], value=a - ar)]
        else:
            m = c["ceiling_branch"]; comps += [comp(f"{key}/ceiling:auroc>={m['auroc_min']}", a >= m["auroc_min"], value=a), comp(f"{key}/ceiling:brier_skill>={m['brier_skill_min']}", bss >= m["brier_skill_min"], value=bss)]
        if rare:
            r = c["rare_label"]
            if branch == "margin": comps.append(comp(f"{key}/rare:brier_skill>={r['brier_skill_min']}", bss >= r["brier_skill_min"], value=bss))
            comps.append(comp(f"{key}/rare:tpr@fpr{r['operating_point']['fpr']}>={r['operating_point']['tpr_min']}", tpr >= r["operating_point"]["tpr_min"], value=tpr))
            comps.append(comp(f"{key}/hard_subset", applicable=False, evaluable=False, reason=f"NOT_APPLICABLE: rare label (prevalence {prev:.3f} < {c['hard_subset']['min_prevalence']})")); d["hard_subset"] = dict(applicable=False, reason="rare label")
        else:
            lo, hi = c["hard_subset"]["ref_prob_range"]; hm = (po >= lo) & (po <= hi); hok, hsuff = V._suff(cfg, y[hm], g[hm]) if hm.any() else (False, dict(pos_groups=0, neg_groups=0, pos_windows=0, neg_windows=0)); d["hard_subset"] = dict(applicable=True, n_windows=int(hm.sum()), sufficiency=hsuff, evaluable=hok)
            nm = f"{key}/hard_subset_auroc>={c['hard_subset']['auroc_min']}"
            if hok: ah = V.auroc(y[hm], p[hm]); d["hard_subset"]["auroc"] = ah; comps.append(comp(nm, ah >= c["hard_subset"]["auroc_min"], value=ah))
            else: comps.append(comp(nm, evaluable=False))
        detail[key] = d
    sec = {}
    for lab in c["secondary"]:
        sets = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; y, p, po, g = V._gather(preds[model], preds[c["reference"]], T, groups, masks, lab, sets); bss, _, _, _ = brier_skill(y, p, po, g, c["brier_skill"]["floor"]); sec[lab] = dict(auroc=V.auroc(y, p), auroc_ref=V.auroc(y, po), brier_skill=bss, prevalence=float(y.mean()))
    return dict(verdict=verdict(comps), conditions=comps, detail=detail, secondary=sec)
# ------------------------------------------------------------------------------------------------ G7 geometry
def _traj(Q_t, rv): Q = Gm.compose(Q_t, rv); return np.concatenate([Gm.qnorm(Q_t)[:, None, :], Q], 1)                  # (n, K+1, 4)
def regimes(Q_t, Y_rv, cfg=CONFIG):
    r = cfg["G7"]["regimes"]; Q = _traj(Q_t, Y_rv); hn = Gm.hnorm(Q); step = Gm.geodesic_sym(Q[:, :-1], Q[:, 1:]); total = Gm.geodesic_sym(Q[:, 0], Q[:, -1])
    sing = hn.min(1) < r["tau_hnorm_singular"]; tumb = ~sing & (step.max(1) > r["tumble_step_geodesic_rad"]); rot = ~sing & ~tumb & (total >= r["rotating_total_geodesic_min_rad"]); stable = ~(sing | tumb | rot)
    return dict(singular=sing, tumbling=tumb, rotating=rot, stable=stable), Q
def yaw_disp(Q, k): return wrap(Gm.yaw_euler(Q[:, k]) - Gm.yaw_euler(Q[:, 0]))
def static_input_mask(W, cfg=CONFIG):
    s = cfg["G7"]["static_input_subset"]; B, C, Af = W["B_hist"], W["C_hist"], W["A_fut"]
    return (B[:, :, 4].max(1) <= s["speed_max_mps"]) & (C[:, :, 1].max(1) == 0) & (C[:, :, 4].max(1) <= s["palm_force_max_N"]) & (Af[:, :, 7].min(1) >= s["gripper_open_min_m"])
def sym_tilt_err(Qp, Qt, k): return np.abs(Gm.tilt_sym(Qp[:, k]) - Gm.tilt_sym(Qt[:, k]))
def tilt_unit_check(stored_tilt, Q_row, tol=1e-3): d = float(np.abs(np.asarray(stored_tilt, float) - Gm.tilt_rad(Q_row)).max()); return d <= tol, d
# ------------------------------------------------------------------------------------------------ G7
def _legacy_yaw(Y_yaw, k): return wrap(np.cumsum(Y_yaw, 1)[:, k])                                              # legacy per-step wrapped-yaw deltas (data contract v1 models): displacement = wrap(sum)
def gate_G7(cfg, preds, T, groups, active, model="wm3b", sets=("test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"), n_boot=10000):
    """Predictions may carry per-step world rotation vectors preds[m][sp]['Y_rv'] (orientation outputs) and/or legacy per-step yaw deltas preds[m][sp]['Y_yaw'] (n,K).  Yaw components need either; orientation (geodesic) components need Y_rv."""
    c = cfg["G7"]; ms = cfg["min_sample"]; K = c["gating_horizon"]; sets = list(sets); names = [model] + c["baselines"]
    has_rv = lambda n: n in preds and all("Y_rv" in preds[n][sp] for sp in sets); has_yaw = lambda n: has_rv(n) or (n in preds and all("Y_yaw" in preds[n][sp] for sp in sets))
    cnames = ["yaw_ratio<=1.10", "yaw_ratio_upper<=1.25", "static_abs<=0.01", "static_diff_vs_persistence_upper<=0.01", "static_relative<=3.0", "orientation_ratio<=1.10", "orientation_ratio_upper<=1.25"]
    if "Q_t" not in T[sets[0]] or "Y_rv" not in T[sets[0]] or not has_yaw(model):
        reason = "model has no orientation outputs (Y_rv / Y_yaw) or windows lack quaternions (contract v2)"; cs = [comp(n, evaluable=False, reason=reason) for n in cnames]; return dict(verdict=verdict(cs), conditions=cs, detail=dict(reason=reason))
    G_ = {}; RG = {}
    for sp in sets:
        rg, Qt = regimes(T[sp]["Q_t"], T[sp]["Y_rv"], cfg); RG[sp] = rg; G_[sp] = dict(true=Qt)
        for n in names:
            if has_rv(n): G_[sp][n] = _traj(T[sp]["Q_t"], preds[n][sp]["Y_rv"])
    def yaw_pred(n, sp, mk, k): return yaw_disp(G_[sp][n][mk], k) if has_rv(n) else _legacy_yaw(preds[n][sp]["Y_yaw"][mk], k - 1)
    def per_group(name, k, mask_by_sp, kind="yaw"):
        ids, vals = [], []
        for sp in sets:
            mk = mask_by_sp[sp]
            if mk.sum() == 0: continue
            Qt = G_[sp]["true"][mk]
            if kind == "yaw": e = np.abs(wrap(yaw_pred(name, sp, mk, k) - yaw_disp(Qt, k)))
            else:
                Qp = G_[sp][name][mk]; e = Gm.geodesic_sym(Qp[:, k], Qt[:, k]) if kind == "geo" else sym_tilt_err(Qp, Qt, k)
            i, v = V.group_means(e, groups[sp][mk]); ids.append(i); vals.append(v)
        if not ids: return np.array([]), np.array([])
        ids, vals = np.concatenate(ids), np.concatenate(vals); o = np.argsort(ids); return ids[o], vals[o]
    def n_(mask_by_sp): return int(sum(m.sum() for m in mask_by_sp.values())), len({int(x) for sp in sets for x in np.unique(groups[sp][mask_by_sp[sp]])})
    def ok_support(mask_by_sp): nw, ng = n_(mask_by_sp); return ng >= ms["groups_error_gates"] and nw >= ms["active_windows"], nw, ng
    def ratio_comps(prefix, mask_by_sp, kind, bl, usable):
        ok, nw, ng = ok_support(mask_by_sp); nm1, nm2 = f"{prefix}_ratio<=1.10", f"{prefix}_ratio_upper<=1.25"
        if not ok or not usable or not bl: return [comp(nm1, evaluable=False), comp(nm2, evaluable=False)], dict(n_windows=nw, n_groups=ng, evaluable=False, reason="insufficient support" if not ok else "model lacks the required output")
        em = per_group(model, K, mask_by_sp, kind)[1]; base = {b: per_group(b, K, mask_by_sp, kind)[1] for b in bl}; best = min(base, key=lambda b: base[b].mean()); fl = c["error_floor_rad"]; R = float(em.mean() / max(base[best].mean(), fl)); rng = np.random.default_rng(0); Gn = len(em); idx = rng.integers(0, Gn, (n_boot, Gn))
        Rb = em[idx].mean(1) / np.maximum(base[best][idx].mean(1), fl); lo, hi = float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))
        return [comp(nm1, R <= c["ratio_max"], value=R), comp(nm2, hi <= c["ratio_upper_bound_max"], value=hi)], dict(E_model=float(em.mean()), E_best=float(base[best].mean()), best_baseline=best, E_by_baseline={b: float(v.mean()) for b, v in base.items()}, ratio=R, ratio_ci=[lo, hi], n_groups=int(Gn), n_windows=nw)
    act_hv = {sp: active[sp] & (RG[sp]["stable"] | RG[sp]["rotating"]) for sp in sets}; comps, detail = [], {}
    cs, detail["yaw_primary"] = ratio_comps("yaw", act_hv, "yaw", [b for b in c["baselines"] if has_yaw(b)], True); comps += cs
    sin = {sp: static_input_mask(T[sp], cfg) & (RG[sp]["stable"] | RG[sp]["rotating"]) for sp in sets}; ok, nw, ng = ok_support(sin); s = c["static_input_subset"]; names_s = ["static_abs<=0.01", "static_diff_vs_persistence_upper<=0.01", "static_relative<=3.0"]
    if ok and has_yaw("persistence"):
        em = per_group(model, K, sin)[1]; ep = per_group("persistence", K, sin)[1]; d_ = em - ep; rng = np.random.default_rng(0); Gn = len(em); idx = rng.integers(0, Gn, (n_boot, Gn)); up = float(np.percentile(d_[idx].mean(1), 97.5)); rel = float(em.mean() / max(ep.mean(), s["relative"]["denominator_floor_rad"]))
        detail["static_input_subset"] = dict(E_model=float(em.mean()), E_persistence=float(ep.mean()), diff=float(d_.mean()), diff_ci=[float(np.percentile(d_[idx].mean(1), 2.5)), up], relative=rel, denominator_used=max(float(ep.mean()), s["relative"]["denominator_floor_rad"]), n_groups=ng, n_windows=nw)
        comps += [comp(names_s[0], em.mean() <= s["abs_max_rad"], value=float(em.mean())), comp(names_s[1], up <= s["diff_vs_persistence_upper_max_rad"], value=up), comp(names_s[2], rel <= s["relative"]["max_ratio"], value=rel)]
    else: detail["static_input_subset"] = dict(n_groups=ng, n_windows=nw, evaluable=False); comps += [comp(n, evaluable=False) for n in names_s]
    st = {sp: active[sp] & (RG[sp]["singular"] | RG[sp]["tumbling"]) for sp in sets}; cs, detail["orientation_singular_tumbling"] = ratio_comps("orientation", st, "geo", [b for b in c["baselines"] if has_rv(b)], has_rv(model)); comps += cs
    rep = {}
    for rn in ("stable", "rotating", "singular", "tumbling"):
        mk = {sp: active[sp] & RG[sp][rn] for sp in sets}; nw, ng = n_(mk); rep[rn] = dict(n_windows=nw, n_groups=ng)
        if nw: rep[rn].update(yaw_E_K20=({n: float(per_group(n, K, mk, "yaw")[1].mean()) for n in names if has_yaw(n) and len(per_group(n, K, mk, "yaw")[1])} if rn in ("stable", "rotating") else None), geodesic_E_K20={n: float(per_group(n, K, mk, "geo")[1].mean()) for n in names if has_rv(n) and len(per_group(n, K, mk, "geo")[1])})
    detail["regimes_reported"] = rep; detail["reported_horizons_yaw_heading_valid"] = {str(k): {n: float(per_group(n, k, act_hv, "yaw")[1].mean()) for n in names if has_yaw(n) and len(per_group(n, k, act_hv, "yaw")[1])} for k in c["horizons_reported"]}
    sgt = {sp: static_input_mask(T[sp], cfg) & (RG[sp]["stable"] | RG[sp]["rotating"]) & (Gm.geodesic_sym(G_[sp]["true"][:, 0], G_[sp]["true"][:, -1]) < 0.003) for sp in sets}; detail["diagnostic_S_gt"] = dict(uses_future_truth=True, n_windows=n_(sgt)[0], yaw_E_K20={n: float(per_group(n, K, sgt)[1].mean()) for n in names if has_yaw(n) and n_(sgt)[0]})
    detail["tilt_reported_only"] = {"metric": c["tilt"]["metric"], "E_K20_rad": {n: float(per_group(n, K, {sp: active[sp] for sp in sets}, "tilt")[1].mean()) for n in names if has_rv(n) and len(per_group(n, K, {sp: active[sp] for sp in sets}, "tilt")[1])}}
    stored = [sp for sp in sets if "tilt_stored" in T[sp]]
    if stored: ok_u, dmax = tilt_unit_check(np.concatenate([T[sp]["tilt_stored"] for sp in stored]), np.concatenate([G_[sp]["true"][:, 0] for sp in stored])); detail["tilt_unit_check"] = dict(ok=ok_u, max_abs_diff_rad=dmax, stored_channel_metric="evaluable" if ok_u else "NOT_EVALUABLE (unit/definition mismatch)")
    return dict(verdict=verdict(comps), conditions=comps, detail=detail)
if __name__ == "__main__":
    if "--draft" in sys.argv:
        out = ROOT / "outputs/gates_v3"; out.mkdir(parents=True, exist_ok=True); p = out / "gates_v3_config.DRAFT.json"; p.write_bytes(config_bytes()); print("DRAFT written (not frozen):", p, "sha256", config_sha())
    else: print("sha256 of current DRAFT CONFIG:", config_sha())
