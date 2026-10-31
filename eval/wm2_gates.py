#!/usr/bin/env python3
"""WM-2 gates G0-G6: frozen definitions (CONFIG), metric functions and verdict logic.  Verdicts are exactly PASS / FAIL / NOT_EVALUABLE per condition and per gate:
  gate = FAIL if any evaluable condition fails; else NOT_EVALUABLE if any required condition is not evaluable; else PASS.
Aggregation (all error/AUC metrics): windows -> per initial-state GROUP mean -> macro mean over groups; 95 % percentile bootstrap over GROUPS (10 000 resamples, seed 0).
The model gated is the GRU prototype; baselines (persistence, constvel, ridge, mlp, ridge_blind, mlp_blind, outcome_only) are only compared against.  Thresholds are read from the frozen config (sha-checked);
nothing here may change after model results are seen.  Run `python eval/wm2_gates.py --freeze` once (before learned training) to write outputs/wm2/gates_config.json and print its sha256."""
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; CFG_PATH = ROOT / "outputs/wm2/gates_config.json"
K = 20
CONFIG = dict(
    version="WM-2 gates v1", model_gated="gru", aggregation=dict(unit="initial-state group", within_group="mean over windows", across_groups="macro mean", bootstrap=dict(resamples=10000, seed=0, level=0.95, unit="groups")),
    windows=dict(primary="active", active_definition="INPUT-ONLY (no target information): max over the K=20 planned actions and the 7 left-arm joints of |A_fut - last past action| > 0.02 rad (commanded motion; independent of the gravity-compensation offset between command and state), OR the left-gripper command crosses 0.028 m between the last past action and the K future actions",
                 active_thresholds=dict(joint_command_gap_rad=0.02, gripper_threshold_m=0.028), secondary="all", horizon_K=20, horizons=[1, 5, 10, 20], H=4, stride=5),
    metrics=dict(box_mm="Euclidean error of cumulative box position change at horizon k, x1000 [mm]", joint_rad="mean over the 7 left-arm joints of |cumulative joint change error| at horizon k [rad]", aoffset_cm="|error| of box-in-TCP a-offset at horizon k [cm]",
                 note="cumulative change = cumsum of the per-step predicted deltas; initial values cancel because every model starts from the observed current row"),
    baseline_pool=["persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind"], best_baseline_rule="per (split, metric) the pool member with the lowest macro error on the POINT estimate; fixed before bootstrapping",
    floors=dict(box_mm=1.0, joint_rad=0.002, aoffset_cm=0.1, rule="if a denominator or baseline error is below its floor the floor is used; relative improvement is then undefined and replaced by absolute difference <= floor"),
    min_sample=dict(groups_error_gates=10, groups_ood_sets=8, active_windows=200, g3_pos_groups=10, g3_neg_groups=10, g3_pos_windows=30, g3_neg_windows=30, paired_poses=10),
    G0=dict(rule="PASS iff: dataset freeze hash matches; 0 groups in >1 split; 0 duplicate action hashes; no seeds 31-33; interventional variants of a pose share one split and no interventional group overlaps an observational group; window tests pass; training code loaded only splits train/val (log checked); originals untouched",
            fail="any item false", not_evaluable="never"),
    G1=dict(split="test_id", horizon=20, subset="active", rel_improvement_min=0.20, ci_lower_gt=0.0, box_abs_max_mm=30.0, joint_abs_max_rad=0.07, metrics=["box_mm", "joint_rad", "aoffset_cm"], abs_applies_to=["box_mm", "joint_rad"],
            rule="PASS iff for box, joint and a-offset: relative improvement r = 1 - E_model/E_best >= 0.20 with bootstrap lower bound > 0, AND box <= 30 mm AND joint <= 0.07 rad; NOT_EVALUABLE if groups < 10 or active windows < 200"),
    G2=dict(sets=["test_low_heldout", "test_ood_low_strict", "test_ood_high"], id_split="test_id", metrics=["box_mm", "joint_rad"], horizon=20, subset="active", noninferior_margin=0.0, ratio_max=1.5,
            ratio_definition="rho = E_model(OOD set) / max(E_model(test_id), floor), same metric, subset, horizon, group-macro", improvement_rule="at least one (set, metric) with relative-improvement lower bound > 0 against the best baseline of that set",
            rule="PASS iff on every OOD set and both metrics: E_model <= E_best (margin 0; if E_best < floor: E_model - E_best <= floor) AND rho <= 1.5, plus the improvement rule; FAIL if any evaluable condition fails; NOT_EVALUABLE if a set has < 8 groups or < 200 active windows (and nothing failed)"),
    G3=dict(subset="at-risk windows per label (input-only): the event has not already happened at t", at_risk=dict(close="left-gripper last past action >= 0.028 m (open) AND box centre farther than 0.05 m from the tray centre", lifted="box z <= 0.4105 m (not lifted > 3 cm above its resting height 0.3805) AND box centre farther than 0.05 m from the tray centre", placed="NOT (box within 0.03 m of the tray centre AND |z - 0.384| < 0.006 m)", palm="palm contact force at t <= 0.01 N"), required=[["close", "test_id"], ["lifted", "test_id"], ["placed", "test_id"], ["palm", "test_low_heldout"], ["palm", "test_ood_low_strict"]], labels=["close", "palm", "lifted", "placed"],
            auroc_min=0.80, auroc_ci_lower_min=0.70, margin_over_outcome_only=0.05,
            sufficiency="(evaluated inside the at-risk subset) evaluable only if >=10 initial-state groups contain >=1 positive window, >=10 groups contain >=1 negative window, and >=30 positive and >=30 negative windows (within the subset); otherwise that combination is NOT_EVALUABLE",
            failure_target_note="'failure' = failure precursors inside the horizon: palm contact (mechanism), and not-lifted / not-placed events; episode-level failure is not predictable from one K=20 window",
            rule="PASS iff every required combination is evaluable and has AUROC >= 0.80, bootstrap lower bound >= 0.70 and AUROC - AUROC(outcome_only) >= 0.05; FAIL if any evaluable combination fails; NOT_EVALUABLE otherwise"),
    G4=dict(blind_split="test_id", metric="box_mm", horizon=20, subset="active", blind_rel_improvement_min=0.20, blind_baselines=["ridge_blind", "mlp_blind"], shuffle_degradation_min=0.30, shuffle_rule="replace A_hist and A_fut of each window by those of a random window of the SAME split/subset from a DIFFERENT group (seed 0)",
            paired=dict(window="t_pair = control close step - 10, same absolute t for all 25 variants of a pose; held-out interventional poses (test_id, test_low_heldout, test_ood_low_strict, test_ood_high)", sign_agreement_min=0.80, sign_min_abs_effect=dict(aoffset_cm=0.5, box_z_mm=5.0),
                        spearman_effect_min=0.70, depth_monotone_spearman_max=-0.80, depth_monotone_pose_fraction_min=0.80, palm_monotone_spearman_min=0.80, palm_monotone_pose_fraction_min=0.80, quantities=["aoffset_cm at t+K", "box_z at t+K"]),
            rule="PASS iff (a) action-conditioned beats the best action-blind learned baseline by >= 20 % (lower bound > 0) AND (b) action-shuffle raises the box error by >= 30 % (lower bound > 0) AND (c) paired-intervention consistency thresholds hold; NOT_EVALUABLE if < 10 paired poses"),
    G5=dict(bins=15, binning="equal width on [0,1], windows pooled", ece_max=0.05, brier_rule="Brier(model) < min(Brier(prevalence-only), Brier(outcome_only)) on the same windows", reported=["ECE", "Brier", "prevalence", "reliability plot"], applies_to="G3 combinations that are evaluable", ensemble_spread="Spearman(ensemble spread, error) >= 0.30 needs an ensemble: NOT_EVALUABLE for the single-seed prototype",
            rule="PASS iff every evaluable G3 combination has ECE <= 0.05 and the Brier rule; NOT_EVALUABLE if no G3 combination is evaluable"),
    G6=dict(batch=1, K=20, threads=1, device="cpu", metric="median wall-clock of one full K=20 prediction from normalised input tensors to output tensors, 1000 timed runs after 100 warm-ups, same process/VM as the ACT query benchmark", median_max_ms=10.0, params_max=5_000_000, report_also=["p95", "p99", "ACT per-query latency measured in the same session", "matmul fingerprint"],
            rule="PASS iff median <= 10 ms AND parameters <= 5 M; FAIL otherwise; never NOT_EVALUABLE"))
def config_bytes(): return json.dumps(CONFIG, indent=1, sort_keys=True).encode()
def config_sha(): return hashlib.sha256(config_bytes()).hexdigest()
def load_config(expect=None):
    if not CFG_PATH.exists(): raise SystemExit("gates config not frozen")
    b = CFG_PATH.read_bytes(); h = hashlib.sha256(b).hexdigest()
    if expect and h != expect: raise SystemExit(f"gates config changed: {h} != {expect}")
    if h != config_sha(): raise SystemExit("gates_config.json differs from the CONFIG in wm2_gates.py (definitions changed after freeze)")
    return json.loads(b)

# ------------------------------------------------------------------------------------------------ input-only window subsets
TRAY = np.array([0.44, 0.30])
def active_mask(W, thr=0.02, gthr=0.028):
    gap = np.abs(W["A_fut"][:, :, :7] - W["A_hist"][:, -1:, :7]).max((1, 2)) > thr; gc = np.concatenate([W["A_hist"][:, -1:, 7], W["A_fut"][:, :, 7]], 1)
    return gap | ((gc.min(1) < gthr) & (gc.max(1) >= gthr))
def at_risk_mask(W, label):
    b = W["B_hist"][:, -1]; far = np.linalg.norm(b[:, :2] - TRAY, axis=1) > 0.05
    if label == "close": return (W["A_hist"][:, -1, 7] >= 0.028) & far
    if label == "lifted": return (b[:, 2] <= 0.4105) & far
    if label == "placed": return ~((np.linalg.norm(b[:, :2] - TRAY, axis=1) < 0.03) & (np.abs(b[:, 2] - 0.384) < 0.006))
    if label == "palm": return W["C_hist"][:, -1, 4] <= 0.01
    raise ValueError(label)
# ------------------------------------------------------------------------------------------------ metrics
def cum(Y): return np.cumsum(Y, axis=1)
def errors(P, T, k=K):
    """P, T: dicts with Y_s (N,K,8), Y_b (N,K,7). Returns dict of per-window errors at horizon k."""
    i = k - 1; cs_p, cs_t, cb_p, cb_t = cum(P["Y_s"]), cum(T["Y_s"]), cum(P["Y_b"]), cum(T["Y_b"])
    return dict(box_mm=np.linalg.norm(cb_p[:, i, :3] - cb_t[:, i, :3], axis=1) * 1000.0, joint_rad=np.abs(cs_p[:, i, :7] - cs_t[:, i, :7]).mean(1), aoffset_cm=np.abs(cb_p[:, i, 6] - cb_t[:, i, 6]) * 100.0)
def group_means(vals, groups):
    g = np.unique(groups); return g, np.array([vals[groups == x].mean() for x in g])
def macro(vals, groups): return float(group_means(vals, groups)[1].mean())
def boot(fn, mats, n=10000, seed=0, level=0.95):
    """mats: dict name -> (G,) per-group arrays aligned on the same groups. fn(dict of resampled arrays) -> scalar. Returns (point, lo, hi)."""
    G = len(next(iter(mats.values()))); rng = np.random.default_rng(seed); pt = fn(mats); idx = rng.integers(0, G, (n, G)); vs = np.array([fn({k: v[i] for k, v in mats.items()}) for i in idx]); a = (1 - level) / 2
    return float(pt), float(np.nanpercentile(vs, 100 * a)), float(np.nanpercentile(vs, 100 * (1 - a)))
def auroc(y, p):
    y = np.asarray(y).astype(bool); p = np.asarray(p, float); npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0: return float("nan")
    r = _rank(p); return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def _rank(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); xs = x[o]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def ece_brier(y, p, bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float); e = 0.0; b = np.minimum((p * bins).astype(int), bins - 1)
    for k in range(bins):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e), float(np.mean((p - y) ** 2)), float(y.mean())
def reliability(y, p, bins=15):
    b = np.minimum((np.asarray(p) * bins).astype(int), bins - 1); return [dict(bin=k, n=int((b == k).sum()), mean_pred=float(np.mean(np.asarray(p)[b == k])) if (b == k).any() else None, frac_pos=float(np.mean(np.asarray(y)[b == k])) if (b == k).any() else None) for k in range(bins)]
def shuffle_partner(groups, seed=0):
    """index of a random window from a DIFFERENT group for each window (never the same group)."""
    rng = np.random.default_rng(seed); n = len(groups); out = np.empty(n, int)
    for i in range(n):
        while True:
            j = int(rng.integers(0, n))
            if groups[j] != groups[i]: out[i] = j; break
    return out
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])
def verdict(conds):
    """conds: list of dict(name, passed(bool|None), evaluable(bool))."""
    ev = [c for c in conds if c["evaluable"]]
    if any(not c["passed"] for c in ev): return "FAIL"
    if len(ev) < len(conds): return "NOT_EVALUABLE"
    return "PASS"
def sub(W_, mask): return {k: (v[mask] if hasattr(v, "__len__") and len(v) == len(mask) else v) for k, v in W_.items()}

# ------------------------------------------------------------------------------------------------ gates
def per_group(P, T, groups, active, metric, k=K):
    m = active; e = errors(sub(P, m), sub(T, m), k)[metric]; return group_means(e, groups[m])
def rel_improve(model_err_g, best_err_g, floor):
    """returns function over (mats) -> r or nan when the baseline is below its floor."""
    def f(M):
        b = M["best"].mean(); return float("nan") if b < floor else float(1 - M["model"].mean() / b)
    return f
def gate_G1(cfg, preds, T, groups, active):
    c = cfg["G1"]; sp = c["split"]; gv = groups[sp]; act = active[sp]; conds = []
    ng = len(np.unique(gv[act])); ev = ng >= cfg["min_sample"]["groups_error_gates"] and act.sum() >= cfg["min_sample"]["active_windows"]; detail = {}
    for met in c["metrics"]:
        gm, em = per_group(preds["gru"][sp], T[sp], gv, act, met); pool = {}
        for b in cfg["baseline_pool"]:
            if b in preds: pool[b] = per_group(preds[b][sp], T[sp], gv, act, met)[1]
        best = min(pool, key=lambda b: pool[b].mean()); r, lo, hi = boot(rel_improve(em, pool[best], cfg["floors"][met]), dict(model=em, best=pool[best]))
        detail[met] = dict(model=float(em.mean()), best_baseline=best, best_value=float(pool[best].mean()), rel_improvement=r, ci_lo=lo, ci_hi=hi)
        conds.append(dict(name=f"{met}_rel_improvement>=0.20_and_ci_lo>0", value=r, ci=[lo, hi], passed=bool(r >= c["rel_improvement_min"] and lo > c["ci_lower_gt"]), evaluable=ev))
    for met, thr in (("box_mm", c["box_abs_max_mm"]), ("joint_rad", c["joint_abs_max_rad"])): conds.append(dict(name=f"{met}_abs<={thr}", value=detail[met]["model"], passed=bool(detail[met]["model"] <= thr), evaluable=ev))
    return dict(verdict=verdict(conds), conditions=conds, detail=detail, groups=ng, active_windows=int(act.sum()))
def gate_G2(cfg, preds, T, groups, active):
    c = cfg["G2"]; conds = []; detail = {}; any_improve = False; idg = groups[c["id_split"]]; ida = active[c["id_split"]]
    for sp in c["sets"]:
        gv, act = groups[sp], active[sp]; ng = len(np.unique(gv[act])); ev = ng >= cfg["min_sample"]["groups_ood_sets"] and act.sum() >= cfg["min_sample"]["active_windows"]; detail[sp] = {}
        for met in c["metrics"]:
            fl = cfg["floors"][met]; em = per_group(preds["gru"][sp], T[sp], gv, act, met)[1]; ide = per_group(preds["gru"][c["id_split"]], T[c["id_split"]], idg, ida, met)[1].mean()
            pool = {b: per_group(preds[b][sp], T[sp], gv, act, met)[1] for b in cfg["baseline_pool"] if b in preds}; best = min(pool, key=lambda b: pool[b].mean()); eb = pool[best].mean()
            r, lo, hi = boot(rel_improve(em, pool[best], fl), dict(model=em, best=pool[best])); rho = float(em.mean() / max(ide, fl))
            near0 = eb < fl; noninf = bool((em.mean() - eb) <= fl) if near0 else bool(em.mean() <= eb + c["noninferior_margin"] * eb)
            detail[sp][met] = dict(model=float(em.mean()), best_baseline=best, best_value=float(eb), baseline_near_zero=bool(near0), rel_improvement=r, ci=[lo, hi], rho=rho, id_error=float(ide), groups=ng)
            conds += [dict(name=f"{sp}/{met}/noninferior", passed=noninf, value=float(em.mean() - eb), evaluable=ev), dict(name=f"{sp}/{met}/rho<=1.5", passed=bool(rho <= c["ratio_max"]), value=rho, evaluable=ev)]
            any_improve |= bool(ev and (not near0) and lo > 0)
    imp = dict(name="improvement_lower_bound>0_in_at_least_one_set_metric", passed=bool(any_improve), value=None, evaluable=all(cd["evaluable"] for cd in conds)); conds.append(imp)
    return dict(verdict=verdict(conds), conditions=conds, detail=detail)
def _g3_sufficient(cfg, y, g):
    m = cfg["min_sample"]; pg = len(np.unique(g[y == 1])); ng = len(np.unique(g[y == 0])); return bool(pg >= m["g3_pos_groups"] and ng >= m["g3_neg_groups"] and (y == 1).sum() >= m["g3_pos_windows"] and (y == 0).sum() >= m["g3_neg_windows"]), dict(pos_groups=int(pg), neg_groups=int(ng), pos_windows=int((y == 1).sum()), neg_windows=int((y == 0).sum()))
def gate_G3(cfg, preds, T, groups, masks, label_names=("close", "palm", "lifted", "placed")):
    c = cfg["G3"]; conds = []; detail = {}
    for lab, sp in c["required"]:
        j = list(label_names).index(lab); act = masks[(lab, sp)]; y = T[sp]["Y_e"][act][:, j].astype(int); g = groups[sp][act]; ok, suff = _g3_sufficient(cfg, y, g); d = dict(sufficiency=suff, evaluable=ok)
        if ok:
            p = preds["gru"][sp]["Y_e"][act][:, j]; po = preds["outcome_only"][sp]["Y_e"][act][:, j]; a = auroc(y, p); ao = auroc(y, po); ug = np.unique(g); rng = np.random.default_rng(cfg["aggregation"]["bootstrap"]["seed"]); vs = []
            members = {x: np.nonzero(g == x)[0] for x in ug}
            for _ in range(2000):
                ii = np.concatenate([members[x] for x in rng.choice(ug, len(ug))]); v = auroc(y[ii], p[ii])
                if not np.isnan(v): vs.append(v)
            lo = float(np.percentile(vs, 2.5)); d.update(auroc=a, ci_lo=lo, auroc_outcome_only=ao, margin=a - ao)
            conds.append(dict(name=f"{lab}/{sp}/auroc", value=a, ci_lo=lo, margin_over_outcome_only=a - ao, passed=bool(a >= c["auroc_min"] and lo >= c["auroc_ci_lower_min"] and a - ao >= c["margin_over_outcome_only"]), evaluable=True))
        else: conds.append(dict(name=f"{lab}/{sp}/auroc", passed=None, evaluable=False))
        detail[f"{lab}/{sp}"] = d
    return dict(verdict=verdict([dict(cd, passed=bool(cd["passed"])) for cd in conds]), conditions=conds, detail=detail)
def gate_G5(cfg, preds, T, groups, masks, g3, label_names=("close", "palm", "lifted", "placed")):
    c = cfg["G5"]; conds = []; detail = {}
    for lab, sp in cfg["G3"]["required"]:
        key = f"{lab}/{sp}"
        if not g3["detail"][key]["evaluable"]: continue
        j = list(label_names).index(lab); act = masks[(lab, sp)]; y = T[sp]["Y_e"][act][:, j]; p = preds["gru"][sp]["Y_e"][act][:, j]; po = preds["outcome_only"][sp]["Y_e"][act][:, j]
        e, b, prev = ece_brier(y, p, c["bins"]); _, bo, _ = ece_brier(y, po, c["bins"]); bprev = float(np.mean((prev - y) ** 2)); ok = bool(e <= c["ece_max"] and b < min(bo, bprev))
        detail[key] = dict(ECE=e, Brier=b, Brier_outcome_only=bo, Brier_prevalence_only=bprev, prevalence=prev, reliability=reliability(y, p, c["bins"])); conds.append(dict(name=key, ECE=e, Brier=b, passed=ok, evaluable=True))
    if not conds: return dict(verdict="NOT_EVALUABLE", conditions=[], detail=detail, note="no G3 combination evaluable")
    return dict(verdict=verdict(conds), conditions=conds, detail=detail, ensemble_spread="NOT_EVALUABLE (single seed)")
def gate_G6(cfg, bench):
    c = cfg["G6"]; ok = bool(bench["median_ms"] <= c["median_max_ms"] and bench["params"] <= c["params_max"]); return dict(verdict="PASS" if ok else "FAIL", conditions=[dict(name="median_ms<=10", value=bench["median_ms"], passed=bench["median_ms"] <= c["median_max_ms"], evaluable=True), dict(name="params<=5M", value=bench["params"], passed=bench["params"] <= c["params_max"], evaluable=True)], bench=bench)

def gate_G4(cfg, preds, T, groups, active, pairs, pair_preds):
    c = cfg["G4"]; conds = []; detail = {}; sp = c["blind_split"]; gv, act = groups[sp], active[sp]; ng = len(np.unique(gv[act])); ev = ng >= cfg["min_sample"]["groups_error_gates"]
    em = per_group(preds["gru"][sp], T[sp], gv, act, c["metric"])[1]; blind = {b: per_group(preds[b][sp], T[sp], gv, act, c["metric"])[1] for b in c["blind_baselines"] if b in preds}; bb = min(blind, key=lambda b: blind[b].mean())
    r, lo, hi = boot(rel_improve(em, blind[bb], cfg["floors"][c["metric"]]), dict(model=em, best=blind[bb])); detail["action_blind"] = dict(best_blind=bb, blind_value=float(blind[bb].mean()), model=float(em.mean()), rel_improvement=r, ci=[lo, hi])
    conds.append(dict(name="beats_best_action_blind_by_20pct", value=r, passed=bool(r >= c["blind_rel_improvement_min"] and lo > 0), evaluable=ev))
    es = per_group(preds["gru_shuffle"][sp], T[sp], gv, act, c["metric"])[1]; d, dlo, dhi = boot(lambda M: float(M["shuf"].mean() / M["model"].mean() - 1), dict(model=em, shuf=es)); detail["action_shuffle"] = dict(shuffled=float(es.mean()), model=float(em.mean()), degradation=d, ci=[dlo, dhi])
    conds.append(dict(name="action_shuffle_degrades_box_error_by_30pct", value=d, passed=bool(d >= c["shuffle_degradation_min"] and dlo > 0), evaluable=ev))
    pc = c["paired"]; K_ = K; fam, lev, pid, vid = pairs["family"], pairs["level"], pairs["pid"], pairs["vid"]; P, Tt = pair_preds["gru"], pairs
    qa = lambda W_: (pairs["B_hist"][:, -1, 6] + cum(W_["Y_b"])[:, K_ - 1, 6]) * 100.0; qz = lambda W_: (pairs["B_hist"][:, -1, 2] + cum(W_["Y_b"])[:, K_ - 1, 2]) * 1000.0     # the current box row is an INPUT shared by truth and prediction
    poses = sorted(set(pid.tolist())); evp = len(poses) >= cfg["min_sample"]["paired_poses"]; Qt = dict(a=qa(Tt), z=qz(Tt)); Qp = dict(a=qa(P), z=qz(P)); agree = {"a": [], "z": []}; dT, dP = [], []; mono = []; pm = []
    for q, thr in (("a", pc["sign_min_abs_effect"]["aoffset_cm"]), ("z", pc["sign_min_abs_effect"]["box_z_mm"])):
        for ps in poses:
            m = pid == ps; ci_ = np.nonzero(m & (vid == 0))[0]
            if len(ci_) == 0: continue
            rows = np.nonzero(m & (vid != 0))[0]; dt = Qt[q][rows] - Qt[q][ci_[0]]; dp = Qp[q][rows] - Qp[q][ci_[0]]; sel = np.abs(dt) >= thr
            if sel.any(): agree[q].append(float(np.mean(np.sign(dt[sel]) == np.sign(dp[sel]))))
            if q == "a": dT += dt.tolist(); dP += dp.tolist()
    for ps in poses:
        m = (pid == ps) & ((fam == 1) | (vid == 0))
        if m.sum() >= 4:
            mono.append(spearman(lev[m] * (fam[m] == 1), Qp["a"][m])); yp = Tt["Y_e"][m][:, 1]
            if yp.min() != yp.max(): pm.append(spearman(lev[m] * (fam[m] == 1), P["Y_e"][m][:, 1]))
    sa, sz, sp_ = (float(np.mean(agree["a"])) if agree["a"] else float("nan")), (float(np.mean(agree["z"])) if agree["z"] else float("nan")), spearman(dP, dT)
    detail["paired"] = dict(poses=len(poses), sign_agreement_aoffset=sa, sign_agreement_box_z=sz, spearman_effect_aoffset=sp_, depth_monotone_fraction=float(np.mean(np.array(mono) <= pc["depth_monotone_spearman_max"])) if mono else None, palm_monotone_fraction=float(np.mean(np.array(pm) >= pc["palm_monotone_spearman_min"])) if pm else None, n_poses_palm_varies=len(pm))
    conds += [dict(name="paired_sign_agreement_aoffset>=0.80", value=sa, passed=bool(sa >= pc["sign_agreement_min"]), evaluable=evp and not np.isnan(sa)), dict(name="paired_sign_agreement_box_z>=0.80", value=sz, passed=bool(sz >= pc["sign_agreement_min"]), evaluable=evp and not np.isnan(sz)),
              dict(name="paired_spearman_effect_aoffset>=0.70", value=sp_, passed=bool(sp_ >= pc["spearman_effect_min"]), evaluable=evp and not np.isnan(sp_)),
              dict(name="depth_monotone_pose_fraction>=0.80", value=detail["paired"]["depth_monotone_fraction"], passed=bool((detail["paired"]["depth_monotone_fraction"] or 0) >= pc["depth_monotone_pose_fraction_min"]), evaluable=evp and bool(mono)),
              dict(name="palm_monotone_pose_fraction>=0.80", value=detail["paired"]["palm_monotone_fraction"], passed=bool((detail["paired"]["palm_monotone_fraction"] or 0) >= pc["palm_monotone_pose_fraction_min"]), evaluable=evp and len(pm) >= 3)]
    return dict(verdict=verdict(conds), conditions=conds, detail=detail)
def gate_G0(cfg, info):
    items = ["dataset_freeze_hash_matches", "groups_in_multiple_splits_zero", "duplicate_action_hashes_zero", "no_forbidden_seeds", "interventional_variants_share_split", "no_interventional_group_overlap_with_observational", "window_tests_pass", "training_loaded_only_train_val", "originals_untouched"]
    conds = [dict(name=k, passed=bool(info.get(k)), evaluable=True) for k in items]; return dict(verdict=verdict(conds), conditions=conds)

if __name__ == "__main__":
    if "--freeze" in sys.argv:
        CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CFG_PATH.exists(): sys.exit(f"{CFG_PATH} exists (frozen); refusing to overwrite")
        CFG_PATH.write_bytes(config_bytes()); print("gates config written, sha256", config_sha())
    else: print("sha256 of current CONFIG:", config_sha())
