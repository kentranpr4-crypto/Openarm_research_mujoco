#!/usr/bin/env python3
"""G2b paired statistical estimators (design implementation; unit-tested on synthetic data only).
Data: arms -> {pose_id: endpoint dict(value in TRUE/FALSE/CENSORED/INVALID/NOT_APPLICABLE, ambiguous bool)} for the primary endpoint E2.  Unit = pose; every comparison is a within-pose paired risk difference RD(X - c0); resampling is over POSES with all arms kept together (the same resampled pose indices for every arm).
Distinguishes (never conflates): MISSING/CENSORED observations, INVALID branches, AMBIGUOUS-but-valid labels (kept with their computed value), the complete-case primary estimand and the pre-specified worst/best-case sensitivity analyses.  No pose subset is dropped silently: every population reports its attrition by reason.
CI levels (fixed before freeze): primary O1-c0 and D1-c0: 97.5 % (Bonferroni, 2 primary tests); every other comparison 95 %; S0 equivalence (TOST, alpha 0.05 per side): 90 % interval inside (-0.10, +0.10), the 95 % interval reported alongside."""
import numpy as np
LEVELS = dict(primary=0.975, secondary=0.95, equivalence=0.90); MARGIN = 0.10; B_DEFAULT = 10000
ARMS = ("c0", "O1", "O3", "D1", "D2", "H1", "S0"); ORACLE = ("O1", "O3"); DEPLOYABLE = ("D1", "D2", "H1", "S0")
def _ep(data, arm, pose): return data.get(arm, {}).get(pose, dict(value="MISSING", ambiguous=False))
def base_population(data, poses):
    """arm-independent rule: a pose is dropped from ALL comparisons when c0 is not a resolved, valid observation"""
    keep, dropped = [], {}
    for p in poses:
        v = _ep(data, "c0", p)["value"]
        if v in ("TRUE", "FALSE"): keep.append(p)
        else: dropped[p] = v
    return keep, dropped
def _val(v, mode, fill):
    if v in ("TRUE", "FALSE"): return 1.0 if v == "TRUE" else 0.0
    return fill if mode in ("worst_case", "best_case") else None                                       # INVALID / CENSORED / MISSING / NOT_APPLICABLE
def paired_values(data, arm, poses, mode="complete_case"):
    """-> (poses used, x_arm, x_c0, attrition dict).  modes: complete_case | worst_case | best_case | all_arms_complete | exclude_ambiguous | adverse_ambiguous | favourable_ambiguous"""
    keep, dropped = base_population(data, poses); used, xa, xc = [], [], []; att = {"c0_not_resolved": len(dropped)}; att_arm = {}
    for p in keep:
        e = _ep(data, arm, p); c = _ep(data, "c0", p); va = e["value"]
        if mode == "all_arms_complete" and any(_ep(data, a, p)["value"] not in ("TRUE", "FALSE") for a in ARMS): att_arm["not_complete_in_all_arms"] = att_arm.get("not_complete_in_all_arms", 0) + 1; continue
        if mode == "exclude_ambiguous" and (e.get("ambiguous") or c.get("ambiguous")): att_arm["ambiguous_excluded"] = att_arm.get("ambiguous_excluded", 0) + 1; continue
        fill = 0.0 if mode == "worst_case" else 1.0
        x = _val(va, mode, fill)
        if x is None: att_arm[f"{va.lower()}_excluded"] = att_arm.get(f"{va.lower()}_excluded", 0) + 1; continue
        xcv = 1.0 if c["value"] == "TRUE" else 0.0
        if mode in ("adverse_ambiguous", "favourable_ambiguous"):
            if e.get("ambiguous") and va in ("TRUE", "FALSE"): x = 0.0 if mode == "adverse_ambiguous" else 1.0                  # flagged candidate branch resolved against / in favour of X
            if c.get("ambiguous"): xcv = 1.0 if mode == "adverse_ambiguous" else 0.0                                            # and the opposite way for the reference
        used.append(p); xa.append(x); xc.append(xcv)
    att["arm_specific"] = att_arm; return used, np.array(xa), np.array(xc), att
def bootstrap_rd(d, level, B=B_DEFAULT, seed=0):
    """percentile interval of the mean paired difference, resampling poses (rows of d are poses; all arms of a pose stay together by construction)"""
    n = len(d)
    if n == 0: return [float("nan")] * 3
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, (B, n)); m = d[idx].mean(1); a = (1 - level) / 2; return [float(d.mean()), float(np.percentile(m, 100 * a)), float(np.percentile(m, 100 * (1 - a)))]
def compare(data, arm, poses, level, mode="complete_case", B=B_DEFAULT, seed=0):
    used, xa, xc, att = paired_values(data, arm, poses, mode); d = xa - xc; rd, lo, hi = bootstrap_rd(d, level, B, seed)
    return dict(arm=arm, mode=mode, n_poses=len(used), rd=rd, ci=[lo, hi], level=level, improved=int((d > 0).sum()), harmed=int((d < 0).sum()), unchanged=int((d == 0).sum()), attrition=att, family="oracle" if arm in ORACLE else "deployable", verdict=("SUPPORTED_BETTER" if lo > 0 else "SUPPORTED_WORSE" if hi < 0 else "INCONCLUSIVE") if len(used) else "NOT_EVALUABLE")
def equivalence_tost(data, poses, arm="S0", margin=MARGIN, mode="complete_case", B=B_DEFAULT, seed=0):
    """TOST (alpha 0.05 per side) as a CI test: equivalent iff the 90 % pose-cluster percentile interval of RD(arm - c0) lies strictly inside (-margin, +margin).  The 95 % interval is reported but is not the test."""
    c90 = compare(data, arm, poses, LEVELS["equivalence"], mode, B, seed); c95 = compare(data, arm, poses, 0.95, mode, B, seed); lo, hi = c90["ci"]
    return dict(arm=arm, margin=margin, n_poses=c90["n_poses"], rd=c90["rd"], ci90=c90["ci"], ci95=c95["ci"], equivalent=bool(c90["n_poses"] > 0 and lo > -margin and hi < margin), harmed=c90["harmed"], improved=c90["improved"], unchanged=c90["unchanged"])
def full_report(data, poses, B=B_DEFAULT, seed=0):
    """all pre-specified analyses; oracle and deployable families in separate tables; nothing is dropped silently"""
    rep = dict(base_population=dict(zip(("kept", "dropped"), (lambda k, d: (len(k), d))(*base_population(data, poses)))), primary={}, secondary={}, sensitivity={}, equivalence=equivalence_tost(data, poses, B=B, seed=seed))
    for arm, fam in (("O1", "oracle"), ("D1", "deployable")): rep["primary"][arm] = compare(data, arm, poses, LEVELS["primary"], B=B, seed=seed)
    for arm in ("O3", "D2", "H1"): rep["secondary"][arm] = compare(data, arm, poses, LEVELS["secondary"], B=B, seed=seed)
    for arm in ("O1", "D1"): rep["sensitivity"][arm] = {m: compare(data, arm, poses, LEVELS["primary"], mode=m, B=B, seed=seed) for m in ("worst_case", "best_case", "all_arms_complete", "exclude_ambiguous", "adverse_ambiguous", "favourable_ambiguous")}
    for arm in ("O1", "D1"):
        s = rep["sensitivity"][arm]; rep["primary"][arm]["claim_supported"] = bool(rep["primary"][arm]["verdict"] == "SUPPORTED_BETTER" and s["worst_case"]["rd"] >= 0 and rep["equivalence"]["equivalent"])           # r1: also needs worst-case point estimate >= 0 and S0 equivalence
    return rep
def equivalence_power(N, d_frac, margin=MARGIN, true_diff=0.0, sims=2000, B=1000, seed=0):
    """Monte-Carlo power of the TOST CI rule for N poses with discordant fraction d_frac (symmetric split unless true_diff != 0)"""
    rng = np.random.default_rng(seed); ok = 0; p1 = (d_frac + true_diff) / 2; p2 = (d_frac - true_diff) / 2
    for _ in range(sims):
        x = rng.choice([1.0, -1.0, 0.0], size=N, p=[p1, p2, 1 - d_frac]); idx = rng.integers(0, N, (B, N)); m = x[idx].mean(1); ok += (np.percentile(m, 5) > -margin) and (np.percentile(m, 95) < margin)
    return ok / sims
