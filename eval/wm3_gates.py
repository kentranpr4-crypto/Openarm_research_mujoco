#!/usr/bin/env python3
"""WM-3 Gate v2 (approved for PROSPECTIVE use only): frozen definitions, metric functions and verdict logic.  Verdicts: PASS / FAIL / NOT_EVALUABLE (gate = FAIL if any evaluable condition fails; else NOT_EVALUABLE
if any required condition is not evaluable; else PASS).  v1 verdicts of WM-2 are NOT altered by this file.
EVALUATION STATUS: every run on the splits train/val/test_* of the frozen wm2 dataset is a DEVELOPMENT evaluation (those test splits were already analysed in WM-2/WM-2.1; re-using them is NOT an independent
validation).  An INDEPENDENT held-out evaluation requires data that were never analysed (see `independent_heldout_plan` in CONFIG) and has NOT been run.
Aggregation: windows -> per initial-state GROUP mean -> macro mean over groups; percentile bootstrap over GROUPS (10 000, seed 0, 95 %).  Changes vs v1 are listed under `changes_vs_v1`.
Freeze once (before training WM3-B):  python eval/wm3_gates.py --freeze   -> outputs/wm3/gates_v2_config.json (+ sha256)."""
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm2_gates as G1
from wm2_gates import errors, macro, group_means, boot, auroc, ece_brier, reliability, shuffle_partner, spearman, verdict, sub, per_group, rel_improve, cum, active_mask, at_risk_mask, K, TRAY
CFG_PATH = ROOT / "outputs/wm3/gates_v2_config.json"
CS = json.loads((ROOT / "outputs/wm3/contact_scales.json").read_text())
CONFIG = dict(
    version="WM-3 gates v2 (prospective)", status="DEVELOPMENT evaluation on re-used wm2 test splits; independent held-out evaluation NOT run", model_gated="wm3b", models_compared_not_gated=["wm2_c0", "hybrid_posthoc"],
    aggregation=dict(unit="initial-state group", within_group="mean over windows", across_groups="macro mean", bootstrap=dict(resamples=10000, seed=0, level=0.95, unit="groups"), note="single training seed: bootstrap reflects data variance only, not training variance => comparisons between configurations are EXPLORATORY"),
    windows=G1.CONFIG["windows"], metrics=dict(box_mm="Euclidean error of cumulative box position change at horizon k [mm]", joint_rad="mean over 7 left-arm joints of |cumulative joint change error| [rad]", aoffset_cm="|error| of box-in-TCP a-offset [cm]",
                                              contact_std="sqrt(mean over channels [finger_force, n_contacts, palm_force, table_force, tray_force] of ((pred-true)/train_std)^2) of the K-th step contact block (absolute values)"),
    contact_scales=dict(channels=CS["channels"], std=CS["std"], used=CS["used_for_gate"], source=CS["source"]),
    baseline_pool=["persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind"], pool_rule="per (split, metric) the pool member with lowest macro error (point estimate, fixed before bootstrapping); gated model, wm2_c0 and hybrid_posthoc are NOT in the pool",
    floors=dict(box_mm=1.0, aoffset_cm=0.1, joint_rad=0.002, contact_std=0.02), min_sample=dict(groups_error_gates=10, groups_ood_sets=8, active_windows=200, g3_pos_groups=10, g3_neg_groups=10, g3_pos_windows=30, g3_neg_windows=30, paired_poses=10),
    G0=dict(rule="PASS iff all: frozen v2 config hash matches; frozen wm2 dataset/window hashes match; 0 groups in >1 split; training loaded only train/val; originals untouched; negative controls behave (persistence FAILS G1; action-shuffled model degrades; label-shuffled AUROC in [0.4,0.6]); the evaluation is labelled DEVELOPMENT"),
    G1=dict(split="test_id", horizon=20, subset="active", joint=dict(rule="E_joint <= max(1.25 x E_ridge_baseline, 0.003 rad) AND E_joint <= 0.07 rad", ridge_factor=1.25, abs_floor_rad=0.003, abs_max_rad=0.07),
            objects=dict(metrics=["box_mm", "aoffset_cm", "contact_std"], rel_improvement_min=0.20, ci_lower_gt=0.0, box_abs_max_mm=30.0, rule="superiority over the best pool baseline: r = 1 - E_model/E_best >= 0.20 with bootstrap lower bound > 0 for box, a-offset and contact; box <= 30 mm"),
            rule="PASS iff the joint rule and all object rules hold; NOT_EVALUABLE if groups < 10 or active windows < 200"),
    G2=dict(sets=["test_low_heldout", "test_ood_low_strict", "test_ood_high"], id_split="test_id", metrics=["box_mm", "aoffset_cm"], horizon=20, subset="active", ratio_max=2.0,
            R_definition="R = [E_model(OOD)/max(E_best(OOD), floor)] / [E_model(ID)/max(E_best(ID), floor)] with E_best the best pool baseline of that split/metric (baseline-relative OOD degradation)", noninferior="E_model(OOD) <= E_best(OOD) (margin 0); if E_best < floor: E_model - E_best <= floor", reported_also=["absolute E_model per split", "raw rho = E_model(OOD)/max(E_model(ID), floor)", "E_model/E_best per split"],
            rule="PASS iff on every OOD set and metric: non-inferior AND R <= 2.0; FAIL if an evaluable condition fails; NOT_EVALUABLE if a set has < 8 groups or < 200 active windows"),
    G3=dict(subset="at-risk windows per label (input-only), definitions as v1", at_risk=G1.CONFIG["G3"]["at_risk"], primary=[dict(label="close", sets=["test_id"]), dict(label="palm", sets=["test_low_heldout", "test_ood_low_strict"], pooled=True)], secondary=["lifted", "placed"],
            auroc_min=0.80, auroc_ci_lower_min=0.70, margin_over_outcome_only=0.05, sufficiency=G1.CONFIG["G3"]["sufficiency"], secondary_reported=["AUROC", "Brier", "Brier skill vs outcome_only = 1 - Brier/Brier_outcome_only", "prevalence"],
            rule="PASS iff both primary targets are evaluable and meet AUROC >= 0.80, CI lower bound >= 0.70, margin over outcome_only >= 0.05; palm is evaluated on the POOLED windows of test_low_heldout and test_ood_low_strict; lifted/placed are reported, not gated"),
    G4=dict(window_rule="phase-aligned windows on held-out interventional poses of the frozen wm2 test splits (31 poses), control included: depth t=close-10 | speed t=(close+release)//2-10 | timing t=close-25 | release_height t=release-10 | noise t=close+10",
            families=dict(depth=dict(quantity="a-offset at t+K [cm]", tau=0.5), speed=dict(quantity="box xy displacement over the horizon [mm]", tau=2.0), timing=dict(quantity="P(close command inside horizon)", tau=0.5), release_height=dict(quantity="box xy displacement over the horizon [mm]", tau=2.0), noise=dict(quantity="box z at t+K [mm]", tau=5.0)),
            eligibility="a family is gated only if >= 50 % of its non-control variants have |TRUE effect vs control| >= tau on the evaluated poses (a data property, independent of any model); otherwise that family is NOT_EVALUABLE", family_pass=dict(sign_agreement_min=0.80, spearman_min=0.70, min_poses=10),
            depth_extra=dict(a_offset_monotone_spearman_max=-0.80, palm_monotone_spearman_min=0.80, pose_fraction_min=0.80),
            blind=dict(split="test_id", metric="box_mm", rel_improvement_min=0.20, baselines=["ridge_blind", "mlp_blind"]), shuffle=dict(degradation_min=0.30, rule="A_hist and A_fut replaced by those of a random window of the SAME split from a DIFFERENT group (seed 0)"),
            rule="PASS iff every eligible family passes (sign agreement and, when the true effects vary, Spearman; an undefined Spearman because the true effect is constant is not applicable), ineligible families are recorded and excluded from the verdict, the depth monotonicity extras hold, the model beats the best action-blind baseline by >= 20 % (CI lower bound > 0) and the action-shuffle raises the box error by >= 30 % (CI lower bound > 0); NOT_EVALUABLE if no family is eligible"),
    G5=dict(bins=15, ece_max=0.05, brier_rule="Brier(model) < min(Brier(prevalence-only), Brier(outcome_only))", applies_to="the evaluable primary G3 targets", reported=["ECE", "Brier", "prevalence", "reliability plot"], rule="PASS iff every evaluable primary target satisfies both; NOT_EVALUABLE if none"),
    G6=G1.CONFIG["G6"],
    changes_vs_v1=["G1 joint: non-inferiority to ridge instead of superiority; G1 adds contact superiority", "G2: baseline-relative OOD degradation R <= 2.0 replaces raw rho <= 1.5 (absolute errors and raw rho reported)", "G3: close/palm primary (palm pooled over the two LOW OOD sets), lifted/placed secondary and not gated",
                   "G4: per action family, family-specific quantity, phase-aligned window, eligibility by true effect size; pooled Spearman removed", "G0: adds development/independent labelling and negative controls", "G5, G6, aggregation, windows, floors, sufficiency rules unchanged"],
    independent_heldout_plan=dict(requirement="data never analysed in WM-2/WM-2.1/WM-3 development: NEW initial poses (seed 61, same region rule, no overlap with the 310 existing groups) with the 25 registered variants, generated and evaluated once by the frozen evaluator; NOT generated, NOT run, needs separate approval", status="NOT RUN"))
def config_bytes(): return json.dumps(CONFIG, indent=1, sort_keys=True).encode()
def config_sha(): return hashlib.sha256(config_bytes()).hexdigest()
def load_config(expect=None):
    if not CFG_PATH.exists(): raise SystemExit("gates v2 config not frozen")
    b = CFG_PATH.read_bytes(); h = hashlib.sha256(b).hexdigest()
    if expect and h != expect: raise SystemExit(f"gates v2 config changed: {h} != {expect}")
    if h != config_sha(): raise SystemExit("gates_v2_config.json differs from CONFIG in wm3_gates.py")
    return json.loads(b)
# ---------------------------------------------------------------------------------------------- metrics
def contact_err(P, T, k=K, cfg=CONFIG):
    c = cfg["contact_scales"]; u = c["used"]; s = np.array(c["std"])[u]; d = (P["Y_c"][:, k - 1][:, u] - T["Y_c"][:, k - 1][:, u]) / s; return np.sqrt((d ** 2).mean(1))
def all_errors(P, T, k=K): e = errors(P, T, k); e["contact_std"] = contact_err(P, T, k); return e
def per_group_all(P, T, groups, mask, metric, k=K): e = all_errors(sub(P, mask), sub(T, mask), k)[metric]; return group_means(e, groups[mask])
def gate_G1(cfg, preds, T, groups, active, model="wm3b"):
    c = cfg["G1"]; sp = c["split"]; gv, act = groups[sp], active[sp]; ng = len(np.unique(gv[act])); ev = ng >= cfg["min_sample"]["groups_error_gates"] and act.sum() >= cfg["min_sample"]["active_windows"]; conds = []; detail = {}
    pool_vals = lambda met: {b: per_group_all(preds[b][sp], T[sp], gv, act, met)[1] for b in cfg["baseline_pool"] if b in preds}
    em = per_group_all(preds[model][sp], T[sp], gv, act, "joint_rad")[1]; pj = pool_vals("joint_rad"); eridge = pj["ridge"].mean(); thr = max(c["joint"]["ridge_factor"] * eridge, c["joint"]["abs_floor_rad"]); detail["joint_rad"] = dict(model=float(em.mean()), ridge=float(eridge), threshold=float(thr))
    conds.append(dict(name="joint_noninferior_to_ridge", value=float(em.mean()), threshold=float(thr), passed=bool(em.mean() <= thr and em.mean() <= c["joint"]["abs_max_rad"]), evaluable=ev))
    for met in c["objects"]["metrics"]:
        em = per_group_all(preds[model][sp], T[sp], gv, act, met)[1]; pool = pool_vals(met); best = min(pool, key=lambda b: pool[b].mean()); r, lo, hi = boot(rel_improve(em, pool[best], cfg["floors"][met]), dict(model=em, best=pool[best]))
        detail[met] = dict(model=float(em.mean()), best_baseline=best, best_value=float(pool[best].mean()), rel_improvement=r, ci=[lo, hi]); conds.append(dict(name=f"{met}_superior_by_20pct", value=r, ci=[lo, hi], passed=bool(r >= c["objects"]["rel_improvement_min"] and lo > c["objects"]["ci_lower_gt"]), evaluable=ev))
    conds.append(dict(name="box_abs<=30mm", value=detail["box_mm"]["model"], passed=bool(detail["box_mm"]["model"] <= c["objects"]["box_abs_max_mm"]), evaluable=ev))
    return dict(verdict=verdict(conds), conditions=conds, detail=detail, groups=ng)
def gate_G2(cfg, preds, T, groups, active, model="wm3b"):
    c = cfg["G2"]; conds = []; detail = {}; idg, ida = groups[c["id_split"]], active[c["id_split"]]
    for sp in c["sets"]:
        gv, act = groups[sp], active[sp]; ng = len(np.unique(gv[act])); ev = ng >= cfg["min_sample"]["groups_ood_sets"] and act.sum() >= cfg["min_sample"]["active_windows"]; detail[sp] = {}
        for met in c["metrics"]:
            fl = cfg["floors"][met]; em = per_group_all(preds[model][sp], T[sp], gv, act, met)[1].mean(); eid = per_group_all(preds[model][c["id_split"]], T[c["id_split"]], idg, ida, met)[1].mean()
            pool = {b: per_group_all(preds[b][sp], T[sp], gv, act, met)[1].mean() for b in cfg["baseline_pool"] if b in preds}; best = min(pool, key=pool.get); eb = pool[best]
            poolid = {b: per_group_all(preds[b][c["id_split"]], T[c["id_split"]], idg, ida, met)[1].mean() for b in cfg["baseline_pool"] if b in preds}; ebid = min(poolid.values())
            R = float((em / max(eb, fl)) / (eid / max(ebid, fl))); noninf = bool(em - eb <= fl) if eb < fl else bool(em <= eb)
            detail[sp][met] = dict(model=float(em), best_baseline=best, best_value=float(eb), model_over_best=float(em / max(eb, fl)), id_model=float(eid), id_best=float(ebid), id_model_over_best=float(eid / max(ebid, fl)), R=R, raw_rho=float(em / max(eid, fl)), groups=ng)
            conds += [dict(name=f"{sp}/{met}/noninferior", passed=noninf, value=float(em - eb), evaluable=ev), dict(name=f"{sp}/{met}/R<=2.0", passed=bool(R <= c["ratio_max"]), value=R, evaluable=ev)]
    return dict(verdict=verdict(conds), conditions=conds, detail=detail)
def _suff(cfg, y, g):
    m = cfg["min_sample"]; pg, ng = len(np.unique(g[y == 1])), len(np.unique(g[y == 0])); ok = pg >= m["g3_pos_groups"] and ng >= m["g3_neg_groups"] and (y == 1).sum() >= m["g3_pos_windows"] and (y == 0).sum() >= m["g3_neg_windows"]
    return bool(ok), dict(pos_groups=int(pg), neg_groups=int(ng), pos_windows=int((y == 1).sum()), neg_windows=int((y == 0).sum()))
LAB = ["close", "palm", "lifted", "placed"]
def _gather(preds_m, preds_o, T, groups, masks, label, sets):
    j = LAB.index(label); ys, ps, os_, gs = [], [], [], []
    for sp in sets: mk = masks[(label, sp)]; ys.append(T[sp]["Y_e"][mk][:, j].astype(int)); ps.append(preds_m[sp]["Y_e"][mk][:, j]); os_.append(preds_o[sp]["Y_e"][mk][:, j]); gs.append(groups[sp][mk])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(os_), np.concatenate(gs)
def gate_G3(cfg, preds, T, groups, masks, model="wm3b"):
    c = cfg["G3"]; conds = []; detail = {}
    for tgt in c["primary"]:
        y, p, po, g = _gather(preds[model], preds["outcome_only"], T, groups, masks, tgt["label"], tgt["sets"]); ok, suff = _suff(cfg, y, g); key = f"{tgt['label']}/{'+'.join(tgt['sets'])}"; d = dict(sufficiency=suff, evaluable=ok)
        if ok:
            a, ao = auroc(y, p), auroc(y, po); ug = np.unique(g); rng = np.random.default_rng(0); members = {x: np.nonzero(g == x)[0] for x in ug}; vs = []
            for _ in range(2000):
                ii = np.concatenate([members[x] for x in rng.choice(ug, len(ug))]); v = auroc(y[ii], p[ii])
                if not np.isnan(v): vs.append(v)
            lo = float(np.percentile(vs, 2.5)); d.update(auroc=a, ci_lo=lo, auroc_outcome_only=ao, margin=a - ao); conds.append(dict(name=key, value=a, ci_lo=lo, margin=a - ao, passed=bool(a >= c["auroc_min"] and lo >= c["auroc_ci_lower_min"] and a - ao >= c["margin_over_outcome_only"]), evaluable=True))
        else: conds.append(dict(name=key, passed=False, evaluable=False))
        detail[key] = d
    sec = {}
    for lab in c["secondary"]:
        sets = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; y, p, po, g = _gather(preds[model], preds["outcome_only"], T, groups, masks, lab, sets); e, b, pr = ece_brier(y, p); _, bo, _ = ece_brier(y, po); sec[lab] = dict(auroc=auroc(y, p), auroc_outcome_only=auroc(y, po), brier=b, brier_outcome_only=bo, brier_skill_vs_outcome_only=1 - b / bo, prevalence=pr)
    return dict(verdict=verdict(conds), conditions=conds, detail=detail, secondary=sec)
def gate_G5(cfg, preds, T, groups, masks, g3, model="wm3b"):
    c = cfg["G5"]; conds = []; detail = {}
    for tgt in cfg["G3"]["primary"]:
        key = f"{tgt['label']}/{'+'.join(tgt['sets'])}"
        if not g3["detail"][key]["evaluable"]: continue
        y, p, po, g = _gather(preds[model], preds["outcome_only"], T, groups, masks, tgt["label"], tgt["sets"]); e, b, prev = ece_brier(y, p, c["bins"]); _, bo, _ = ece_brier(y, po, c["bins"]); bp = float(np.mean((prev - y) ** 2))
        detail[key] = dict(ECE=e, Brier=b, Brier_outcome_only=bo, Brier_prevalence_only=bp, prevalence=prev, reliability=reliability(y, p, c["bins"])); conds.append(dict(name=key, ECE=e, Brier=b, passed=bool(e <= c["ece_max"] and b < min(bo, bp)), evaluable=True))
    return dict(verdict=verdict(conds) if conds else "NOT_EVALUABLE", conditions=conds, detail=detail)
def family_quantity(name, W, is_pred=False, Wref=None):
    """Quantity per family; W is a window dict (truth) or a prediction dict (needs the inputs from Wref for the current box row)."""
    inp = Wref if Wref is not None else W; Yb = W["Y_b"]
    if name == "depth": return (inp["B_hist"][:, -1, 6] + cum(Yb)[:, 19, 6]) * 100.0
    if name in ("speed", "release_height"): return np.linalg.norm(cum(Yb)[:, 19, :2], axis=1) * 1000.0
    if name == "timing": return W["Y_e"][:, 0].astype(float)
    if name == "noise": return (inp["B_hist"][:, -1, 2] + cum(Yb)[:, 19, 2]) * 1000.0
def gate_G4(cfg, preds, T, groups, active, fam_windows, fam_preds, model="wm3b", shuffled=None):
    c = cfg["G4"]; conds = []; detail = {}; sp = c["blind"]["split"]; gv, act = groups[sp], active[sp]; ev0 = len(np.unique(gv[act])) >= cfg["min_sample"]["groups_error_gates"]
    em = per_group_all(preds[model][sp], T[sp], gv, act, c["blind"]["metric"])[1]; blind = {b: per_group_all(preds[b][sp], T[sp], gv, act, c["blind"]["metric"])[1] for b in c["blind"]["baselines"]}; bb = min(blind, key=lambda b: blind[b].mean())
    r, lo, hi = boot(rel_improve(em, blind[bb], cfg["floors"]["box_mm"]), dict(model=em, best=blind[bb])); conds.append(dict(name="beats_best_action_blind_by_20pct", value=r, ci=[lo, hi], passed=bool(r >= c["blind"]["rel_improvement_min"] and lo > 0), evaluable=ev0))
    es = per_group_all(shuffled[sp], T[sp], gv, act, c["blind"]["metric"])[1]; d, dlo, dhi = boot(lambda M_: float(M_["shuf"].mean() / M_["model"].mean() - 1), dict(model=em, shuf=es)); conds.append(dict(name="action_shuffle_degrades_box_error_by_30pct", value=d, ci=[dlo, dhi], passed=bool(d >= c["shuffle"]["degradation_min"] and dlo > 0), evaluable=ev0))
    elig_any = False
    for name, spec in c["families"].items():
        W = fam_windows[name]; pid, vid, lev = W["pid"], W["vid"], W["level"]; Qt = family_quantity(name, W); Qp = family_quantity(name, fam_preds[model][name], Wref=W); poses = sorted(set(pid.tolist())); ctl = {p: i for p, i in zip(pid[vid == 0], np.nonzero(vid == 0)[0])}
        rows = [i for i in range(len(pid)) if vid[i] != 0 and pid[i] in ctl]; dT = np.array([Qt[i] - Qt[ctl[pid[i]]] for i in rows]); dP = np.array([Qp[i] - Qp[ctl[pid[i]]] for i in rows]); big = np.abs(dT) >= spec["tau"]; frac = float(big.mean()) if len(big) else 0.0
        eligible = bool(frac >= 0.5 and len(poses) >= c["family_pass"]["min_poses"]); elig_any |= eligible; sa = float(np.mean(np.sign(dP[big]) == np.sign(dT[big]))) if big.any() else float("nan"); sr = spearman(dP, dT)
        detail[name] = dict(quantity=spec["quantity"], tau=spec["tau"], poses=len(poses), true_abs_effect_mean=float(np.abs(dT).mean()) if len(dT) else None, frac_true_effect_ge_tau=frac, eligible=eligible, sign_agreement=sa, spearman=sr)
        if not eligible: continue                                  # ineligible family: recorded in detail, excluded from the verdict
        conds += [dict(name=f"{name}/sign_agreement>=0.80", value=sa, passed=bool(sa >= c["family_pass"]["sign_agreement_min"]), evaluable=True), dict(name=f"{name}/spearman>=0.70", value=sr, passed=bool(sr >= c["family_pass"]["spearman_min"]), evaluable=True)] if not np.isnan(sr) else [dict(name=f"{name}/sign_agreement>=0.80", value=sa, passed=bool(sa >= c["family_pass"]["sign_agreement_min"]), evaluable=True)]
        if name == "depth":
            mono, pm = [], []
            for p in poses:
                m = (pid == p) & ((np.array([True] * len(pid))))
                if m.sum() >= 4:
                    mono.append(spearman(lev[m], Qp[m])); yp = W["Y_e"][m][:, 1]
                    if yp.min() != yp.max(): pm.append(spearman(lev[m], fam_preds[model][name]["Y_e"][m][:, 1]))
            fm = float(np.mean(np.array(mono) <= c["depth_extra"]["a_offset_monotone_spearman_max"])) if mono else float("nan"); fp = float(np.mean(np.array(pm) >= c["depth_extra"]["palm_monotone_spearman_min"])) if pm else float("nan")
            detail[name].update(a_offset_monotone_fraction=fm, palm_monotone_fraction=fp, n_pose_palm_varies=len(pm))
            conds += [dict(name="depth/a_offset_monotone_pose_fraction>=0.80", value=fm, passed=bool(fm >= c["depth_extra"]["pose_fraction_min"]), evaluable=bool(mono)), dict(name="depth/palm_monotone_pose_fraction>=0.80", value=fp, passed=bool(fp >= c["depth_extra"]["pose_fraction_min"]), evaluable=len(pm) >= 3)]
    if not elig_any: return dict(verdict="NOT_EVALUABLE", conditions=conds, detail=detail)
    return dict(verdict=verdict(conds), conditions=conds, detail=detail)
def gate_G0(cfg, info):
    items = ["v2_config_hash_matches", "wm2_dataset_and_window_hashes_match", "groups_in_multiple_splits_zero", "training_loaded_only_train_val", "originals_untouched", "negative_control_persistence_fails_G1", "negative_control_action_shuffle_degrades", "negative_control_label_shuffle_auroc_in_0p4_0p6", "labelled_development_evaluation"]
    conds = [dict(name=k, passed=bool(info.get(k)), evaluable=True) for k in items]; return dict(verdict=verdict(conds), conditions=conds)
def gate_G6(cfg, bench): return G1.gate_G6(dict(G6=cfg["G6"]), bench)
if __name__ == "__main__":
    if "--freeze" in sys.argv:
        if CFG_PATH.exists(): sys.exit(f"{CFG_PATH} exists (frozen)")
        CFG_PATH.write_bytes(config_bytes()); print("gates v2 config written, sha256", config_sha())
    else: print("sha256 of current CONFIG:", config_sha())
