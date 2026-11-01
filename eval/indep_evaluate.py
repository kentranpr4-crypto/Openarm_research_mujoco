#!/usr/bin/env python3
"""INDEPENDENT held-out evaluation: INDEPENDENT-PROCESS EVALUATOR.  Imports no model/training code (only the frozen gate module eval/wm3_gates.py).  Reads the frozen configs, the independent windows (labels), the frozen family windows and the
prediction files written by eval/indep_predict.py.  Unit of statistical analysis = initial-state GROUP (a window is never a sample; per-group mean -> macro mean; percentile bootstrap over groups, 10 000 resamples, seed 0).
Modes: default = independent set (outputs/indep) ; --dry-run = plumbing check on the DEVELOPMENT windows (outputs/indep_dry), whose results must NOT be reported as independent.   Run (ACT venv): python eval/indep_evaluate.py [--dry-run]"""
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); DRY = "--dry-run" in sys.argv
import wm3_gates as V
BASE = ROOT / ("outputs/indep_dry" if DRY else "outputs/indep"); PRED = BASE / "preds"; WIN = ROOT / "outputs/wm2/windows" if DRY else BASE / "windows"; FAMF = ROOT / ("outputs/wm3/family_windows.npz" if DRY else "outputs/indep/family_windows.npz")
EVAL = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; FAMS = ["depth", "speed", "timing", "release_height", "noise"]; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
DEV = json.loads((ROOT / "outputs/wm3/freeze.json").read_text()); cfg = V.load_config(DEV["gates_v2_config_sha256"]); NB = 10000
STATUS = "DEVELOPMENT dry-run (plumbing check on already-analysed wm2 test windows; NOT independent)" if DRY else "INDEPENDENT held-out evaluation (50 new initial-state groups; never used for training, validation, model selection, gate design or normalisation)"
# ---------------------------------------------------------------------------- integrity
g0 = {}; extra = {}
if not DRY:
    F = json.loads((BASE / "protocol_freeze.json").read_text()); A = json.loads((BASE / "dataset_audit.json").read_text()); PR = json.loads((BASE / "predict_report.json").read_text())
    ok_files = {k: sha(ROOT / k) == v for k, v in F["files_sha256"].items()}; extra.update(frozen_files_unchanged=all(ok_files.values()), frozen_files_changed=[k for k, v in ok_files.items() if not v], model_checkpoints_match=PR["model_checkpoint_sha256"] == F["model_checkpoint_sha256"], closed_form_baselines_match=PR["closed_form_hashes"] == F["closed_form_baselines_sha256"],
                 independent_window_hashes_match=all(sha(WIN / f"{sp}.npz") == A["windows_sha256"][sp] for sp in EVAL) and sha(FAMF) == A["family_windows_sha256"], dataset_index_match=sha(BASE / "dataset/index.json") == A["index_sha256"], groups_overlapping_development=A["groups_overlapping_development"],
                 dataset_audit_clean=(A["n_episodes"] == A["expected"] and A["rate_limit_violations"] == 0 and A["ctrlrange_violations"] == 0 and not A["forbidden_seeds_used"] and A["poses_with_all_25_variants"] == A["groups"]), manifest_sha_matches_freeze=sha(BASE / "manifest.json") == F["files_sha256"]["outputs/indep/manifest.json"])
    g0["v2_config_hash_matches"] = sha(ROOT / "outputs/wm3/gates_v2_config.json") == DEV["gates_v2_config_sha256"] and ok_files["outputs/wm3/gates_v2_config.json"]
    g0["wm2_dataset_and_window_hashes_match"] = sha(ROOT / "outputs/wm2/dataset/index.json") == DEV["wm2_dataset_index_sha256"] and all(sha(ROOT / f"outputs/wm2/windows/{sp}.npz") == h for sp, h in DEV["wm2_windows_sha256"].items()) and extra["independent_window_hashes_match"]
    g0["training_loaded_only_train_val"] = set(PR["splits_loaded_for_baseline_refit"]) <= {"train", "val"}
else:
    PR = json.loads((BASE / "predict_report.json").read_text()); g0["v2_config_hash_matches"] = sha(ROOT / "outputs/wm3/gates_v2_config.json") == DEV["gates_v2_config_sha256"]; g0["wm2_dataset_and_window_hashes_match"] = all(sha(WIN / f"{sp}.npz") == DEV["wm2_windows_sha256"][sp] for sp in EVAL); g0["training_loaded_only_train_val"] = set(PR["splits_loaded_for_baseline_refit"]) <= {"train", "val"}
g0["originals_untouched"] = sha(ROOT / "outputs/wm1/dataset/index.json") == json.loads((ROOT / "outputs/wm2/freeze.json").read_text())["wm1_index_sha256"]
def lp(n): z = np.load(PRED / f"{n}.npz"); return {k: z[k] for k in z.files}
Wd = {sp: {k: np.load(WIN / f"{sp}.npz")[k] for k in np.load(WIN / f"{sp}.npz").files} for sp in EVAL}; T = {sp: {k: Wd[sp][k] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e")} for sp in EVAL}
groups = {sp: Wd[sp]["group"] for sp in EVAL}; active = {sp: V.active_mask(Wd[sp]) for sp in EVAL}; masks = {(lab, sp): V.at_risk_mask(Wd[sp], lab) for lab in V.LAB for sp in EVAL}
g0["groups_in_multiple_splits_zero"] = all(not (set(groups[a].tolist()) & set(groups[b].tolist())) for a in EVAL for b in EVAL if a < b) and (DRY or extra["groups_overlapping_development"] == 0)
MODELS = ["persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind", "outcome_only", "wm3b", "wm2_c0", "hybrid_posthoc"]; MAIN = ["persistence", "constvel", "ridge", "ridge_blind", "mlp", "mlp_blind", "wm2_c0", "hybrid_posthoc", "wm3b"]
preds = {m: {sp: lp(f"{m}_{sp}") for sp in EVAL} for m in MODELS}; shuf = {"wm3b": {"test_id": lp("wm3b_shuffle_test_id")}, "wm2_c0": {"test_id": lp("wm2_c0_shuffle_test_id")}}
fw_raw = np.load(FAMF); fam_windows = {f: {k.split("/", 1)[1]: fw_raw[k] for k in fw_raw.files if k.startswith(f + "/")} for f in FAMS}; fam_preds = {m: {f: lp(f"{m}_family_{f}") for f in FAMS} for m in MODELS}
# ---------------------------------------------------------------------------- per-group error arrays and vectorised group bootstrap (identical resampling scheme to V.boot for mean statistics)
_c = {}
def yaw_err(P, Tt, k): return np.abs(V.cum(P["Y_b"])[:, k - 1, 3] - V.cum(Tt["Y_b"])[:, k - 1, 3])
def PG(m, sp, met, k, fam=None):
    key = (m, sp, met, k, fam)
    if key not in _c:
        a = active[sp] if fam is None else active[sp] & (Wd[sp]["family"] == fam); P, Tt = V.sub(preds[m][sp], a), V.sub(T[sp], a)
        e = yaw_err(P, Tt, k) if met == "yaw_rad" else V.all_errors(P, Tt, k)[met]; _c[key] = V.group_means(e, groups[sp][a])
    return _c[key]
def pooled(m, met, k, sps=EVAL, fam=None):
    parts = [PG(m, sp, met, k, fam) for sp in sps]; ids = np.concatenate([p[0] for p in parts]); vals = np.concatenate([p[1] for p in parts]); o = np.argsort(ids); return ids[o], vals[o]
def bci(x, level=0.95, n=NB, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (n, len(x))); v = x[idx].mean(1); a = (1 - level) / 2; return [float(x.mean()), float(np.percentile(v, 100 * a)), float(np.percentile(v, 100 * (1 - a)))]
def label(lo, hi): return "SUPPORTED_LOWER_ERROR" if hi < 0 else ("SUPPORTED_HIGHER_ERROR" if lo > 0 else "INCONCLUSIVE")
SCOPES = {"ALL_50_groups": EVAL, **{sp: [sp] for sp in EVAL}}; MET = ["box_mm", "yaw_rad", "joint_rad", "aoffset_cm", "contact_std"]; KS = [1, 5, 10, 20]
# ---------------------------------------------------------------------------- negative controls
neg = {}; pc = dict(preds); pc["wm3b"] = preds["persistence"]; neg["persistence_as_candidate_G1"] = V.gate_G1(cfg, pc, T, groups, active)["verdict"]; g0["negative_control_persistence_fails_G1"] = neg["persistence_as_candidate_G1"] == "FAIL"
e_c = V.per_group_all(preds["wm3b"]["test_id"], T["test_id"], groups["test_id"], active["test_id"], "box_mm")[1].mean(); e_s = V.per_group_all(shuf["wm3b"]["test_id"], T["test_id"], groups["test_id"], active["test_id"], "box_mm")[1].mean(); neg["action_shuffle_box_mm"] = [float(e_c), float(e_s)]; g0["negative_control_action_shuffle_degrades"] = bool(e_s > 1.3 * e_c)
mk = masks[("close", "test_id")]; y = T["test_id"]["Y_e"][mk][:, 0].astype(int); p = preds["wm3b"]["test_id"]["Y_e"][mk][:, 0]; rng = np.random.default_rng(0); neg["label_shuffle_auroc"] = float(np.mean([V.auroc(rng.permutation(y), p) for _ in range(20)])); g0["negative_control_label_shuffle_auroc_in_0p4_0p6"] = 0.4 <= neg["label_shuffle_auroc"] <= 0.6
g0["labelled_development_evaluation"] = STATUS.startswith("INDEPENDENT") or STATUS.startswith("DEVELOPMENT")      # pre-registered adaptation: the frozen G0 item 'labelled DEVELOPMENT' is satisfied by a correct evaluation label (INDEPENDENT here)
# ---------------------------------------------------------------------------- Gate v2 (frozen config, applied unchanged)
def run_gates(model):
    g1 = V.gate_G1(cfg, preds, T, groups, active, model); g2 = V.gate_G2(cfg, preds, T, groups, active, model); g3 = V.gate_G3(cfg, preds, T, groups, masks, model); g4 = V.gate_G4(cfg, preds, T, groups, active, fam_windows, fam_preds, model, shuffled=shuf[model]) if model in shuf else None
    return dict(G1=g1, G2=g2, G3=g3, G4=g4, G5=V.gate_G5(cfg, preds, T, groups, masks, g3, model))
GATES = {m: run_gates(m) for m in ("wm3b", "wm2_c0", "hybrid_posthoc")}; GATES["wm3b"]["G0"] = V.gate_G0(cfg, g0); GATES["wm3b"]["G6"] = V.gate_G6(cfg, dict(median_ms=PR["latency"]["wm3b"]["median_ms"], params=PR["latency"]["wm3b"]["params"]))
# ---------------------------------------------------------------------------- error tables with group-bootstrap CIs
tab = {m: {sc: {f"K={k}": {met: dict(zip(("mean", "lo", "hi"), bci(pooled(m, met, k, sps)[1]))) | dict(n_groups=int(len(pooled(m, met, k, sps)[1]))) for met in MET} for k in KS} for sc, sps in SCOPES.items()} for m in MAIN}
# ---------------------------------------------------------------------------- paired comparisons (pre-registered primary: P1 box WM3-B vs C0, P2 box WM3-B vs hybrid, P3 joint WM3-B vs C0, all K=20, ALL groups)
def pair(a, b, met, k, sps, level):
    _, va = pooled(a, met, k, sps); _, vb = pooled(b, met, k, sps); d = bci(va - vb, level); return dict(diff_mean=d[0], lo=d[1], hi=d[2], level=level, a_mean=float(va.mean()), b_mean=float(vb.mean()), rel_diff=float(d[0] / vb.mean()), verdict=label(d[1], d[2]), n_groups=int(len(va)))
PRIM = {"P1_box_K20_wm3b_vs_c0": ("wm3b", "wm2_c0", "box_mm"), "P2_box_K20_wm3b_vs_hybrid": ("wm3b", "hybrid_posthoc", "box_mm"), "P3_joint_K20_wm3b_vs_c0": ("wm3b", "wm2_c0", "joint_rad")}
paired = {"primary": {n: {"95": pair(a, b, met, 20, EVAL, 0.95), "bonferroni_98.33": pair(a, b, met, 20, EVAL, 1 - 0.05 / 3)} for n, (a, b, met) in PRIM.items()}, "secondary": {}}
for other in ("wm2_c0", "hybrid_posthoc", "mlp", "ridge"):
    for sc, sps in SCOPES.items():
        for k in KS:
            for met in MET: paired["secondary"][f"wm3b_vs_{other}/{sc}/K={k}/{met}"] = pair("wm3b", other, met, k, sps, 0.95)
# ---------------------------------------------------------------------------- per-family (variant family) K=20 box error on standard windows; NOT_EVALUABLE if < 10 groups
FAMIDX = {"control": 0, "depth": 1, "timing": 2, "noise": 3, "speed": 4, "release_height": 5}; perfam = {}
for fn, fi in FAMIDX.items():
    ng = len({int(g) for sp in EVAL for g in np.unique(groups[sp][active[sp] & (Wd[sp]["family"] == fi)])})
    if ng < 10: perfam[fn] = dict(verdict="NOT_EVALUABLE", groups=ng); continue
    perfam[fn] = dict(groups=ng, **{m: dict(zip(("mean", "lo", "hi"), bci(pooled(m, "box_mm", 20, EVAL, fam=fi)[1]))) for m in ("wm2_c0", "hybrid_posthoc", "mlp", "ridge", "wm3b")})
# ---------------------------------------------------------------------------- classification / calibration (at-risk windows, all four strata pooled; windows are only used to compute the statistic, CIs resample GROUPS)
cls = {}
for lab_i, lab in enumerate(V.LAB):
    for m in ("outcome_only", "wm2_c0", "hybrid_posthoc", "wm3b"):
        ys, ps, gs = [], [], []
        for sp in EVAL: mkk = masks[(lab, sp)]; ys.append(T[sp]["Y_e"][mkk][:, lab_i].astype(int)); ps.append(preds[m][sp]["Y_e"][mkk][:, lab_i]); gs.append(groups[sp][mkk])
        y, p, g = np.concatenate(ys), np.concatenate(ps), np.concatenate(gs); ok, suff = V._suff(cfg, y, g); d = dict(sufficiency=suff, evaluable=ok)
        if ok:
            e, b, pr = V.ece_brier(y, p, 15); ug = np.unique(g); rg = np.random.default_rng(0); mem = {x: np.nonzero(g == x)[0] for x in ug}; vs = []
            for _ in range(1000):
                ii = np.concatenate([mem[x] for x in rg.choice(ug, len(ug))]); v = V.auroc(y[ii], p[ii]); vs.append(v) if not np.isnan(v) else None
            d.update(auroc=V.auroc(y, p), auroc_ci=[float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))], ECE=e, Brier=b, prevalence=pr, Brier_prevalence_only=float(np.mean((pr - y) ** 2)), n_groups=int(len(ug)))
        cls[f"{lab}/{m}"] = d
# ---------------------------------------------------------------------------- paired action-effect (family windows): direction, rank agreement, magnitude (bootstrap over initial-state groups = poses)
def effects(model, name):
    W = fam_windows[name]; pid, vid = W["pid"], W["vid"]; Qt = V.family_quantity(name, W); Qp = V.family_quantity(name, fam_preds[model][name], Wref=W); ctl = {p: i for p, i in zip(pid[vid == 0], np.nonzero(vid == 0)[0])}
    rows = [i for i in range(len(pid)) if vid[i] != 0 and pid[i] in ctl]; return np.array([pid[i] for i in rows]), np.array([Qt[i] - Qt[ctl[pid[i]]] for i in rows]), np.array([Qp[i] - Qp[ctl[pid[i]]] for i in rows])
act = {}
for m in ("wm3b", "wm2_c0", "hybrid_posthoc"):
    act[m] = {}
    for name in FAMS:
        p_, dT, dP = effects(m, name); tau = cfg["G4"]["families"][name]["tau"]; big = np.abs(dT) >= tau; ps = np.unique(p_); rg = np.random.default_rng(0); sa, mr, sl = [], [], []
        def stat(sel):
            b = big[sel]; t_, q_ = dT[sel], dP[sel]; return (float(np.mean(np.sign(q_[b]) == np.sign(t_[b]))) if b.any() else np.nan, float(np.abs(q_).mean() / max(np.abs(t_).mean(), 1e-12)), float(np.sum(q_ * t_) / max(np.sum(t_ * t_), 1e-12)))
        pt = stat(np.arange(len(p_))); mem = {x: np.nonzero(p_ == x)[0] for x in ps}
        for _ in range(1000): s_ = stat(np.concatenate([mem[x] for x in rg.choice(ps, len(ps))])); sa.append(s_[0]); mr.append(s_[1]); sl.append(s_[2])
        ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        act[m][name] = dict(n_poses=int(len(ps)), n_pairs=int(len(p_)), frac_true_effect_ge_tau=float(big.mean()) if len(big) else None, eligible=bool(GATES[m]["G4"]["detail"][name]["eligible"]) if GATES[m]["G4"] else None, true_abs_effect_mean=float(np.abs(dT).mean()), pred_abs_effect_mean=float(np.abs(dP).mean()),
                            sign_agreement=dict(point=pt[0], ci=ci(sa)), magnitude_ratio=dict(point=pt[1], ci=ci(mr)), regression_slope_pred_on_true=dict(point=pt[2], ci=ci(sl)))
# ---------------------------------------------------------------------------- ablations on test_id K=20 (WM3-B only) and result file
abl = {n: dict(zip(("mean", "lo", "hi"), bci(V.per_group_all(lp(f"wm3b_{n}_test_id"), T["test_id"], groups["test_id"], active["test_id"], "box_mm")[1]))) for n in ("ablate_rfc_zero", "ablate_rfc_shuffled")}; abl["shuffle_actions"] = dict(zip(("mean", "lo", "hi"), bci(V.per_group_all(shuf["wm3b"]["test_id"], T["test_id"], groups["test_id"], active["test_id"], "box_mm")[1]))); abl["clean"] = dict(zip(("mean", "lo", "hi"), bci(V.per_group_all(preds["wm3b"]["test_id"], T["test_id"], groups["test_id"], active["test_id"], "box_mm")[1])))
sup = {sp: dict(groups=int(len(np.unique(groups[sp]))), windows=int(len(groups[sp])), active_windows=int(active[sp].sum()), active_groups=int(len(np.unique(groups[sp][active[sp]])))) for sp in EVAL}
res = dict(status=STATUS, dry_run=DRY, support=sup, integrity=dict(g0_items=g0, extra=extra), negative_controls=neg, gate_verdicts={k: {g: (v[g]["verdict"] if v.get(g) else None) for g in v} for k, v in GATES.items()},
           gates={k: {g: dict(conditions=v[g]["conditions"], detail=v[g].get("detail"), secondary=v[g].get("secondary")) for g in v if v.get(g)} for k, v in GATES.items()}, errors=tab, paired=paired, per_variant_family_box_K20=perfam, classification_calibration=cls, action_effects=act, ablations_test_id_K20=abl, latency_ms=PR["latency"])
(BASE / ("evaluation_dry.json" if DRY else "evaluation_independent.json")).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); COL = dict(persistence="gray", ridge="C0", mlp="C1", wm2_c0="C2", hybrid_posthoc="C4", wm3b="k")
for j, met in enumerate(("box_mm", "joint_rad", "aoffset_cm")):
    for m, c in COL.items():
        mm = [bci(pooled(m, met, k)[1], n=2000) for k in range(1, 21)]; ax[j].plot(range(1, 21), [x[0] for x in mm], color=c, label=m, lw=2 if m == "wm3b" else 1); ax[j].fill_between(range(1, 21), [x[1] for x in mm], [x[2] for x in mm], color=c, alpha=0.12)
    ax[j].set_title(f"{met}, 50 groups, 95% group-bootstrap band"); ax[j].set_xlabel("horizon k")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(BASE / "I1_horizon_errors.png", dpi=120); plt.close()
print(STATUS); print(json.dumps(res["gate_verdicts"], indent=1)); print(neg); print(json.dumps(paired["primary"], indent=1)[:2500])
