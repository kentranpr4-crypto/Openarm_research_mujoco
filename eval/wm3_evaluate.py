#!/usr/bin/env python3
"""WM-3 INDEPENDENT-PROCESS EVALUATOR (Gate v2).  Reads ONLY: frozen configs, the frozen wm2 test windows / family windows, and the prediction files written by the trainer.  Imports no training or model code.
All results are DEVELOPMENT evaluation (the wm2 test splits were already analysed in WM-2/WM-2.1) and are single-seed => comparisons between configurations are EXPLORATORY.
Run (ACT venv): python eval/wm3_evaluate.py"""
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm3_gates as V
OUT = ROOT / "outputs/wm3"; PRED = OUT / "preds"; WIN = ROOT / "outputs/wm2/windows"; EVAL = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; FAMS = ["depth", "speed", "timing", "release_height", "noise"]
FROZEN = json.loads((OUT / "freeze.json").read_text()); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg = V.load_config(FROZEN["gates_v2_config_sha256"]); g0 = {}
g0["v2_config_hash_matches"] = sha(OUT / "gates_v2_config.json") == FROZEN["gates_v2_config_sha256"]
g0["wm2_dataset_and_window_hashes_match"] = sha(ROOT / "outputs/wm2/dataset/index.json") == FROZEN["wm2_dataset_index_sha256"] and all(sha(WIN / f"{sp}.npz") == h for sp, h in FROZEN["wm2_windows_sha256"].items()) and sha(OUT / "family_windows.npz") == FROZEN["family_windows_sha256"]
g0["originals_untouched"] = sha(ROOT / "outputs/wm1/dataset/index.json") == json.loads((ROOT / "outputs/wm2/freeze.json").read_text())["wm1_index_sha256"]
tr_rep = json.loads((OUT / "trainer_report.json").read_text()); g0["training_loaded_only_train_val"] = set(tr_rep["splits_loaded_during_fit"]) <= {"train", "val"}; g0["labelled_development_evaluation"] = "DEVELOPMENT" in FROZEN["status"]
def lp(n): z = np.load(PRED / f"{n}.npz"); return {k: z[k] for k in z.files}
Wd = {sp: {k: np.load(WIN / f"{sp}.npz")[k] for k in np.load(WIN / f"{sp}.npz").files} for sp in EVAL}; T = {sp: {k: Wd[sp][k] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e")} for sp in EVAL}
groups = {sp: Wd[sp]["group"] for sp in EVAL}; active = {sp: V.active_mask(Wd[sp]) for sp in EVAL}; masks = {(lab, sp): V.at_risk_mask(Wd[sp], lab) for lab in V.LAB for sp in EVAL}
g0["groups_in_multiple_splits_zero"] = all(not (set(groups[a].tolist()) & set(groups[b].tolist())) for a in EVAL for b in EVAL if a < b)
MODELS = ["persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind", "outcome_only", "wm3b", "wm2_c0", "hybrid_posthoc"]
preds = {m: {sp: lp(f"{m}_{sp}") for sp in EVAL} for m in MODELS}; shuf = {"wm3b": {"test_id": lp("wm3b_shuffle_test_id")}, "wm2_c0": {"test_id": lp("wm2_c0_shuffle_test_id")}}
fw_raw = np.load(OUT / "family_windows.npz"); fam_windows = {f: {k.split("/", 1)[1]: fw_raw[k] for k in fw_raw.files if k.startswith(f + "/")} for f in FAMS}
fam_preds = {m: {f: lp(f"{m}_family_{f}") for f in FAMS} for m in ("wm3b", "wm2_c0", "hybrid_posthoc")}
# ---------------- negative controls
neg = {}
pers_as_cand = dict(preds); pers_as_cand["wm3b"] = preds["persistence"]; neg["persistence_as_candidate_G1"] = V.gate_G1(cfg, pers_as_cand, T, groups, active)["verdict"]; g0["negative_control_persistence_fails_G1"] = neg["persistence_as_candidate_G1"] == "FAIL"
e_clean = V.per_group_all(preds["wm3b"]["test_id"], T["test_id"], groups["test_id"], active["test_id"], "box_mm")[1].mean(); e_shuf = V.per_group_all(shuf["wm3b"]["test_id"], T["test_id"], groups["test_id"], active["test_id"], "box_mm")[1].mean(); neg["action_shuffle_box_mm"] = [float(e_clean), float(e_shuf)]; g0["negative_control_action_shuffle_degrades"] = bool(e_shuf > 1.3 * e_clean)
mk = masks[("close", "test_id")]; y = T["test_id"]["Y_e"][mk][:, 0].astype(int); p = preds["wm3b"]["test_id"]["Y_e"][mk][:, 0]; rng = np.random.default_rng(0); neg["label_shuffle_auroc"] = float(np.mean([V.auroc(rng.permutation(y), p) for _ in range(20)])); g0["negative_control_label_shuffle_auroc_in_0p4_0p6"] = 0.4 <= neg["label_shuffle_auroc"] <= 0.6
def run_gates(model):
    g1 = V.gate_G1(cfg, preds, T, groups, active, model); g2 = V.gate_G2(cfg, preds, T, groups, active, model); g3 = V.gate_G3(cfg, preds, T, groups, masks, model); g4 = V.gate_G4(cfg, preds, T, groups, active, fam_windows, fam_preds, model, shuffled={sp: v for sp, v in shuf[model].items()}) if model in shuf else None
    g5 = V.gate_G5(cfg, preds, T, groups, masks, g3, model); return dict(G1=g1, G2=g2, G3=g3, G4=g4, G5=g5)
bench = dict(median_ms=tr_rep["latency"]["median_ms"], params=tr_rep["latency"]["params"]); GATES = {"wm3b": run_gates("wm3b"), "wm2_c0": run_gates("wm2_c0")}
GATES["wm3b"]["G0"] = V.gate_G0(cfg, g0); GATES["wm3b"]["G6"] = V.gate_G6(cfg, bench)
# hybrid has no shuffle run -> evaluate G1/G2/G3/G5 only
hy = dict(G1=V.gate_G1(cfg, preds, T, groups, active, "hybrid_posthoc"), G2=V.gate_G2(cfg, preds, T, groups, active, "hybrid_posthoc")); hy["G3"] = V.gate_G3(cfg, preds, T, groups, masks, "hybrid_posthoc"); hy["G5"] = V.gate_G5(cfg, preds, T, groups, masks, hy["G3"], "hybrid_posthoc"); GATES["hybrid_posthoc"] = hy
# ---------------- error tables
tab = {}
for m in MODELS:
    if m == "outcome_only": continue
    tab[m] = {sp: {f"K={k}": {met: float(V.macro(v, groups[sp][active[sp]])) for met, v in V.all_errors(V.sub(preds[m][sp], active[sp]), V.sub(T[sp], active[sp]), k).items()} for k in (1, 5, 10, 20)} for sp in EVAL}
curves = {}
for m in ("persistence", "ridge", "mlp", "wm2_c0", "hybrid_posthoc", "wm3b"):
    curves[m] = {met: [float(V.macro(V.all_errors(V.sub(preds[m]["test_id"], active["test_id"]), V.sub(T["test_id"], active["test_id"]), k)[met], groups["test_id"][active["test_id"]])) for k in range(1, 21)] for met in ("box_mm", "joint_rad", "aoffset_cm", "contact_std")}
abl = {n: {met: float(V.macro(v, groups["test_id"][active["test_id"]])) for met, v in V.all_errors(V.sub(lp(f"wm3b_{n}_test_id") if n != "shuffle" else shuf["wm3b"]["test_id"], active["test_id"]), V.sub(T["test_id"], active["test_id"]), 20).items()} for n in ("ablate_rfc_zero", "ablate_rfc_shuffled", "shuffle")}
abl["clean"] = tab["wm3b"]["test_id"]["K=20"]
# per-variable K=20 (box channels, contact channels) for the four main models on test_id and strict OOD-low
CN = ["finger_force", "n_contacts", "a_contact", "a_valid", "palm_force", "table_force", "tray_force"]; OB = ["box_x_mm", "box_y_mm", "box_z_mm", "yaw_rad", "speed_mps", "tilt_rad", "aoffset_cm"]; sc = np.array([1000, 1000, 1000, 1, 1, 1, 100]); pv = {}
for sp in ("test_id", "test_ood_low_strict"):
    a = active[sp]; g = groups[sp][a]; pv[sp] = {}
    for m in ("ridge", "mlp", "wm2_c0", "hybrid_posthoc", "wm3b"):
        db = np.abs(np.cumsum(preds[m][sp]["Y_b"][a], 1)[:, 19] - np.cumsum(T[sp]["Y_b"][a], 1)[:, 19]) * sc; dc = np.abs(preds[m][sp]["Y_c"][a][:, 19] - T[sp]["Y_c"][a][:, 19]); ds = np.abs(np.cumsum(preds[m][sp]["Y_s"][a], 1)[:, 19] - np.cumsum(T[sp]["Y_s"][a], 1)[:, 19])
        pv[sp][m] = dict(object={n: float(V.macro(db[:, i], g)) for i, n in enumerate(OB)}, contact={n: float(V.macro(dc[:, i], g)) for i, n in enumerate(CN)}, joints_mean=float(V.macro(ds[:, :7].mean(1), g)))
strata = {}
for sp in ("test_id", "test_ood_low_strict"):
    W = Wd[sp]; a = active[sp]; palm = T[sp]["Y_e"][:, 1] > 0; fc = W["C_hist"][:, -1, 1] > 0; strata[sp] = {}
    for nm, mkk in (("palm_contact_in_horizon", palm), ("no_palm_contact_in_horizon", ~palm), ("finger_contact_at_t", fc), ("no_finger_contact_at_t", ~fc)):
        mm = a & mkk
        if len(np.unique(groups[sp][mm])) >= 3: strata[sp][nm] = {m: float(V.macro(V.all_errors(V.sub(preds[m][sp], mm), V.sub(T[sp], mm), 20)["box_mm"], groups[sp][mm])) for m in ("persistence", "ridge", "mlp", "wm2_c0", "hybrid_posthoc", "wm3b")} | dict(n=int(mm.sum()))
res = dict(status=FROZEN["status"], frozen=dict(gates_v2=FROZEN["gates_v2_config_sha256"], model_config=FROZEN["model_config_wm3b_sha256"], family_windows=FROZEN["family_windows_sha256"]), negative_controls=neg, gate_verdicts={k: {g: (v[g]["verdict"] if v.get(g) else None) for g in v} for k, v in GATES.items()},
           gates={k: {g: dict(conditions=v[g]["conditions"], detail=v[g].get("detail"), secondary=v[g].get("secondary")) for g in v if v.get(g)} for k, v in GATES.items()}, errors=tab, horizon_curves_test_id=curves, ablations_test_id_K20=abl, per_variable_K20=pv, contact_strata_box_K20=strata, trainer=dict(train_seconds=tr_rep["train_seconds"], best_epoch=tr_rep["best_epoch"], latency=tr_rep["latency"], total_wall_s=tr_rep["total_wall_s"]))
(OUT / "evaluation_v2.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8)); COL = dict(persistence="gray", ridge="C0", mlp="C1", wm2_c0="C2", hybrid_posthoc="C4", wm3b="k")
for j, met in enumerate(("box_mm", "joint_rad", "aoffset_cm", "contact_std")):
    for m, c in curves.items(): ax[j].plot(range(1, 21), c[met], color=COL[m], label=m, lw=2 if m == "wm3b" else 1)
    ax[j].set_title(f"test_id, {met}"); ax[j].set_xlabel("horizon k")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "V1_horizon_errors.png", dpi=120); plt.close()
ev = [(k, v) for k, v in GATES["wm3b"]["G5"]["detail"].items() if v.get("reliability")]
if ev:
    fig, ax = plt.subplots(1, len(ev), figsize=(4.2 * len(ev), 3.8), squeeze=False)
    for a_, (k, v) in zip(ax[0], ev): rr = [r for r in v["reliability"] if r["n"] > 0]; a_.plot([0, 1], [0, 1], "k:"); a_.plot([r["mean_pred"] for r in rr], [r["frac_pos"] for r in rr], "-o"); a_.set_title(f"{k}\nECE {v['ECE']:.3f} Brier {v['Brier']:.3f} prev {v['prevalence']:.3f}", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "V2_reliability.png", dpi=120); plt.close()
print(json.dumps(res["gate_verdicts"], indent=1)); print(neg)
