#!/usr/bin/env python3
"""Verification of Gate v2 definitions on synthetic data with known answers (run BEFORE freezing).  python eval/test_wm3_gates.py"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import wm3_gates as V, wm2_gates as G1
res = {}; rng = np.random.default_rng(0)
def ok(n, c, info=None): res[n] = bool(c); print(("PASS " if c else "FAIL ") + n, info if info is not None else "")
cfg = json.loads(V.config_bytes()); per = 40
def make(ng, seed):
    r = np.random.default_rng(seed); n = ng * per; g = np.repeat(np.arange(ng), per)
    return g, dict(Y_s=r.normal(0, .01, (n, 20, 8)).astype(np.float32), Y_b=r.normal(0, .005, (n, 20, 7)).astype(np.float32), Y_c=r.normal(0, 1, (n, 20, 7)).astype(np.float32), Y_e=(r.random((n, 4)) < .3).astype(np.float32), S_hist=np.zeros((n, 4, 8), np.float32), B_hist=np.zeros((n, 4, 7), np.float32))
def pred(T, nj, nb, nc, seed, ev_noise=0.05):
    r = np.random.default_rng(seed); return dict(Y_s=T["Y_s"] + r.normal(0, nj, T["Y_s"].shape).astype(np.float32), Y_b=T["Y_b"] + r.normal(0, nb, T["Y_b"].shape).astype(np.float32), Y_c=T["Y_c"] + r.normal(0, nc, T["Y_c"].shape).astype(np.float32), Y_e=np.clip(T["Y_e"] * 0.8 + 0.1 + r.normal(0, ev_noise, T["Y_e"].shape), 0, 1).astype(np.float32))
groups, active, T, preds = {}, {}, {}, {m: {} for m in ["persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind", "outcome_only", "wm3b", "shuf"]}
for sp, ng in (("test_id", 14), ("test_low_heldout", 12), ("test_ood_low_strict", 9), ("test_ood_high", 12)):
    g, Tr = make(ng, {'test_id': 11, 'test_low_heldout': 12, 'test_ood_low_strict': 13, 'test_ood_high': 14}[sp]); groups[sp] = g; active[sp] = np.ones(len(g), bool); T[sp] = Tr
    preds["persistence"][sp] = dict(Y_s=np.zeros_like(Tr["Y_s"]), Y_b=np.zeros_like(Tr["Y_b"]), Y_c=np.zeros_like(Tr["Y_c"]), Y_e=np.full_like(Tr["Y_e"], .3)); preds["constvel"][sp] = preds["persistence"][sp]
    preds["ridge"][sp] = pred(Tr, 0.0003, 0.02, 0.5, 1); preds["mlp"][sp] = pred(Tr, 0.004, 0.01, 0.4, 2); preds["ridge_blind"][sp] = pred(Tr, 0.002, 0.03, 0.8, 3); preds["mlp_blind"][sp] = pred(Tr, 0.004, 0.02, 0.6, 4)
    preds["wm3b"][sp] = pred(Tr, 0.0003, 0.003, 0.1, 5); preds["outcome_only"][sp] = dict(Y_e=np.full_like(Tr["Y_e"], .3)); preds["shuf"][sp] = pred(Tr, 0.002, 0.04, 0.5, 6)
g1 = V.gate_G1(cfg, preds, T, groups, active); ok("G1_pass_when_joint_equals_ridge_and_objects_much_better", g1["verdict"] == "PASS", [(c["name"], c["passed"]) for c in g1["conditions"] if not c["passed"]])
bad = {k: dict(v) for k, v in preds.items()}; bad["wm3b"] = {sp: pred(T[sp], 0.01, 0.004, 0.1, 7) for sp in T}; ok("G1_fails_when_joint_error_above_threshold", V.gate_G1(cfg, bad, T, groups, active)["verdict"] == "FAIL")
bad2 = {k: dict(v) for k, v in preds.items()}; bad2["wm3b"] = {sp: pred(T[sp], 0.0003, 0.02, 0.5, 8) for sp in T}; ok("G1_fails_without_superiority_on_objects", V.gate_G1(cfg, bad2, T, groups, active)["verdict"] == "FAIL")
ok("G1_not_evaluable_few_groups", V.gate_G1(cfg, preds, T, {**groups, "test_id": np.repeat(np.arange(3), (len(groups["test_id"]) // 3) + 1)[:len(groups["test_id"])]}, active)["verdict"] in ("NOT_EVALUABLE", "FAIL"))
g2 = V.gate_G2(cfg, preds, T, groups, active); ok("G2_baseline_relative_R_reported_and_pass_when_similar_degradation", g2["verdict"] == "PASS" and all("R" in v for s in g2["detail"].values() for v in s.values()), g2["verdict"])
bad3 = {k: dict(v) for k, v in preds.items()}; bad3["wm3b"] = dict(preds["wm3b"]); bad3["wm3b"]["test_ood_high"] = pred(T["test_ood_high"], 0.0003, 0.05, 0.1, 9); ok("G2_fails_when_ood_error_explodes", V.gate_G2(cfg, bad3, T, groups, active)["verdict"] == "FAIL")
# floor rule: baseline error below floor
fl = {k: dict(v) for k, v in preds.items()}
for b in ("persistence", "constvel", "ridge", "mlp", "ridge_blind", "mlp_blind"): fl[b] = {sp: {**T[sp], "Y_e": T[sp]["Y_e"]} for sp in T}                                   # baselines = truth => error 0 < floor
fl["wm3b"] = {sp: pred(T[sp], 0.0, 1e-6, 0.0, 10) for sp in T}; g2f = V.gate_G2(cfg, fl, T, groups, active); ok("G2_floor_rule_when_baseline_near_zero", all(v["best_value"] < cfg["floors"][m] for s in g2f["detail"].values() for m, v in s.items()) and g2f["verdict"] == "PASS")
# G3 pooled palm + secondary
masks = {(lab, sp): active[sp] for lab in V.LAB for sp in groups}
good = {k: dict(v) for k, v in preds.items()}; good["wm3b"] = {sp: dict(preds["wm3b"][sp], Y_e=np.clip(T[sp]["Y_e"] * 0.9 + 0.05, 0, 1)) for sp in T}; good["outcome_only"] = {sp: dict(Y_e=np.full_like(T[sp]["Y_e"], .3)) for sp in T}
g3 = V.gate_G3(cfg, good, T, groups, masks); ok("G3_primary_close_and_pooled_palm_evaluable_and_secondary_reported", g3["verdict"] == "PASS" and "lifted" in g3["secondary"] and "brier_skill_vs_outcome_only" in g3["secondary"]["placed"], g3["verdict"])
T2 = {k: dict(v) for k, v in T.items()}
for sp in ("test_low_heldout", "test_ood_low_strict"): T2[sp]["Y_e"] = T2[sp]["Y_e"].copy(); T2[sp]["Y_e"][:, 1] = 0
ok("G3_palm_not_evaluable_without_positives", V.gate_G3(cfg, good, T2, groups, masks)["verdict"] == "NOT_EVALUABLE")
few = {k: v.copy() if hasattr(v, "copy") else v for k, v in groups.items()}; T3 = {k: dict(v) for k, v in T.items()}
for sp in ("test_low_heldout", "test_ood_low_strict"): Y = np.zeros_like(T[sp]["Y_e"]); Y[:per * 3, 1] = 1; T3[sp]["Y_e"] = Y                                                                          # only 3 groups positive per set -> pooled 6 < 10
ok("G3_pooled_palm_insufficient_groups_not_evaluable", V.gate_G3(cfg, good, T3, groups, masks)["detail"]["palm/test_low_heldout+test_ood_low_strict"]["evaluable"] is False)
g5 = V.gate_G5(cfg, good, T, groups, masks, g3); ok("G5_runs", g5["verdict"] in ("PASS", "FAIL", "NOT_EVALUABLE"), g5["verdict"])
# G4 per family: build synthetic family windows with known effects
def fam_windows(effect_scale):
    fw = {}; fp = {"wm3b": {}, "oracle": {}}
    for name in cfg["G4"]["families"]:
        rows = {k: [] for k in ("B_hist", "Y_b", "Y_e", "pid", "vid", "level")}; pr = {k: [] for k in ("Y_b", "Y_e")}
        for pid in range(12):
            for vid in range(7):
                lev = 0.0 if vid == 0 else float(vid) * 0.01; eff = effect_scale.get(name, 0.0) * lev * 100; Yb = np.zeros((20, 7), np.float32)
                if name == "depth": Yb[-1, 6] = -eff / 100.0
                elif name in ("speed", "release_height"): Yb[-1, 0] = eff / 1000.0
                elif name == "noise": Yb[-1, 2] = eff / 1000.0
                Ye = np.array([1.0 if (name == "timing" and eff > 0.4) else 0.0, 1.0 if (name == "depth" and lev >= 0.04) else 0.0, 0, 0], np.float32)
                rows["B_hist"].append(np.zeros((4, 7), np.float32)); rows["Y_b"].append(Yb); rows["Y_e"].append(Ye); rows["pid"].append(pid); rows["vid"].append(vid); rows["level"].append(lev); pr["Y_b"].append(Yb); pr["Y_e"].append(Ye)
        fw[name] = {k: np.asarray(v) for k, v in rows.items()}; fp["oracle"][name] = {k: np.asarray(v) for k, v in pr.items()}; fp["wm3b"][name] = fp["oracle"][name]
    return fw, fp
fw, fp = fam_windows(dict(depth=1.0, speed=5.0, release_height=5.0, noise=20.0, timing=1.0))
g4 = V.gate_G4(cfg, preds, T, groups, active, fw, fp, shuffled=preds["shuf"]); ok("G4_oracle_passes_eligible_families", g4["verdict"] == "PASS", [(c["name"], c["passed"]) for c in g4["conditions"] if c["evaluable"] and not c["passed"]])
fw0, fp0 = fam_windows(dict(depth=1.0, speed=0.0, release_height=0.0, noise=0.0, timing=0.0)); g40 = V.gate_G4(cfg, preds, T, groups, active, fw0, fp0, shuffled=preds["shuf"])
ok("G4_families_without_true_effect_are_not_evaluable_not_failed", not g40["detail"]["speed"]["eligible"] and not g40["detail"]["release_height"]["eligible"] and g40["verdict"] in ("PASS", "FAIL"), {k: v["eligible"] for k, v in g40["detail"].items()})
fpb = {"wm3b": {n: dict(Y_b=np.zeros_like(fp["oracle"][n]["Y_b"]), Y_e=np.zeros_like(fp["oracle"][n]["Y_e"])) for n in fw}}; ok("G4_action_blind_predictions_fail_family_conditions", V.gate_G4(cfg, preds, T, groups, active, fw, fpb, shuffled=preds["shuf"])["verdict"] == "FAIL")
fw_n, fp_n = fam_windows({}); ok("G4_no_eligible_family_not_evaluable", V.gate_G4(cfg, preds, T, groups, active, fw_n, fp_n, shuffled=preds["shuf"])["verdict"] == "NOT_EVALUABLE")
ok("G6_pass_fail", V.gate_G6(cfg, dict(median_ms=3.0, params=8e5))["verdict"] == "PASS" and V.gate_G6(cfg, dict(median_ms=12.0, params=8e5))["verdict"] == "FAIL")
ok("G0_requires_all_items", V.gate_G0(cfg, {k: True for k in ["v2_config_hash_matches", "wm2_dataset_and_window_hashes_match", "groups_in_multiple_splits_zero", "training_loaded_only_train_val", "originals_untouched", "negative_control_persistence_fails_G1", "negative_control_action_shuffle_degrades", "negative_control_label_shuffle_auroc_in_0p4_0p6", "labelled_development_evaluation"]})["verdict"] == "PASS" and V.gate_G0(cfg, {})["verdict"] == "FAIL")
ok("config_declares_development_not_independent", "DEVELOPMENT" in cfg["status"] and cfg["independent_heldout_plan"]["status"] == "NOT RUN"); ok("config_sha_stable", V.config_sha() == V.config_sha())
(Path(__file__).resolve().parent.parent / "outputs/wm3/gates_v2_unit_tests.json").write_text(json.dumps(res, indent=1)); f = [k for k, v in res.items() if not v]; print("ALL PASS" if not f else f"FAILED: {f}"); sys.exit(1 if f else 0)
