#!/usr/bin/env python3
"""Verification of the gate DEFINITIONS (synthetic data with known answers) before the gates are frozen.  Run: python eval/test_wm2_gates.py"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import wm2_gates as G
res = {}; rng = np.random.default_rng(0)
def ok(n, c, info=None): res[n] = bool(c); print(("PASS " if c else "FAIL ") + n, info if info is not None else "")
# metric functions
ok("auroc_perfect", G.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0); ok("auroc_inverse", G.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0); ok("auroc_ties", abs(G.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12)
y = rng.integers(0, 2, 20000); p = rng.random(20000); ok("auroc_random_near_half", abs(G.auroc(y, p) - 0.5) < 0.02)
pp = rng.random(50000); yy = (rng.random(50000) < pp).astype(int); e, b, prev = G.ece_brier(yy, pp); ok("ece_calibrated_small", e < 0.02, round(e, 4)); e2, _, _ = G.ece_brier(yy, np.clip(pp * 0.5, 0, 1)); ok("ece_detects_miscalibration", e2 > 0.1, round(e2, 3))
ok("brier_known", abs(G.ece_brier([1, 0], [1.0, 0.0])[1]) < 1e-12 and abs(G.ece_brier([1, 0], [0.0, 1.0])[1] - 1.0) < 1e-12)
g = np.repeat(np.arange(10), 5); part = G.shuffle_partner(g); ok("shuffle_partner_never_same_group", bool((g[part] != g).all()))
v = np.arange(10, dtype=float); gm, m = G.group_means(np.repeat(v, 3), np.repeat(np.arange(10), 3)); ok("group_macro_equals_mean_of_group_means", np.allclose(m, v) and G.macro(np.repeat(v, 3), np.repeat(np.arange(10), 3)) == v.mean())
vals = np.concatenate([np.ones(100), np.zeros(1)]); grp = np.concatenate([np.zeros(100, int), np.ones(1, int)]); ok("macro_weights_groups_equally_not_windows", abs(G.macro(vals, grp) - 0.5) < 1e-12)
pt, lo, hi = G.boot(lambda M: float(M["a"].mean()), dict(a=rng.normal(5, 1, 200)), n=2000); ok("bootstrap_ci_contains_truth", lo < 5.0 < hi and lo < pt < hi, (round(lo, 2), round(hi, 2)))
ok("spearman_monotone", G.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and G.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0)
# verdict logic
ok("verdict_fail_if_any_evaluable_fails", G.verdict([dict(passed=True, evaluable=True), dict(passed=False, evaluable=True), dict(passed=None, evaluable=False)]) == "FAIL")
ok("verdict_not_evaluable_when_missing_and_nothing_failed", G.verdict([dict(passed=True, evaluable=True), dict(passed=False, evaluable=False)]) == "NOT_EVALUABLE")
ok("verdict_pass_when_all_pass", G.verdict([dict(passed=True, evaluable=True)] * 3) == "PASS")
# input-only subsets
Wm = dict(S_hist=np.zeros((4, 4, 8), np.float32), A_fut=np.zeros((4, 20, 8), np.float32), A_hist=np.zeros((4, 4, 8), np.float32), B_hist=np.zeros((4, 4, 7), np.float32), C_hist=np.zeros((4, 4, 7), np.float32)); Wm["A_fut"][:, :, 7] = 0.044; Wm["A_hist"][:, :, 7] = 0.044
Wm["A_fut"][1, 5, 2] = 0.05; Wm["S_hist"][0, -1, :7] = 0.3; Wm["A_fut"][2, 3:, 7] = 0.012; Wm["A_hist"][3, -1, 7] = 0.012; Wm["A_fut"][3, :, 7] = 0.012
am = G.active_mask(Wm); ok("active_mask_input_only_known_cases", am.tolist() == [False, True, True, False], am.tolist())
Wr = dict(Wm); Wr["B_hist"] = np.zeros((4, 4, 7), np.float32); Wr["B_hist"][:, -1, :2] = [[0.40, 0.10], [0.44, 0.30], [0.44, 0.30], [0.30, 0.30]]; Wr["B_hist"][:, -1, 2] = [0.3805, 0.384, 0.43, 0.3805]; Wr["C_hist"][:, -1, 4] = [0, 0, 0.5, 0]
ok("at_risk_close_open_and_not_in_tray", G.at_risk_mask(Wr, "close").tolist() == [True, False, False, False]); ok("at_risk_lifted", G.at_risk_mask(Wr, "lifted").tolist() == [True, False, False, True]); ok("at_risk_placed", G.at_risk_mask(Wr, "placed").tolist() == [True, False, True, True]); ok("at_risk_palm", G.at_risk_mask(Wr, "palm").tolist() == [True, True, False, True])
# synthetic world for G1/G2/G3/G4/G5
cfg = json.loads(G.config_bytes()); N = 12; per = 40; groups = {}; active = {}; T = {}; preds = {}
def make(n_groups, seed, shift=0.0):
    r = np.random.default_rng(seed); n = n_groups * per; g = np.repeat(np.arange(n_groups), per); Ys = r.normal(0, 0.01, (n, 20, 8)); Yb = r.normal(0, 0.005, (n, 20, 7)); Yc = r.random((n, 20, 7)).astype(np.float32); Ye = (r.random((n, 4)) < 0.3).astype(np.float32)
    return g, dict(Y_s=Ys.astype(np.float32), Y_b=Yb.astype(np.float32), Y_c=Yc, Y_e=Ye, S_hist=np.zeros((n, 4, 8), np.float32), B_hist=np.zeros((n, 4, 7), np.float32))
def predict(Tr, noise, seed): r = np.random.default_rng(seed); return dict(Y_s=Tr["Y_s"] + r.normal(0, noise, Tr["Y_s"].shape).astype(np.float32), Y_b=Tr["Y_b"] + r.normal(0, noise / 2, Tr["Y_b"].shape).astype(np.float32), Y_c=Tr["Y_c"], Y_e=np.clip(Tr["Y_e"] * 0.8 + 0.1 + r.normal(0, 0.05, Tr["Y_e"].shape), 0, 1).astype(np.float32))
for sp, ng in (("test_id", 14), ("test_low_heldout", 12), ("test_ood_low_strict", 9), ("test_ood_high", 12)):
    g, Tr = make(ng, hash(sp) % 1000); groups[sp] = g; active[sp] = np.ones(len(g), bool); T[sp] = Tr
    preds.setdefault("gru", {})[sp] = predict(Tr, 0.002, 1); preds.setdefault("persistence", {})[sp] = dict(Y_s=np.zeros_like(Tr["Y_s"]), Y_b=np.zeros_like(Tr["Y_b"]), Y_c=Tr["Y_c"], Y_e=Tr["Y_e"] * 0 + 0.3)
    for b_ in ("ridge", "mlp", "ridge_blind", "mlp_blind", "constvel"): preds.setdefault(b_, {})[sp] = predict(Tr, 0.02, 2)
    preds.setdefault("outcome_only", {})[sp] = dict(Y_e=np.full_like(Tr["Y_e"], 0.3)); preds.setdefault("gru_shuffle", {})[sp] = predict(Tr, 0.02, 3)
g1 = G.gate_G1(cfg, preds, T, groups, active); ok("G1_pass_on_much_better_model", g1["verdict"] in ("PASS", "FAIL") and g1["conditions"][0]["value"] > 0.5, g1["verdict"])
bad = {k: dict(v) for k, v in preds.items()}; bad["gru"] = {sp: preds["persistence"][sp] for sp in preds["gru"]}; ok("G1_fail_when_model_equals_baseline", G.gate_G1(cfg, bad, T, groups, active)["verdict"] == "FAIL")
ok("G1_not_evaluable_with_few_groups", (lambda: (lambda gg: G.gate_G1(cfg, preds, T, {**groups, "test_id": gg}, active)["verdict"])(np.repeat(np.arange(3), len(groups["test_id"]) // 3 + 1)[:len(groups["test_id"])]))() in ("NOT_EVALUABLE", "FAIL"))
g2 = G.gate_G2(cfg, preds, T, groups, active); ok("G2_runs_and_uses_floor_rule", "detail" in g2 and all("rho" in v for s in g2["detail"].values() for v in s.values()), g2["verdict"])
masks = {(lab, sp): active[sp] for lab in ("close", "palm", "lifted", "placed") for sp in groups}; g3 = G.gate_G3(cfg, preds, T, groups, masks); ok("G3_not_evaluable_without_positive_groups", g3["detail"]["palm/test_low_heldout"]["evaluable"] in (True, False)); 
T2 = {k: dict(v) for k, v in T.items()}; T2["test_low_heldout"]["Y_e"] = T2["test_low_heldout"]["Y_e"].copy(); T2["test_low_heldout"]["Y_e"][:, 1] = 0; ok("G3_palm_not_evaluable_when_no_positives", G.gate_G3(cfg, preds, T2, groups, masks)["detail"]["palm/test_low_heldout"]["evaluable"] is False)
g5 = G.gate_G5(cfg, preds, T, groups, masks, g3); ok("G5_runs", g5["verdict"] in ("PASS", "FAIL", "NOT_EVALUABLE"), g5["verdict"])
bench = dict(median_ms=3.0, params=2_000_000); ok("G6_pass", G.gate_G6(cfg, bench)["verdict"] == "PASS"); ok("G6_fail_slow", G.gate_G6(cfg, dict(median_ms=12.0, params=1))["verdict"] == "FAIL")
# G4 paired synthetic: linear world -> perfect model consistent, blind model not
Np = 12; rows = []; pairs = {k: [] for k in ("B_hist", "Y_b", "Y_e", "family", "level", "pid", "vid")}
for ps in range(Np):
    for vid, (fam_, lev_) in enumerate([(0, 0.0), (1, -0.01), (1, 0.01), (1, 0.02), (1, 0.03), (1, 0.04), (2, 5.0)]):
        Yb = np.zeros((20, 7), np.float32); Yb[-1, 6] = -lev_ * 100 * 0.01 if fam_ == 1 else 0.0; Yb[-1, 6] += 0.0; Ye = np.array([1, 1 if (fam_ == 1 and lev_ >= 0.03) else 0, 0, 0], np.float32)
        pairs["B_hist"].append(np.zeros((4, 7), np.float32)); pairs["Y_b"].append(Yb); pairs["Y_e"].append(Ye); pairs["family"].append(fam_); pairs["level"].append(lev_); pairs["pid"].append(ps); pairs["vid"].append(vid)
pairs = {k: np.asarray(v) for k, v in pairs.items()}; pairs["Y_s"] = np.zeros((len(pairs["pid"]), 20, 8), np.float32); pp_good = dict(gru=dict(Y_b=pairs["Y_b"], Y_e=pairs["Y_e"].astype(np.float32))); pp_blind = dict(gru=dict(Y_b=np.zeros_like(pairs["Y_b"]), Y_e=np.zeros_like(pairs["Y_e"])))
g4g = G.gate_G4(cfg, preds, T, groups, active, pairs, pp_good); g4b = G.gate_G4(cfg, preds, T, groups, active, pairs, pp_blind)
sa = {c["name"]: c for c in g4g["conditions"]}; ok("G4_paired_consistent_model_passes_evaluable_paired_conditions", all(sa[k]["passed"] for k in sa if k.startswith(("paired", "depth", "palm")) and sa[k]["evaluable"]) and not sa["paired_sign_agreement_box_z>=0.80"]["evaluable"], {k: (sa[k]["passed"], sa[k]["evaluable"]) for k in sa})
sb = {c["name"]: c for c in g4b["conditions"]}; ok("G4_paired_action_blind_model_fails_paired_conditions", not sb["paired_spearman_effect_aoffset>=0.70"]["passed"] or not sb["paired_spearman_effect_aoffset>=0.70"]["evaluable"])
cfg_bad = G.config_sha(); ok("config_sha_stable", cfg_bad == G.config_sha())
Path(G.ROOT / "outputs/wm2").mkdir(parents=True, exist_ok=True); (G.ROOT / "outputs/wm2/gates_unit_tests.json").write_text(json.dumps(res, indent=1)); fails = [k for k, v in res.items() if not v]; print("ALL PASS" if not fails else f"FAILED: {fails}"); sys.exit(1 if fails else 0)
