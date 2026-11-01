#!/usr/bin/env python3
"""Synthetic-fixture tests with known ground truth for the Gate v3 DRAFT (G3v3, G7).  Deterministic seeds.  Writes outputs/gates_v3/unit_tests.json.  Run: python eval/test_wm4_gates.py"""
import sys, json, copy, numpy as np
from statistics import NormalDist
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm3_gates as V, wm4_gates as G
SETS = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; NG = dict(test_id=18, test_low_heldout=10, test_ood_low_strict=10, test_ood_high=12); RES = {}
def rec(name, ok, info=""): RES[name] = bool(ok); print(("PASS " if ok else "FAIL "), name, info)
def sig(x): return 1 / (1 + np.exp(-x))
def make_ref(y, target_auc, rng):
    d = np.sqrt(2) * NormalDist().inv_cdf(target_auc) / 2; return sig(d * (2 * y - 1) * 2 + rng.standard_normal(len(y)) * 1.0 * 1.0 + 0)   # logit separation 2d, unit noise => AUROC = Phi(sqrt(2)*d*2/ (sqrt2*... )) calibrated below
def fixture3(ref_auc, win=200, seed=0, groups=NG, palm_prev=0.06):
    rng = np.random.default_rng(seed); T, groups_, preds = {}, {}, {m: {} for m in ("oracle", "ref", "shuffle", "outcome_only")}; masks = {}; gid = 0
    for sp in SETS:
        g = np.repeat(np.arange(gid, gid + groups[sp]), win); gid += groups[sp]; n = len(g); y = np.zeros((n, 4), np.float32); y[:, 0] = rng.random(n) < 0.4; y[:, 1] = (rng.random(n) < palm_prev) if sp in ("test_low_heldout", "test_ood_low_strict") else 0
        y[:, 2] = rng.random(n) < 0.3; y[:, 3] = rng.random(n) < 0.2; T[sp] = dict(Y_e=y); groups_[sp] = g
        po = np.stack([sig(((2 * y[:, j] - 1) * NormalDist().inv_cdf(ref_auc) / np.sqrt(2)) + rng.standard_normal(n)) for j in range(4)], 1).astype(np.float32)   # AUROC_ref ~ ref_auc (binormal, unit variance)
        preds["outcome_only"][sp] = dict(Y_e=po); preds["ref"][sp] = dict(Y_e=po.copy()); preds["oracle"][sp] = dict(Y_e=np.clip(y * 0.98 + 0.01 + 0.005 * rng.standard_normal((n, 4)), 0, 1).astype(np.float32)); preds["shuffle"][sp] = dict(Y_e=y[rng.permutation(n)] * 0.9 + 0.05)
        for lab in V.LAB: masks[(lab, sp)] = np.ones(n, bool)
    return preds, T, groups_, masks
CFG = copy.deepcopy(G.CONFIG); CFG["G3"]["bootstrap_resamples_auroc"] = 100       # fewer AUROC bootstrap resamples in tests only (speed); the CI is not a gating quantity in the ceiling branch
def run3(preds, T, groups, masks, model, cfg=CFG): return G.gate_G3_v3(cfg, preds, T, groups, masks, model)
def cond(r, frag): return [c for c in r["conditions"] if frag in c["name"]]
# ------------------------------------------------------------------ G3 v3
pr, T, g, mk = fixture3(0.75); r_or = run3(pr, T, g, mk, "oracle"); r_ref = run3(pr, T, g, mk, "ref"); r_sh = run3(pr, T, g, mk, "shuffle")
d = r_or["detail"]["close/test_id"]; rec("G3_margin_branch_selected_when_ref_auroc_le_0.90", d["branch"] == "margin" and 0.7 < d["auroc_ref"] < 0.8, f"A_ref={d['auroc_ref']:.3f}")
rec("G3_margin_branch_oracle_PASS", r_or["verdict"] == "PASS", r_or["verdict"]); rec("G3_margin_branch_reference_as_model_FAIL", r_ref["verdict"] == "FAIL" and any(not c["passed"] and "margin" in c["name"] for c in r_ref["conditions"])); rec("G3_margin_branch_label_shuffle_FAIL", r_sh["verdict"] == "FAIL")
pr, T, g, mk = fixture3(0.97, seed=1); r_or = run3(pr, T, g, mk, "oracle"); r_ref = run3(pr, T, g, mk, "ref"); r_sh = run3(pr, T, g, mk, "shuffle"); d = r_or["detail"]["close/test_id"]
rec("G3_ceiling_branch_selected_when_ref_auroc_gt_0.90", d["branch"] == "ceiling" and d["auroc_ref"] > 0.95, f"A_ref={d['auroc_ref']:.3f}")
rec("G3_v2_margin_rule_is_unsatisfiable_at_this_ceiling_but_v3_oracle_PASSES", (1 - d["auroc_ref"]) < 0.05 and r_or["verdict"] == "PASS", f"max possible v2 margin={1-d['auroc_ref']:.3f}, v3={r_or['verdict']}")
rec("G3_ceiling_branch_reference_as_model_FAIL_brier_skill_zero", r_ref["verdict"] == "FAIL" and abs(r_ref["detail"]["close/test_id"]["brier_skill"]) < 1e-9); rec("G3_ceiling_branch_label_shuffle_FAIL", r_sh["verdict"] == "FAIL")
# Brier skill definition: macro over groups, floor
y = np.r_[np.ones(10), np.zeros(1000)]; gg = np.r_[np.zeros(10), np.ones(1000)]; p = np.r_[np.full(10, 0.0), np.full(1000, 0.0)]; po = np.r_[np.full(10, 0.5), np.full(1000, 0.1)]
bss, bd, bm, br = G.brier_skill(y, p, po, gg, 1e-3); exp_ref = np.mean([0.25 * 1.0 if False else ((0.5 - 1) ** 2), (0.1 - 0) ** 2]); exp_model = np.mean([1.0, 0.0])
rec("G3_brier_skill_is_group_macro_not_window_pooled", abs(bd["B_ref"] - exp_ref) < 1e-9 and abs(bd["B_model"] - exp_model) < 1e-9 and abs(bss - (1 - exp_model / exp_ref)) < 1e-9, f"B_ref={bd['B_ref']:.4f}")
y = np.r_[np.ones(500), np.zeros(500)]; gg = np.repeat(np.arange(20), 50); po = np.where(y == 1, 0.999, 0.001); pm = np.where(y == 1, 0.9995, 0.0005); b1, d1, _, _ = G.brier_skill(y, pm, po, gg, 1e-3)
rec("G3_brier_floor_when_reference_nearly_perfect", d1["floor_used"] and abs(d1["denominator"] - 1e-3) < 1e-12 and abs(b1 - (1 - d1["B_model"] / 1e-3)) < 1e-9, f"B_ref={d1['B_ref']:.2e}, BSS={b1:.3f}")
b0, d0, _, _ = G.brier_skill(y, po, po, gg, 1e-3); rec("G3_brier_skill_zero_for_reference_even_with_floor", abs(b0) < 1.0 and d0["floor_used"], f"BSS(ref)={b0:.3f}")
# hard subset behaviours
pr, T, g, mk = fixture3(0.985, seed=2, win=400); lo, hi = CFG["G3"]["hard_subset"]["ref_prob_range"]
for sp in SETS:                                   # model: oracle outside the reference-uncertain zone, random inside it
    po = pr["outcome_only"][sp]["Y_e"]; hm = (po >= lo) & (po <= hi); q = pr["oracle"][sp]["Y_e"].copy(); rnd = np.random.default_rng(3).random(q.shape).astype(np.float32); q[hm] = rnd[hm]; pr.setdefault("hardfail", {})[sp] = dict(Y_e=q)
r_h = run3(pr, T, g, mk, "hardfail"); dh = r_h["detail"]["close/test_id"]; hc = cond(r_h, "hard_subset")[0]
rec("G3_hard_subset_catches_model_that_is_random_where_reference_is_uncertain", dh["hard_subset"]["evaluable"] and not hc["passed"] and hc["evaluable"] and r_h["verdict"] == "FAIL", f"hard AUROC={dh['hard_subset'].get('auroc'):.3f}, n={dh['hard_subset']['n_windows']}")
pr, T, g, mk = fixture3(0.985, seed=4, win=400); r_o = run3(pr, T, g, mk, "oracle"); rec("G3_hard_subset_oracle_passes", r_o["verdict"] == "PASS" and cond(r_o, "hard_subset")[0]["passed"], r_o["verdict"])
pr, T, g, mk = fixture3(0.75, seed=5)                                 # force an EMPTY hard subset: reference probabilities pushed outside [0.2, 0.8]
for sp in SETS:
    y = T[sp]["Y_e"]; pr["outcome_only"][sp]["Y_e"] = np.where(y > 0, 0.95, 0.05).astype(np.float32) + 0.001 * np.random.default_rng(0).random(y.shape).astype(np.float32)
r_ne = run3(pr, T, g, mk, "oracle"); hc = cond(r_ne, "hard_subset")
rec("G3_insufficient_hard_subset_support_is_NOT_EVALUABLE_not_PASS", r_ne["verdict"] == "NOT_EVALUABLE" and all(not c["evaluable"] for c in hc) and len(hc) == 2 and all(c["passed"] for c in r_ne["conditions"] if c["evaluable"]), f"{r_ne['verdict']}")
r_ne2 = run3(pr, T, g, mk, "shuffle"); rec("G3_FAIL_dominates_NOT_EVALUABLE", r_ne2["verdict"] == "FAIL", r_ne2["verdict"])
pr, T, g, mk = fixture3(0.75, seed=6, groups=dict(test_id=5, test_low_heldout=3, test_ood_low_strict=3, test_ood_high=3)); r_f = run3(pr, T, g, mk, "oracle")
rec("G3_insufficient_groups_overall_is_NOT_EVALUABLE", r_f["verdict"] == "NOT_EVALUABLE" and all(not c["evaluable"] for c in r_f["conditions"] if c["name"].startswith("close")), r_f["verdict"])
pr, T, g, mk = fixture3(0.75, seed=7); a_ref = r_or2 = run3(pr, T, g, mk, "oracle")["detail"]["close/test_id"]["auroc_ref"]; cfg2 = copy.deepcopy(CFG); cfg2["G3"]["branch_threshold_auroc_ref"] = a_ref
rec("G3_branch_boundary_is_inclusive_margin_branch_at_equality", run3(pr, T, g, mk, "oracle", cfg2)["detail"]["close/test_id"]["branch"] == "margin"); cfg2["G3"]["branch_threshold_auroc_ref"] = a_ref - 1e-6
rec("G3_branch_boundary_ceiling_branch_just_above", run3(pr, T, g, mk, "oracle", cfg2)["detail"]["close/test_id"]["branch"] == "ceiling")
pr, T, g, mk = fixture3(0.8, seed=8); rec("G3_deterministic", json.dumps(run3(pr, T, g, mk, "oracle"), default=str) == json.dumps(run3(pr, T, g, mk, "oracle"), default=str))
rec("G3_lifted_placed_reported_without_gating", set(run3(pr, T, g, mk, "oracle")["secondary"]) == {"lifted", "placed"})
# ------------------------------------------------------------------ G7
def fixture7(win=60, seed=0, groups=NG, move_frac=0.4):
    rng = np.random.default_rng(seed); T, grp, act = {}, {}, {}; gid = 0; pr = {m: {} for m in ("persistence", "ridge", "oracle")}
    for sp in SETS:
        g = np.repeat(np.arange(gid, gid + groups[sp]), win); gid += groups[sp]; n = len(g); Yb = np.zeros((n, 20, 7), np.float32); mv = rng.random(n) < move_frac; step = rng.uniform(0.001, 0.003, n) * rng.choice([-1, 1], n); Yb[:, :, 3] = np.where(mv[:, None], step[:, None], 0.0)
        B = np.zeros((n, 4, 7), np.float32); C = np.zeros((n, 4, 7), np.float32); Af = np.zeros((n, 20, 8), np.float32); Af[:, :, 7] = 0.044; B[mv, :, 4] = 0.05            # moving windows: box speed > threshold => not in S_in
        T[sp] = dict(Y_b=Yb, B_hist=B, C_hist=C, A_fut=Af); grp[sp] = g; act[sp] = np.ones(n, bool); z = np.zeros_like(Yb)
        pr["persistence"][sp] = dict(Y_b=z.copy()); pr["ridge"][sp] = dict(Y_b=-Yb); pr["oracle"][sp] = dict(Y_b=Yb.copy())
    return pr, T, grp, act
def add(pr, name, base, fn):
    pr[name] = {sp: dict(Y_b=fn(sp, pr[base][sp]["Y_b"].copy())) for sp in SETS}
def run7(pr, T, grp, act, model, cfg=CFG): return G.gate_G7(cfg, pr, T, grp, act, model, n_boot=2000)
pr, T, grp, act = fixture7(); P_ = lambda sp: pr["oracle"][sp]
x = dict(Y_b=np.zeros((2, 20, 7), np.float32)); x["Y_b"][0, :, 3] = 3.1 / 20; xp = dict(Y_b=np.zeros((2, 20, 7), np.float32)); xp["Y_b"][0, :, 3] = 3.2 / 20
rec("G7_angle_error_wraps_across_pi", abs(G.ang_err(xp, x, 20)[0] - 0.1) < 1e-5 and abs(G.theta(xp["Y_b"], 20)[0] - (3.2 - 2 * np.pi)) < 1e-5, f"{G.ang_err(xp, x, 20)[0]:.4f}")
xq = dict(Y_b=np.zeros((2, 20, 7), np.float32)); xq["Y_b"][0, 0, 3] = 0.5; xr = dict(Y_b=np.zeros((2, 20, 7), np.float32)); xr["Y_b"][0, 0, 3] = 0.5 - 2 * np.pi
rec("G7_error_invariant_to_2pi_representation", G.ang_err(xq, xr, 20)[0] < 1e-5)
r = run7(pr, T, grp, act, "oracle"); rec("G7_oracle_PASS", r["verdict"] == "PASS", r["verdict"]); r = run7(pr, T, grp, act, "persistence"); rec("G7_persistence_is_baseline_ratio_1_PASS", r["verdict"] == "PASS" and abs(r["detail"]["primary"]["ratio"] - 1.0) < 1e-6, f"ratio={r['detail']['primary']['ratio']:.3f}")
rng = np.random.default_rng(11); add(pr, "noisy", "oracle", lambda sp, y: np.concatenate([y[:, :, :3], (0.01 * rng.standard_normal(y[:, :, 3:4].shape)).astype(np.float32), y[:, :, 4:]], 2))
r = run7(pr, T, grp, act, "noisy"); rec("G7_noisy_rotation_predictor_FAILS_ratio_and_static", r["verdict"] == "FAIL" and not cond(r, "yaw_ratio<=1.10")[0]["passed"] and not cond(r, "static_hallucination")[0]["passed"], f"ratio={r['detail']['primary']['ratio']:.2f}, static={r['detail']['static_input_subset']['E_model']:.3f}")
def bias_on_static(b): return lambda sp, y: _bias(sp, y, b)
def _bias(sp, y, b):
    m = G.static_input_mask(T[sp]); y[m, 0, 3] += b; return y
add(pr, "halluc_big", "oracle", bias_on_static(0.02)); r = run7(pr, T, grp, act, "halluc_big"); c = cond(r, "static_hallucination")[0]; rec("G7_static_hallucination_0.02_rad_FAILS_static_condition", (not c["passed"]) and c["evaluable"] and abs(r["detail"]["static_input_subset"]["E_model"] - 0.02) < 1e-4, f"E={r['detail']['static_input_subset']['E_model']:.4f}")
add(pr, "halluc_small", "oracle", bias_on_static(0.005)); r = run7(pr, T, grp, act, "halluc_small"); rec("G7_static_hallucination_0.005_rad_passes_static_condition", cond(r, "static_hallucination")[0]["passed"], r["verdict"])
# ratio point estimate OK but upper bound too wide
rngb = np.random.default_rng(5); prw, Tw, gw, aw = fixture7(groups=dict(test_id=3, test_low_heldout=3, test_ood_low_strict=3, test_ood_high=3), win=40, seed=9)
for sp in SETS: Tw[sp]["Y_b"][:, :, 3] = 0; Tw[sp]["Y_b"][:, 0, 3] = 0.02 * np.where(rngb.random(len(gw[sp])) < 0.5, -1, 1); prw["oracle"][sp]["Y_b"] = Tw[sp]["Y_b"].copy(); prw["ridge"][sp]["Y_b"] = np.zeros_like(Tw[sp]["Y_b"]); Tw[sp]["B_hist"][:] = 0
cnt = 0
def wide(sp, y):
    global cnt
    for gi in np.unique(gw[sp]):
        if cnt % 2 == 1: y[gw[sp] == gi, 0, 3] += 0.042 * np.sign(y[gw[sp] == gi, 0, 3])
        cnt += 1
    return y
add(prw, "wide", "oracle", wide); r = run7(prw, Tw, gw, aw, "wide"); p_ = r["detail"]["primary"]
rec("G7_ratio_point_ok_but_bootstrap_upper_bound_fails", p_["ratio"] <= 1.10 and p_["ratio_ci"][1] > 1.25 and not cond(r, "upper_bound")[0]["passed"] and cond(r, "ratio<=1.10")[0]["passed"], f"ratio={p_['ratio']:.3f}, upper={p_['ratio_ci'][1]:.3f}")
prs, Ts, gs, as_ = fixture7(groups=dict(test_id=3, test_low_heldout=2, test_ood_low_strict=2, test_ood_high=2)); r = run7(prs, Ts, gs, as_, "oracle"); rec("G7_insufficient_groups_NOT_EVALUABLE", r["verdict"] == "NOT_EVALUABLE" and all(not c["evaluable"] for c in r["conditions"]), r["verdict"])
pr2, T2, g2, a2 = fixture7()                                      # S_in empty (all windows have contact) -> static condition NOT_EVALUABLE, ratio passes -> NOT_EVALUABLE (never PASS)
for sp in SETS: T2[sp]["C_hist"][:, :, 1] = 1
r = run7(pr2, T2, g2, a2, "oracle"); rec("G7_insufficient_static_support_NOT_EVALUABLE_not_PASS", r["verdict"] == "NOT_EVALUABLE" and not cond(r, "static")[0]["evaluable"] and cond(r, "ratio<=1.10")[0]["passed"], r["verdict"])
# event windows (|true step| > pi/2): excluded from gating, reported separately
pr3, T3, g3, a3 = fixture7(seed=2)
for sp in SETS:
    ev = np.zeros(len(g3[sp]), bool); ev[::25] = True; T3[sp]["Y_b"][ev, 3, 3] = 3.0; pr3["oracle"][sp]["Y_b"] = T3[sp]["Y_b"].copy()
r3 = run7(pr3, T3, g3, a3, "persistence"); d3 = r3["detail"]; n_all = sum(len(g3[sp]) for sp in SETS)
rec("G7_event_windows_excluded_from_gating_and_reported", d3["reported_event_windows"]["n_windows"] > 0 and d3["primary"]["n_windows"] == n_all - d3["reported_event_windows"]["n_windows"] and d3["reported_event_windows"]["E_K20"]["persistence"] > 2.0, f"events={d3['reported_event_windows']['n_windows']}")
rec("G7_all_window_ratio_reported_next_to_gated", d3["reported_all_windows_K20"]["ratio"] is not None and set(d3["reported_all_windows_K20"]["E_by_baseline"]) == {"persistence", "ridge"} and abs(d3["primary"]["ratio"] - 1.0) < 1e-6)
ev_only = G.event_mask(dict(Y_b=np.array([[[0, 0, 0, 3.1] + [0] * 3] + [[0] * 7] * 19], np.float32))); rec("G7_pi_discontinuity_step_flagged_as_event", bool(ev_only[0]))
# tilt report-only; gate independent of tilt channel
pr4 = copy.deepcopy(pr); 
for sp in SETS: pr4["oracle"][sp]["Y_b"][:, :, 5] = 123.0
rec("G7_gate_ignores_tilt_channel", json.dumps(run7(pr, T, grp, act, "oracle")["conditions"], default=str) == json.dumps(run7(pr4, T, grp, act, "oracle")["conditions"], default=str) and "tilt" in G.CONFIG["G7"] and G.CONFIG["G7"]["tilt"]["reported_only"])
tr = G.tilt_report(pr4, T, grp, act, "oracle"); rec("G7_tilt_report_runs_no_verdict", "oracle" in tr and tr["oracle"] > 10)
# static subset is input-only
T5 = copy.deepcopy(T); m0 = {sp: G.static_input_mask(T[sp]) for sp in SETS}
for sp in SETS: T5[sp]["Y_b"] = np.random.default_rng(1).standard_normal(T5[sp]["Y_b"].shape).astype(np.float32)
rec("G7_static_subset_independent_of_future_ground_truth", all(np.array_equal(m0[sp], G.static_input_mask(T5[sp])) for sp in SETS))
rec("G7_horizons_1_5_10_20_reported", set(run7(pr, T, grp, act, "oracle")["detail"]["reported_horizons"]) == {"1", "5", "10", "20"})
rec("G7_diagnostic_S_gt_flagged_as_using_future_truth", run7(pr, T, grp, act, "oracle")["detail"]["diagnostic_S_gt"]["uses_future_truth"] is True)
rec("G7_deterministic", json.dumps(run7(pr, T, grp, act, "noisy"), default=str) == json.dumps(run7(pr, T, grp, act, "noisy"), default=str))
rec("CONFIG_declares_draft_not_frozen_and_not_applied_to_I1", "NOT FROZEN" in G.CONFIG["status"] and any("I-1" in s for s in G.CONFIG["not_applied_to"]))
rec("CONFIG_inherits_v2_hash_and_sample_sizes", G.CONFIG["inherits"]["gates_v2_config_sha256"] == "f6c323914a99b4bc884ffc88c4c538615ad3a1ba10649fd95e38282acbaa0627" and G.CONFIG["min_sample"] == V.CONFIG["min_sample"])
(ROOT / "outputs/gates_v3").mkdir(parents=True, exist_ok=True); json.dump(dict(results=RES, config_sha256=G.config_sha(), n=len(RES), all_pass=all(RES.values())), open(ROOT / "outputs/gates_v3/unit_tests.json", "w"), indent=1); print("ALL PASS" if all(RES.values()) else "SOME FAILED", len(RES))
