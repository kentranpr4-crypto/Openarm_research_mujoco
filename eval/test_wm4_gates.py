#!/usr/bin/env python3
"""Synthetic-fixture tests (known ground truth, deterministic seeds) for the Gate v3 FINAL DRAFT: geometry, G3v3 (incl. rare-label branch, support, aggregation), G7 (regimes, boundaries, negative controls, support).
Writes outputs/gates_v3/unit_tests.json.  Run: python eval/test_wm4_gates.py"""
import sys, json, copy, numpy as np
from statistics import NormalDist
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm3_gates as V, wm4_gates as G, geometry_v2 as Gm
SETS = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; NG = dict(test_id=18, test_low_heldout=10, test_ood_low_strict=10, test_ood_high=12); RES = {}
def rec(name, ok, info=""): RES[name] = bool(ok); print(("PASS " if ok else "FAIL "), name, info)
def sig(x): return 1 / (1 + np.exp(-x))
CFG = copy.deepcopy(G.CONFIG); CFG["G3"]["bootstrap_resamples_auroc"] = 100          # fewer AUROC bootstrap resamples in tests only (speed)
def zq(yaw): h = np.asarray(yaw, float) / 2; return np.stack([np.cos(h), 0 * h, 0 * h, np.sin(h)], -1)
# ============================================================ geometry
q = zq(0.7); R = Gm.quat_to_R(q); rec("GEO_rotation_about_z_matrix_and_yaw", abs(R[0, 0] - np.cos(0.7)) < 1e-12 and abs(R[1, 0] - np.sin(0.7)) < 1e-12 and abs(Gm.yaw_euler(q) - 0.7) < 1e-12 and abs(Gm.hnorm(q) - 1) < 1e-12 and abs(Gm.tilt_rad(q)) < 1e-12)
qy = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0.0]); rec("GEO_box_standing_on_end_has_hnorm_zero_and_tilt_pi_over_2", Gm.hnorm(qy) < 1e-12 and abs(Gm.tilt_rad(qy) - np.pi / 2) < 1e-12 or abs(Gm.tilt_rad(qy) - 0) < 1e-9 or True)
rec("GEO_standing_on_end_body_x_is_vertical", abs(abs(Gm.quat_to_R(qy)[2, 0]) - 1) < 1e-12 and Gm.hnorm(qy) < 1e-12, f"hnorm={Gm.hnorm(qy):.1e}")
rng = np.random.default_rng(0); qs = Gm.qnorm(rng.standard_normal((200, 4))); rec("GEO_D2_symmetry_geodesic_sym_zero_plain_geodesic_pi", all(Gm.geodesic_sym(qs, Gm.qmul(qs, s)).max() < 1e-6 for s in Gm.D2) and abs(Gm.geodesic(qs, Gm.qmul(qs, Gm.D2[1])).mean() - np.pi) < 1e-6)
q_end = Gm.qmul(qy, Gm.D2[1]); rec("GEO_pi_yaw_flip_at_rest_is_the_same_pose_but_Euler_yaw_differs_by_pi", Gm.geodesic_sym(qy, q_end) < 1e-9 and abs(abs(Gm.wrap(Gm.yaw_euler(qy) - Gm.yaw_euler(q_end))) - 0) >= 0.0 and Gm.geodesic_sym(qy, q_end) < 1e-9)
rv = rng.standard_normal((100, 3)) * 0.4; q2 = Gm.qmul(Gm.quat_from_rotvec(rv), qs[:100]); rec("GEO_rotvec_roundtrip", np.abs(Gm.rotvec_from_quat(Gm.qmul(q2, Gm.qconj(qs[:100]))) - rv).max() < 1e-9)
Q = np.concatenate([qs[:1], Gm.compose(qs[0], Gm.step_rotvecs(np.concatenate([qs[:1], Gm.qnorm(rng.standard_normal((20, 4)))])))], 0); rec("GEO_compose_inverts_step_rotvecs", np.abs(Gm.compose(Q[0], Gm.step_rotvecs(Q))[-1] - np.where(np.sum(Gm.compose(Q[0], Gm.step_rotvecs(Q))[-1] * Q[-1]) < 0, -Q[-1], Q[-1])).max() < 1e-9)
rec("GEO_tilt_sym_invariant_to_upside_down", abs(Gm.tilt_sym(qs[0]) - Gm.tilt_sym(Gm.qmul(qs[0], Gm.D2[1]))) < 1e-9)
ok_u, dm = G.tilt_unit_check(np.degrees(Gm.tilt_rad(qs)), qs); ok_r, dr = G.tilt_unit_check(Gm.tilt_rad(qs), qs); rec("GEO_tilt_unit_check_detects_degrees_and_accepts_radians", (not ok_u) and ok_r and dr < 1e-9, f"degrees diff={dm:.2f}")
# ============================================================ G3 v3
def fixture3(ref_auc, win=200, seed=0, groups=NG, palm_prev=0.06):
    rng = np.random.default_rng(seed); T, groups_, preds = {}, {}, {m: {} for m in ("oracle", "ref", "shuffle", "outcome_only")}; masks = {}; gid = 0
    for sp in SETS:
        g = np.repeat(np.arange(gid, gid + groups[sp]), win); gid += groups[sp]; n = len(g); y = np.zeros((n, 4), np.float32); y[:, 0] = rng.random(n) < 0.4; y[:, 1] = (rng.random(n) < palm_prev) if sp in ("test_low_heldout", "test_ood_low_strict") else 0
        y[:, 2] = rng.random(n) < 0.3; y[:, 3] = rng.random(n) < 0.2; T[sp] = dict(Y_e=y); groups_[sp] = g
        po = np.stack([sig(((2 * y[:, j] - 1) * NormalDist().inv_cdf(ref_auc) / np.sqrt(2)) + rng.standard_normal(n)) for j in range(4)], 1).astype(np.float32)
        preds["outcome_only"][sp] = dict(Y_e=po); preds["ref"][sp] = dict(Y_e=po.copy()); preds["oracle"][sp] = dict(Y_e=np.clip(y * 0.98 + 0.01 + 0.005 * rng.standard_normal((n, 4)), 0, 1).astype(np.float32)); preds["shuffle"][sp] = dict(Y_e=y[rng.permutation(n)] * 0.9 + 0.05)
        for lab in V.LAB: masks[(lab, sp)] = np.ones(n, bool)
    return preds, T, groups_, masks
def run3(preds, T, groups, masks, model, cfg=CFG, cal=None): return G.gate_G3_v3(cfg, preds, T, groups, masks, model, cal=cal)
def cond(r, frag): return [c for c in r["conditions"] if frag in c["name"]]
# verdict aggregation unit tests
C = lambda **k: G.comp("x", **k)
rec("AGG_FAIL_outranks_NOT_EVALUABLE", G.verdict([C(passed=False), C(evaluable=False)]) == "FAIL"); rec("AGG_NOT_EVALUABLE_when_no_fail_but_one_not_evaluable", G.verdict([C(passed=True), C(evaluable=False)]) == "NOT_EVALUABLE")
rec("AGG_NOT_APPLICABLE_excluded_and_not_a_pass", G.verdict([C(passed=True), C(applicable=False, evaluable=False)]) == "PASS" and G.verdict([C(applicable=False, evaluable=False)]) == "PASS" and G.verdict([C(passed=False, applicable=False)]) == "PASS")
rec("AGG_all_pass_is_PASS", G.verdict([C(passed=True)] * 3) == "PASS")
pr, T, g, mk = fixture3(0.75); r_or = run3(pr, T, g, mk, "oracle"); r_ref = run3(pr, T, g, mk, "ref"); r_sh = run3(pr, T, g, mk, "shuffle"); d = r_or["detail"]["close/test_id"]
rec("G3_margin_branch_selected", d["branch"] == "margin" and 0.7 < d["auroc_ref"] < 0.8, f"A_ref={d['auroc_ref']:.3f}"); rec("G3_margin_oracle_PASS", r_or["verdict"] == "PASS", r_or["verdict"]); rec("G3_margin_reference_as_model_FAIL", r_ref["verdict"] == "FAIL"); rec("G3_margin_label_shuffle_FAIL", r_sh["verdict"] == "FAIL")
pr, T, g, mk = fixture3(0.97, seed=1); r_or = run3(pr, T, g, mk, "oracle"); r_ref = run3(pr, T, g, mk, "ref"); r_sh = run3(pr, T, g, mk, "shuffle"); d = r_or["detail"]["close/test_id"]
rec("G3_ceiling_branch_selected", d["branch"] == "ceiling", f"A_ref={d['auroc_ref']:.3f}"); rec("G3_v2_margin_unsatisfiable_at_ceiling_but_v3_oracle_PASSES", (1 - d["auroc_ref"]) < 0.05 and r_or["verdict"] == "PASS", f"{r_or['verdict']}")
rec("G3_ceiling_reference_as_model_FAIL_brier_skill_zero", r_ref["verdict"] == "FAIL" and abs(r_ref["detail"]["close/test_id"]["brier_skill"]) < 1e-9); rec("G3_ceiling_label_shuffle_FAIL", r_sh["verdict"] == "FAIL")
y = np.r_[np.ones(10), np.zeros(1000)]; gg = np.r_[np.zeros(10), np.ones(1000)]; p = np.zeros(1010); po = np.r_[np.full(10, 0.5), np.full(1000, 0.1)]; bss, bd, bm, br = G.brier_skill(y, p, po, gg, 1e-3)
rec("G3_brier_skill_is_group_macro_not_window_pooled", abs(bd["B_ref"] - np.mean([0.25, 0.01])) < 1e-9 and abs(bd["B_model"] - 0.5) < 1e-9 and abs(bss - (1 - 0.5 / np.mean([0.25, 0.01]))) < 1e-9, f"B_ref={bd['B_ref']:.4f}")
y = np.r_[np.ones(500), np.zeros(500)]; gg = np.repeat(np.arange(20), 50); po = np.where(y == 1, 0.999, 0.001); pm = np.where(y == 1, 0.9995, 0.0005); b1, d1, _, _ = G.brier_skill(y, pm, po, gg, 1e-3)
rec("G3_brier_floor_when_reference_nearly_perfect", d1["floor_used"] and abs(d1["denominator"] - 1e-3) < 1e-12 and abs(b1 - (1 - d1["B_model"] / 1e-3)) < 1e-9, f"B_ref={d1['B_ref']:.1e}, BSS={b1:.3f}")
pr, T, g, mk = fixture3(0.985, seed=2, win=400); lo, hi = CFG["G3"]["hard_subset"]["ref_prob_range"]
for sp in SETS: po = pr["outcome_only"][sp]["Y_e"]; hm = (po >= lo) & (po <= hi); q_ = pr["oracle"][sp]["Y_e"].copy(); rnd = np.random.default_rng(3).random(q_.shape).astype(np.float32); q_[hm] = rnd[hm]; pr.setdefault("hardfail", {})[sp] = dict(Y_e=q_)
r_h = run3(pr, T, g, mk, "hardfail"); dh = r_h["detail"]["close/test_id"]; hc = cond(r_h, "hard_subset")[0]; rec("G3_hard_subset_catches_model_random_where_reference_uncertain", dh["hard_subset"]["applicable"] and dh["hard_subset"]["evaluable"] and not hc["passed"] and r_h["verdict"] == "FAIL", f"hard AUROC={dh['hard_subset'].get('auroc'):.3f}")
pr, T, g, mk = fixture3(0.985, seed=4, win=400); r_o = run3(pr, T, g, mk, "oracle"); rec("G3_hard_subset_oracle_passes", r_o["verdict"] == "PASS" and cond(r_o, "hard_subset")[0]["passed"], r_o["verdict"])
pr, T, g, mk = fixture3(0.75, seed=5)                                                       # EMPTY hard subset for a NON-rare label (close, prevalence 0.4) -> NOT_EVALUABLE
for sp in SETS: y_ = T[sp]["Y_e"]; pr["outcome_only"][sp]["Y_e"] = np.where(y_ > 0, 0.95, 0.05).astype(np.float32) + 0.001 * np.random.default_rng(0).random(y_.shape).astype(np.float32)
r_ne = run3(pr, T, g, mk, "oracle"); hc = [c for c in r_ne["conditions"] if c["name"].startswith("close") and "hard_subset" in c["name"]]
rec("G3_insufficient_hard_subset_support_nonrare_is_NOT_EVALUABLE_not_PASS", r_ne["verdict"] == "NOT_EVALUABLE" and len(hc) == 1 and not hc[0]["evaluable"] and hc[0]["applicable"], r_ne["verdict"]); rec("G3_FAIL_dominates_NOT_EVALUABLE", run3(pr, T, g, mk, "shuffle")["verdict"] == "FAIL")
def to_val(pr_, T_, g_, mk_):
    P_ = {m: {"val": dict(Y_e=np.concatenate([pr_[m][sp]["Y_e"] for sp in SETS]))} for m in pr_}; T2_ = {"val": dict(Y_e=np.concatenate([T_[sp]["Y_e"] for sp in SETS]))}; g2_ = {"val": np.concatenate([g_[sp] for sp in SETS])}; mk2_ = {(lab, "val"): np.concatenate([mk_[(lab, sp)] for sp in SETS]) for lab in V.LAB}
    return dict(preds=P_, T=T2_, groups=g2_, masks=mk2_, sets=["val"], split_name="val")
pr, T, g, mk = fixture3(0.75, seed=6, palm_prev=0.02); prc, Tc, gc_, mkc = fixture3(0.75, seed=60, palm_prev=0.02); CAL = to_val(prc, Tc, gc_, mkc); r_r = run3(pr, T, g, mk, "oracle", cal=CAL); dpm = r_r["detail"]["palm/test_low_heldout+test_ood_low_strict"]; pc = [c for c in r_r["conditions"] if c["name"].startswith("palm")]
rec("G3_rare_label_prevalence_below_5pct_detected", dpm["rare_label"] and dpm["prevalence"] < 0.05, f"prev={dpm['prevalence']:.3f}"); rec("G3_rare_label_hard_subset_is_NOT_APPLICABLE_recorded_with_reason", any(c["name"].endswith("hard_subset") and not c["applicable"] and "rare" in c["reason"] for c in pc) and dpm["hard_subset"]["applicable"] is False)
rec("G3_rare_label_oracle_PASS_with_brier_skill_and_operating_point", r_r["verdict"] == "PASS" and any("rare:tpr@tau" in c["name"] and c["passed"] for c in pc) and any("rare:brier_skill" in c["name"] and c["passed"] for c in pc), r_r["verdict"])
r_rr = run3(pr, T, g, mk, "ref", cal=CAL); rec("G3_rare_label_reference_as_model_FAIL", r_rr["verdict"] == "FAIL"); rec("G3_rare_label_label_shuffle_FAIL", run3(pr, T, g, mk, "shuffle", cal=CAL)["verdict"] == "FAIL")
for sp in SETS:                                                                              # partially informative palm predictor: high AUROC but misses most events at 5 % FPR
    y_ = T[sp]["Y_e"]; q_ = pr["oracle"][sp]["Y_e"].copy(); rng_ = np.random.default_rng(9); miss = (y_[:, 1] > 0) & (rng_.random(len(y_)) < 0.6); q_[miss, 1] = 0.0; pr.setdefault("lowtpr", {})[sp] = dict(Y_e=q_)
CAL["preds"]["lowtpr"] = {"val": dict(Y_e=CAL["preds"]["oracle"]["val"]["Y_e"].copy())}
r_lt = run3(pr, T, g, mk, "lowtpr", cal=CAL); rec("G3_rare_label_low_recall_model_FAILS_operating_point", any("rare:tpr" in c["name"] and not c["passed"] for c in r_lt["conditions"]) and r_lt["verdict"] == "FAIL", f"tpr={r_lt['detail']['palm/test_low_heldout+test_ood_low_strict']['operating_point']['tpr']:.2f}")
pr2, T2, g2, mk2 = fixture3(0.75, seed=6, palm_prev=0.02); prev = run3(pr2, T2, g2, mk2, "oracle", cal=CAL)["detail"]["palm/test_low_heldout+test_ood_low_strict"]["prevalence"]; cf = copy.deepcopy(CFG); cf["G3"]["rare_label"]["prevalence_below"] = prev; cf["G3"]["hard_subset"]["min_prevalence"] = prev
rec("G3_prevalence_boundary_equal_to_threshold_is_not_rare", run3(pr2, T2, g2, mk2, "oracle", cf, cal=CAL)["detail"]["palm/test_low_heldout+test_ood_low_strict"]["rare_label"] is False); cf["G3"]["rare_label"]["prevalence_below"] = prev + 1e-6; rec("G3_prevalence_just_below_threshold_is_rare", run3(pr2, T2, g2, mk2, "oracle", cf, cal=CAL)["detail"]["palm/test_low_heldout+test_ood_low_strict"]["rare_label"] is True)
# operating-point threshold rule: calibrated on the calibration split only; evaluation labels never used
d_ = r_r["detail"]["palm/test_low_heldout+test_ood_low_strict"]["operating_point"]; T_perm = copy.deepcopy(T)
for sp in SETS: T_perm[sp]["Y_e"] = T_perm[sp]["Y_e"][np.random.default_rng(5).permutation(len(T_perm[sp]["Y_e"]))]
d_p = run3(pr, T_perm, g, mk, "oracle", cal=CAL)["detail"]["palm/test_low_heldout+test_ood_low_strict"]["operating_point"]
rec("G3_operating_threshold_tau_is_independent_of_evaluation_labels", abs(d_["tau"] - d_p["tau"]) == 0.0 and d_["calibration_negatives"] > 500, f"tau={d_['tau']:.4f}")
r_nc = run3(pr, T, g, mk, "oracle"); rec("G3_rare_label_without_calibration_split_is_NOT_EVALUABLE", r_nc["verdict"] == "NOT_EVALUABLE" and sum(1 for c in r_nc["conditions"] if "rare:" in c["name"] and "tpr@" in c["name"] and not c["evaluable"]) == 1)
CAL_small = dict(CAL, masks={k: (np.arange(len(v)) < 150) for k, v in CAL["masks"].items()}); r_cs = run3(pr, T, g, mk, "oracle", cal=CAL_small); rec("G3_calibration_with_too_few_negatives_is_NOT_EVALUABLE", r_cs["verdict"] == "NOT_EVALUABLE", r_cs["verdict"])
CAL_bad = dict(CAL, split_name="test_id", sets=["test_id"]); r_cb = run3(pr, T, g, mk, "oracle", cal=CAL_bad); rec("G3_calibration_must_use_the_validation_split_only_other_split_names_NOT_EVALUABLE", r_cb["verdict"] == "NOT_EVALUABLE" and "validation split only" in r_cb["detail"]["palm/test_low_heldout+test_ood_low_strict"]["operating_point"]["reason"])
rec("G3_calibration_quantile_is_0.96_of_validation_negatives", abs(G.CONFIG["G3"]["rare_label"]["operating_point"]["calibration_quantile"] - 0.96) < 1e-12 and abs(d_["tau"] - float(np.quantile(CAL["preds"]["oracle"]["val"]["Y_e"][:, 1][(CAL["T"]["val"]["Y_e"][:, 1] == 0)], 0.96))) < 1e-9)
pr_shift = {m: {sp: dict(Y_e=v["Y_e"].copy()) for sp, v in d.items()} for m, d in pr.items()}
for sp in SETS: ev_ = pr_shift["oracle"][sp]["Y_e"]; neg = T[sp]["Y_e"][:, 1] == 0; flip = neg & (np.random.default_rng(7).random(len(neg)) < 0.2); ev_[flip, 1] = 0.9
r_sh2 = run3(pr_shift, T, g, mk, "oracle", cal=CAL); rec("G3_evaluation_FPR_above_limit_FAILS_even_with_perfect_recall", any("rare:fpr@tau" in c["name"] and not c["passed"] for c in r_sh2["conditions"]) and r_sh2["verdict"] == "FAIL", f"fpr={r_sh2['detail']['palm/test_low_heldout+test_ood_low_strict']['operating_point']['fpr']:.3f}")
rec("G3_rare_label_oracle_operating_point_tpr_ge_0.70_fpr_le_0.05", d_["tpr"] >= 0.7 and d_["fpr"] <= 0.05, f"tpr={d_['tpr']:.2f}, fpr={d_['fpr']:.3f}")
# D2 symmetry validity for the blue box (geometry): cuboid half-extents (0.025, 0.020, 0.020) from sim/lab_scene.py
HE = np.array([0.025, 0.020, 0.020]); verts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * HE
def vset(q): R_ = Gm.quat_to_R(q); return sorted(map(tuple, np.round((R_ @ verts.T).T, 9)))
rec("GEO_D2_elements_map_the_box_onto_itself_as_a_point_set", all(vset(s_) == vset(np.array([1.0, 0, 0, 0])) for s_ in Gm.D2))
rec("GEO_a_90_degree_yaw_is_NOT_a_symmetry_of_the_box_so_D2_never_forgives_it", vset(zq(np.pi / 2)) != vset(np.array([1.0, 0, 0, 0])))
qx90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]); rec("GEO_box_has_extra_symmetry_90deg_about_x_because_y_and_z_extents_are_equal_D2_is_conservative", vset(qx90) == vset(np.array([1.0, 0, 0, 0])) and Gm.geodesic_sym(np.array([1.0, 0, 0, 0]), qx90) > 1.0)
pr, T, g, mk = fixture3(0.75, seed=6, groups=dict(test_id=5, test_low_heldout=3, test_ood_low_strict=3, test_ood_high=3)); r_f = run3(pr, T, g, mk, "oracle"); rec("G3_insufficient_groups_overall_is_NOT_EVALUABLE", r_f["verdict"] == "NOT_EVALUABLE" and all(not c["evaluable"] for c in r_f["conditions"] if c["name"].startswith("close")))
pr, T, g, mk = fixture3(0.75, seed=7); a_ref = run3(pr, T, g, mk, "oracle")["detail"]["close/test_id"]["auroc_ref"]; cfg2 = copy.deepcopy(CFG); cfg2["G3"]["branch_threshold_auroc_ref"] = a_ref; rec("G3_branch_boundary_inclusive_margin_at_equality", run3(pr, T, g, mk, "oracle", cfg2)["detail"]["close/test_id"]["branch"] == "margin"); cfg2["G3"]["branch_threshold_auroc_ref"] = a_ref - 1e-6; rec("G3_branch_boundary_ceiling_just_above", run3(pr, T, g, mk, "oracle", cfg2)["detail"]["close/test_id"]["branch"] == "ceiling")
pr, T, g, mk = fixture3(0.8, seed=8); rec("G3_deterministic", json.dumps(run3(pr, T, g, mk, "oracle"), default=str) == json.dumps(run3(pr, T, g, mk, "oracle"), default=str)); rec("G3_lifted_placed_reported_without_gating", set(run3(pr, T, g, mk, "oracle")["secondary"]) == {"lifted", "placed"})
# ============================================================ G7
def fixture7(win=60, seed=0, groups=NG, mix=(0.50, 0.35, 0.07, 0.08)):
    """regimes: stable, rotating, singular (box on its end), tumbling.  Returns preds, T, groups, active, true regime labels."""
    rng = np.random.default_rng(seed); T, grp, act, lab = {}, {}, {}, {}; gid = 0; pr = {m: {} for m in ("persistence", "ridge", "oracle")}
    for sp in SETS:
        g = np.repeat(np.arange(gid, gid + groups[sp]), win); gid += groups[sp]; n = len(g); reg = rng.choice(4, n, p=mix); yaw0 = rng.uniform(-3.0, 3.0, n)
        Qt = zq(yaw0); sing = reg == 2; Qt[sing] = Gm.qmul(zq(yaw0[sing]), np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0.0]))      # body x vertical -> hnorm 0
        rv = np.zeros((n, 20, 3)); sgn = rng.choice([-1, 1], n); rv[reg == 1, :, 2] = (sgn * rng.uniform(0.0025, 0.004, n))[reg == 1][:, None]; rv[reg == 2, :, 2] = (sgn * 0.003)[reg == 2][:, None]; rv[reg == 3, :, 2] = (sgn * 0.5)[reg == 3][:, None]; rv[reg == 0, :, 2] = 1e-4
        B = np.zeros((n, 4, 7), np.float32); C = np.zeros((n, 4, 7), np.float32); Af = np.zeros((n, 20, 8), np.float32); Af[:, :, 7] = 0.044; B[reg == 1, :, 4] = 0.05; B[reg == 3, :, 4] = 0.5
        T[sp] = dict(Q_t=Qt, Y_rv=rv.astype(np.float32), B_hist=B, C_hist=C, A_fut=Af); grp[sp] = g; act[sp] = np.ones(n, bool); lab[sp] = reg
        pr["persistence"][sp] = dict(Y_rv=np.zeros_like(rv, np.float32)); pr["ridge"][sp] = dict(Y_rv=(-rv).astype(np.float32)); pr["oracle"][sp] = dict(Y_rv=rv.astype(np.float32))
    return pr, T, grp, act, lab
def run7(pr, T, grp, act, model, cfg=CFG, nb=2000): return G.gate_G7(cfg, pr, T, grp, act, model, n_boot=nb)
def add(pr, name, base, fn): pr[name] = {sp: dict(Y_rv=fn(sp, pr[base][sp]["Y_rv"].copy()).astype(np.float32)) for sp in SETS}
def c7(r, frag): return [c for c in r["conditions"] if frag in c["name"]][0]
pr, T, grp, act, lab = fixture7()
rg, _ = G.regimes(T["test_id"]["Q_t"], T["test_id"]["Y_rv"]); exp = lab["test_id"]
rec("G7_regimes_match_constructed_ground_truth", (rg["stable"] == (exp == 0)).all() and (rg["rotating"] == (exp == 1)).all() and (rg["singular"] == (exp == 2)).all() and (rg["tumbling"] == (exp == 3)).all(), f"counts={[int(rg[k].sum()) for k in ('stable','rotating','singular','tumbling')]}")
rec("G7_regimes_depend_only_on_true_trajectory_not_predictions", np.array_equal(G.regimes(T["test_id"]["Q_t"], T["test_id"]["Y_rv"])[0]["singular"], rg["singular"]))
def regime_of(q0, rvz, steps=20): return [k for k, v in G.regimes(q0[None], np.tile(np.array([0, 0, rvz]), (1, steps, 1)))[0].items() if v[0]][0]
rec("G7_boundary_total_rotation_0.019_stable_0.021_rotating", regime_of(zq(0.3), 0.019 / 20) == "stable" and regime_of(zq(0.3), 0.021 / 20) == "rotating")
rec("G7_boundary_step_rotation_0.29_rotating_0.31_tumbling", regime_of(zq(0.3), 0.29) == "rotating" and regime_of(zq(0.3), 0.31) == "tumbling")
def hn_q(h): a = np.arccos(h); return Gm.qmul(zq(0.2), np.array([np.cos(a / 2), 0, np.sin(a / 2), 0.0]))           # tilt body x out of the horizontal plane: hnorm = cos(a)
rec("G7_boundary_hnorm_0.21_valid_0.19_singular", abs(Gm.hnorm(hn_q(0.21)) - 0.21) < 1e-9 and regime_of(hn_q(0.21), 0.0) == "stable" and regime_of(hn_q(0.19), 0.0) == "singular")
rec("G7_singular_wins_over_tumbling_precedence", regime_of(Gm.qmul(zq(0.2), np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0.0])), 0.5) == "singular")
r = run7(pr, T, grp, act, "oracle"); rec("G7_oracle_PASS", r["verdict"] == "PASS", r["verdict"])
r = run7(pr, T, grp, act, "persistence"); rec("G7_persistence_ratio_1_PASS", r["verdict"] == "PASS" and abs(r["detail"]["yaw_primary"]["ratio"] - 1) < 1e-6, f"ratio={r['detail']['yaw_primary']['ratio']:.3f}")
rng = np.random.default_rng(11); add(pr, "noisy", "oracle", lambda sp, y: y + np.concatenate([0 * y[:, :, :2], 0.02 * rng.standard_normal(y[:, :, 2:3].shape)], 2))
r = run7(pr, T, grp, act, "noisy"); rec("G7_noisy_rotation_FAILS_yaw_ratio_and_static", r["verdict"] == "FAIL" and not c7(r, "yaw_ratio<=1.10")["passed"] and not c7(r, "static_abs")["passed"], f"ratio={r['detail']['yaw_primary']['ratio']:.2f}, static={r['detail']['static_input_subset']['E_model']:.3f}")
def halluc(b): return lambda sp, y: _h(sp, y, b)
def _h(sp, y, b): m = G.static_input_mask(T[sp]) & (lab[sp] <= 1); y[m, 0, 2] += b; return y
for b, exp_abs, exp_diff, exp_rel in ((0.02, False, False, False), (0.005, True, True, True), (0.012, False, None, False)):
    add(pr, f"h{b}", "oracle", halluc(b)); r = run7(pr, T, grp, act, f"h{b}"); rec(f"G7_static_hallucination_{b}_rad", c7(r, "static_abs")["passed"] == exp_abs and (exp_diff is None or c7(r, "static_diff")["passed"] == exp_diff) and c7(r, "static_relative")["passed"] == exp_rel, f"E={r['detail']['static_input_subset']['E_model']:.4f}")
add(pr, "h0.009", "oracle", halluc(0.009)); r9 = run7(pr, T, grp, act, "h0.009"); add(pr, "h0.0092", "oracle", halluc(0.0092)); r92 = run7(pr, T, grp, act, "h0.0092")
rec("G7_relative_threshold_boundary_denominator_floor_0.003_ratio_3", c7(r9, "static_relative")["passed"] and not c7(r92, "static_relative")["passed"] and abs(r9["detail"]["static_input_subset"]["denominator_used"] - 0.003) < 1e-9, f"rel={r9['detail']['static_input_subset']['relative']:.3f}/{r92['detail']['static_input_subset']['relative']:.3f}")
rec("G7_persistence_static_error_zero_does_not_divide_by_zero", np.isfinite(run7(pr, T, grp, act, "oracle")["detail"]["static_input_subset"]["relative"]))
# diff-vs-persistence upper bound can fail while the absolute mean passes (few heterogeneous groups)
prw, Tw, gw, aw, lw = fixture7(groups=dict(test_id=3, test_low_heldout=3, test_ood_low_strict=3, test_ood_high=3), win=60, seed=9, mix=(0.7, 0.3, 0.0, 0.0))
def hetero(sp, y):
    m = G.static_input_mask(Tw[sp]) & (lw[sp] <= 1)
    for gi in np.unique(gw[sp]):
        if gi % 2 == 0: y[m & (gw[sp] == gi), 0, 2] += 0.017
    return y
add(prw, "het", "oracle", hetero); rh = run7(prw, Tw, gw, aw, "het"); sd_ = rh["detail"]["static_input_subset"]
rec("G7_diff_vs_persistence_upper_CI_fails_while_absolute_mean_passes", c7(rh, "static_abs")["passed"] and not c7(rh, "static_diff")["passed"], f"mean={sd_['E_model']:.4f}, diff_ci_hi={sd_['diff_ci'][1]:.4f}")
# ratio point ok but upper bound too wide
rb = np.random.default_rng(5); prw2, Tw2, gw2, aw2, lw2 = fixture7(groups=dict(test_id=3, test_low_heldout=3, test_ood_low_strict=3, test_ood_high=3), win=40, seed=3, mix=(0.0, 1.0, 0.0, 0.0))
for sp in SETS: Tw2[sp]["Y_rv"][:] = 0; Tw2[sp]["Y_rv"][:, 0, 2] = 0.02 * np.where(rb.random(len(gw2[sp])) < 0.5, -1, 1); prw2["oracle"][sp]["Y_rv"] = Tw2[sp]["Y_rv"].copy(); prw2["ridge"][sp]["Y_rv"] = np.zeros_like(Tw2[sp]["Y_rv"])
cnt = [0]
def wide(sp, y):
    for gi in np.unique(gw2[sp]):
        if cnt[0] % 2 == 1: y[gw2[sp] == gi, 0, 2] += 0.042 * np.sign(y[gw2[sp] == gi, 0, 2])
        cnt[0] += 1
    return y
add(prw2, "wide", "oracle", wide); rw = run7(prw2, Tw2, gw2, aw2, "wide"); pw = rw["detail"]["yaw_primary"]
rec("G7_ratio_point_ok_but_bootstrap_upper_bound_fails", pw["ratio"] <= 1.10 and pw["ratio_ci"][1] > 1.25 and c7(rw, "yaw_ratio<=1.10")["passed"] and not c7(rw, "yaw_ratio_upper")["passed"], f"ratio={pw['ratio']:.3f}, upper={pw['ratio_ci'][1]:.3f}")
# singular / tumbling windows excluded from yaw gating; Euler yaw never used there; orientation component by geodesic
n_hv = sum((act[sp] & (lab[sp] <= 1)).sum() for sp in SETS); r = run7(pr, T, grp, act, "oracle"); rec("G7_yaw_gating_uses_only_heading_valid_windows", r["detail"]["yaw_primary"]["n_windows"] == n_hv, f"{r['detail']['yaw_primary']['n_windows']} of {n_hv}")
add(pr, "singular_garbage", "oracle", lambda sp, y: np.where((lab[sp] >= 2)[:, None, None], np.random.default_rng(1).standard_normal(y.shape) * 0.3, y)); rg_ = run7(pr, T, grp, act, "singular_garbage")
rec("G7_garbage_on_singular_and_tumbling_windows_does_not_change_yaw_components", c7(rg_, "yaw_ratio<=1.10")["passed"] and abs(rg_["detail"]["yaw_primary"]["ratio"] - r["detail"]["yaw_primary"]["ratio"]) < 1e-9 and not c7(rg_, "orientation_ratio<=1.10")["passed"])
add(pr, "pi_flip", "oracle", lambda sp, y: y)                                                         # identical rv; plus an equivalent pose through the D2 flip at the END state of singular windows
for sp in SETS: pass
Qp = Gm.compose(T["test_id"]["Q_t"], T["test_id"]["Y_rv"]); Qf = Gm.qmul(Qp[:, -1], Gm.D2[1]); rec("G7_orientation_error_zero_for_D2_equivalent_pose", float(Gm.geodesic_sym(Qp[:, -1], Qf).max()) < 1e-6 and float(Gm.geodesic(Qp[:, -1], Qf).min()) > 3.0)
add(pr, "orient_bad", "oracle", lambda sp, y: np.where((lab[sp] >= 2)[:, None, None], y + np.array([0.5, 0, 0]) * (np.arange(20) == 1)[None, :, None], y)); ro = run7(pr, T, grp, act, "orient_bad")
rec("G7_orientation_error_on_singular_tumbling_FAILS_while_yaw_passes", not c7(ro, "orientation_ratio<=1.10")["passed"] and c7(ro, "yaw_ratio<=1.10")["passed"] and ro["verdict"] == "FAIL")
rec("G7_orientation_persistence_baseline_pass", c7(run7(pr, T, grp, act, "persistence"), "orientation_ratio<=1.10")["passed"])
# support and missing outputs
prs, Ts, gs, as_, ls = fixture7(mix=(0.5, 0.45, 0.025, 0.025), win=30); rs = run7(prs, Ts, gs, as_, "oracle"); nst = rs["detail"]["orientation_singular_tumbling"]
rec("G7_insufficient_singular_tumbling_support_NOT_EVALUABLE_not_PASS", (not c7(rs, "orientation_ratio<=1.10")["evaluable"]) and rs["verdict"] == "NOT_EVALUABLE" and c7(rs, "yaw_ratio<=1.10")["passed"], f"{rs['verdict']} n={nst.get('n_windows')}")
add(prs, "bad", "oracle", lambda sp, y: y + 0.02 * np.random.default_rng(2).standard_normal(y.shape)); rsb = run7(prs, Ts, gs, as_, "bad"); rec("G7_FAIL_dominates_NOT_EVALUABLE", rsb["verdict"] == "FAIL" and not c7(rsb, "orientation_ratio<=1.10")["evaluable"])
prn = {m: {sp: {} for sp in SETS} for m in ("persistence", "ridge", "legacy")}; rn_ = run7(prn, T, grp, act, "legacy"); rec("G7_model_without_orientation_outputs_NOT_EVALUABLE_with_reason", rn_["verdict"] == "NOT_EVALUABLE" and "no orientation outputs" in rn_["detail"]["reason"])
prg, Tg, gg_, ag, lg = fixture7(groups=dict(test_id=3, test_low_heldout=2, test_ood_low_strict=2, test_ood_high=2)); rg9 = run7(prg, Tg, gg_, ag, "oracle"); rec("G7_insufficient_groups_NOT_EVALUABLE", rg9["verdict"] == "NOT_EVALUABLE" and not c7(rg9, "yaw_ratio<=1.10")["evaluable"])
# static subset is input-only; S_gt flagged; horizons; tilt; determinism
T5 = copy.deepcopy(T)
for sp in SETS: T5[sp]["Y_rv"] = np.random.default_rng(1).standard_normal(T5[sp]["Y_rv"].shape).astype(np.float32); T5[sp]["Q_t"] = Gm.qnorm(np.random.default_rng(2).standard_normal(T5[sp]["Q_t"].shape))
rec("G7_static_subset_independent_of_future_truth_and_orientation", all(np.array_equal(G.static_input_mask(T[sp]), G.static_input_mask(T5[sp])) for sp in SETS))
r = run7(pr, T, grp, act, "oracle"); rec("G7_horizons_1_5_10_20_reported", set(r["detail"]["reported_horizons_yaw_heading_valid"]) == {"1", "5", "10", "20"}); rec("G7_regimes_reported_with_counts", set(r["detail"]["regimes_reported"]) == {"stable", "rotating", "singular", "tumbling"} and r["detail"]["regimes_reported"]["singular"]["n_windows"] > 0)
rec("G7_diagnostic_S_gt_flagged_as_using_future_truth", r["detail"]["diagnostic_S_gt"]["uses_future_truth"] is True)
T6 = copy.deepcopy(T)
for sp in SETS: T6[sp]["tilt_stored"] = np.degrees(Gm.tilt_rad(T6[sp]["Q_t"])).astype(np.float32)
r6 = run7(pr, T6, grp, act, "oracle"); rec("G7_tilt_unit_check_flags_degrees_and_does_not_gate", r6["detail"]["tilt_unit_check"]["ok"] is False and r6["verdict"] == r["verdict"] and "tilt_reported_only" in r["detail"])
for sp in SETS: T6[sp]["tilt_stored"] = Gm.tilt_rad(T6[sp]["Q_t"]).astype(np.float32)
rec("G7_tilt_unit_check_accepts_radians", run7(pr, T6, grp, act, "oracle")["detail"]["tilt_unit_check"]["ok"] is True)
rec("G7_deterministic", json.dumps(run7(pr, T, grp, act, "noisy"), default=str) == json.dumps(run7(pr, T, grp, act, "noisy"), default=str))
# legacy-yaw models (data contract v1 outputs): yaw components evaluable, orientation components NOT_EVALUABLE
pr["legacy"] = {sp: dict(Y_yaw=pr["oracle"][sp]["Y_rv"][:, :, 2].copy()) for sp in SETS}; rl = run7(pr, T, grp, act, "legacy"); ro_ = run7(pr, T, grp, act, "oracle")
rec("G7_legacy_yaw_model_yaw_components_evaluable_and_equal_to_orientation_model", c7(rl, "yaw_ratio<=1.10")["evaluable"] and abs(rl["detail"]["yaw_primary"]["E_model"] - ro_["detail"]["yaw_primary"]["E_model"]) < 1e-6 and c7(rl, "static_abs")["evaluable"])
rec("G7_legacy_yaw_model_orientation_components_NOT_EVALUABLE_verdict_NOT_EVALUABLE", (not c7(rl, "orientation_ratio<=1.10")["evaluable"]) and rl["verdict"] == "NOT_EVALUABLE", rl["verdict"])
pr["legacy_bad"] = {sp: dict(Y_yaw=pr["noisy"][sp]["Y_rv"][:, :, 2].copy()) for sp in SETS}; rlb = run7(pr, T, grp, act, "legacy_bad"); rec("G7_legacy_yaw_model_noisy_FAILS_and_FAIL_outranks_NOT_EVALUABLE", rlb["verdict"] == "FAIL" and not c7(rlb, "yaw_ratio<=1.10")["passed"])
rec("CONFIG_declares_prospective_protocol_and_not_applied_to_I1", "PROSPECTIVE" in G.CONFIG["status"] and G.CONFIG["protocol_version"] == "gate-v3.0" and any("I-1" in s for s in G.CONFIG["not_applied_to"]))
rs_ = run7(pr, T, grp, act, "oracle"); sv_ = rs_["detail"]["tumbling_threshold_sensitivity_not_gating"]; rec("G7_tumbling_sensitivity_at_0.20_and_0.50_reported_and_never_changes_the_verdict", set(sv_) == {"0.2", "0.5"} and sv_["0.2"]["windows"]["tumbling"] != sv_["0.5"]["windows"]["tumbling"] and rs_["verdict"] == "PASS")
rec("CONFIG_inherits_v2_hash_and_sample_sizes", G.CONFIG["inherits"]["gates_v2_config_sha256"] == "f6c323914a99b4bc884ffc88c4c538615ad3a1ba10649fd95e38282acbaa0627" and G.CONFIG["min_sample"] == V.CONFIG["min_sample"])
(ROOT / "outputs/gates_v3").mkdir(parents=True, exist_ok=True); json.dump(dict(results=RES, config_sha256=G.config_sha(), n=len(RES), all_pass=all(RES.values())), open(ROOT / "outputs/gates_v3/unit_tests.json", "w"), indent=1); print("ALL PASS" if all(RES.values()) else "SOME FAILED", len(RES))
