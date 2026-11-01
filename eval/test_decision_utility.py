#!/usr/bin/env python3
"""Synthetic tests for the decision-utility library (scoring, tie-break, regret, pairwise accuracy, AUROC, calibration, bootstrap, rollout correctness, leakage by perturbation).  Run: python3 eval/test_decision_utility.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import decision_utility_lib as L
RES = {}
def rec(n, ok, info=""): RES[n] = bool(ok); print(("PASS " if ok else "FAIL "), n, info)
b = np.zeros((5, 7)); b[:, 2] = 0.38; b[-1, :3] = [0.44, 0.30, 0.384]; b[2, 2] = 0.44; c = np.zeros((5, 7)); t = L.terms(b, c)
rec("score_perfect_placement_with_lift_and_no_palm_is_zero", t["placed"] and t["lifted"] and not t["palm"] and abs(L.score(t)) < 1e-12)
c2 = c.copy(); c2[3, 4] = 0.5; rec("score_palm_penalty_0.05", abs(L.score(L.terms(b, c2)) - 0.05) < 1e-12); b2 = b.copy(); b2[:, 2] = np.minimum(b2[:, 2], 0.40); b2[-1, 2] = 0.384; rec("score_not_lifted_penalty_0.05", abs(L.score(L.terms(b2, c)) - 0.05) < 1e-12)
b3 = b.copy(); b3[-1, :2] = [0.471, 0.30]; rec("score_xy_error_in_metres_and_placed_flag_boundary", abs(L.score(L.terms(b3, c)) - 0.031) < 1e-9 and (not L.terms(b3, c)["placed"]) )
vids = [0, 1, 2, 3]; J = [0.1, 0.1, 0.2, 0.3]; s1 = L.select(J, "p1", vids); s2 = L.select(J, "p1", vids); rec("tie_break_deterministic_and_among_tied_only", s1 == s2 and s1 in (0, 1)); picks = {L.select(J, f"pose{i}", vids) for i in range(60)}; rec("tie_break_is_not_biased_towards_the_first_candidate", picks == {0, 1})
rec("select_min", L.select([0.3, 0.1, 0.2], "x", [0, 1, 2]) == 1)
r, nr = L.regret([0.3, 0.1, 0.2, 0.9], 2); rec("regret_and_normalised_regret", abs(r - 0.1) < 1e-12 and abs(nr - (0.1 / (np.median([0.3, 0.1, 0.2, 0.9]) - 0.1))) < 1e-12); r0, nr0 = L.regret([0.1, 0.1, 0.1], 0); rec("normalised_regret_undefined_when_no_spread", r0 == 0 and np.isnan(nr0))
Jt = np.array([0.0, 0.05, 0.10, 0.20]); acc, n = L.pairwise_accuracy(Jt, Jt); rec("pairwise_accuracy_perfect_ranker_is_1", acc == 1.0 and n == 6); acc2, _ = L.pairwise_accuracy(-Jt, Jt); rec("pairwise_accuracy_reversed_ranker_is_0", acc2 == 0.0); acc3, _ = L.pairwise_accuracy(np.zeros(4), Jt); rec("pairwise_accuracy_all_ties_is_0.5", acc3 == 0.5)
acc4, n4 = L.pairwise_accuracy([0, 1, 2], [0.0, 0.005, 0.0099]); rec("pairwise_pairs_below_meaningful_difference_are_ignored", np.isnan(acc4) and n4 == 0)
rec("kendall_perfect_and_reversed", L.kendall(Jt, Jt) == 1.0 and L.kendall(-Jt, Jt) == -1.0)
y = np.array([0, 0, 1, 1]); rec("auroc_perfect_reversed_ties", L.auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0 and L.auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0 and L.auroc(y, [0.5] * 4) == 0.5)
e, br = L.ece_brier([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]); rec("ece_brier_perfect", e == 0 and br == 0); e2, _ = L.ece_brier([0, 1] * 50, [0.5] * 100); rec("ece_calibrated_constant_is_0", e2 < 1e-12)
m, lo, hi = L.boot_mean(np.arange(20.0)); m2 = L.boot_mean(np.arange(20.0)); rec("bootstrap_deterministic_and_contains_mean", [m, lo, hi] == m2 and lo < m < hi)
# ---- rollout correctness: an oracle predictor that returns the TRUE next-K deltas reproduces the true trajectory exactly; a persistence predictor leaves the box where it is
rng = np.random.default_rng(0); N = 3; T = 90; Tl = np.array([90, 70, 50]); S_true = np.cumsum(rng.standard_normal((N, T, 8)) * 0.01, 1); B_true = np.cumsum(rng.standard_normal((N, T, 7)) * 0.01, 1); C_true = rng.random((N, T, 7)); A = rng.standard_normal((N, T, 8))
state = {"i": None}
def oracle_predict_factory(S_true, B_true, C_true, ids):
    def f(W):
        t = int(W["t"][0]); n = len(W["t"]); out = dict(Y_s=np.zeros((n, 20, 8)), Y_b=np.zeros((n, 20, 7)), Y_c=np.zeros((n, 20, 7)), Y_e=np.full((n, 4), 0.5))
        for k in range(n):
            i = ids[k]; out["Y_s"][k] = S_true[i, t + 1:t + 21] - S_true[i, t:t + 20]; out["Y_b"][k] = B_true[i, t + 1:t + 21] - B_true[i, t:t + 20]; out["Y_c"][k] = C_true[i, t + 1:t + 21]
        return out
    return f
class Oracle:
    """oracle that identifies the candidate by matching A_fut with the stored actions (test double only)"""
    def __init__(self): pass
def pred_by_action(W):
    n = len(W["t"]); t = int(W["t"][0]); out = dict(Y_s=np.zeros((n, 20, 8)), Y_b=np.zeros((n, 20, 7)), Y_c=np.zeros((n, 20, 7)), Y_e=np.full((n, 4), 0.5))
    for k in range(n):
        i = [j for j in range(N) if np.allclose(A[j, t:t + 20], W["A_fut"][k])][0]; out["Y_s"][k] = S_true[i, t + 1:t + 21] - S_true[i, t:t + 20]; out["Y_b"][k] = B_true[i, t + 1:t + 21] - B_true[i, t:t + 20]; out["Y_c"][k] = C_true[i, t + 1:t + 21]
    return out
hist = dict(S=S_true[:, 1:5].copy(), B=B_true[:, 1:5].copy(), C=C_true[:, 1:5].copy()); R = L.rollout(pred_by_action, hist, A, Tl, [0.4, 0.1, 0.0])
ok = True
for i in range(N):
    nb = (Tl[i] - 1 - 4) // 20; ok &= R[i]["n_blocks"] == nb and np.allclose(R[i]["b"], B_true[i, 4:4 + 20 * nb + 1]) and np.allclose(R[i]["c"][1:], C_true[i, 5:4 + 20 * nb + 1])
rec("rollout_with_true_deltas_reproduces_the_true_rows_and_uses_the_right_number_of_blocks", ok, f"blocks={[r['n_blocks'] for r in R]}")
zero = lambda W: dict(Y_s=np.zeros((len(W['t']), 20, 8)), Y_b=np.zeros((len(W['t']), 20, 7)), Y_c=np.repeat(W["C_hist"][:, -1:], 20, 1), Y_e=np.full((len(W['t']), 4), 0.2)); Rz = L.rollout(zero, hist, A, Tl, [0.4, 0.1, 0.0])
rec("rollout_with_zero_deltas_keeps_the_box_where_it_was", all(np.allclose(r["b"], B_true[i, 4]) for i, r in enumerate(Rz)))
# leakage: change every true quantity after row t0 (states, contacts) -> predictions identical; change actions of candidate 0 -> rollout of candidate 0 changes
cap = []
def recorder(W): cap.append({k: v.copy() for k, v in W.items()}); return zero(W)
L.rollout(recorder, hist, A, Tl, [0.4, 0.1, 0.0]); S2, B2, C2 = S_true.copy(), B_true.copy(), C_true.copy(); S2[:, 5:] += 5; B2[:, 5:] += 5; C2[:, 5:] += 5; cap2 = []
def recorder2(W): cap2.append({k: v.copy() for k, v in W.items()}); return zero(W)
hist2 = dict(S=S2[:, 1:5].copy(), B=B2[:, 1:5].copy(), C=C2[:, 1:5].copy()); L.rollout(recorder2, hist2, A, Tl, [0.4, 0.1, 0.0])
rec("leakage_predictor_inputs_do_not_depend_on_ground_truth_rows_after_t0", all(all(np.array_equal(a[k], b_[k]) for k in a) for a, b_ in zip(cap, cap2)))
rec("predictor_input_keys_exclude_labels_and_metadata", set(cap[0]) == {"S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "t", "Y_e", "pose"} and not cap[0]["Y_e"].any())
Rp = L.rollout(lambda W: dict(Y_s=np.zeros((len(W['t']), 20, 8)), Y_b=np.full((len(W['t']), 20, 7), 0.1), Y_c=np.zeros((len(W['t']), 20, 7)), Y_e=np.zeros((len(W['t']), 4))), hist, A, Tl, [0, 0, 0], yaw_mode="persist")
rec("yaw_persist_mode_zeroes_only_the_yaw_delta", all(np.allclose(np.diff(r["b"][:, 3]), 0) and np.allclose(np.diff(r["b"][:, 0]), 0.1) for r in Rp))
Re = L.rollout(lambda W: dict(Y_s=np.zeros((len(W['t']), 20, 8)), Y_b=np.full((len(W['t']), 20, 7), 0.1), Y_c=np.zeros((len(W['t']), 20, 7)), Y_e=np.zeros((len(W['t']), 4))), hist, A, Tl, [0, 0, 0], yaw_mode="external", yaw_predict=lambda W: np.full((len(W['t']), 20), 0.02))
rec("yaw_external_mode_uses_the_supplied_yaw", all(np.allclose(np.diff(r["b"][:, 3]), 0.02) for r in Re))
json.dump(dict(results=RES, n=len(RES), all_pass=all(RES.values())), open(ROOT / "outputs/decision_utility/unit_tests.json", "w"), indent=1); print("ALL PASS" if all(RES.values()) else "SOME FAILED", len(RES))
