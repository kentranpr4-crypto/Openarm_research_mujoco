#!/usr/bin/env python3
"""Tests for data contract v2 (prototype) and the orientation losses: leakage (no future ground truth in model inputs), exactness of the tilt repair, rotation-vector consistency on REPLAYED sample episodes, loss properties.  Run (ACT venv): python eval/test_contract_v2.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "eval"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import geometry_v2 as Gm, contract_v2 as C, torch, orientation_loss as L
RES = {}
def rec(n, ok, info=""): RES[n] = bool(ok); print(("PASS " if ok else "FAIL "), n, info)
rng = np.random.default_rng(0); T = 80
def synth(T):
    rv = np.cumsum(rng.standard_normal((T, 3)) * 0.05, 0); q = np.concatenate([np.array([[1, 0, 0, 0.0]]), Gm.compose(np.array([1, 0, 0, 0.0]), rv)], 0); pos = np.cumsum(rng.standard_normal((T + 1, 3)) * 0.002, 0); sp = np.abs(rng.standard_normal(T + 1)); ao = rng.standard_normal(T + 1)
    return q, pos, sp, ao
q, pos, sp, ao = synth(T); b2 = C.row_features(q, pos, sp, ao); Y, aux = C.targets(q, pos, sp, ao)
ep = dict(b2=b2, Y=Y, aux=aux, s=rng.standard_normal((T, 8)).astype(np.float32), c=rng.standard_normal((T, 7)).astype(np.float32), a=rng.standard_normal((T, 8)).astype(np.float32))
rec("C2_shapes_b2_14_targets_9", b2.shape == (T + 1, 14) and Y.shape == (T, 9) and len(C.B2_NAMES) == 14 and aux["yaw_step_valid"].shape == (T,) and np.isfinite(b2).all() and np.isfinite(Y).all())
t0 = 30; X, Yw, Ya = C.window(ep, t0)
# leakage: change EVERY quantity after row t0 -> inputs must be bitwise identical, targets must change
q2, pos2, sp2, ao2 = q.copy(), pos.copy(), sp.copy(), ao.copy(); q2[t0 + 1:] = Gm.qnorm(rng.standard_normal(q2[t0 + 1:].shape)); pos2[t0 + 1:] += 1.0; sp2[t0 + 1:] += 3.0; ao2[t0 + 1:] -= 2.0
b2b = C.row_features(q2, pos2, sp2, ao2); Yb, auxb = C.targets(q2, pos2, sp2, ao2); epb = dict(ep, b2=b2b, Y=Yb, aux=auxb); Xb, Ywb, Yab = C.window(epb, t0)
rec("C2_LEAKAGE_inputs_bitwise_identical_when_all_future_rows_are_replaced", all(np.array_equal(X[k], Xb[k]) for k in X) and not np.array_equal(Yw["Y_b2"], Ywb["Y_b2"]))
rec("C2_LEAKAGE_target_time_masks_are_not_in_the_input_dict", set(X) == {"S_hist", "B2_hist", "C_hist", "A_hist", "A_fut"} and "yaw_step_valid" in Ya and not any("valid" in k and k != "B2_hist" for k in X))
rec("C2_row_features_use_only_the_current_row", np.array_equal(C.row_features(q[10:11], pos[10:11], sp[10:11], ao[10:11])[0], b2[10]))
hv = C.row_features(Gm.qmul(np.array([[1, 0, 0, 0.0]]), np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0.0])), np.zeros((1, 3)), np.zeros(1), np.zeros(1))[0]
rec("C2_heading_valid_flag_zero_for_box_on_its_end_one_for_flat_box", hv[12] == 0 and C.row_features(np.array([[1, 0, 0, 0.0]]), np.zeros((1, 3)), np.zeros(1), np.zeros(1))[0][12] == 1)
Qc = np.concatenate([q[:1], Gm.compose(q[0], Y[:, 3:6])], 0); rec("C2_rotation_vector_targets_recompose_the_orientation_trajectory", np.max(Gm.geodesic(Qc, q)) < 1e-5)
# real replayed episodes
z = np.load(ROOT / "outputs/contract_v2/replay_sample.npz"); eids = sorted({k.split("/")[0] for k in z.files})[:60]; idx = {e["eid"]: e for e in json.loads((ROOT / "outputs/wm2/dataset/index.json").read_text())}; bad = 0; maxrec = 0; maxtilt = 0; nvalid = ntot = 0
for e in eids:
    qq, pp, spd = z[f"{e}/q"], z[f"{e}/pos"], z[f"{e}/speed"]; ao_ = np.load(ROOT / f"outputs/wm2/dataset/episodes/{e}.npz")["b"][:, 6]; ao_ = np.concatenate([ao_, ao_[-1:]])[:len(qq)]; b2_ = C.row_features(qq, pp, spd, ao_); Y_, aux_ = C.targets(qq, pp, spd, ao_)
    bad += int(not (np.isfinite(b2_).all() and np.isfinite(Y_).all())); Qc_ = np.concatenate([qq[:1], Gm.compose(qq[0], Y_[:, 3:6])], 0); maxrec = max(maxrec, float(Gm.geodesic(Qc_, qq).max())); ntot += len(aux_["yaw_step_valid"]); nvalid += int(aux_["yaw_step_valid"].sum())
rec("C2_real_replayed_episodes_finite_and_rotvec_roundtrip", bad == 0 and maxrec < 1e-4, f"n={len(eids)}, max recomposition geodesic={maxrec:.1e}, valid steps {nvalid}/{ntot}")
v = json.loads((ROOT / "outputs/contract_v2/validation.json").read_text()); rec("C2_tilt_repair_exact_to_float32_on_replayed_episodes", v["V6_tilt_repair"]["max_abs_diff_rad_after_unit_conversion"] < 1e-5, f"{v['V6_tilt_repair']['max_abs_diff_rad_after_unit_conversion']:.1e}")
rec("C2_conventions_quaternion_matrix_orthonormal_verified", v["V1_conventions_max_abs_diff"]["q_to_xmat"] < 1e-12 and v["V1_conventions_max_abs_diff"]["orth"] < 1e-12)
# losses
torch.manual_seed(0); pred = torch.randn(8, 20, requires_grad=True) * 0.01; pred.retain_grad(); true = torch.randn(8, 20) * 0.01; valid = torch.rand(8, 20) > 0.3; l = L.yaw_huber_masked(pred, true, valid, 0.0166); l.backward()
rec("L_yaw_masked_loss_finite_and_zero_gradient_on_masked_steps", torch.isfinite(l) and bool((pred.grad[~valid] == 0).all()) and bool((pred.grad[valid] != 0).any()))
rec("L_yaw_loss_invariant_to_2pi_offset_of_the_target", abs(float(L.yaw_huber_masked(pred.detach(), true + 2 * np.pi, valid, 0.0166) - L.yaw_huber_masked(pred.detach(), true, valid, 0.0166))) < 1e-4)
big = torch.zeros(1, 20); big[0, 0] = 3.1; tt = torch.zeros(1, 20); vv = torch.ones(1, 20, dtype=torch.bool); lh = L.yaw_huber_masked(big, tt, vv, 0.0166, 0.05); lm = L.yaw_huber_masked(big, tt, vv, 0.0166, 1e9)
rec("L_huber_bounds_the_influence_of_a_spike_relative_to_mse", float(lh) < 0.05 * float(lm), f"huber={float(lh):.1f}, mse-like={float(lm):.1f}")
rec("L_yaw_loss_masked_singular_spike_contributes_zero", float(L.yaw_huber_masked(big, tt, torch.zeros(1, 20, dtype=torch.bool), 0.0166)) == 0.0)
rv = torch.randn(4, 20, 3, requires_grad=True); lr = L.rotvec_loss(rv, torch.zeros(4, 20, 3), 0.1); lr.backward(); rec("L_rotvec_loss_finite_with_gradient", torch.isfinite(lr) and bool(torch.isfinite(rv.grad).all()))
rec("L_clip_only_ignores_target_beyond_clip", abs(float(L.clip_only(torch.zeros(1, 20), torch.full((1, 20), 3.0), 0.0166, 0.2)) - (0.2 / 0.0166) ** 2) < 1e-3)
json.dump(dict(results=RES, n=len(RES), all_pass=all(RES.values())), open(ROOT / "outputs/contract_v2/unit_tests.json", "w"), indent=1); print("ALL PASS" if all(RES.values()) else "SOME FAILED", len(RES))
