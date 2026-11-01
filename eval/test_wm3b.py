#!/usr/bin/env python3
"""Unit tests and smoke tests for WM3-B (run before training).  python eval/test_wm3b.py"""
import sys, json, time, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm3b_model as B
res = {}
def ok(n, c, info=None): res[n] = dict(passed=bool(c), info=info); print(("PASS " if c else "FAIL ") + n, info if info is not None else "")
tr, va = R.load("train"), R.load("val"); nrm = M.Norm(tr); t0 = time.time(); ridge = M.Ridge("cond").fit(nrm, tr); print("ridge fitted", round(time.time() - t0, 1), "s")
sub = {k: v[:256] for k, v in tr.items()}; subv = {k: v[:256] for k, v in va.items()}
m = B.WM3B(ridge, nrm); log = []; m.fit(sub, subv, log, seed=0, epochs=1, max_steps=2)
# 1 shapes
p = m.predict({k: v[:32] for k, v in sub.items()}); shp = {k: list(v.shape) for k, v in p.items()}
ok("1_shapes", shp == dict(Y_s=[32, 20, 8], Y_b=[32, 20, 7], Y_c=[32, 20, 7], Y_e=[32, 4]) and sum(q.numel() for q in m.model.parameters()) == m.n_params, dict(shapes=shp, params=m.n_params))
hist, af, rn, raw = m.tensors({k: v[:8] for k, v in sub.items()}); ok("1b_input_tensor_shapes", tuple(hist.shape) == (8, 120) and tuple(af.shape) == (8, 20, 8) and tuple(rn.shape) == (8, 20, 8))
# 2 no future leakage: replace all targets by NaN, predictions identical; also changing Y_* does not change outputs
W0 = {k: v[:64].copy() for k, v in sub.items()}; Wn = {k: v.copy() for k, v in W0.items()}
for k in ("Y_s", "Y_b", "Y_c", "Y_e"): Wn[k] = np.full_like(Wn[k], np.nan)
pa, pb = m.predict(W0), m.predict(Wn); ok("2_no_future_leakage_targets_nan", all(np.array_equal(pa[k], pb[k]) and np.isfinite(pb[k]).all() for k in pa))
src = open(ROOT / "eval/wm3b_model.py").read(); ok("2b_forecast_path_reads_only_inputs", 'ridge.predict(W)["Y_s"]' in src and "Y_s\"]" not in src.split("def ridge_forecast")[1].split("class WM3B")[0].replace('ridge.predict(W)["Y_s"]', ""))
# 3 ridge forecast actually used
pz = m.predict(W0, rfc_override=np.zeros_like(m.tensors(W0)[2].numpy())); pr = m.predict(W0, rfc_override=m.tensors(W0)[2].numpy()[::-1].copy()); d0 = float(np.abs(pa["Y_b"] - pz["Y_b"]).mean()); d1 = float(np.abs(pa["Y_b"] - pr["Y_b"]).mean())
ok("3a_ridge_forecast_changes_output", d0 > 1e-6 and d1 > 1e-6, dict(zeroed=d0, shuffled=d1))
mm = B.GRUOC(); hist, af, rn, _ = m.tensors({k: v[:64] for k, v in sub.items()}); rn = rn.clone().requires_grad_(True); out, lg = mm(hist, af, rn); (out.sum() + lg.sum()).backward(); ok("3b_gradient_reaches_ridge_input_and_sin_weights", rn.grad is not None and float(rn.grad.abs().sum()) > 0 and float(mm.sin.weight.grad.abs().sum()) > 0)
# 4 train/inference contract
raw_a, rn_a = B.ridge_forecast(ridge, nrm, {k: v[:64] for k, v in sub.items()}); tt = m.tensors({k: v[:64] for k, v in sub.items()}); ok("4_train_inference_use_same_forecast_function_bitwise", np.array_equal(tt[2].numpy(), rn_a.astype(np.float32)) and np.array_equal(tt[3], raw_a) and "ridge_forecast(self.ridge, self.nrm, W)" in src)
# 5 finite loss and gradients
Y = m.targets({k: v[:64] for k, v in sub.items()}); E = torch.tensor(sub["Y_e"][:64]); mm2 = B.GRUOC(); hist, af, rn, _ = m.tensors({k: v[:64] for k, v in sub.items()}); o, l = mm2(hist, af, rn); L = B.loss_oc(o, Y, E, l); L.backward(); ok("5_finite_loss_and_gradients", bool(torch.isfinite(L)) and all(torch.isfinite(q.grad).all() for q in mm2.parameters() if q.grad is not None), float(L))
# 6 determinism
def losses(seed): lg = []; mx = B.WM3B(ridge, nrm); mx.fit(sub, subv, lg, seed=seed, epochs=1, max_steps=5); return lg[0]["train_loss"]
a, b = losses(0), losses(0); ok("6_determinism_same_seed_identical_loss", a == b, (a, b))
# 7 action sensitivity
Wa = {k: v[:64].copy() for k, v in sub.items()}; Wb2 = {k: v.copy() for k, v in Wa.items()}; Wb2["A_fut"] = Wb2["A_fut"] + 0.1; pa2, pb2 = m.predict(Wa), m.predict(Wb2); ok("7_action_sensitivity", float(np.abs(pa2["Y_b"] - pb2["Y_b"]).mean()) > 1e-6 and float(np.abs(pa2["Y_s"] - pb2["Y_s"]).mean()) > 1e-6, float(np.abs(pa2["Y_b"] - pb2["Y_b"]).mean()))
# 8 overfit tiny (400 steps) + smoke training
lg = []; mo = B.WM3B(ridge, nrm); mo.fit(sub, subv, lg, seed=0, epochs=200)                                  # 200 epochs x 2 batches = 400 steps
drop = 1 - min(r["train_loss"] for r in lg) / lg[0]["train_loss"]; ok("8a_overfit_tiny_400_steps_drop>=80pct", drop >= 0.8, round(drop, 4))
big = {k: v[:3840] for k, v in tr.items()}; lg2 = []; ms = B.WM3B(ridge, nrm); ms.fit(big, subv, lg2, seed=0, epochs=1, max_steps=30); ok("8b_smoke_training_30_steps_finite", all(r["finite"] for r in lg2) and np.isfinite(lg2[0]["val_loss"]), lg2[0])
json.dump(res, open(ROOT / "outputs/wm3/wm3b_unit_tests.json", "w"), indent=1, default=str); f = [k for k, v in res.items() if not v["passed"]]; print("ALL PASS" if not f else f"FAILED: {f}"); sys.exit(1 if f else 0)
