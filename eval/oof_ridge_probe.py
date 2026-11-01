#!/usr/bin/env python3
"""DIAGNOSTIC (closed-form only, no network training): size of the ridge-forecast train/test mismatch of WM3-B.  Re-fits the frozen WM-2 ridge joint block (same normaliser, alpha, features) in 5 GROUP folds of the train windows,
and compares the K=20 joint error of (a) in-sample forecasts on train, (b) out-of-fold forecasts on train, (c) full-train ridge on val / independent windows.  Also reports planning statistics (per-group SD of paired differences) from the independent evaluation for sample-size design.
Run (ACT venv): python eval/oof_ridge_probe.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.argv = [sys.argv[0], "none"]
import wm2_models as M, wm2_run as R
def fit(nrm, W, idx):
    X = nrm.feat({k: v[idx] for k, v in W.items() if hasattr(v, "__len__") and len(v) == len(W["t"])}, "cond").astype(np.float64); mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd; X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = nrm.targets({k: v[idx] for k, v in W.items() if hasattr(v, "__len__") and len(v) == len(W["t"])}).astype(np.float64)[:, :160]; Rg = np.eye(X1.shape[1]) * M.CFG["ridge"]["alpha"]; Rg[-1, -1] = 0; return mu, sd, np.linalg.solve(X1.T @ X1 + Rg, X1.T @ Y)
def pred(nrm, W, m):
    mu, sd, Wt = m; X = (nrm.feat(W, "cond").astype(np.float64) - mu) / sd; X1 = np.concatenate([X, np.ones((len(X), 1))], 1); return (X1 @ Wt).reshape(len(X), 20, 8) * nrm.Ys[1] + nrm.Ys[0]
def jerr(P, W, k=20): return float(np.abs(np.cumsum(P, 1)[:, k - 1, :7] - np.cumsum(W["Y_s"], 1)[:, k - 1, :7]).mean())
tr = R.load("train"); va = R.load("val"); nrm = M.Norm(tr); g = tr["group"]; ug = np.unique(g); rng = np.random.default_rng(0); perm = rng.permutation(ug); folds = np.array_split(perm, 5); full = fit(nrm, tr, np.arange(len(g)))
insample = jerr(pred(nrm, tr, full), tr); oof = np.zeros((len(g), 20, 8), np.float32)
for f in folds:
    te = np.isin(g, f); m = fit(nrm, tr, np.nonzero(~te)[0]); oof[te] = pred(nrm, {k: v[te] for k, v in tr.items() if hasattr(v, "__len__") and len(v) == len(g)}, m)
res = dict(n_train_windows=int(len(g)), n_train_groups=int(len(ug)), K20_joint_rad=dict(in_sample_train=insample, out_of_fold_train=jerr(oof, tr), val_full_train_ridge=jerr(pred(nrm, va, full), va)))
ind = {sp: {k: v for k, v in np.load(ROOT / f"outputs/indep/windows/{sp}.npz").items()} for sp in ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]}
res["K20_joint_rad"]["independent_full_train_ridge_window_mean"] = float(np.mean([jerr(pred(nrm, ind[sp], full), ind[sp]) for sp in ind])); res["mismatch_ratio_oof_over_insample"] = res["K20_joint_rad"]["out_of_fold_train"] / insample
# per-step normalised forecast residual scale (what the GRU sees as input noise): in-sample vs OOF vs val
nz = lambda P, W: float(np.sqrt(((nrm.n(P, nrm.Ys) - nrm.n(W["Y_s"], nrm.Ys)) ** 2).mean())); res["normalised_forecast_rmse_all_steps"] = dict(in_sample=nz(pred(nrm, tr, full), tr), out_of_fold=nz(oof, tr), val=nz(pred(nrm, va, full), va))
# planning statistics from the independent evaluation (paired group-level differences, K=20 box)
sys.path.insert(0, str(ROOT / "eval")); import wm2_gates as G
P = lambda m: {sp: {k: v for k, v in np.load(ROOT / f"outputs/indep/preds/{m}_{sp}.npz").items()} for sp in ind}; d = {}
for m in ("wm3b", "wm2_c0"):
    pp = P(m); vals = []
    for sp in ind:
        a = G.active_mask(ind[sp]); e = G.errors(G.sub(pp[sp], a), G.sub({k: ind[sp][k] for k in ("Y_s", "Y_b", "Y_c", "Y_e")}, a), 20)["box_mm"]; vals.append(G.group_means(e, ind[sp]["group"][a]))
    ids = np.concatenate([v[0] for v in vals]); x = np.concatenate([v[1] for v in vals]); o = np.argsort(ids); d[m] = x[o]
diff = d["wm3b"] - d["wm2_c0"]; sdv = float(diff.std(ddof=1)); n = len(diff)
res["planning_box_K20_paired_group_diff"] = dict(n_groups=n, mean=float(diff.mean()), sd=sdv, groups_needed_for_95pct_CI_half_width={str(h): int(np.ceil((1.96 * sdv / h) ** 2)) for h in (0.1, 0.15, 0.2, 0.3)}, frac_groups_where_wm3b_lower=float((diff < 0).mean()))
(ROOT / "outputs/yaw_audit/oof_ridge_probe.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
