#!/usr/bin/env python3
"""YAW ROOT-CAUSE, stages C and D (offline; forward pass of the FROZEN WM3-B / WM2-C0 checkpoints only; nothing is trained or modified).
C: per angular-motion regime, how much of the yaw target energy and of the models' normalised squared error comes from where; zero-predictor vs model per regime on TRAIN (in-sample) and VAL windows.
D: element-level effect of alternative scalings / representations on (i) the share of target energy carried by unpredictable spikes, (ii) the normalised penalty of a spurious 0.01 rad per-step rotation.
Run (ACT venv): python eval/yaw_lossprobe.py"""
import sys, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm3b_model as B
OUT = ROOT / "outputs/yaw_audit"; REG = [("static(<0.005)", 0, 0.005), ("small(0.005-0.05)", 0.005, 0.05), ("moderate(0.05-0.5)", 0.05, 0.5), ("fast(0.5-pi/2)", 0.5, np.pi / 2), ("spike(>pi/2)", np.pi / 2, 99)]
torch.set_num_threads(3); tr = R.load("train"); va = R.load("val"); nrm = M.Norm(tr); ridge = M.Ridge("cond").fit(nrm, tr); m = B.WM3B(ridge, nrm); m.model = B.GRUOC(); m.model.load_state_dict(torch.load(ROOT / "outputs/wm3/models/wm3b.pt")); m.model.eval()
c0 = M.NN("gru"); c0.nrm = nrm; c0.model = M.GRUModel(); c0.model.load_state_dict(torch.load(ROOT / "outputs/wm2_1/models/gru_c0.pt")); c0.model.eval()
sd = nrm.Yb[1][3]; res = dict(yaw_normaliser_std=float(sd), regimes=[r[0] for r in REG])
def regime_stats(W, name):
    y = W["Y_b"][:, :, 3]; mx = np.abs(y).max(1); out = {}
    P = {"wm3b": m.predict(W)["Y_b"][:, :, 3], "wm2_c0": c0.predict(W)["Y_b"][:, :, 3], "ridge": ridge.predict(W)["Y_b"][:, :, 3]}
    se_zero = ((y / sd) ** 2).sum(1); tot_energy = se_zero.sum(); tot_err = {k: (((v - y) / sd) ** 2).sum(1) for k, v in P.items()}
    for rn, lo, hi in REG:
        mk = (mx >= lo) & (mx < hi); o = dict(n_windows=int(mk.sum()), frac_windows=float(mk.mean()), share_of_yaw_target_energy=float(se_zero[mk].sum() / tot_energy), mean_window_sse_zero_predictor=float(se_zero[mk].mean()) if mk.any() else None)
        for k, e in tot_err.items(): o[f"share_of_{k}_yaw_error"] = float(e[mk].sum() / e.sum()); o[f"mean_window_sse_{k}"] = float(e[mk].mean()) if mk.any() else None
        out[rn] = o
    out["_all"] = {k: float(v.mean()) for k, v in tot_err.items()} | dict(zero=float(se_zero.mean())); out["_all_excluding_spike_regime"] = {k: float(v[mx < np.pi / 2].mean()) for k, v in tot_err.items()} | dict(zero=float(se_zero[mx < np.pi / 2].mean())); return out
res["C_train_in_sample"] = regime_stats(tr, "train"); res["C_val"] = regime_stats(va, "val")
# per-channel concentration of target energy (top 0.1 % of elements)
Yb = tr["Y_b"].reshape(-1, 7); e2 = (Yb - Yb.mean(0)) ** 2; res["channel_energy_concentration_top0p1pct_elements"] = dict(zip(["x", "y", "z", "yaw", "speed", "tilt(mixed units)", "aoffset"], [float(np.sort(e2[:, c])[::-1][:max(1, int(0.001 * len(e2)))].sum() / e2[:, c].sum()) for c in range(7)]))
# D: alternative scalings / representations on the element-level yaw deltas
d = tr["Y_b"][:, :, 3].ravel().astype(np.float64); a = np.abs(d); sp = a > np.pi / 2; bands = [("static(<0.005)", a < 0.005), ("small(0.005-0.05)", (a >= 0.005) & (a < 0.05)), ("moderate(0.05-0.5)", (a >= 0.05) & (a < 0.5)), ("fast(0.5-pi/2)", (a >= 0.5) & ~sp), ("spike(>pi/2)", sp)]
def table(tf, label, scale=None):
    t = tf(d); t = t if t.ndim == 2 else t[:, None]; s = scale if scale is not None else float(t.std(0).mean()); e = (t ** 2).sum(1)
    spur = float((np.array([0.01]) / s) ** 2) if t.shape[1] == 1 else None
    return dict(scale_used=s, energy_share_by_band={n: float(e[mk].sum() / e.sum()) for n, mk in bands}, normalised_penalty_of_0p01_rad_spurious_step=spur, normalised_value_of_p90_delta=float(np.percentile(a, 90) / s) if t.shape[1] == 1 else None)
res["D_alternatives"] = {"current: std of raw wrapped deltas": table(lambda x: x, "cur"), "robust: std excluding |delta|>pi/2 (scale only, spikes kept)": table(lambda x: x, "rob", scale=float(d[~sp].std())), "clip to +-0.2 rad then std": table(lambda x: np.clip(x, -0.2, 0.2), "clip"),
                         "wrap-aware angle: wrap(delta) then std": table(lambda x: np.arctan2(np.sin(x), np.cos(x)), "wrap"), "circular (sin d, 1-cos d) per step": table(lambda x: np.stack([np.sin(x), 1 - np.cos(x)], 1), "circ")}
(OUT / "yaw_lossprobe.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: res[k] for k in ("C_val", "D_alternatives", "channel_energy_concentration_top0p1pct_elements")}, indent=1)[:6000])
