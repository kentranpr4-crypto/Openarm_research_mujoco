#!/usr/bin/env python3
"""YAW AUDIT (post-hoc DIAGNOSTIC, read-only).  Analyses the FROZEN predictions of the independent evaluation (outputs/indep, INDEPENDENT) and of the development dry-run predictions (outputs/indep_dry, DEVELOPMENT windows).
Nothing is trained, tuned or re-evaluated for selection; no checkpoint, dataset, gate or official evaluator is modified.  Output: outputs/yaw_audit/yaw_audit.json (+ figures).  Run: python eval/yaw_audit.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "data"))
import wm2_gates as G
OUT = ROOT / "outputs/yaw_audit"; EV = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; MOD = ["persistence", "ridge", "mlp", "wm2_c0", "wm3b"]; FAMN = {0: "control", 1: "depth", 2: "timing", 3: "noise", 4: "speed", 5: "release_height"}
wrap = lambda x: np.arctan2(np.sin(x), np.cos(x))
def load(setname):
    win = ROOT / ("outputs/indep/windows" if setname == "indep" else "outputs/wm2/windows"); pr = ROOT / ("outputs/indep/preds" if setname == "indep" else "outputs/indep_dry/preds"); W, P = {}, {}
    for sp in EV:
        z = np.load(win / f"{sp}.npz"); W[sp] = {k: z[k] for k in z.files}
        P[sp] = {m: {k: v for k, v in np.load(pr / f"{m}_{sp}.npz").items()} for m in MOD}
    cat = lambda f: np.concatenate([f(sp) for sp in EV]); A = {k: cat(lambda sp, k=k: W[sp][k]) for k in ("B_hist", "C_hist", "A_hist", "Y_b", "group", "family", "active")}
    A["stratum"] = np.concatenate([np.full(len(W[sp]["group"]), i) for i, sp in enumerate(EV)]); A["Yb_pred"] = {m: cat(lambda sp, m=m: P[sp][m]["Y_b"]) for m in MOD}; return A
def gmean(vals, g):
    u = np.unique(g); return float(np.mean([vals[g == x].mean() for x in u]))
def analyse(A, label):
    R = {}; act = A["active"]; g = A["group"]; Yb = A["Y_b"]; B0 = A["B_hist"][:, -1]
    # 1. representation / wrap -----------------------------------------------------------------
    yaw_hist = A["B_hist"][:, :, 3]; dy = Yb[:, :, 3]; cum_t = np.cumsum(dy, 1)
    R["representation"] = dict(yaw_channel_index=3, units="rad, yaw_of(quaternion)=atan2(2(wz+xy),1-2(y^2+z^2)) in (-pi,pi]; targets Y_b[:,k,3] are per-step DELTAS of that angle", box_yaw_abs_range=[float(yaw_hist.min()), float(yaw_hist.max())], true_per_step_delta_abs_max=float(np.abs(dy).max()),
                               n_per_step_deltas_gt_pi_over_2=int((np.abs(dy) > np.pi / 2).sum()), true_cum_yaw_K20_abs_mean=float(np.abs(cum_t[:, 19]).mean()), true_cum_yaw_K20_abs_p99=float(np.percentile(np.abs(cum_t[:, 19]), 99)), true_cum_yaw_K20_abs_max=float(np.abs(cum_t[:, 19]).max()))
    # 7. metric verification: recompute the K=20 error from ABSOLUTE angles with explicit wrapping, compare with the evaluator's definition (|cum_pred - cum_true|)
    ver = {}
    for m in MOD:
        cp = np.cumsum(A["Yb_pred"][m][:, :, 3], 1)[:, 19]; ev = np.abs(cp - cum_t[:, 19]); abs_t = wrap(yaw_hist[:, -1] + cum_t[:, 19]); abs_p = wrap(yaw_hist[:, -1] + cp); w = np.abs(wrap(abs_p - abs_t)); ver[m] = dict(max_abs_diff_between_definitions=float(np.abs(w - ev).max()), evaluator_mean_all_windows=float(ev.mean()), wrapped_absolute_mean_all_windows=float(w.mean()), deg_mean=float(np.degrees(w.mean())))
    R["metric_verification_K20"] = ver
    # 2. horizons ------------------------------------------------------------------------------
    err = {m: np.abs(np.cumsum(A["Yb_pred"][m][:, :, 3], 1) - cum_t) for m in MOD}; R["horizon_yaw_error_active_group_macro"] = {m: {f"K={k}": gmean(err[m][act][:, k - 1], g[act]) for k in (1, 5, 10, 20)} for m in MOD}
    R["horizon_curve_active_group_macro"] = {m: [gmean(err[m][act][:, k], g[act]) for k in range(20)] for m in MOD}
    R["by_stratum_K20"] = {EV[i]: {m: gmean(err[m][(A["stratum"] == i) & act][:, 19], g[(A["stratum"] == i) & act]) for m in MOD} for i in range(4)}
    R["by_variant_family_K20"] = {FAMN[f]: {m: gmean(err[m][(A["family"] == f) & act][:, 19], g[(A["family"] == f) & act]) for m in MOD} | dict(n_windows=int(((A["family"] == f) & act).sum())) for f in FAMN if ((A["family"] == f) & act).sum() > 50}
    # phases from INPUT-only information
    fc = A["C_hist"][:, -1, 1] > 0; gopen = A["A_hist"][:, -1, 7] >= 0.028; lifted = B0[:, 2] > 0.3905; far = np.linalg.norm(B0[:, :2] - np.array([0.44, 0.30]), axis=1) > 0.05
    ph = {"approach(no finger contact, gripper open, on table)": ~fc & gopen & ~lifted & far, "grasp(finger contact, not lifted)": fc & ~lifted, "carry(finger contact, lifted)": fc & lifted, "other(post-release / in tray / closed no contact)": ~(~fc & gopen & ~lifted & far) & ~(fc & ~lifted) & ~(fc & lifted)}
    R["by_contact_phase_K20"] = {}
    for n, mk in ph.items():
        mk = mk & act
        if mk.sum() < 50: continue
        R["by_contact_phase_K20"][n] = {m: float(err[m][mk][:, 19].mean()) for m in MOD} | dict(true_abs_cum_yaw_mean=float(np.abs(cum_t[mk][:, 19]).mean()), n_windows=int(mk.sum()), n_groups=int(len(np.unique(g[mk]))))
    R["by_phase_note"] = "window means (not group-macro), descriptive; phases from input blocks at time t"
    # 4. bias / drift / variance -----------------------------------------------------------------
    bias = {}
    for m in MOD:
        cp = np.cumsum(A["Yb_pred"][m][:, :, 3], 1); d = (cp - cum_t)[act]; bias[m] = dict(signed_mean_K20=float(d[:, 19].mean()), signed_mean_K5=float(d[:, 4].mean()), pred_std_K20=float(cp[act][:, 19].std()), true_std_K20=float(cum_t[act][:, 19].std()), corr_pred_true_K20=float(np.corrcoef(cp[act][:, 19], cum_t[act][:, 19])[0, 1]) if cp[act][:, 19].std() > 0 else None,
                                                                          slope_pred_on_true_K20=float(np.polyfit(cum_t[act][:, 19], cp[act][:, 19], 1)[0]) if cp[act][:, 19].std() > 0 else None, rmse_K20=float(np.sqrt((d[:, 19] ** 2).mean())), mae_K20=float(np.abs(d[:, 19]).mean()),
                                                                          error_growth_ratio_K20_over_K5=float(np.abs(d[:, 19]).mean() / max(np.abs(d[:, 4]).mean(), 1e-12)))
    R["bias_variance_drift_active"] = bias
    # 6. tails / stationary windows
    tail = {}; stat = np.abs(cum_t[:, 19]) < 0.003
    for m in MOD:
        e = err[m][act][:, 19]; s = np.sort(e)[::-1]; k1 = max(1, int(0.01 * len(s))); k10 = max(1, int(0.10 * len(s)))
        tail[m] = dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), p99=float(np.percentile(e, 99)), mean=float(e.mean()), share_of_total_error_top1pct_windows=float(s[:k1].sum() / s.sum()), share_top10pct=float(s[:k10].sum() / s.sum()), mean_error_when_true_yaw_nearly_static=float(err[m][act & stat][:, 19].mean()),
                       frac_windows_worse_than_persistence=float((err[m][act][:, 19] > err["persistence"][act][:, 19]).mean()))
    R["tails_and_static"] = dict(per_model=tail, frac_windows_true_yaw_static_lt_0p003=float(stat[act].mean()), true_abs_cum_yaw_quantiles_K20={q: float(np.percentile(np.abs(cum_t[act][:, 19]), q)) for q in (50, 75, 90, 95, 99)})
    # 5. per-channel R2 of per-step deltas (are object channels fit?)
    r2 = {}
    for m in ("ridge", "mlp", "wm2_c0", "wm3b"):
        P_ = A["Yb_pred"][m][act]; T_ = Yb[act]; r2[m] = [float(1 - ((P_[:, :, c] - T_[:, :, c]) ** 2).sum() / ((T_[:, :, c] - T_[:, :, c].mean()) ** 2).sum()) for c in range(7)]
    R["R2_per_step_delta_by_Yb_channel[x,y,z,yaw,speed,tilt,aoffset]"] = r2
    # robustness: windows without tumbling / angular wrap jumps (no per-step true yaw delta > pi/2 in the horizon, absolute yaw within +-1 rad); wrapped angular error
    clean = ~(np.abs(dy) > np.pi / 2).any(1) & (np.abs(yaw_hist).max(1) < 1.0) & act; R["clean_subset"] = dict(n_windows=int(clean.sum()), frac_of_active=float(clean.sum() / act.sum()), n_groups=int(len(np.unique(g[clean]))), K20_wrapped_error_group_macro={}, mean_abs_true_cum_yaw=float(np.abs(cum_t[clean][:, 19]).mean()), signed_R2_K20_vs_zero={})
    for m in MOD:
        cp = np.cumsum(A["Yb_pred"][m][:, :, 3], 1)[:, 19]; w = np.abs(wrap(cp - cum_t[:, 19])); R["clean_subset"]["K20_wrapped_error_group_macro"][m] = gmean(w[clean], g[clean]); R["clean_subset"]["signed_R2_K20_vs_zero"][m] = float(1 - ((cp[clean] - cum_t[clean][:, 19]) ** 2).sum() / (cum_t[clean][:, 19] ** 2).sum())
    R["wrap_vs_raw_mean_all_active"] = {m: dict(raw=float(err[m][act][:, 19].mean()), wrapped=float(np.abs(wrap(np.cumsum(A["Yb_pred"][m][:, :, 3], 1)[:, 19] - cum_t[:, 19]))[act].mean())) for m in MOD}
    st = act & (np.abs(cum_t[:, 19]) < 0.003); R["static_windows_mean_abs_predicted_per_step_yaw_delta_K1"] = {m: float(np.abs(A["Yb_pred"][m][st][:, 0, 3]).mean()) for m in MOD}; R["n_static_windows"] = int(st.sum())
    return R
def train_stats():
    z = np.load(ROOT / "outputs/wm2/windows/train.npz"); Yb = z["Y_b"].reshape(-1, 7); sd = Yb.std(0); d = z["Y_b"][:, :, 3].ravel(); a = np.abs(d)
    big = np.abs(z["Y_b"][:, :, 3]) > np.pi / 2; sd_rob = float(z["Y_b"][:, :, 3][~big].std()); yaw_abs = np.abs(z["B_hist"][:, :, 3]); sd_all = float(z["Y_b"][:, :, 3].std())
    spike = dict(n_deltas=int(d.size), n_deltas_gt_pi_over_2=int(big.sum()), share_of_total_squared_delta_from_gt_pi_over_2=float((d[big.ravel()] ** 2).sum() / (d ** 2).sum()), std_all=sd_all, std_excluding_gt_pi_over_2=sd_rob, normaliser_inflation_factor=sd_all / sd_rob, normalised_value_of_one_2pi_wrap_spike=float(2 * np.pi / sd_all),
                 normalised_value_of_typical_abs_delta_p90=float(np.percentile(a, 90) / sd_all), frac_train_windows_with_abs_yaw_gt_2p5=float((yaw_abs.max(1) > 2.5).mean()), n_train_windows=int(len(yaw_abs)))
    return dict(spike_analysis=spike, channels=["x", "y", "z", "yaw", "speed", "tilt", "aoffset"], per_step_delta_std_train=sd.tolist(), per_step_delta_mean_train=Yb.mean(0).tolist(), yaw_delta_quantiles_abs={q: float(np.percentile(a, q)) for q in (50, 90, 99, 99.9)}, yaw_delta_kurtosis=float(((d - d.mean()) ** 4).mean() / d.var() ** 2),
                yaw_delta_std_over_median_abs=float(d.std() / np.median(a)), frac_variance_from_top0p1pct=float(np.sort(d ** 2)[::-1][:max(1, int(0.001 * len(d)))].sum() / (d ** 2).sum()),
                loss_note="loss_oc: mse(normalised Y_b flattened 140 dims) + 0.5 mse(Y_c) + 0.5 bce(Y_e); every Y_b channel is divided by its train per-step-delta std, so each of the 7 object channels has EQUAL weight 1/7 of the object term; no per-channel or cumulative-yaw weighting; targets are per-step deltas (not cumulative angle)")
if __name__ == "__main__":
    res = dict(status="post-hoc diagnostic on frozen predictions; independent = INDEPENDENT windows (already evaluated), dev = DEVELOPMENT windows; no training, no selection", train_statistics=train_stats())
    for s in ("indep", "dev"): res[s] = analyse(load(s), s)
    (OUT / "yaw_audit.json").write_text(json.dumps(res, indent=1)); print("written")
