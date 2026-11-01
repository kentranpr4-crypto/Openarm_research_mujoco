#!/usr/bin/env python3
"""Offline counterfactual action-RANKING evaluation (decision utility) — pure numpy library: scoring function, autoregressive block rollout, selection, regret, pairwise accuracy, classification, calibration, bootstrap.
Candidate = one executed action sequence (one of the 25 registered variants of an initial state).  The world model (WM) is rolled out autoregressively in K=20-step blocks from t0 = H using ONLY (i) the true current-state rows t0-H+1..t0 (privileged-state WM input, disclosed) and (ii) the candidate's own action sequence;
no later ground-truth row, outcome label or intervention metadata is passed to the predictor (tested by perturbation).
FIXED SCORING FUNCTION (pre-registered; lower is better):  J = d_xy + |z_end - z_tray| + 0.05 * 1[palm force > 0.01 N at any row] + 0.05 * 1[max rise < 0.05 m]
  d_xy = distance of the box centre to the tray centre (0.44, 0.30) m at the last rolled-out row, z_tray = 0.384 m, palm = contact channel 4, rise = z - z(t0).  Weights are physical-unit choices fixed before any ranking result; no weight is fitted.
Deterministic tie-break: among candidates whose J differs by < 1e-9, the smallest SHA-1 of f"{pose_id}|{variant_id}|tie".   Tolerances come from eval/metrics.py (place 0.03 m / 0.006 m, pick lift 0.05 m)."""
import hashlib, numpy as np
TRAY = np.array([0.44, 0.30]); Z_TRAY = 0.384; PALM_THR = 0.01; LIFT_M = 0.05; W_PALM = 0.05; W_LIFT = 0.05; PLACE_XY = 0.03; PLACE_Z = 0.006; H, K = 4, 20; MEANINGFUL_PAIR = 0.01
def terms(b, c):
    """b (R,7), c (R,7): rows t0..t_end.  Returns dict(dxy, dz, palm, lifted, placed)."""
    fin = b[-1]; dxy = float(np.linalg.norm(fin[:2] - TRAY)); dz = float(abs(fin[2] - Z_TRAY)); palm = bool((c[:, 4] > PALM_THR).any()); lifted = bool((b[:, 2] - b[0, 2]).max() >= LIFT_M)
    return dict(dxy=dxy, dz=dz, palm=palm, lifted=lifted, placed=bool(dxy < PLACE_XY and dz < PLACE_Z))
def score(t): return t["dxy"] + t["dz"] + W_PALM * float(t["palm"]) + W_LIFT * float(not t["lifted"])
def tie_key(pose_id, vid): return int(hashlib.sha1(f"{pose_id}|{vid}|tie".encode()).hexdigest()[:12], 16)
def select(J, pose_id, vids, tol=1e-9):
    J = np.asarray(J, float); m = J.min(); cand = [i for i in range(len(J)) if J[i] - m < tol]; return min(cand, key=lambda i: tie_key(pose_id, vids[i]))
def regret(Jt, i): Jt = np.asarray(Jt, float); r = float(Jt[i] - Jt.min()); den = float(np.median(Jt) - Jt.min()); return r, (r / den if den > 1e-9 else float("nan"))
def pairwise_accuracy(Jp, Jt, thr=MEANINGFUL_PAIR):
    Jp, Jt = np.asarray(Jp, float), np.asarray(Jt, float); n = len(Jp); acc = []; 
    for i in range(n):
        for j in range(i + 1, n):
            dt = Jt[i] - Jt[j]
            if abs(dt) < thr: continue
            dp = Jp[i] - Jp[j]; acc.append(1.0 if dp * dt > 0 else (0.5 if abs(dp) < 1e-9 else 0.0))
    return float(np.mean(acc)) if acc else float("nan"), len(acc)
def kendall(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a); s = 0.0; c = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if abs(da) < 1e-12 or abs(db) < 1e-12: continue
            s += 1.0 if da * db > 0 else -1.0; c += 1
    return s / c if c else float("nan")
def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0; allv = np.concatenate([pos, neg])
    # average ranks for ties
    order = np.argsort(allv, kind="mergesort"); ranks = np.empty(len(allv)); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
def ece_brier(y, p, bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float); e = 0.0; b = np.minimum((p * bins).astype(int), bins - 1)
    for k in range(bins):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e), float(np.mean((p - y) ** 2))
def boot_mean(x, n=10000, seed=0, level=0.95):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return [float("nan")] * 3
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (n, len(x))); v = x[idx].mean(1); a = (1 - level) / 2; return [float(x.mean()), float(np.percentile(v, 100 * a)), float(np.percentile(v, 100 * (1 - a)))]
def boot_paired(x, y, n=10000, seed=0): d = np.asarray(x, float) - np.asarray(y, float); return boot_mean(d, n, seed)
# ------------------------------------------------------------------------------------------------ autoregressive block rollout (batched over candidates)
def rollout(predict, hist, A, lengths, pose, t0=H, yaw_mode="model", yaw_predict=None):
    """predict(W)->dict(Y_s (n,K,8) deltas, Y_b (n,K,7) deltas, Y_c (n,K,7) absolute, Y_e (n,4)).  hist: dict(S (N,H,8), B (N,H,7), C (N,H,7)) = TRUE rows t0-H+1..t0 only.  A (N,Tmax,8): candidate action sequences (zero-padded);
    lengths (N,): T_i.  Candidate i is rolled while t + K <= T_i - 1.  yaw_mode: 'model' (use the predicted yaw delta) | 'persist' (zero yaw delta) | 'external' (yaw_predict(W) -> (n,K) per-step yaw deltas).
    Returns per candidate dict(b (R,7), c (R,7), p_placed (float|nan), n_blocks) with row 0 = the true row t0 (copied from the input history) and predicted rows after it."""
    N = len(A); S = hist["S"].copy(); B = hist["B"].copy(); C = hist["C"].copy(); rows_b = [[B[i, -1].copy()] for i in range(N)]; rows_c = [[C[i, -1].copy()] for i in range(N)]; p_pl = np.full(N, np.nan); nb = np.zeros(N, int); t = t0
    while True:
        act = np.nonzero(t + K <= lengths - 1)[0]
        if len(act) == 0: break
        W = dict(S_hist=S[act], B_hist=B[act], C_hist=C[act], A_hist=A[act, t - H:t], A_fut=A[act, t:t + K], t=np.full(len(act), t), Y_e=np.zeros((len(act), 4), np.float32), pose=np.repeat(np.asarray(pose, np.float32)[None], len(act), 0))
        P = predict(W); dS = np.asarray(P["Y_s"], np.float64); dB = np.asarray(P["Y_b"], np.float64).copy(); Cn = np.asarray(P["Y_c"], np.float64)
        if yaw_mode == "persist": dB[:, :, 3] = 0.0
        elif yaw_mode == "external": dB[:, :, 3] = np.asarray(yaw_predict(W), np.float64)
        Sn = S[act, -1:, :] + np.cumsum(dS, 1); Bn = B[act, -1:, :] + np.cumsum(dB, 1)
        for k, i in enumerate(act): rows_b[i] += list(Bn[k]); rows_c[i] += list(Cn[k]); p_pl[i] = float(P["Y_e"][k, 3]); nb[i] += 1
        S[act] = Sn[:, -H:]; B[act] = Bn[:, -H:]; C[act] = Cn[:, -H:]; t += K
    return [dict(b=np.array(rows_b[i]), c=np.array(rows_c[i]), p_placed=p_pl[i], n_blocks=int(nb[i])) for i in range(N)]
