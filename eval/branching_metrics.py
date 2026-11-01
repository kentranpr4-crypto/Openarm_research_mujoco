#!/usr/bin/env python3
"""Numpy-only metrics for the offline branching replay (unit-tested on synthetic cases in eval/test_branching_replay.py).  Unit of resampling = initial-state pose; branches/states are never treated as independent."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import decision_utility_lib as L
THR_PAIR = 0.01; TAU = 0.01; TOL = 1e-9
def pair_counts(Jp, Jt, thr=THR_PAIR):
    """(n_correct, n_pairs) over pairs with |dJ_true| >= thr; a predicted tie (|dJp|<1e-9) scores 0.5."""
    Jp, Jt = np.asarray(Jp, float), np.asarray(Jt, float); c = 0.0; n = 0
    for i in range(len(Jp)):
        for j in range(i + 1, len(Jp)):
            dt = Jt[i] - Jt[j]
            if abs(dt) < thr: continue
            dp = Jp[i] - Jp[j]; n += 1; c += 1.0 if dp * dt > 0 else (0.5 if abs(dp) < 1e-9 else 0.0)
    return c, n
def state_metrics(Jp, Jt, pose_id):
    """Jp/Jt (3,) predicted and realised J of the candidates [default, ...]. Regrets are vs the realised oracle (min Jt)."""
    Jp, Jt = np.asarray(Jp, float), np.asarray(Jt, float); mn = float(Jt.min()); sel = L.select(Jp, pose_id, list(range(len(Jp))), tol=TOL); s = np.sort(Jp); margin = float(s[1] - s[0]); den = float(np.median(Jt) - mn)
    gated = sel if margin >= TAU else 0
    f = lambda i: float(Jt[i] - mn)
    return dict(headroom=f(0), regret_wm=f(sel), regret_default=f(0), regret_random=float(Jt.mean() - mn), regret_gated=f(gated), selected=int(sel), gated_selected=int(gated), margin=margin, norm_den=den if den > 1e-9 else None,
                norm_regret_wm=(f(sel) / den if den > 1e-9 else None), norm_regret_default=(f(0) / den if den > 1e-9 else None), norm_regret_random=(float(Jt.mean() - mn) / den if den > 1e-9 else None), has_headroom=bool(f(0) >= THR_PAIR), pair=pair_counts(Jp, Jt))
def boot_ratio(num_by_pose, den_by_pose, n=10000, seed=0):
    """Pooled ratio sum(num)/sum(den) with percentile CI, resampling poses (dict pose -> value)."""
    ks = sorted(set(num_by_pose) | set(den_by_pose)); a = np.array([num_by_pose.get(k, 0.0) for k in ks], float); b = np.array([den_by_pose.get(k, 0.0) for k in ks], float)
    if b.sum() == 0: return [float("nan")] * 3
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ks), (n, len(ks))); sa, sb = a[idx].sum(1), b[idx].sum(1); ok = sb > 0; v = sa[ok] / sb[ok]
    return [float(a.sum() / b.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def boot_mean_by_pose(vals_by_pose, n=10000, seed=0):
    """Mean over poses of the per-pose mean, percentile CI over poses (dict pose -> list of per-state values)."""
    ks = sorted(vals_by_pose); m = np.array([np.mean(vals_by_pose[k]) for k in ks], float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ks), (n, len(ks))); v = m[idx].mean(1); return [float(m.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def boot_paired_by_pose(a_by_pose, b_by_pose, n=10000, seed=0):
    ks = sorted(set(a_by_pose) & set(b_by_pose)); d = {k: [x - y for x, y in zip(a_by_pose[k], b_by_pose[k])] for k in ks}; return boot_mean_by_pose(d, n, seed)
