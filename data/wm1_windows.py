#!/usr/bin/env python3
"""WM-1 window rule and loader (contract v1).  Episode record rows are aligned: row t of s/b/c is the observation BEFORE action t; a (T,8) executed left-arm targets.
Window at t (H=4, K=20): history rows t-H+1..t of s,b,c; PAST actions t-H..t-1; FUTURE actions t..t+K-1; targets rows t+1..t+K.  Valid t in [H, T-1-K]; never crosses an episode boundary."""
import numpy as np
H, K, STRIDE = 4, 20, 5
def window_starts(T, h=H, k=K, stride=STRIDE): return list(range(h, T - 1 - k + 1, stride))
def get_window(ep, t, h=H, k=K):
    s, b, c, a = ep["s"], ep["b"], ep["c"], ep["a"]; T = len(a); assert h <= t <= T - 1 - k, (t, T)
    X = dict(S_hist=s[t - h + 1:t + 1], B_hist=b[t - h + 1:t + 1], C_hist=c[t - h + 1:t + 1], A_hist=a[t - h:t], A_fut=a[t:t + k])
    Yd = lambda x: x[t + 1:t + k + 1] - x[t:t + k]                       # delta w.r.t. the previous row
    Y = dict(Y_s=Yd(s), Y_b=Yd(b), Y_c=c[t + 1:t + k + 1])
    gcmd = a[t:t + k, 7]; rise = b[t + k, 2] - b[0, 2]
    place = (np.linalg.norm(b[t + k, :2] - np.array([0.44, 0.30])) < 0.03) and abs(b[t + k, 2] - (0.364 + 0.02)) < 0.006
    Y["Y_e"] = np.array([float((gcmd < 0.028).any()), float((c[t + 1:t + k + 1, 4] > 0.01).any()), float(rise > 0.03), float(place)], np.float32)   # close cmd, palm contact, lifted>3cm, placed (at horizon end)
    return X, Y
