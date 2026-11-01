#!/usr/bin/env python3
"""DATA CONTRACT v2 (design + prototype; NOT used to rebuild any official dataset).  Row features, window builder with an explicit input/target-time separation, and the geometric yaw-validity mask.
Per-row object block b2 (14): [pos(3), Rb_x(3), Rb_y(3) (body x and y axes in the world: continuous, no wrap, no Euler singularity), speed(1), tilt_rad(1)=arccos(R22) in RADIANS, a_offset(1), heading_valid(1), hnorm(1)]  (heading_valid and hnorm use ONLY the current row).
Per-step targets (K): d pos(3), world rotation vector of q_{k+1} (x) conj(q_k) (3), d speed, d tilt_rad, d a_offset, + contact block (7, unchanged) + events (4, unchanged).  Legacy wrapped yaw delta is kept as AUXILIARY array (yaw_legacy) with a target-time validity mask for the loss-only ablations.
Input/target separation: X contains rows <= t only; masks that use rows > t (yaw_step_valid, event flags) live in Y_aux and are used only for LOSS WEIGHTING and diagnostic stratification, never as model inputs."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import geometry_v2 as Gm
TAU_H = 0.2          # heading-validity threshold on hnorm (set from the replay validation; see docs); heading amplification ~ 1/hnorm
H, K = 4, 20
def row_features(q, pos, speed, a_offset, tau=TAU_H):
    """q (T,4), pos (T,3), speed (T,), a_offset (T,) -> b2 (T,14), all from the same row only."""
    R = Gm.quat_to_R(q); hn = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2); b2 = np.concatenate([pos, R[:, :, 0], R[:, :, 1], speed[:, None], Gm.tilt_rad(q)[:, None], a_offset[:, None], (hn >= tau)[:, None].astype(float), hn[:, None]], 1)
    return b2.astype(np.float32)
B2_NAMES = ["x", "y", "z", "bx_x", "bx_y", "bx_z", "by_x", "by_y", "by_z", "speed", "tilt_rad", "a_offset", "heading_valid", "hnorm"]
def targets(q, pos, speed, a_offset, tau=TAU_H):
    """Per-step targets for transitions k=0..T-1 and target-time yaw-step validity mask (both ends valid)."""
    rv = Gm.step_rotvecs(q); R = Gm.quat_to_R(q); hn = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2); valid = (hn >= tau); yaw = Gm.yaw_euler(q)
    Y = np.concatenate([np.diff(pos, axis=0), rv, np.diff(speed)[:, None], np.diff(Gm.tilt_rad(q))[:, None], np.diff(a_offset)[:, None]], 1).astype(np.float32)      # (T,9)
    aux = dict(yaw_legacy=np.diff(yaw).astype(np.float32), yaw_step_valid=(valid[:-1] & valid[1:]), rot_step_geodesic_sym=Gm.geodesic_sym(q[:-1], q[1:]).astype(np.float32)); return Y, aux
def window(ep, t, h=H, k=K):
    """ep: dict with b2 (T+1,14), Y (T,9), aux, s (T,8), c (T,7), a (T,8).  Inputs use rows <= t (history) and actions (past and candidate future) only."""
    X = dict(S_hist=ep["s"][t - h + 1:t + 1], B2_hist=ep["b2"][t - h + 1:t + 1], C_hist=ep["c"][t - h + 1:t + 1], A_hist=ep["a"][t - h:t], A_fut=ep["a"][t:t + k])
    Yw = dict(Y_b2=ep["Y"][t:t + k]); Yaux = {kk: v[t:t + k] for kk, v in ep["aux"].items()}; return X, Yw, Yaux
