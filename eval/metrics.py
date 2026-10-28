"""Task labels and rollout metrics. Ground truth (box/TCP poses) is used HERE for evaluation and labelling only."""
import numpy as np

PICK_LIFT_M, HOLD_DIST_M = 0.05, 0.06          # box raised >=5 cm while within 6 cm of the TCP
PLACE_XY_TOL, PLACE_Z_TOL, REST_SPEED = 0.03, 0.006, 0.02   # box centre within 3 cm of tray centre, resting on its base

def task_labels(box_pos, tcp_pos, box_speed, tray_xy, tray_base_top_z, box_half_z=0.02):
    """box_pos/tcp_pos (T,3), box_speed (T,). Returns dict of bools + the numbers behind them (no hidden thresholds)."""
    box_pos, tcp_pos = np.asarray(box_pos), np.asarray(tcp_pos)
    rise = box_pos[:, 2] - box_pos[0, 2]
    dist = np.linalg.norm(box_pos - tcp_pos, axis=1)
    pick = bool(np.any((rise >= PICK_LIFT_M) & (dist < HOLD_DIST_M)))
    f = box_pos[-1]; dxy = float(np.linalg.norm(f[:2] - np.asarray(tray_xy)))
    dz = float(f[2] - (tray_base_top_z + box_half_z))
    place = bool(dxy < PLACE_XY_TOL and abs(dz) < PLACE_Z_TOL and box_speed[-1] < REST_SPEED)
    return dict(pick_success=pick, place_success=place, task_success=pick and place, max_rise_m=float(rise.max()),
                final_box_to_tray_xy_m=dxy, final_box_z_err_m=dz, final_box_speed=float(box_speed[-1]))

def tracking_error(state, action):
    """Joint tracking: |commanded target - measured joint| per step (rad for arm joints, m for gripper). state/action (T,16)."""
    e = np.abs(np.asarray(action) - np.asarray(state))
    return dict(mean_arm=float(e[:, [i for i in range(16) if i not in (7, 15)]].mean()), max_arm=float(e[:, [i for i in range(16) if i not in (7, 15)]].max()),
                mean_gripper=float(e[:, [7, 15]].mean()))

def smoothness(action, hz):
    """Mean |jerk| (third difference / dt^3) over arm joints, and mean |action step|; lower = smoother."""
    a = np.asarray(action)[:, [i for i in range(16) if i not in (7, 15)]]; dt = 1.0 / hz
    jerk = np.abs(np.diff(a, 3, axis=0)).mean() / dt**3 if len(a) > 4 else float("nan")
    return dict(mean_abs_jerk=float(jerk), mean_abs_step=float(np.abs(np.diff(a, axis=0)).mean()))

def latency_percentiles(ms):
    ms = np.asarray(ms); return {f"p{p}": float(np.percentile(ms, p)) for p in (50, 95, 99)}
