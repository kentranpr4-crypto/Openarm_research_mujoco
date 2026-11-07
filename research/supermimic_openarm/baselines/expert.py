"""Scripted Jacobian (damped least squares) reaching controller used ONLY to generate demonstrations for the behaviour-cloning baseline in the NEW reaching task.
It is unrelated to the ACT expert / datasets of the existing study and reads no stored data.  Pushing / pick-and-place experts are NOT implemented (BC is reaching-only)."""
import numpy as np
def reach_delta_action(env, gain=4.0, damping=0.05):
    """returns the normalised action a in [-1,1]^8 (delta parameterisation, action_scale 1) moving the TCP toward the goal"""
    mj, m, d, A = env.mj, env.m, env.d, env.adapter; jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mj.mj_jacBody(m, d, jp, jr, env.tcp_id); cols = A.vadr[:7]; J = jp[:, cols]
    err = env.goal - d.xpos[env.tcp_id]; v = gain * err; dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), v)                  # rad/s
    step = dq * (1.0 / env.hz); max_step = A.limits.max_speed[:7] * (1.0 / env.hz); a = np.zeros(8); a[:7] = np.clip(step / max_step, -1, 1); a[7] = 1.0       # gripper stays open
    return a
