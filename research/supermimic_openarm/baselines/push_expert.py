"""Scripted pusher for the P1 pushing task (development / diagnostics only; it is NOT used to train the SuperMimic policy).  Standalone: works on `OpenArmTaskEnv`.
Strategy: hover behind the box on the box->goal line, descend slowly, push along the line at a capped TCP speed aiming 3 cm behind the goal point, then back away horizontally and rise.
Why speed caps: the failure analysis (docs/ai-native/tasks/2026-11-11-supermimic-p1-env-repair-report.md) found that uncapped approach speeds launched the box (0.3-0.6 m/s) and tipped it more than 1 cm.
Returns the normalised action a in [-1,1]^8 (delta parameterisation, identical to the policy action semantics).  Acceptance is MEASURED by p1_expert_eval.py / p1_validate_env.py."""
import numpy as np
MAX_STEP = np.array([2.0 * 0.05] * 7 + [0.20 * 0.05])                 # slew limits of control/safety_limits.py at 20 Hz (rad / step for the arm, m / step for the gripper)
DEFAULT_PARAMS = dict(z_hover=0.47, z_push=0.384, aim_offset=0.030, push_lead=0.01, behind=0.075, v_hover=0.40, v_descend=0.12, v_push=0.10, v_retreat=0.30, v_min=0.05, k_taper=1.5, gain=5.0, xy_tol=0.006, overshoot_tol=0.015, z_tol=0.008, goal_stop=0.018)
def jac_action(env, target, gain=4.0, damping=0.05, grip=1.0, vmax=None):
    mj, m, d, A = env.mj, env.m, env.d, env.adapter; jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mj.mj_jacBody(m, d, jp, jr, env.tcp_id); J = jp[:, A.vadr[:7]]
    v = gain * (np.asarray(target) - d.xpos[env.tcp_id])
    if vmax is not None:
        n = float(np.linalg.norm(v))
        if n > vmax: v = v * (vmax / n)                                                                   # Cartesian TCP speed cap (m/s)
    dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), v); a = np.zeros(8, np.float32); a[:7] = np.clip(dq * 0.05 / MAX_STEP[:7], -1, 1); a[7] = grip; return a
class PushExpert:
    """explicit phase machine (the earlier gate-based version could limit-cycle between 'push' and 'rise' once the descent was speed-capped):
    traverse (hover, to the point behind the box) -> descend (hold xy, lower to z_push) -> push (aim at a point `push_lead` ahead of the BOX CENTRE on the box->goal line, so lateral error is corrected continuously) -> retreat (back away horizontally, then rise)."""
    def __init__(self, **params): self.p = {**DEFAULT_PARAMS, **params}; self.phase = "traverse"
    def act(self, env):
        p = self.p; tcp = env.d.xpos[env.tcp_id].astype(float); box = env.d.qpos[env.box_adr:env.box_adr + 3].astype(float); g = np.asarray(env.goal, float)[:2]; dvec = g - box[:2]; n = np.linalg.norm(dvec); dirn = dvec / max(n, 1e-6)
        rel = tcp[:2] - box[:2]; along = float(rel @ dirn); lat = abs(float(rel[0] * dirn[1] - rel[1] * dirn[0])); behind = box[:2] - dirn * p["behind"]; G = p["gain"]
        if env.t == 0: self.phase = "traverse"; self.line0 = dirn.copy()
        overshoot = float((box[:2] - g) @ self.line0)                                                      # > 0: the box passed the goal along the original push line
        if self.phase == "push" and overshoot > p["overshoot_tol"]: self.phase = "retreat"
        if self.phase == "retreat" and n >= p["goal_stop"] and tcp[2] >= p["z_hover"] - 0.015: self.phase = "traverse"      # try again from the other side
        if n < p["goal_stop"] and self.phase != "retreat": self.phase = "retreat"; self.done_pushing = True
        if self.phase == "traverse" and np.linalg.norm(tcp[:2] - behind) < p["xy_tol"] and tcp[2] > p["z_hover"] - 0.03: self.phase = "descend"
        if self.phase == "descend" and tcp[2] < p["z_push"] + p["z_tol"]: self.phase = "push"
        if self.phase == "descend" and np.linalg.norm(tcp[:2] - behind) > 3 * p["xy_tol"]: self.phase = "traverse"          # box moved / drifted: re-align
        if self.phase == "retreat":
            if along > -0.09 and tcp[2] < p["z_hover"] - 0.02: return jac_action(env, np.array([*(tcp[:2] - dirn * 0.04), tcp[2]]), gain=G, grip=-1.0, vmax=p["v_retreat"])
            return jac_action(env, np.array([tcp[0], tcp[1], p["z_hover"]]), gain=G, grip=-1.0, vmax=p["v_retreat"])
        if self.phase == "push": return jac_action(env, np.array([*(box[:2] + dirn * p["push_lead"]), p["z_push"]]), gain=G, grip=-1.0, vmax=min(p["v_push"], p["v_min"] + p["k_taper"] * n))
        if self.phase == "descend": return jac_action(env, np.array([*behind, p["z_push"]]), gain=G, grip=-1.0, vmax=p["v_descend"])
        return jac_action(env, np.array([*behind, p["z_hover"]]), gain=G, grip=-1.0, vmax=p["v_hover"])
