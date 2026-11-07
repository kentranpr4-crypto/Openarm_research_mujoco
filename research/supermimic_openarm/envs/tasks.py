"""Task definitions: goal sampling, resets, differentiable reward (torch), deterministic success criteria and the five specs every task exposes
(observation_spec, action_spec, reward_spec, termination_spec, success_spec, randomization_spec).
Rewards are analytic functions of the (predicted or real) state, so the same function is used for the simulator reward and inside imagined rollouts."""
import numpy as np, torch
from . import layout as L
TABLE_TOP_Z = 0.36; BOX_HALF = (0.025, 0.020, 0.020); BOX_REST_Z = TABLE_TOP_Z + BOX_HALF[2] + 0.0005; TRAY_C = (0.44, 0.30); TRAY_BASE_T = 0.004; TRAY_REST_Z = TABLE_TOP_Z + TRAY_BASE_T + BOX_HALF[2]
FALL_Z = TABLE_TOP_Z - 0.03; HOME_TCP = (0.402, 0.152, 0.456); FAR_BOX_XY = (0.60, -0.30)
def _f(x, like): return torch.as_tensor(x, dtype=like.dtype, device=like.device)
class Task:
    name = "base"; goal_dim = 0; max_steps = 100; uses_privileged_obs = False
    deployable_without_perception = True
    def __init__(self, **params): self.p = {**self.DEFAULTS, **params}
    DEFAULTS = {}
    # ---- sampling (numpy, deterministic given rng) --------------------------------------------------------------
    def sample(self, rng): raise NotImplementedError          # -> dict(box_xy, box_yaw, friction_scale, goal(np.ndarray))
    # ---- reward / actor observation (torch) -----------------------------------------------------------------------
    def reward_torch(self, s, a_exec, goal, tcp): raise NotImplementedError
    def actor_obs(self, s, goal, tcp):
        parts = [s[:, L.Q], s[:, L.DQ], s[:, L.PREV], tcp, goal]
        if self.uses_privileged_obs: parts += [s[:, L.BOX_POS], s[:, L.BOX_QUAT], s[:, L.BOX_VEL]]
        return torch.cat(parts, -1)
    @property
    def actor_obs_dim(self): return 8 + 8 + 8 + 3 + self.goal_dim + (10 if self.uses_privileged_obs else 0)
    def deployment_obs_keys(self): return ["q", "dq", "prev_ctrl", "tcp_pos", "goal"]
    # ---- success (numpy, simulator truth) ------------------------------------------------------------------------
    def success_now(self, ctx): raise NotImplementedError     # ctx: dict of numpy quantities from the simulator -> bool
    hold_steps = 5
    def failed(self, ctx): return bool(ctx["box_pos"][2] < FALL_Z)
    # ---- specs -----------------------------------------------------------------------------------------------------
    def observation_spec(self):
        t = [dict(name="q", dim=8, unit="rad|m", privileged=False, source="joint encoders"), dict(name="dq", dim=8, unit="rad/s|m/s", privileged=False, source="joint velocity"),
             dict(name="prev_ctrl", dim=8, unit="rad|m", privileged=False, source="last executed ctrl (known to the controller)"), dict(name="tcp_pos", dim=3, unit="m", privileged=False, source="forward kinematics of q"),
             dict(name="goal", dim=self.goal_dim, unit="m", privileged=False, source="task command")]
        if self.uses_privileged_obs: t += [dict(name="box_pos", dim=3, unit="m", privileged=True, source="simulator ground truth"), dict(name="box_quat", dim=4, unit="-", privileged=True, source="simulator ground truth"), dict(name="box_vel", dim=3, unit="m/s", privileged=True, source="simulator ground truth")]
        return dict(terms=t, actor_obs_dim=self.actor_obs_dim, uses_privileged_obs=self.uses_privileged_obs, deployable_without_perception=self.deployable_without_perception,
                    note="privileged terms come from the simulator; a real robot would need a perception system to provide them (NOT Sim2Real)")
    def action_spec(self):
        return dict(dim=L.ACT_DIM, space="[-1,1] (tanh)", meaning="left joints 1..7 + left gripper; delta or absolute joint-position target (config actor.action_param) -> clip to ctrlrange -> slew limit (control/safety_limits.py)",
                    executed_action="ctrl after clip+slew; the world model is trained and rolled out on this executed ctrl", right_arm="held at its home ctrl", control_hz=20, controller="MuJoCo position servos (OpenArm v1 actuators)")
    def reward_spec(self): return dict(kind="dense, analytic function of the next state", terms=self.REWARD_TERMS, differentiable=True)
    def termination_spec(self): return dict(terminated="box fell below the table (z < %.3f m) or non-finite simulation state" % FALL_Z, truncated=f"time limit {self.max_steps} control steps")
    def success_spec(self): return dict(definition=self.SUCCESS_TEXT, hold_steps=self.hold_steps, evaluated_with="simulator ground truth (never a policy input)")
    def randomization_spec(self): return dict(self.RANDOMIZATION)
class Reaching(Task):
    name = "reaching"; goal_dim = 3; max_steps = 100; uses_privileged_obs = False; deployable_without_perception = True
    DEFAULTS = dict(x=(0.32, 0.50), y=(0.05, 0.25), z=(0.44, 0.58), tol=0.02, speed_tol=0.10)
    REWARD_TERMS = ["-3*||tcp-goal||", "+2*exp(-(||tcp-goal||/0.08)^2)", "-0.002*||executed_step||^2/max_step^2 (smoothness)"]
    SUCCESS_TEXT = "||tcp - goal|| < 0.02 m and TCP speed < 0.10 m/s"
    RANDOMIZATION = dict(goal_x_m=(0.32, 0.50), goal_y_m=(0.05, 0.25), goal_z_m=(0.44, 0.58), box="parked far from the workspace", friction="n/a")
    def sample(self, rng):
        g = np.array([rng.uniform(*self.p["x"]), rng.uniform(*self.p["y"]), rng.uniform(*self.p["z"])]); return dict(box_xy=FAR_BOX_XY, box_yaw=0.0, friction_scale=1.0, goal=g)
    def reward_torch(self, s, a_exec, goal, tcp):
        d = (tcp - goal).norm(dim=-1); step = (a_exec - s[:, L.PREV]); return -3.0 * d + 2.0 * torch.exp(-(d / 0.08) ** 2) - 0.002 * (step ** 2).sum(-1) / (0.1 ** 2)
    def success_now(self, ctx): return bool(np.linalg.norm(ctx["tcp"] - ctx["goal"]) < self.p["tol"] and ctx["tcp_speed"] < self.p["speed_tol"])
    def failed(self, ctx): return False
class Pushing(Task):
    name = "pushing"; goal_dim = 2; max_steps = 150; uses_privileged_obs = True; deployable_without_perception = False
    DEFAULTS = dict(box_x=(0.38, 0.48), box_y=(0.06, 0.18), dist=(0.07, 0.13), angle_deg=(-45.0, 45.0), friction=(0.7, 1.3), tol=0.03, speed_tol=0.03, max_rise=0.01)
    REWARD_TERMS = ["-3*||box_xy-goal||", "-0.5*||tcp-box||", "+2*exp(-(||box_xy-goal||/0.04)^2)", "-1*max(0, box_z-rest-0.01)/0.01*0.1 (no lifting)"]
    SUCCESS_TEXT = "||box_xy - goal|| < 0.03 m, box speed < 0.03 m/s, box never lifted > 0.01 m (pushed, not carried)"
    RANDOMIZATION = dict(box_x_m=(0.38, 0.48), box_y_m=(0.06, 0.18), goal_distance_m=(0.07, 0.13), goal_angle_deg=(-45, 45), box_friction_scale=(0.7, 1.3))
    def sample(self, rng):
        b = np.array([rng.uniform(*self.p["box_x"]), rng.uniform(*self.p["box_y"])]); ang = np.deg2rad(rng.uniform(*self.p["angle_deg"])); r = rng.uniform(*self.p["dist"])
        g = np.clip(b + r * np.array([np.cos(ang), np.sin(ang)]), [0.33, 0.0], [0.56, 0.26]); return dict(box_xy=tuple(b), box_yaw=float(rng.uniform(-0.3, 0.3)), friction_scale=float(rng.uniform(*self.p["friction"])), goal=g)
    def reward_torch(self, s, a_exec, goal, tcp):
        bp = s[:, L.BOX_POS]; d = (bp[:, :2] - goal).norm(dim=-1); dt = (tcp - bp).norm(dim=-1); lift = torch.relu(bp[:, 2] - BOX_REST_Z - 0.01) / 0.01
        return -3.0 * d - 0.5 * dt + 2.0 * torch.exp(-(d / 0.04) ** 2) - 0.1 * lift
    def success_now(self, ctx): return bool(np.linalg.norm(ctx["box_pos"][:2] - ctx["goal"]) < self.p["tol"] and ctx["box_speed"] < self.p["speed_tol"] and ctx["box_max_rise"] <= self.p["max_rise"])
class PickPlace(Task):
    name = "pick_place"; goal_dim = 2; max_steps = 200; uses_privileged_obs = True; deployable_without_perception = False; hold_steps = 10
    DEFAULTS = dict(box_x=(0.38, 0.48), box_y=(0.06, 0.18), friction=(0.8, 1.2), region=0.05, z_tol=0.008, speed_tol=0.02, lift_required=0.03, open_m=0.03)
    REWARD_TERMS = ["-0.5*||tcp-box|| (until lifted)", "+1*closed*near", "+3*closed*clip((box_z-rest)/0.08,0,1)", "-2*||box_xy-tray|| * lifted", "+2*exp(-(||box_xy-tray||/0.04)^2) * open"]
    SUCCESS_TEXT = ("box lifted >= 0.03 m earlier, then: box centre within 0.05 m (xy) of the tray centre, |z - tray rest height| < 0.008 m, speed < 0.02 m/s, "
                    "left fingers open (>= 0.03 m) and NOT touching the box, held for 10 consecutive steps. Final position alone never counts.")
    RANDOMIZATION = dict(box_x_m=(0.38, 0.48), box_y_m=(0.06, 0.18), box_yaw_rad=(-0.3, 0.3), box_friction_scale=(0.8, 1.2), goal="tray centre (fixed)")
    def sample(self, rng):
        b = (float(rng.uniform(*self.p["box_x"])), float(rng.uniform(*self.p["box_y"]))); return dict(box_xy=b, box_yaw=float(rng.uniform(-0.3, 0.3)), friction_scale=float(rng.uniform(*self.p["friction"])), goal=np.array(TRAY_C))
    def reward_torch(self, s, a_exec, goal, tcp):
        bp = s[:, L.BOX_POS]; dtb = (tcp - bp).norm(dim=-1); closed = torch.clamp(1.0 - s[:, 7] / L.GRIPPER_OPEN_M, 0.0, 1.0); near = (dtb < 0.04).to(s.dtype); rise = torch.clamp((bp[:, 2] - BOX_REST_Z) / 0.08, 0.0, 1.0)
        dxy = (bp[:, :2] - goal).norm(dim=-1); lifted = torch.clamp((bp[:, 2] - BOX_REST_Z) / 0.03, 0.0, 1.0); opened = 1.0 - closed
        return -0.5 * dtb * (1.0 - rise) + closed * near + 3.0 * closed * rise - 2.0 * dxy * lifted + 2.0 * torch.exp(-(dxy / 0.04) ** 2) * opened
    def success_now(self, ctx):
        p = self.p; return bool(ctx["box_max_rise"] >= p["lift_required"] and np.linalg.norm(ctx["box_pos"][:2] - ctx["goal"]) < p["region"] and abs(ctx["box_pos"][2] - TRAY_REST_Z) < p["z_tol"]
                                and ctx["box_speed"] < p["speed_tol"] and ctx["gripper_open_m"] >= p["open_m"] and ctx["finger_contacts"] == 0)
TASK_CLASSES = {"reaching": Reaching, "pushing": Pushing, "pick_place": PickPlace}
def make_task(name, **params):
    if name not in TASK_CLASSES: raise ValueError(f"unknown task {name!r}; available: {sorted(TASK_CLASSES)}")
    t = TASK_CLASSES[name](**params); return t
