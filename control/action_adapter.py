"""Action / state contract between learned policies, scripted expert and the MuJoCo simulator (OpenArm v1 bimanual).

ACTION (16,)  = absolute joint-position targets in ACTUATOR order, i.e. exactly what is written to `data.ctrl`:
    [left_joint1..7 (rad), left_finger1 (m), right_joint1..7 (rad), right_finger1 (m)]
  The actuators are position servos (audit: docs/openarm-mujoco-model-audit.md); v1 gripper unit is METRES (v2 would be rad,
  so v1 checkpoints must never be reused on v2). The expert's ctrl already contains a model-based gravity-compensation
  offset (ctrl = q + qfrc_bias/kp), so a policy trained on it learns that offset implicitly from the state.
STATE  (16,)  = qpos of the joint each actuator drives (actuator_trnid -> jnt_qposadr), same order/units as the action.
          (nq=18 contains a redundant finger_joint2 tied by an equality constraint; it is dropped on purpose.)
VEL    (16,)  = qvel of the same joints.
Never index ctrl and qpos with the same integer (indices differ from the right arm onward)."""
import numpy as np, mujoco
from control.safety_limits import SafetyLimits

class ActionAdapter:
    def __init__(self, model, control_hz=20):
        self.m = model; self.dt = 1.0 / control_hz
        assert model.nu == 16, f"expects OpenArm v1 bimanual (nu=16), got nu={model.nu}"
        self.names = [model.actuator(i).name for i in range(model.nu)]
        jid = model.actuator_trnid[:, 0]
        assert np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        self.qadr = model.jnt_qposadr[jid].copy(); self.vadr = model.jnt_dofadr[jid].copy()
        self.limits = SafetyLimits.from_model(model)
        self.gripper_idx = [i for i, n in enumerate(self.names) if "finger" in n]
        self.joint_names = [model.joint(j).name for j in jid]

    def state(self, data):      return data.qpos[self.qadr].astype(np.float32).copy()
    def velocity(self, data):   return data.qvel[self.vadr].astype(np.float32).copy()

    def to_ctrl(self, action, prev_ctrl):
        """Policy action -> safe ctrl. Rejects NaN/Inf and wrong shape (fail loud), clips to ctrlrange, limits slew."""
        a = np.asarray(action, float)
        if a.shape != (16,) or not np.isfinite(a).all(): raise ValueError(f"bad action {a.shape} finite={np.isfinite(a).all()}")
        return self.limits.clip_and_slew(a, np.asarray(prev_ctrl, float), self.dt)
