"""Safety limits for the 16-D position-target action (ctrl order of openarm_bimanual v1).

Hard limits come from the MJCF ctrlrange (read from the model, not hard-coded). Slew limits are PLACEHOLDER engineering
values chosen for the simulation, NOT hardware specifications: max joint speed 2.0 rad/s (arms), 0.20 m/s (v1 gripper
finger slide). Validate against the real motor/driver limits before any use on hardware."""
from dataclasses import dataclass
import numpy as np

ARM_MAX_SPEED_RAD_S = 2.0
GRIPPER_MAX_SPEED_M_S = 0.20

@dataclass(frozen=True)
class SafetyLimits:
    lo: np.ndarray            # (16,) ctrlrange low
    hi: np.ndarray            # (16,) ctrlrange high
    max_speed: np.ndarray     # (16,) units/s (rad/s for arm joints, m/s for v1 gripper)

    @classmethod
    def from_model(cls, model):
        names = [model.actuator(i).name for i in range(model.nu)]
        lo, hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
        speed = np.array([GRIPPER_MAX_SPEED_M_S if "finger" in n else ARM_MAX_SPEED_RAD_S for n in names])
        return cls(lo, hi, speed)

    def clip_and_slew(self, target, prev, dt):
        """Clamp to ctrlrange, then limit |target-prev| <= max_speed*dt. Returns (ctrl, n_clipped_range, n_slewed)."""
        t = np.asarray(target, float)
        c = np.clip(t, self.lo, self.hi); n_rng = int(np.count_nonzero(c != t))
        step = np.clip(c - prev, -self.max_speed * dt, self.max_speed * dt); n_slew = int(np.count_nonzero(step != c - prev))
        return prev + step, n_rng, n_slew
