"""Policy action -> executed ctrl.  A differentiable torch mirror of the repository's `control.safety_limits.SafetyLimits.clip_and_slew`
(clip to ctrlrange, then limit |ctrl - prev| <= max_speed * dt).  The simulator-side path always goes through the ORIGINAL numpy `ActionAdapter.to_ctrl`;
this mirror is used only inside imagined rollouts and is tested for exact parity (tests/test_adapter.py).
The learned actor controls the LEFT arm + left gripper (8 dims); the right arm is held at its home ctrl (documented in the task action_spec)."""
import numpy as np, torch
LEFT = slice(0, 8)
class TorchAdapter:
    def __init__(self, limits, dt, action_param="delta", action_scale=1.0, device="cpu", dtype=torch.float32):
        t = lambda a: torch.as_tensor(np.asarray(a)[LEFT], dtype=dtype, device=device)
        self.lo, self.hi, self.max_step = t(limits.lo), t(limits.hi), t(limits.max_speed) * dt; self.action_param, self.action_scale = action_param, float(action_scale); self.dt = dt
    def to(self, device, dtype=None):
        for k in ("lo", "hi", "max_step"): setattr(self, k, getattr(self, k).to(device=device, dtype=dtype or getattr(self, k).dtype))
        return self
    def target_from_action(self, a, prev):
        """a in [-1,1]^8 -> absolute joint-position target (before clip/slew)."""
        if self.action_param == "delta": return prev + a * self.max_step * self.action_scale
        return self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)
    def execute(self, target, prev):
        c = torch.minimum(torch.maximum(target, self.lo), self.hi); step = torch.minimum(torch.maximum(c - prev, -self.max_step), self.max_step); return prev + step
    def act(self, a, prev): return self.execute(self.target_from_action(a, prev), prev)
