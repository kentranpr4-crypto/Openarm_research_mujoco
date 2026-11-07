"""Batched, differentiable forward kinematics of the left-arm TCP, built from the MuJoCo model (hinge chains only; no upstream file is touched).
Verified against `mj_forward` in tests/test_kinematics.py.  Used by the world-model imagination so the TCP position is a differentiable function of the predicted joints."""
import numpy as np, torch
def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
class TorchFK(torch.nn.Module):
    """qpos of the chain joints (B, n_joints) in chain order -> TCP position (B, 3) in the world frame."""
    def __init__(self, model, tcp_body, joint_actuator_index):
        super().__init__(); import mujoco
        chain = []; b = model.body(tcp_body).id
        while b > 0: chain.append(b); b = int(model.body_parentid[b])
        chain = chain[::-1]; self.steps = []
        for bid in chain:
            R = _quat_to_mat(model.body_quat[bid]); p = model.body_pos[bid].copy(); js = None
            if model.body_jntnum[bid] > 1: raise ValueError("TorchFK supports at most one joint per body")
            if model.body_jntnum[bid] == 1:
                j = int(model.body_jntadr[bid]); 
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE: raise ValueError("TorchFK supports hinge joints only")
                act = joint_actuator_index[int(model.jnt_qposadr[j])]; js = (act, model.jnt_axis[j].copy(), model.jnt_pos[j].copy())
            self.steps.append((torch.tensor(R, dtype=torch.float64), torch.tensor(p, dtype=torch.float64), js))
        self.n_joints = len(joint_actuator_index)
    def forward(self, q):
        B = q.shape[0]; dt = q.dtype; Rw = torch.eye(3, dtype=dt, device=q.device).expand(B, 3, 3); pw = torch.zeros(B, 3, dtype=dt, device=q.device)
        for R, p, js in self.steps:
            R = R.to(dtype=dt, device=q.device); p = p.to(dtype=dt, device=q.device)
            pw = pw + torch.einsum("bij,j->bi", Rw, p); Rw = Rw @ R
            if js is not None:
                act, axis, anchor = js; a = torch.tensor(axis, dtype=dt, device=q.device); anc = torch.tensor(anchor, dtype=dt, device=q.device); th = q[:, act]; c, s = torch.cos(th), torch.sin(th)
                K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=dt, device=q.device)
                Rj = c[:, None, None] * torch.eye(3, dtype=dt, device=q.device) + s[:, None, None] * K + (1 - c)[:, None, None] * torch.outer(a, a)
                # rotation about the joint anchor: p_body' = anchor - Rj @ anchor
                pw = pw + torch.einsum("bij,bj->bi", Rw, anc - torch.einsum("bij,j->bi", Rj, anc)); Rw = Rw @ Rj
        return pw
