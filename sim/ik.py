"""Damped least-squares IK for one OpenArm arm (7 hinge joints), position + orientation, with joint-limit clipping.
Pure geometry on a private MjData: it never touches the live simulation. Shared by lab_scene checks and the expert."""
import numpy as np, mujoco

class ArmIK:
    def __init__(self, model, side):
        self.m, self.side = model, side
        self.tcp = model.body(f"openarm_{side}_hand_tcp").id
        self.jids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                     and model.body(model.jnt_bodyid[j]).name.startswith(f"openarm_{side}_link")]
        assert len(self.jids) == 7, (side, len(self.jids))
        self.qadr = [model.jnt_qposadr[j] for j in self.jids]
        self.dofs = [model.jnt_dofadr[j] for j in self.jids]
        self.act = [model.actuator(f"{side}_joint{i}_ctrl").id for i in range(1, 8)]
        self.kp = np.array([model.actuator_gainprm[a][0] for a in self.act])

    def solve(self, dd, pos, R_target, iters=400, tol_p=5e-4, tol_r=5e-3, orient=True):
        """Mutates dd.qpos (arm joints only) towards TCP pose. Returns True if converged."""
        m = self.m
        qt = np.zeros(4); mujoco.mju_mat2Quat(qt, np.asarray(R_target, float).ravel())
        for _ in range(iters):
            mujoco.mj_forward(m, dd)
            ep = pos - dd.xpos[self.tcp]
            qc = np.zeros(4); mujoco.mju_mat2Quat(qc, dd.xmat[self.tcp]); er = np.zeros(3)
            mujoco.mju_subQuat(er, qt, qc); er = dd.xmat[self.tcp].reshape(3, 3) @ er
            if np.linalg.norm(ep) < tol_p and (not orient or np.linalg.norm(er) < tol_r): return True
            Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, dd, Jp, Jr, self.tcp)
            J = np.vstack([Jp, Jr] if orient else [Jp])[:, self.dofs]
            e = np.r_[ep, er] if orient else ep
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            for k, j in enumerate(self.jids):
                lo, hi = m.jnt_range[j]; dd.qpos[self.qadr[k]] = np.clip(dd.qpos[self.qadr[k]] + 0.5 * dq[k], lo, hi)
        return False

    def q(self, dd): return np.array([dd.qpos[a] for a in self.qadr])

    def gravity_comp(self, q):
        """Static balance of a position servo: kp*(ctrl-q) = qfrc_bias  ->  ctrl = q + bias/kp (model-based, see grasp report)."""
        dd = mujoco.MjData(self.m); mujoco.mj_resetData(self.m, dd)
        for a, v in zip(self.qadr, q): dd.qpos[a] = v
        mujoco.mj_forward(self.m, dd)
        return q + dd.qfrc_bias[self.dofs] / self.kp
