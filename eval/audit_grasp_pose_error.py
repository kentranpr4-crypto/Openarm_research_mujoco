#!/usr/bin/env python3
"""For every audited ACT rollout: compare ACT's left-arm command at the moment it starts closing the gripper with the scripted
expert's command at ITS closing start for the SAME box pose. Forward kinematics turns the joint-space difference into a TCP
difference and attributes the height (z) error to individual joints. Diagnostics only (uses expert plans = privileged info).
Run: MUJOCO_GL=osmesa python eval/audit_grasp_pose_error.py --rollouts outputs/eval/audit_v1"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, numpy as np, mujoco
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", type=Path, default=ROOT / "outputs/eval/audit_v1"); a = ap.parse_args()
from lab_env import LabEnv
from policies.baseline import ScriptedExpert
from ik import ArmIK
env = LabEnv(cameras=()); m = env.m; ex = ScriptedExpert(m); ik = ArmIK(m, "left"); dd = mujoco.MjData(m)
def q_from_ctrl(c):                       # invert ctrl = q + bias/kp by fixed point
    q = c.copy()
    for _ in range(5): q = c - (ik.gravity_comp(q) - q)
    return q
def tcp(q):
    mujoco.mj_resetData(m, dd)
    for ad, v in zip(ik.qadr, q): dd.qpos[ad] = v
    mujoco.mj_forward(m, dd); return dd.xpos[ik.tcp].copy(), dd.xmat[ik.tcp].reshape(3, 3).copy()
rows = []
for s in (7, 8, 9):
    for cond in ("nominal", "unseen_pose"):
        d = a.rollouts / f"seed{s}" / "act" / cond
        for e in map(json.loads, (d / "episodes.jsonl").read_text().splitlines()):
            z = np.load(d / "rollouts" / f"ep{e['episode_index']:04d}.npz"); act = z["action"]; tcl = np.nonzero(act[:, 7] < 0.028)[0]
            plan = ex.plan(tuple(e["box_xy"]), e["box_yaw"])["ctrl"]; tce = np.nonzero(plan[:, 7] < 0.028)[0]
            if not len(tcl) or not len(tce): continue
            ia, ie = max(tcl[0] - 1, 0), max(tce[0] - 1, 0); ca, ce = act[ia, ik.act], plan[ie, ik.act]; qe = q_from_ctrl(ce); qa = qe + (ca - ce)
            pe, Re = tcp(qe); pa, Ra = tcp(qa); per_joint_dz = []
            for j in range(7):
                q = qe.copy(); q[j] = qa[j]; per_joint_dz.append(float((tcp(q)[0] - pe)[2] * 100))
            cosang = np.clip((np.trace(Re.T @ Ra) - 1) / 2, -1, 1)
            rows.append(dict(seed=s, cond=cond, idx=e["episode_index"], success=bool(e["task_success"]), box_xy=e["box_xy"], t_close_act=int(tcl[0]), t_close_expert=int(tce[0]), joint_diff_rad=(ca - ce).round(4).tolist(),
                             fk_tcp_diff_cm=((pa - pe) * 100).round(2).tolist(), hand_rotation_error_deg=float(np.degrees(np.arccos(cosang))), per_joint_dz_cm=np.round(per_joint_dz, 2).tolist()))
J = np.array([r["joint_diff_rad"] for r in rows]); ok = np.array([r["success"] for r in rows]); F = np.array([r["fk_tcp_diff_cm"] for r in rows]); PJ = np.array([r["per_joint_dz_cm"] for r in rows])
summ = dict(n=len(rows), mean_abs_joint_diff_rad_success=np.abs(J[ok]).mean(0).round(4).tolist(), mean_abs_joint_diff_rad_fail=np.abs(J[~ok]).mean(0).round(4).tolist() if (~ok).any() else None,
            mean_fk_tcp_diff_cm_success=F[ok].mean(0).round(2).tolist(), mean_fk_tcp_diff_cm_fail=F[~ok].mean(0).round(2).tolist() if (~ok).any() else None,
            mean_abs_dz_cm_success=float(np.abs(F[ok, 2]).mean()), mean_abs_dz_cm_fail=float(np.abs(F[~ok, 2]).mean()) if (~ok).any() else None,
            hand_rotation_error_deg_success=float(np.mean([r["hand_rotation_error_deg"] for r in rows if r["success"]])), hand_rotation_error_deg_fail=float(np.mean([r["hand_rotation_error_deg"] for r in rows if not r["success"]])) if (~ok).any() else None,
            mean_per_joint_dz_cm_fail=PJ[~ok].mean(0).round(2).tolist() if (~ok).any() else None, mean_per_joint_dz_cm_success=PJ[ok].mean(0).round(2).tolist(),
            joint_names=[f"left_joint{i}" for i in range(1, 8)], note="ctrl-space difference (gravity-comp offsets cancel); FK on the expert joint solution + difference")
(a.rollouts / "grasp_pose_error.json").write_text(json.dumps(dict(summary=summ, episodes=rows), indent=1)); print(json.dumps(summ, indent=1))
