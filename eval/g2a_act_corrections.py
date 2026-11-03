#!/usr/bin/env python3
"""G2a: ACT-anchored grasp CORRECTION experiment (DEVELOPMENT; ORACLE-ASSISTED causal intervention; simulation only; no training; NOT a deployable policy).
At ONE decision state per pose (start of the 20-step ACT chunk at t=40) the unmodified ACT (c0_seed0) chunk is turned into 9 candidates: ACT raw joint targets -> remove gravity-comp offset -> forward kinematics -> geometric correction of the TCP target (depth / lateral / yaw in the box frame) -> IK ->
gravity-comp -> absolute actuator targets -> ctrlrange clip -> sequential slew.  Candidate c0 is the untouched ACT.  Corrections persist as a constant joint-space offset on the following ACT chunks (60, 80, 100) which are queried normally.  Branches start from a bit-exact MuJoCo snapshot (mj_copyData).
Oracle elements: the box pose used for the geometric reference.  Run (ACT venv): python eval/g2a_act_corrections.py run"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
import mujoco, lab_scene
import closed_loop_pilot as P
import branching_replay as BR
from policies.baseline import ScriptedExpert, G_OPEN, GRASP_DZ, _rz
OUT = ROOT / "outputs/g2a_act_corrections"; FREEZE = OUT / "protocol_freeze.json"; LEDGER = OUT / "cpu_ledger.json"; CAP_TOTAL_S = 0.25 * 3600; DT = 0.05
POLICY = "c0_s0"; T_DEC = 40; T_END = 100; CHUNK = 20; N_POSES = 48; CLOSE_TRIG = 0.040
CLIP = dict(df=(-0.0364, 0.0366), dn=(-0.0156, 0.0127), dpsi=(-0.182, 0.187))        # pooled ACT-expert error p5/p95 from the READ-ONLY audit (outcome-agnostic)
RAMP_EARLY = 6; RAMP_LATE = 4; DELAY_STEPS = 8; FIXED_DEPTH = 0.020
# candidate table: kind in {nominal, cancel, fixed, delay}; dims = which geometric components are corrected; timing in {early, late}
CANDS = [dict(id="c0", kind="nominal"), dict(id="c1", kind="cancel", dims=("f",), timing="early"), dict(id="c2", kind="cancel", dims=("f",), timing="late"), dict(id="c3", kind="cancel", dims=("f", "n", "psi"), timing="early"),
         dict(id="c4", kind="delay"), dict(id="c5", kind="fixed", dims=("f",), value=-FIXED_DEPTH, timing="early"), dict(id="c6", kind="fixed", dims=("f",), value=+FIXED_DEPTH, timing="early"), dict(id="c7", kind="cancel", dims=("n",), timing="early"), dict(id="c8", kind="cancel", dims=("psi",), timing="early")]
RISE_MIN, HOLD_DIST, SLIP_MAX, DROP_RISE, PALM_THR = 0.05, 0.06, 0.01, 0.02, 0.01; CLOSURE_AFTER, HOLD_FROM, HOLD_LEN = 14, 8, 8; DISTINCT = dict(arm=0.02, tcp=0.003)
def ledger_used(): return float(sum(e["cpu_s"] for e in json.loads(LEDGER.read_text()))) if LEDGER.exists() else 0.0
def ledger_add(tag, cpu_s):
    L = json.loads(LEDGER.read_text()) if LEDGER.exists() else []; L.append(dict(tag=tag, cpu_s=float(cpu_s), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))); P.atomic_json(LEDGER, L)
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def sample_poses():
    """48 development poses: separate RNG namespace; rejected within Chebyshev 0.003 (x,y,yaw) of ANY previously used initial state (all episodes.jsonl poses, wm datasets, indep manifest, G1 poses); 32 'hard' (x 0.34-0.38) + 16 'mid' (0.38-0.47); no outcome/model looked at."""
    import indep_manifest as IM
    used = [s for v in IM.used_states().values() for s in v] + [(p["x"], p["y"], p["yaw"]) for p in json.loads((ROOT / "outputs/indep/manifest.json").read_text())["poses"]] + [(p["x"], p["y"], p["yaw"]) for p in json.loads((ROOT / "outputs/g1_grasp/protocol_freeze.json").read_text())["poses"]]
    U = np.array(used); rng = np.random.default_rng([20261103, 9201]); poses = []; pid = 3000
    for stratum, xr, n in (("hard", (0.34, 0.38), 32), ("mid", (0.38, 0.47), 16)):
        got = 0
        while got < n:
            p = np.round(np.array([rng.uniform(*xr), rng.uniform(0.07, 0.13), rng.uniform(-0.20, 0.20)]), 6); cur = np.array([[q["x"], q["y"], q["yaw"]] for q in poses]) if poses else np.zeros((0, 3))
            if np.abs(U - p).max(1).min() < 0.003 or (len(cur) and np.abs(cur - p).max(1).min() < 0.003): continue
            poses.append(dict(idx=pid, stratum=stratum, x=float(p[0]), y=float(p[1]), yaw=float(p[2]))); pid += 1; got += 1
    return poses, float(np.abs(U[:, None, :] - np.array([[q["x"], q["y"], q["yaw"]] for q in poses])[None]).max(2).min())
# ------------------------------------------------------------------------------------------------------------------------------ kinematics + correction
class Kin:
    """FK / gravity-comp inversion / IK on private MjData (never touches the live simulation)."""
    def __init__(self, ex):
        self.ex, self.ik, self.m = ex, ex.ik, ex.m; self.dd = mujoco.MjData(self.m); mujoco.mj_resetData(self.m, self.dd); self.dq = mujoco.MjData(self.m); mujoco.mj_resetData(self.m, self.dq)
    def _set(self, d, q):
        for a, v in zip(self.ik.qadr, q): d.qpos[a] = v
    def bias(self, q): self._set(self.dd, q); mujoco.mj_forward(self.m, self.dd); return self.dd.qfrc_bias[self.ik.dofs] / self.ik.kp
    def q_from_ctrl(self, c):
        q = np.asarray(c, float).copy()
        for _ in range(3): q = np.asarray(c, float) - self.bias(q)
        return q
    def ctrl_from_q(self, q): return np.asarray(q, float) + self.bias(q)
    def fk(self, q): self._set(self.dd, q); mujoco.mj_forward(self.m, self.dd); return self.dd.xpos[self.ik.tcp].copy(), self.dd.xmat[self.ik.tcp].reshape(3, 3).copy()
    def ik_to(self, q0, p, R):
        self._set(self.dq, q0); ok = self.ik.solve(self.dq, p, R); mujoco.mj_forward(self.m, self.dq); qt = np.zeros(4); mujoco.mju_mat2Quat(qt, np.asarray(R).ravel()); qc = np.zeros(4); mujoco.mju_mat2Quat(qc, self.dq.xmat[self.ik.tcp]); er = np.zeros(3); mujoco.mju_subQuat(er, qt, qc)
        return self.ik.q(self.dq), float(np.linalg.norm(p - self.dq.xpos[self.ik.tcp])), float(np.linalg.norm(er)), bool(ok)
def box_axes(yaw): return np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
def close_step(raw): idx = np.where(raw[:, 7] < CLOSE_TRIG)[0]; return int(idx[0]) if len(idx) else None
def oracle_error(kin, raw, box_xy, yaw):
    """ACT-commanded TCP at its own close step vs the expert grasp point (box frame): depth e_f [m] (expert target df=+0.01), lateral e_n [m], hand-yaw error e_psi [rad]. Uses the raw chunk and the initial box pose only."""
    j = close_step(raw)
    if j is None: return None
    q = kin.q_from_ctrl(raw[j, kin.ik.act]); p, R = kin.fk(q); f, n = box_axes(yaw); c = np.array(box_xy); rel = R @ kin.ex.R0.T; psi = float(np.arctan2(rel[1, 0], rel[0, 0]))
    return dict(j_close=j, e_f=float((p[:2] - c) @ f - 0.01), e_n=float((p[:2] - c) @ n), e_psi=float(wrap(psi - yaw)))
def weights(timing, j_close):
    s = np.arange(CHUNK)
    if timing == "early": return np.minimum(1.0, (s + 1) / RAMP_EARLY)
    st = max(0, j_close - RAMP_LATE); return np.clip((s - st + 1) / RAMP_LATE, 0.0, 1.0)
def correction_values(cd, err):
    if cd["kind"] == "cancel": return dict(f=(-float(np.clip(err["e_f"], *CLIP["df"])) if "f" in cd["dims"] else 0.0), n=(-float(np.clip(err["e_n"], *CLIP["dn"])) if "n" in cd["dims"] else 0.0), psi=(-float(np.clip(err["e_psi"], *CLIP["dpsi"])) if "psi" in cd["dims"] else 0.0))
    if cd["kind"] == "fixed": return dict(f=cd["value"], n=0.0, psi=0.0)
    return dict(f=0.0, n=0.0, psi=0.0)
def correct_chunk(kin, raw, box_xy, yaw, cd, err):
    """-> (modified raw chunk (20,16), dq_off (7) persistent joint offset or None, residual stats). Only left-arm joint targets and (delay) the left gripper channel are modified."""
    out = raw.copy(); res = dict(max_pos_err_m=0.0, max_rot_err_rad=0.0, ik_fail=0, n_ik=0, applied=correction_values(cd, err) if err else None); act = kin.ik.act; g = kin.ex.g_l
    if cd["kind"] == "nominal": return out, None, res
    if cd["kind"] == "delay":
        j = err["j_close"]; k = DELAY_STEPS; open_val = raw[max(j - 1, 0), g]; col = raw[:, g].copy()
        for t in range(j, CHUNK): col[t] = raw[t - k, g] if t - k >= j else open_val
        out[:, g] = col; return out, None, res
    v = res["applied"]; f, n = box_axes(yaw); delta = np.r_[v["f"] * f + v["n"] * n, 0.0]; w = weights(cd["timing"], err["j_close"]); dq_off = None
    for s in range(CHUNK):
        if w[s] <= 0: continue
        q = kin.q_from_ctrl(raw[s, act]); p, R = kin.fk(q); R2 = _rz(w[s] * v["psi"]) @ R; q2, pe, re, ok = kin.ik_to(q, p + w[s] * delta, R2)
        out[s, act] = kin.ctrl_from_q(q2); res["max_pos_err_m"] = max(res["max_pos_err_m"], pe); res["max_rot_err_rad"] = max(res["max_rot_err_rad"], re); res["ik_fail"] += int(not (ok and pe < 2e-3 and re < 0.03)); res["n_ik"] += 1
    dq_off = out[CHUNK - 1, act] - raw[CHUNK - 1, act]
    return out, dq_off, res
# ------------------------------------------------------------------------------------------------------------------------------ execution + outcome
PEN_MAX = 0.002          # collision validity: robot-obstacle (desk/tray/room) penetration deeper than 2 mm at any step makes a candidate invalid; shallower grazes are recorded only
def NEW_ROWS(): return dict(box=[], tcp=[], hf=[], nf=[], ff=[], coll=[], pen=[])
class Runner:
    def __init__(self):
        import torch; torch.set_num_threads(1); from control.safety_limits import SafetyLimits
        self.policy = P.load_chunk_policies(torch, [POLICY])[POLICY]; self.env = P.make_env(self.policy.cams); self.m = self.env.m; self.ex = ScriptedExpert(self.m, control_hz=20); self.kin = Kin(self.ex); self.lim = SafetyLimits.from_model(self.m); self.obs = P.LiveObserver(self.env)
        bn = [self.m.body(self.m.geom_bodyid[g]).name for g in range(self.m.ngeom)]; self.robot_g = np.array([n.startswith("openarm_") for n in bn]); self.obst_g = np.array([n in ("desk", "tray", "room") for n in bn]); assert self.robot_g.sum() > 10 and self.obst_g.sum() >= 3
    def obstacle_penetration(self):
        """max penetration depth [m] over robot-obstacle contact points (0.0 when no robot-obstacle contact)."""
        d = self.env.d; pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (self.robot_g[c.geom1] and self.obst_g[c.geom2]) or (self.robot_g[c.geom2] and self.obst_g[c.geom1]): pen = max(pen, -float(c.dist))
        return pen
    def obstacle_contact(self): return self.obstacle_penetration() > 0.0 or self._any_contact()
    def _any_contact(self):
        d = self.env.d
        for i in range(d.ncon):
            c = d.contact[i]
            if (self.robot_g[c.geom1] and self.obst_g[c.geom2]) or (self.robot_g[c.geom2] and self.obst_g[c.geom1]): return True
        return False
    def record(self, R):
        i = self.env.info(); b, c = self.obs.row(); R["box"].append(i["box_pos"]); R["tcp"].append(i["tcp_pos"]); R["hf"].append(float(c[4])); R["nf"].append(float(c[1])); R["ff"].append(float(c[0] * 50.0)); pen = self.obstacle_penetration(); R["pen"].append(pen); R["coll"].append(pen > 0.0 or self._any_contact())
    def run_chunk(self, raw_mod, prev, R):
        """clip->slew chain then execute 20 steps (rows recorded at step starts); returns executed (20,16) and the last command."""
        exe, nr, ns = P.slew_chain(self.lim, raw_mod, prev)
        for u in exe: self.record(R); P.lean_step(self.env, u, False)
        return exe, exe[-1], nr, ns
    def prefix(self, pose):
        """unmodified ACT chunks at t=0 and t=20, then the observation at t=40 and the shared raw chunk of the decision state."""
        env = self.env; env.set_perturbation(); o = env.reset((pose["x"], pose["y"]), pose["yaw"]); prev = np.asarray(env.prev_ctrl, float).copy(); R = NEW_ROWS(); exes = []
        for t in range(0, T_DEC, CHUNK):
            raw = self.policy.chunk(o["state"], o["images"]); exe, prev, _, _ = self.run_chunk(raw, prev, R); exes.append(exe); o = BR.obs_full(env)
        return o, prev, R, np.concatenate(exes)
    def branch(self, snap, prev, raw40_mod, dq_off, o40_unused=None):
        """execute one candidate from the snapshot: corrected chunk at t=40, then ACT chunks at 60/80/100 (queried from the live observation) + persistent joint offset."""
        env = self.env; BR.restore(env, snap); R = NEW_ROWS(); exe, prev_c, nr, ns = self.run_chunk(raw40_mod, prev, R); exes = [exe]; n_clip, n_slew = nr, ns; act = self.kin.ik.act
        for t in range(T_DEC + CHUNK, T_END, CHUNK):
            o = BR.obs_full(env); raw = self.policy.chunk(o["state"], o["images"])
            if dq_off is not None: raw = raw.copy(); raw[:, act] += dq_off
            exe, prev_c, nr, ns = self.run_chunk(raw, prev_c, R); exes.append(exe); n_clip += nr; n_slew += ns
        self.record(R); return {k: np.array(v) for k, v in R.items()}, np.concatenate(exes), n_clip, n_slew
def assess(R, exe, z0, t0=T_DEC):
    """Frozen ACT-compatible lift-and-hold definition. Rows: row k = observation before step t0+k (81 rows); exe = executed targets of steps t0..T_END-1."""
    n = len(R["box"]); tc = np.where(exe[:, 7] < CLOSE_TRIG)[0]; rise = R["box"][:, 2] - z0; sep = R["box"] - R["tcp"]; out = dict(t_close=(int(tc[0]) + t0 if len(tc) else None))
    closed = bool(len(tc) and tc[0] + CLOSURE_AFTER < n and R["nf"][tc[0] + CLOSURE_AFTER] >= 2); up = np.where(rise >= RISE_MIN)[0]; kl = int(up[0]) if len(up) else None; hold_ok = kl is not None and kl + HOLD_FROM + HOLD_LEN < n
    lifted = near = slipped = dropped = False; slip = None
    if hold_ok:
        W = slice(kl + HOLD_FROM, kl + HOLD_FROM + HOLD_LEN + 1); lifted = bool((rise[W] >= RISE_MIN).all()); near = bool((np.linalg.norm(sep[W], axis=1) < HOLD_DIST).all()); slip = float(np.linalg.norm(sep[W] - sep[W].mean(0), axis=1).max()); slipped = slip > SLIP_MAX
    if kl is not None: dropped = bool((rise[kl:] < DROP_RISE).any())
    success = bool(closed and hold_ok and lifted and near and not slipped and not dropped)
    reason = "success" if success else ("not_closed" if not closed else "not_lifted" if kl is None else "late_lift" if not hold_ok else "dropped" if dropped else "not_lifted" if not (lifted and near) else "slipped")
    hf = np.where(R["hf"] > PALM_THR)[0]; fc = np.where(R["nf"] > 0)[0]; k = int(tc[0]) if len(tc) else None
    out.update(success=success, reason=reason, closed=closed, t_lift=(kl + t0 if kl is not None else None), slip_m=slip, dropped=dropped, palm=bool(len(hf)), palm_first_step=(int(hf[0]) + t0 if len(hf) else None), finger_first_step=(int(fc[0]) + t0 if len(fc) else None), palm_before_finger=(None if not len(hf) else bool((not len(fc)) or hf[0] < fc[0])),
               palm_force_max_N=float(R["hf"].max()), finger_force_max_N=float(R["ff"].max()), obstacle_contact=bool(R["coll"].any()), obstacle_penetration_max_m=float(R["pen"].max()), rise_max_m=float(rise.max()))
    if k is not None and k < n:
        box_yaw_unused = None; d = R["tcp"][k] - R["box"][k]; out.update(tcp_box_dist_cm_at_close=float(np.linalg.norm(d) * 100), tcp_speed_ms_at_close=float(np.linalg.norm(R["tcp"][min(k + 1, n - 1)] - R["tcp"][max(k - 1, 0)]) / (2 * DT)))
    return out
def tcp_frame_at_close(R, exe, box_xy, yaw):
    tc = np.where(exe[:, 7] < CLOSE_TRIG)[0]
    if not len(tc): return None
    k = int(tc[0]); f, n = box_axes(yaw); d = R["tcp"][k] - R["box"][0]; return dict(df_cm=float(d[:2] @ f * 100), dn_cm=float(d[:2] @ n * 100), dz_cm=float(d[2] * 100))
def process_pose(Rn, pose, guard):
    guard(margin=8.0); o, prev, Rp, exe_pre = Rn.prefix(pose); env = Rn.env; z0 = float(Rp["box"][0][2]); snap = BR.snapshot(env); box_xy = (pose["x"], pose["y"]); yaw = pose["yaw"]
    raw40 = Rn.policy.chunk(o["state"], o["images"]); err = oracle_error(Rn.kin, raw40, box_xy, yaw); applicable = bool(err is not None and exe_pre[-1, 7] >= CLOSE_TRIG); out = dict(pose=pose, applicable=applicable, raw_j_close=(err["j_close"] if err else None), oracle_error=err, executed_gripper_before_decision=float(exe_pre[-1, 7]), candidates=[])
    if not applicable: out["not_applicable_reason"] = "no close command inside the t=40 chunk" if err is None else "gripper already closing before the decision state"; return out
    nom = None
    for cd in CANDS:
        guard(margin=4.0); raw_mod, dq_off, res = correct_chunk(Rn.kin, raw40, box_xy, yaw, cd, err); r = dict(cand=cd["id"], kind=cd["kind"], timing=cd.get("timing"), dims=list(cd.get("dims", [])), applied={k: float(v) for k, v in (res["applied"] or {}).items()}, ik_max_pos_err_m=res["max_pos_err_m"], ik_max_rot_err_rad=res["max_rot_err_rad"], ik_fail=res["ik_fail"], n_ik=res["n_ik"])
        R, exe, n_clip, n_slew = Rn.branch(snap, prev, raw_mod, dq_off); legal = bool(P.check_legal(Rn.lim, exe, prev)); finite = bool(np.isfinite(exe).all() and np.isfinite(R["box"]).all()); coll = bool(R["pen"].max() > PEN_MAX); ikok = res["ik_fail"] == 0
        r.update(valid=bool(legal and finite and not coll and ikok), legal=legal, finite=finite, obstacle_contact=coll, clipped=int(n_clip), slewed=int(n_slew), outcome=assess(R, exe, z0) if (legal and finite) else None, frame_at_close=tcp_frame_at_close(R, exe, box_xy, yaw))
        if cd["id"] == "c0": nom = dict(exe=exe[:CHUNK], tcp=R["tcp"][:CHUNK + 1])
        r["_exe"], r["_tcp"] = exe[:CHUNK], R["tcp"][:CHUNK + 1]; out["candidates"].append(r)
    for r in out["candidates"]:
        r["arm_linf_vs_nominal_rad"] = float(np.abs(r["_exe"][:, :7] - nom["exe"][:, :7]).max()); r["tcp_dev_vs_nominal_m"] = float(np.linalg.norm(r["_tcp"] - nom["tcp"], axis=1).max()); r["gripper_linf_vs_nominal_m"] = float(np.abs(r["_exe"][:, 7] - nom["exe"][:, 7]).max())
        r["distinct_vs_nominal"] = bool(r["cand"] != "c0" and (r["arm_linf_vs_nominal_rad"] >= DISTINCT["arm"] or r["tcp_dev_vs_nominal_m"] >= DISTINCT["tcp"] or r["gripper_linf_vs_nominal_m"] >= 0.002)); r.pop("_exe"); r.pop("_tcp")
    return out
def verify_freeze():
    F = json.loads(FREEZE.read_text()); bad = [f for f, h in F["files_sha256"].items() if P.sha256(ROOT / f) != h]
    if bad: raise SystemExit(f"frozen files changed: {bad}")
    return F
def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run": raise SystemExit("usage: g2a_act_corrections.py run")
    F = verify_freeze(); rep = json.loads((OUT / "integration_report.json").read_text())
    if rep["verdict"] != "PASS": raise SystemExit("integration gates not PASS")
    if rep["code_sha256"]["eval/g2a_act_corrections.py"] != P.sha256(__file__): raise SystemExit("runner changed after the integration gates")
    resf = OUT / "results/g2a_results.json"
    if resf.exists(): raise SystemExit("results exist; refusing to overwrite")
    c0 = P.cpu(); used = ledger_used(); cap = CAP_TOTAL_S - used - 20.0; guard = P.make_guard(c0, cap, 0); res = dict(status="RUNNING", cap_s=cap, ledger_used_before_s=used, poses=[], stop_reason=None)
    try:
        Rn = Runner(); res["setup_cpu_s"] = P.cpu() - c0
        for pose in F["poses"]:
            guard(margin=30.0); res["poses"].append(process_pose(Rn, pose, guard)); res["cpu_s"] = P.cpu() - c0; P.atomic_json(resf, res)
        res["status"] = "COMPLETE"
    except P.CapReached as e: res["status"] = "PARTIAL"; res["stop_reason"] = f"cap reached: {e}"
    except Exception as e:
        import traceback; res["status"] = "BLOCKED"; res["stop_reason"] = f"{type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()[-1500:]
    res["cpu_s"] = P.cpu() - c0; P.atomic_json(resf, res); ledger_add("g2a_run", res["cpu_s"]); print(json.dumps({k: res[k] for k in ("status", "stop_reason", "cpu_s") if k in res}), flush=True)
if __name__ == "__main__": main()
