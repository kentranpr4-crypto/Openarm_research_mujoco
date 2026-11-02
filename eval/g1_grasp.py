#!/usr/bin/env python3
"""G1: physically parameterised GRASP CANDIDATES - validity, outcome diversity and oracle headroom (DEVELOPMENT; ORACLE-POSE mode: candidate IK uses the simulator box pose; simulation only; no model, no training).
Candidates are 2-level designs over geometry/actuator parameters (orthogonal fractional factorials + the nominal scripted grasp as the default), executed from bit-exact MuJoCo snapshots (mj_copyData) at three predefined grasp decision states.
Run (ACT venv): python eval/g1_grasp.py run     (needs outputs/g1_grasp/{integration_report,protocol_freeze}.json)"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")): sys.path.insert(0, p)
import mujoco, lab_scene
import closed_loop_pilot as P
import branching_replay as BR
from policies.baseline import ScriptedExpert, G_OPEN, G_CLOSE, Z_LIFT, GRASP_DZ, _smooth, _rz
OUT = ROOT / "outputs/g1_grasp"; FREEZE = OUT / "protocol_freeze.json"; LEDGER = OUT / "cpu_ledger.json"; CAP_TOTAL_S = 0.25 * 3600; DT = 0.05
N_POSES = 24; STATES = ("D1", "D2", "D3"); DECISION_SEG = {"D1": "pregrasp", "D2": "approach", "D3": "close"}
NOMINAL = dict(dd=0.0, lat=0.0, yaw=0.0, pre=0.09, g_close=G_CLOSE, delay=0, close_dur=0.7)
LEVELS = dict(dd=(-0.010, 0.010), lat=(-0.005, 0.005), yaw=(-0.05, 0.05), pre=(0.07, 0.11), g_close=(0.008, 0.016), delay=(2, 8), close_dur=(0.4, 1.0))        # geometry / actuator anchored; chosen before any G1 outcome
FACTORS = {"D1": ("dd", "lat", "yaw", "pre"), "D2": ("dd", "lat", "yaw", "g_close"), "D3": ("delay", "close_dur", "g_close")}; YAW_FROM = {"D1": "pregrasp", "D2": "approach", "D3": None}
RISE_MIN, HOLD_DIST, SLIP_MAX, DROP_RISE, PALM_THR, N_HOLD = 0.05, 0.06, 0.01, 0.02, 0.01, 8          # frozen outcome definition (rise/hold distance as eval/metrics.py)
DISTINCT = dict(arm=0.02, grip=0.002, tcp=0.003)
def ledger_used(): return float(sum(e["cpu_s"] for e in json.loads(LEDGER.read_text()))) if LEDGER.exists() else 0.0
def ledger_add(tag, cpu_s):
    L = json.loads(LEDGER.read_text()) if LEDGER.exists() else []; L.append(dict(tag=tag, cpu_s=float(cpu_s), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))); P.atomic_json(LEDGER, L)
def design(state):
    """9 candidates: c0 = nominal (the DEFAULT), c1..c8 = 2-level fractional factorial (D = ABC for 4 factors; full 2^3 for 3 factors) in a fixed order."""
    fs = FACTORS[state]; combos = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]; cands = [dict(id="c0", levels={f: 0 for f in fs}, th=dict(NOMINAL))]
    for i, (a, b, c) in enumerate(combos):
        lv = dict(zip(fs, (a, b, c) + ((a * b * c,) if len(fs) == 4 else ()))); th = dict(NOMINAL)
        for f, v in lv.items(): th[f] = LEVELS[f][0 if v < 0 else 1]
        cands.append(dict(id=f"c{i + 1}", levels=lv, th=th))
    return cands
def sample_poses():
    """24 development poses, separate RNG namespace; rejected if within Chebyshev 0.003 (x[m], y[m], yaw[rad]) of ANY previously used initial state; does not look at any model/outcome. 12 'hard' (x in [0.34,0.38)) + 12 'mid' (x in [0.38,0.47])."""
    import indep_manifest as IM
    used = [s for v in IM.used_states().values() for s in v] + [(p["x"], p["y"], p["yaw"]) for p in json.loads((ROOT / "outputs/indep/manifest.json").read_text())["poses"]]; U = np.array(used); rng = np.random.default_rng([20261102, 9001]); poses = []; pid = 2000
    for stratum, xr, n in (("hard", (0.34, 0.38), 12), ("mid", (0.38, 0.47), 12)):
        got = 0
        while got < n:
            p = np.round(np.array([rng.uniform(*xr), rng.uniform(0.07, 0.13), rng.uniform(-0.20, 0.20)]), 6)
            cur = np.array([[q["x"], q["y"], q["yaw"]] for q in poses]) if poses else np.zeros((0, 3))
            if np.abs(U - p).max(1).min() < 0.003 or (len(cur) and np.abs(cur - p).max(1).min() < 0.003): continue
            poses.append(dict(idx=pid, stratum=stratum, x=float(p[0]), y=float(p[1]), yaw=float(p[2]))); pid += 1; got += 1
    return poses, float(np.abs(U[:, None, :] - np.array([[q["x"], q["y"], q["yaw"]] for q in poses])[None]).max(2).min())
# ------------------------------------------------------------------------------------------------------------------------------ plan builder (mirrors ScriptedExpert.plan; nominal parameters reproduce it bit-for-bit)
def build_segments(ex, box_xy, yaw, th, state=None):
    bx, by = box_xy; f = np.array([np.cos(yaw), np.sin(yaw), 0.0]); nl = np.array([-np.sin(yaw), np.cos(yaw), 0.0]); zg = lab_scene.TABLE_TOP_Z + 0.020 + GRASP_DZ; c = np.array([bx, by, zg]); h = ex.home_tcp
    pre_a = c - 0.09 * f; pre_a[1], pre_a[2] = h[1], h[2]; pregrasp = c + (th["lat"] if state == "D1" else 0.0) * nl - (th["pre"] if state == "D1" else 0.09) * f; grasp = c + (0.01 + th["dd"]) * f + th["lat"] * nl; up = np.array([*grasp[:2], Z_LIFT])
    segs = [("settle", h, G_OPEN, 0.4), ("pre_a", pre_a, G_OPEN, None), ("pregrasp", pregrasp, G_OPEN, None), ("approach", grasp, G_OPEN, None)]
    if th["delay"] > 0: segs.append(("close_wait", grasp, G_OPEN, th["delay"] / ex.hz))
    return segs + [("close", grasp, th["g_close"], th["close_dur"]), ("grip_hold", grasp, th["g_close"], 0.4), ("lift", up, th["g_close"], None), ("lift_hold", up, th["g_close"], 0.4)]
def plan_variant(ex, box_xy, box_yaw, th, yaw_from=None, state=None):
    dd = mujoco.MjData(ex.m); lab_scene.reset_home(ex.m, dd); segs = build_segments(ex, box_xy, box_yaw, th, state); names_all = [s[0] for s in segs]; i_yaw = names_all.index(yaw_from) if yaw_from else 10 ** 6
    cur, g_cur = ex.home_tcp.copy(), G_OPEN; qs, gs, ph, fails, names, pe, re = [], [], [], 0, [], [], []; y_prev = box_yaw
    for si, (name, pos, g, dur) in enumerate(segs):
        names.append(name); dist = np.linalg.norm(pos - cur); n = max(1, int(round((dur if dur else max(ex.min_dur, dist / ex.speed)) * ex.hz))); y_t = box_yaw + (th["yaw"] if si >= i_yaw else 0.0)
        for i in range(1, n + 1):
            s = _smooth(i / n); p = cur + (pos - cur) * s; R = _rz(y_prev + (y_t - y_prev) * s) @ ex.R0; qt = np.zeros(4); mujoco.mju_mat2Quat(qt, R.ravel())
            fails += 0 if ex.ik.solve(dd, p, R) else 1
            qc = np.zeros(4); er = np.zeros(3); mujoco.mju_mat2Quat(qc, dd.xmat[ex.ik.tcp]); mujoco.mju_subQuat(er, qt, qc)
            pe.append(float(np.linalg.norm(p - dd.xpos[ex.ik.tcp]))); re.append(float(np.linalg.norm(er))); qs.append(ex.ik.q(dd)); gs.append(g_cur + (g - g_cur) * s); ph.append(si)
        cur, g_cur, y_prev = pos.copy(), g, y_t
    ctrl = np.zeros((len(qs), 16)); ctrl[:, ex.ik.act] = [ex.ik.gravity_comp(q) for q in qs]; ctrl[:, ex.g_l] = gs; ctrl[:, ex.ikr.act] = ex.ctrl_right; ctrl[:, ex.g_r] = ex.g_r_home
    lo, hi = ex.m.actuator_ctrlrange[:, 0], ex.m.actuator_ctrlrange[:, 1]; ctrl = np.clip(ctrl, lo, hi); ph = np.array(ph); starts = {n_: int(np.argmax(ph == i)) for i, n_ in enumerate(names)}; ends = {n_: int(len(ph) - np.argmax(ph[::-1] == i)) for i, n_ in enumerate(names)}
    return dict(ctrl=ctrl, names=names, starts=starts, ends=ends, ok=bool(max(pe) < 2e-3 and max(re) < 0.03 and fails == 0), ik_fail_steps=fails, max_pos_err=float(max(pe)), max_rot_err=float(max(re)))
# ------------------------------------------------------------------------------------------------------------------------------ execution + outcome
class Env:
    def __init__(self):
        from lab_env import LabEnv
        from control.safety_limits import SafetyLimits
        self.env = LabEnv(cameras=()); self.m = self.env.m; self.ex = ScriptedExpert(self.m, control_hz=20); self.lim = SafetyLimits.from_model(self.m); self.obs = P.LiveObserver(self.env); m = self.m
        bn = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]; self.robot_g = np.array([n.startswith("openarm_") for n in bn]); self.obst_g = np.array([n in ("desk", "tray", "room") for n in bn])
        assert self.robot_g.sum() > 10 and self.obst_g.sum() >= 3, (self.robot_g.sum(), self.obst_g.sum())
    def obstacle_contact(self):
        d = self.env.d
        for i in range(d.ncon):
            c = d.contact[i]
            if (self.robot_g[c.geom1] and self.obst_g[c.geom2]) or (self.robot_g[c.geom2] and self.obst_g[c.geom1]): return True
        return False
    def record(self, R):
        i = self.env.info(); b, c = self.obs.row(); R["box"].append(i["box_pos"]); R["tcp"].append(i["tcp_pos"]); R["hf"].append(float(c[4])); R["nf"].append(float(c[1])); R["ff"].append(float(c[0] * 50.0)); R["coll"].append(self.obstacle_contact())
    def run(self, ctrl_seq, prev):
        R = dict(box=[], tcp=[], hf=[], nf=[], ff=[], coll=[]); exe = []; prev = np.asarray(prev, float)
        for u in ctrl_seq:
            self.record(R); c, _, _ = self.lim.clip_and_slew(u, prev, DT); P.lean_step(self.env, c, False); prev = c; exe.append(c)
        self.record(R); return {k: np.array(v) for k, v in R.items()}, np.array(exe)
def assess(R, plan, tD, z0):
    """Frozen outcome definition on rows R (row k = state before plan step tD+k; last row = after the final step)."""
    k0 = lambda name: plan["starts"][name] - tD; hold0 = k0("lift_hold"); lift0 = k0("lift"); rise = R["box"][:, 2] - z0; sep = R["box"] - R["tcp"]; W = slice(hold0, None)
    closed = bool(R["nf"][lift0] >= 2); lifted = bool((rise[W] >= RISE_MIN).all()); near = bool((np.linalg.norm(sep[W], axis=1) < HOLD_DIST).all()); slip = float(np.linalg.norm(sep[W] - sep[W].mean(0), axis=1).max()); slipped = slip > SLIP_MAX
    reached = np.where(rise >= RISE_MIN)[0]; dropped = bool(len(reached) and (rise[reached[0]:] < DROP_RISE).any()); success = bool(closed and lifted and near and not slipped and not dropped)
    reason = "success" if success else ("not_closed" if not closed else "dropped" if dropped else "not_lifted" if not (lifted and near) else "slipped")
    palm_idx = np.where(R["hf"] > PALM_THR)[0]
    return dict(success=success, reason=reason, closed=closed, lifted=lifted, near=near, slip_m=slip, dropped=dropped, palm=bool(len(palm_idx)), palm_first_step=int(palm_idx[0] + tD) if len(palm_idx) else None, palm_force_max_N=float(R["hf"].max()), finger_contacts_at_lift=float(R["nf"][lift0]), finger_force_max_N=float(R["ff"].max()),
                rise_hold_min_m=float(rise[W].min()), box_tcp_dist_hold_max_m=float(np.linalg.norm(sep[W], axis=1).max()), obstacle_contact=bool(R["coll"].any()))
def process_pose(E, pose, guard):
    ex, env, lim = E.ex, E.env, E.lim; box_xy = (pose["x"], pose["y"]); yaw = pose["yaw"]; nom = plan_variant(ex, box_xy, yaw, NOMINAL); D = {s: nom["starts"][DECISION_SEG[s]] for s in STATES}; env.set_perturbation(); env.reset(box_xy, yaw); prev = np.asarray(env.prev_ctrl, float).copy()
    snaps = {}; z0 = None; Rp = dict(box=[], tcp=[], hf=[], nf=[], ff=[], coll=[])
    for t in range(max(D.values()) + 1):
        for s in STATES:
            if D[s] == t: snaps[s] = (BR.snapshot(env), prev.copy())
        if t == max(D.values()): break
        E.record(Rp); c, _, _ = lim.clip_and_slew(nom["ctrl"][t], prev, DT); P.lean_step(env, c, False); prev = c
    z0 = float(Rp["box"][0][2]); out = dict(pose=pose, nominal_plan_ok=nom["ok"], decision_steps=D, states=[])
    for s in STATES:
        snap, prev_s = snaps[s]; tD = D[s]; cands = design(s); recs = []; nom_trace = None
        for cd in cands:
            guard(margin=3.0); plan = plan_variant(ex, box_xy, yaw, cd["th"], YAW_FROM[s], s); prefix_ok = bool(np.array_equal(plan["ctrl"][:tD], nom["ctrl"][:tD])); r = dict(state_id=f"p{pose['idx']}_{s}", cand_id=f"p{pose['idx']}_{s}_{cd['id']}", cand=cd["id"], levels=cd["levels"], th={k: float(v) for k, v in cd["th"].items()}, plan_ok=plan["ok"], prefix_ok=prefix_ok, n_steps=int(len(plan["ctrl"]) - tD))
            if not (plan["ok"] and prefix_ok): r.update(valid=False, invalid_reason="ik_or_prefix", outcome=None); recs.append(r); continue
            BR.restore(env, snap); R, exe = E.run(plan["ctrl"][tD:], prev_s); legal = bool(P.check_legal(lim, exe, prev_s)); finite = bool(np.isfinite(exe).all() and np.isfinite(R["box"]).all()); coll = bool(R["coll"].any())
            r.update(valid=bool(legal and finite and not coll), invalid_reason=None if (legal and finite and not coll) else ("illegal" if not legal else "non_finite" if not finite else "obstacle_collision"), legal=legal, finite=finite)
            r["outcome"] = assess(R, plan, tD, z0) if (legal and finite) else None
            if cd["id"] == "c0": nom_trace = dict(exe=exe, tcp=R["tcp"])
            r["_exe"] = exe; r["_tcp"] = R["tcp"]; recs.append(r)
        for r in (recs if nom_trace is not None else []):       # diversity vs the nominal branch of the same state (common length, hold last)
            if "_exe" not in r: continue
            n = min(len(r["_exe"]), len(nom_trace["exe"])); r["div_arm_linf"] = float(np.abs(r["_exe"][:n, :7] - nom_trace["exe"][:n, :7]).max()); r["div_grip_linf"] = float(np.abs(r["_exe"][:n, 7] - nom_trace["exe"][:n, 7]).max()); r["div_tcp_max_m"] = float(np.linalg.norm(r["_tcp"][:n] - nom_trace["tcp"][:n], axis=1).max())
            r["distinct_vs_nominal"] = bool(r["cand"] != "c0" and (r["div_arm_linf"] >= DISTINCT["arm"] or r["div_grip_linf"] >= DISTINCT["grip"] or r["div_tcp_max_m"] >= DISTINCT["tcp"]))
        keep = [r for r in recs if "_exe" in r]; reps = []
        for r in keep:
            if not any(max(np.abs(r["_exe"][:min(len(r["_exe"]), len(q["_exe"])), :7] - q["_exe"][:min(len(r["_exe"]), len(q["_exe"])), :7]).max(), 10 * np.abs(r["_exe"][:min(len(r["_exe"]), len(q["_exe"])), 7] - q["_exe"][:min(len(r["_exe"]), len(q["_exe"])), 7]).max()) < DISTINCT["arm"] for q in reps): reps.append(r)
        for r in recs: r.pop("_exe", None); r.pop("_tcp", None)
        out["states"].append(dict(state_id=f"p{pose['idx']}_{s}", state=s, decision_step=int(tD), n_distinct_clusters=len(reps), candidates=recs))
    return out
def verify_freeze():
    F = json.loads(FREEZE.read_text()); bad = [f for f, h in F["files_sha256"].items() if P.sha256(ROOT / f) != h]
    if bad: raise SystemExit(f"frozen files changed: {bad}")
    return F
def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run": raise SystemExit("usage: g1_grasp.py run")
    F = verify_freeze(); rep = json.loads((OUT / "integration_report.json").read_text())
    if rep["verdict"] != "PASS": raise SystemExit("integration gates not PASS")
    if rep["code_sha256"]["eval/g1_grasp.py"] != P.sha256(__file__): raise SystemExit("runner changed after the integration gates")
    resf = OUT / "results/g1_results.json"
    if resf.exists(): raise SystemExit("results exist; refusing to overwrite")
    c0 = P.cpu(); used = ledger_used(); cap = CAP_TOTAL_S - used - 15.0; guard = P.make_guard(c0, cap, 0); poses = F["poses"]; res = dict(status="RUNNING", cap_s=cap, ledger_used_before_s=used, poses=[], stop_reason=None)
    try:
        E = Env(); res["setup_cpu_s"] = P.cpu() - c0
        for pose in poses:
            guard(margin=25.0); res["poses"].append(process_pose(E, pose, guard)); res["cpu_s"] = P.cpu() - c0; P.atomic_json(resf, res)
        res["status"] = "COMPLETE"
    except P.CapReached as e: res["status"] = "PARTIAL"; res["stop_reason"] = f"cap reached: {e}"
    except Exception as e:
        import traceback; res["status"] = "BLOCKED"; res["stop_reason"] = f"{type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()[-1500:]
    res["cpu_s"] = P.cpu() - c0; P.atomic_json(resf, res); ledger_add("g1_run", res["cpu_s"]); print(json.dumps({k: res[k] for k in ("status", "stop_reason", "cpu_s") if k in res}), flush=True)
if __name__ == "__main__": main()
