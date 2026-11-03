#!/usr/bin/env python3
"""G2a INTEGRATION GATES (mandatory before the experiment).  Run once (ACT venv): python eval/test_g2a_act_corrections.py
Writes outputs/g2a_act_corrections/integration_report.json and adds its CPU to the G2a ledger (cap for ALL of G2a: 0.25 CPU-h).  Plumbing runs use xsweep poses 0 and 4 only and are never counted as results."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, copy, inspect, tempfile, traceback, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import closed_loop_pilot as P, branching_replay as BR, g2a_act_corrections as G, mujoco
C0 = P.cpu(); GATES = []; CAP_TEST = 140.0; NOGUARD = lambda margin=0.0: None
def gate(n, name):
    def deco(fn):
        c0 = P.cpu()
        try:
            if P.cpu() - C0 > CAP_TEST: raise P.CapReached("test cap")
            ok, ev = fn()
        except Exception as e: ok, ev = False, f"EXCEPTION {type(e).__name__}: {e}\n{traceback.format_exc()[-700:]}"
        GATES.append(dict(gate=n, name=name, result="PASS" if ok else "FAIL", evidence=ev, cpu_s=round(P.cpu() - c0, 2))); print(f"G{n:02d} {'PASS' if ok else 'FAIL'}  {name}  {ev if not ok else ''}", flush=True); return fn
    return deco
Rn = G.Runner(); env = Rn.env; XP = {p["idx"]: p for p in json.loads((ROOT / "outputs/xsweep/xsweep_config.json").read_text())["poses"]}; POSE4 = dict(idx=4, **{k: XP[4][k] for k in ("x", "y", "yaw")}); POSE0 = dict(idx=0, **{k: XP[0][k] for k in ("x", "y", "yaw")})
def straight(pose, steps=G.T_END):
    env.set_perturbation(); o = env.reset((pose["x"], pose["y"]), pose["yaw"]); prev = np.asarray(env.prev_ctrl, float).copy(); R = G.NEW_ROWS(); exes = []
    for t in range(0, steps, G.CHUNK):
        raw = Rn.policy.chunk(o["state"], o["images"]); exe, prev, _, _ = Rn.run_chunk(raw, prev, R); exes.append(exe); o = BR.obs_full(env)
    Rn.record(R); return {k: np.array(v) for k, v in R.items()}, np.concatenate(exes)
def cmp(a, b): return max(float(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)).max()) for k in ("box", "tcp", "hf", "nf", "ff"))
def mk_branch_inputs(pose):
    o, prev, Rp, exe_pre = Rn.prefix(pose); snap = BR.snapshot(env); raw40 = Rn.policy.chunk(o["state"], o["images"]); err = G.oracle_error(Rn.kin, raw40, (pose["x"], pose["y"]), pose["yaw"]); return o, prev, Rp, exe_pre, snap, raw40, err
@gate(1, "snapshot/restore bit-exact (repeat, after another candidate, fresh env instance) including ACT re-queries")
def _():
    o, prev, Rp, exe_pre, snap, raw40, err = mk_branch_inputs(POSE0); cd = G.CANDS[3]; mod, dq, _ = G.correct_chunk(Rn.kin, raw40, (POSE0["x"], POSE0["y"]), POSE0["yaw"], cd, err)
    r1, e1, _, _ = Rn.branch(snap, prev, raw40, None); r2, e2, _, _ = Rn.branch(snap, prev, raw40, None); Rn.branch(snap, prev, mod, dq); r3, e3, _, _ = Rn.branch(snap, prev, raw40, None)
    env2 = P.make_env(Rn.policy.cams); R2 = copy.copy(Rn); R2.env = env2; R2.obs = P.LiveObserver(env2); env2.set_perturbation(); env2.reset((0.5, 0.2), 0.3); r4, e4, _, _ = R2.branch(snap, prev, raw40, None); env2.close()
    d = dict(repeat=cmp(r1, r2), after_other=cmp(r1, r3), fresh_env=cmp(r1, r4), exe=float(max(np.abs(e1 - e2).max(), np.abs(e1 - e3).max(), np.abs(e1 - e4).max()))); return max(d.values()) == 0.0, d
@gate(2, "ACT nominal reproduction: branch c0 == uninterrupted unmodified ACT rollout (bit-exact); rollout == stored Stage-1 A0 on the same pose")
def _():
    Rs, es = straight(POSE4); o, prev, Rp, exe_pre, snap, raw40, err = mk_branch_inputs(POSE4); rb, eb, _, _ = Rn.branch(snap, prev, raw40, None); sl = {k: Rs[k][G.T_DEC:] for k in Rs}
    T = np.load(P.OUT / "results/traj/A0_p04_r0.npz"); d_stage1 = float(np.abs(es[:G.T_END] - T["action"][:G.T_END]).max()); d_box = float(np.abs(Rs["box"][:G.T_END + 1] - T["box_pos"][:G.T_END + 1]).max())
    return cmp(sl, rb) == 0.0 and float(np.abs(es[G.T_DEC:] - eb).max()) == 0.0 and d_stage1 <= 1e-6 and d_box <= 1e-6, dict(branch_vs_straight_rows=cmp(sl, rb), exe=float(np.abs(es[G.T_DEC:] - eb).max()), straight_vs_stage1_actions=d_stage1, straight_vs_stage1_box=d_box)
@gate(3, "executed-control history and policy state: prev_ctrl/time/ctrl restored; chain sensitivity; ACT queries stateless/deterministic")
def _():
    o, prev, Rp, exe_pre, snap, raw40, err = mk_branch_inputs(POSE4); Rn.branch(snap, prev, raw40, None); BR.restore(env, snap); a, _, _ = P.slew_chain(Rn.lim, raw40 + 0.01, env.prev_ctrl); b, _, _ = P.slew_chain(Rn.lim, raw40 + 0.01, prev); w, _, _ = P.slew_chain(Rn.lim, raw40 + 0.01, np.zeros(16))
    q1 = Rn.policy.chunk(o["state"], o["images"]); o2 = BR.obs_full(env); Rn.policy.chunk(o2["state"] * 0 + 0.1, o2["images"]); q2 = Rn.policy.chunk(o["state"], o["images"])
    ok = np.array_equal(env.prev_ctrl, prev) and env.t == snap["t"] and np.array_equal(env.d.ctrl, snap["d"].ctrl) and np.array_equal(a, b) and not np.array_equal(a, w) and np.array_equal(q1, q2); return bool(ok), dict(chain=bool(np.array_equal(a, b)), neg_control_differs=bool(not np.array_equal(a, w)), policy_stateless=bool(np.array_equal(q1, q2)))
RES0 = {}
@gate(4, "IK residuals (pos<2 mm, rot<0.03 rad) for all 9 candidates on a real ACT chunk; IK round-trip fidelity; full process_pose works")
def _():
    t0 = P.cpu(); RES0["r"] = G.process_pose(Rn, POSE0, NOGUARD); RES0["cpu"] = P.cpu() - t0; r = RES0["r"]; c = r["candidates"]
    ok = r["applicable"] and all(x["ik_fail"] == 0 and x["ik_max_pos_err_m"] < 2e-3 and x["ik_max_rot_err_rad"] < 0.03 for x in c)
    o, prev, Rp, exe_pre, snap, raw40, err = mk_branch_inputs(POSE0); rt, _, _ = G.correct_chunk(Rn.kin, raw40, (POSE0["x"], POSE0["y"]), POSE0["yaw"], dict(id="z", kind="fixed", dims=("f",), value=0.0, timing="early"), err); rtf = float(np.abs(rt[:, Rn.kin.ik.act] - raw40[:, Rn.kin.ik.act]).max())
    return bool(ok and rtf < 5e-3), dict(applicable=r["applicable"], oracle_error=r["oracle_error"], max_pos_err_m=max(x["ik_max_pos_err_m"] for x in c), max_rot_err_rad=max(x["ik_max_rot_err_rad"] for x in c), roundtrip_zero_correction_max_joint_dev_rad=rtf, process_pose_cpu_s=RES0["cpu"])
@gate(5, "joint/actuator limits: every candidate executed chain legal and finite; gravity-comp inversion consistent")
def _():
    c = RES0["r"]["candidates"]; legal = all(x["legal"] and x["finite"] for x in c); q = np.array([0.3, -0.5, 0.2, 1.0, 0.1, 0.2, -0.1]); inv = float(np.abs(Rn.kin.ctrl_from_q(Rn.kin.q_from_ctrl(Rn.kin.ctrl_from_q(q))) - Rn.kin.ctrl_from_q(q)).max())
    return legal and inv < 1e-5, dict(all_legal=legal, gravcomp_roundtrip=inv, clipped=[x["clipped"] for x in c], slewed=[x["slewed"] for x in c])
@gate(6, "collision validity: penetration detector fires (>2 mm) on a constructed violation, not at home; no candidate penetrates an obstacle deeper than 2 mm")
def _():
    env.set_perturbation(); env.reset((0.4, 0.1), 0.0); home_free = Rn.obstacle_penetration() == 0.0; dd = mujoco.MjData(env.m); import lab_scene; lab_scene.reset_home(env.m, dd); R = G._rz(0.0) @ Rn.ex.R0; pen = 0.0
    for z in (0.34, 0.30):
        Rn.ex.ik.solve(dd, np.array([0.40, 0.10, z]), R); env.d.qpos[Rn.ex.ik.qadr] = Rn.ex.ik.q(dd); mujoco.mj_forward(env.m, env.d); pen = max(pen, Rn.obstacle_penetration())
    pens = [x["outcome"]["obstacle_penetration_max_m"] for x in RES0["r"]["candidates"] if x["outcome"]]; ok = all(x["valid"] for x in RES0["r"]["candidates"]) and max(pens) <= G.PEN_MAX
    return home_free and pen > G.PEN_MAX and ok, dict(home_free=home_free, constructed_violation_penetration_m=pen, candidate_max_penetration_m=max(pens), any_contact_flags=[x["outcome"]["obstacle_contact"] for x in RES0["r"]["candidates"]])
@gate(7, "gripper channel semantics: arm-only corrections leave the executed gripper identical to ACT; delay shifts the close event by exactly 8 steps; close detection")
def _():
    o, prev, Rp, exe_pre, snap, raw40, err = mk_branch_inputs(POSE0); j = err["j_close"]; same = []
    for cd in G.CANDS:
        if cd["kind"] in ("cancel", "fixed"): m, _, _ = G.correct_chunk(Rn.kin, raw40, (POSE0["x"], POSE0["y"]), POSE0["yaw"], cd, err); same.append(bool(np.array_equal(m[:, 7], raw40[:, 7])))
    md, _, _ = G.correct_chunk(Rn.kin, raw40, (POSE0["x"], POSE0["y"]), POSE0["yaw"], G.CANDS[4], err); jd = G.close_step(md); shifted = (jd == j + G.DELAY_STEPS) if j + G.DELAY_STEPS < G.CHUNK else (jd is None)
    gl = [x["gripper_linf_vs_nominal_m"] for x in RES0["r"]["candidates"] if x["kind"] in ("cancel", "fixed")]; return all(same) and shifted and max(gl) == 0.0, dict(j_close=j, j_close_delayed=jd, arm_only_gripper_unchanged=all(same), executed_gripper_linf=max(gl))
@gate(8, "frames and sign conventions: FK of the commanded targets recovers the expert grasp point; +depth moves the TCP +f, +lateral +n, +psi rotates the hand +z")
def _():
    ex = Rn.ex; xy = (0.40, 0.10); yaw = 0.10; pl = ex.plan(xy, yaw); idx = int(np.where(np.array(pl["phase_names"])[np.array(pl["phase"])] == "grip_hold")[0][0]); ctrl = pl["ctrl"][idx]; tg = pl["tcp_target"][idx]
    q = Rn.kin.q_from_ctrl(ctrl[Rn.kin.ik.act]); p, R = Rn.kin.fk(q); f, n = G.box_axes(yaw); rel = R @ ex.R0.T; psi0 = float(np.arctan2(rel[1, 0], rel[0, 0])); fk_err = float(np.linalg.norm(p - tg))
    raw = np.tile(ctrl, (G.CHUNK, 1)); raw[:, 7] = 0.044; err = dict(j_close=19, e_f=0, e_n=0, e_psi=0); out = {}
    for name, cd in (("f", dict(id="t", kind="fixed", dims=("f",), value=0.01, timing="early")), ("n", dict(id="t", kind="cancel", dims=("n",), timing="early")), ("psi", dict(id="t", kind="cancel", dims=("psi",), timing="early"))):
        e2 = dict(err, e_n=-0.005, e_psi=-0.10) ; m, _, res = G.correct_chunk(Rn.kin, raw, xy, yaw, cd, e2); q2 = Rn.kin.q_from_ctrl(m[19, Rn.kin.ik.act]); p2, R2 = Rn.kin.fk(q2); d = p2 - p; rel2 = R2 @ ex.R0.T; out[name] = dict(df=float(d[:2] @ f), dn=float(d[:2] @ n), dz=float(d[2]), dpsi=float(G.wrap(np.arctan2(rel2[1, 0], rel2[0, 0]) - psi0)))
    ok = fk_err < 1.5e-3 and abs(out["f"]["df"] - 0.01) < 1e-3 and abs(out["f"]["dn"]) < 1e-3 and abs(out["n"]["dn"] - 0.005) < 1e-3 and abs(out["n"]["df"]) < 1e-3 and abs(out["psi"]["dpsi"] - 0.10) < 0.01 and abs(out["f"]["dz"]) < 1e-3
    return bool(ok), dict(fk_vs_expert_target_m=fk_err, effects=out)
@gate(9, "no future/outcome leakage: correction uses only the raw chunk, the initial box pose and registered constants; deterministic")
def _():
    src = inspect.getsource(G.correct_chunk) + inspect.getsource(G.oracle_error) + inspect.getsource(G.correction_values) + inspect.getsource(G.weights); clean = all(w not in src for w in ("outcome", "assess", "env.", "info(", 'R["box"]', "success"))
    o, prev, Rp, exe_pre, snap, raw40, err = mk_branch_inputs(POSE0); a = G.correct_chunk(Rn.kin, raw40, (POSE0["x"], POSE0["y"]), POSE0["yaw"], G.CANDS[3], err); b = G.correct_chunk(Rn.kin, raw40, (POSE0["x"], POSE0["y"]), POSE0["yaw"], G.CANDS[3], err); det = bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    return clean and det, dict(no_outcome_or_env_access_in_correction=clean, deterministic=det)
@gate(10, "outcome extraction (synthetic + palm-force replay of a Stage-1 palm episode) and atomic writer")
def _():
    n = G.T_END - G.T_DEC + 1; exe = np.full((n - 1, 16), 0.044); exe[3:, 7] = 0.012
    def mk(nf=2, rise=0.06, lift_at=10, wob=0.0, dip=None):
        z = np.full(n, 0.38); z[lift_at:] += rise
        if dip is not None: z[lift_at + 30] = 0.38 + dip
        box = np.zeros((n, 3)); box[:, 2] = z; tcp = box.copy(); tcp[:, 0] += 0.01; tcp[lift_at + 8:, 1] += wob * (np.arange(n - lift_at - 8) % 2); return dict(box=box, tcp=tcp, hf=np.zeros(n), nf=np.full(n, nf), ff=np.zeros(n), coll=np.zeros(n, bool), pen=np.zeros(n))
    a = lambda **k: G.assess(mk(**k), exe, 0.38)["reason"]; syn = [a() == "success", a(nf=1) == "not_closed", a(rise=0.02) == "not_lifted", a(lift_at=48) == "late_lift", a(wob=0.03) == "slipped", a(dip=0.01) == "dropped"]
    R = json.loads((P.OUT / "results/pilot_results.json").read_text()); e = next(x for x in R["episodes"] if x["status"] == "OK" and x["palm_contact"]); T = np.load(P.OUT / f"results/traj/{e['arm']}_p{e['pose_idx']:02d}_r{e['repeat']}.npz"); env.set_perturbation(); env.reset((e["x"], e["y"]), e["yaw"]); Rr = G.NEW_ROWS()
    for u in T["action"].astype(float): Rn.record(Rr); P.lean_step(env, u, False)
    hf = np.array(Rr["hf"]); ref = T["c_rows"][:, 4]; d = float(np.abs(hf[:200] - ref).max())
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "r.json"; P.atomic_json(f, dict(a=1)); P.atomic_json(f, dict(a=2)); back = json.loads(f.read_text()); left = [q.name for q in Path(td).iterdir()]
    return all(syn) and d <= 1e-3 and bool(((hf[:200] > 0.01) == (ref > 0.01)).all()) and back == dict(a=2) and left == ["r.json"], dict(synthetic=syn, palm_force_replay_maxdiff_N=d, atomic=back == dict(a=2))
@gate(11, "CPU guard, deterministic reset, deterministic pose generator (48 poses: 32 hard + 16 mid, min distance, no reserved seeds)")
def _():
    g = P.make_guard(P.cpu() - 5.0, 1.0, 9); raised = False
    try: g()
    except P.CapReached: raised = True
    a, ma = G.sample_poses(); b, _ = G.sample_poses(); strata = [sum(p["stratum"] == s for p in a) for s in ("hard", "mid")]; env.set_perturbation(); env.reset((0.4, 0.1), 0.0); x1 = env.d.qpos.copy(); env.reset((0.4, 0.1), 0.0); x2 = env.d.qpos.copy()
    return raised and a == b and len(a) == 48 and strata == [32, 16] and ma >= 0.003 and np.allclose(x1, x2, atol=1e-12) and G.CAP_TOTAL_S == 900.0, dict(guard=raised, deterministic=a == b, strata=strata, min_chebyshev=ma, reset_stable=bool(np.allclose(x1, x2, atol=1e-12)))
@gate(12, "branch horizon and candidate timing: 61 rows, 60 executed steps, ACT queried exactly at t=60/80, ramp weights, continuity at chunk boundaries (no snap)")
def _():
    calls = []; orig = Rn.policy.chunk; Rn.policy.chunk = lambda s, i: (calls.append(1), orig(s, i))[1]
    try:
        o, prev, Rp, exe_pre, snap, raw40, err = mk_branch_inputs(POSE0); calls.clear(); cd = G.CANDS[3]; mod, dq, _ = G.correct_chunk(Rn.kin, raw40, (POSE0["x"], POSE0["y"]), POSE0["yaw"], cd, err); R, exe, _, _ = Rn.branch(snap, prev, mod, dq)
    finally: Rn.policy.chunk = orig
    we = G.weights("early", 5); wl = G.weights("late", 7); jump = float(np.abs(exe[G.CHUNK] - exe[G.CHUNK - 1])[Rn.kin.ik.act].max())
    return len(R["box"]) == 61 and len(exe) == 60 and len(calls) == 2 and we[0] == 1 / 6 and we[5] == 1.0 and wl[2] == 0.0 and wl[3] == 0.25 and wl[6] == 1.0 and jump <= 0.1 + 1e-9 and len(mod) == G.CHUNK, dict(rows=len(R["box"]), exe=len(exe), act_queries=len(calls), early_w=[float(we[0]), float(we[5])], late_w=[float(wl[2]), float(wl[3]), float(wl[6])], chunk_boundary_jump_rad=jump)
cpu_used = P.cpu() - C0; G.ledger_add("integration_tests", cpu_used)
rep = dict(verdict="PASS" if all(g["result"] == "PASS" for g in GATES) and len(GATES) == 12 else "FAIL", n_gates=len(GATES), gates=GATES, cpu_s=cpu_used, ledger_total_s=G.ledger_used(), measured_process_pose_cpu_s=RES0.get("cpu"), code_sha256={f: P.sha256(ROOT / f) for f in ("eval/g2a_act_corrections.py", "eval/test_g2a_act_corrections.py")}, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
P.atomic_json(G.OUT / "integration_report.json", rep); print(rep["verdict"], f"cpu={cpu_used:.1f}s ledger_total={rep['ledger_total_s']:.1f}s process_pose_cpu={RES0.get('cpu')}", flush=True)
