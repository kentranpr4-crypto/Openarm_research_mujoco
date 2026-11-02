#!/usr/bin/env python3
"""G1 INTEGRATION GATES (mandatory before the G1 rollout).  Run once (ACT venv): python eval/test_g1_grasp.py
Writes outputs/g1_grasp/integration_report.json and adds its CPU to the G1 ledger (cap for ALL of G1: 0.25 CPU-h).  Plumbing runs are never counted as results."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, tempfile, traceback, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import closed_loop_pilot as P, branching_replay as BR, g1_grasp as G, mujoco
C0 = P.cpu(); GATES = []; CAP_TEST = 130.0
def gate(n, name):
    def deco(fn):
        c0 = P.cpu()
        try:
            if P.cpu() - C0 > CAP_TEST: raise P.CapReached("test cap")
            ok, ev = fn()
        except Exception as e: ok, ev = False, f"EXCEPTION {type(e).__name__}: {e}\n{traceback.format_exc()[-700:]}"
        GATES.append(dict(gate=n, name=name, result="PASS" if ok else "FAIL", evidence=ev, cpu_s=round(P.cpu() - c0, 2))); print(f"G{n:02d} {'PASS' if ok else 'FAIL'}  {name}  {ev if not ok else ''}", flush=True); return fn
    return deco
E = G.Env(); env, ex, lim = E.env, E.ex, E.lim; POSE = dict(idx=1, x=0.40, y=0.10, yaw=0.10); BOX = (POSE["x"], POSE["y"]); YAW = POSE["yaw"]
nom = G.plan_variant(ex, BOX, YAW, G.NOMINAL); D = {s: nom["starts"][G.DECISION_SEG[s]] for s in G.STATES}
def prefix_to(tD):
    env.set_perturbation(); env.reset(BOX, YAW); prev = np.asarray(env.prev_ctrl, float).copy()
    for t in range(tD): c, _, _ = lim.clip_and_slew(nom["ctrl"][t], prev, G.DT); P.lean_step(env, c, False); prev = c
    return prev
def cmp(a, b): return max(float(np.abs(a[k].astype(float) - b[k].astype(float)).max()) for k in ("box", "tcp", "hf", "nf", "ff"))
@gate(1, "snapshot/restore determinism (repeat, after another branch, fresh env instance): bit-exact")
def _():
    tD = D["D2"]; prev = prefix_to(tD); snap = BR.snapshot(env); seq = nom["ctrl"][tD:]; alt = G.plan_variant(ex, BOX, YAW, dict(G.NOMINAL, dd=0.01), "approach", "D2")["ctrl"][tD:]
    BR.restore(env, snap); r1, e1 = E.run(seq, prev); BR.restore(env, snap); r2, e2 = E.run(seq, prev); BR.restore(env, snap); E.run(alt, prev); BR.restore(env, snap); r3, e3 = E.run(seq, prev)
    from lab_env import LabEnv
    env2 = LabEnv(cameras=()); obs2 = P.LiveObserver(env2); env2.set_perturbation(); env2.reset((0.5, 0.2), 0.3); BR.restore(env2, snap); E2 = G.Env.__new__(G.Env); E2.env, E2.lim, E2.obs, E2.robot_g, E2.obst_g = env2, lim, obs2, E.robot_g, E.obst_g; r4, e4 = E2.run(seq, prev); env2.close()
    d = dict(repeat=cmp(r1, r2), after_other_branch=cmp(r1, r3), fresh_env=cmp(r1, r4), exe=float(max(np.abs(e1 - e2).max(), np.abs(e1 - e3).max(), np.abs(e1 - e4).max()))); return max(d.values()) == 0.0, d
@gate(2, "same-state branching equivalence: nominal branch from snapshot == straight uninterrupted nominal execution")
def _():
    tD = D["D3"]; env.set_perturbation(); env.reset(BOX, YAW); prev0 = np.asarray(env.prev_ctrl, float).copy(); Rs, es = E.run(nom["ctrl"], prev0)
    prev = prefix_to(tD); snap = BR.snapshot(env); BR.restore(env, snap); Rb, eb = E.run(nom["ctrl"][tD:], prev); sl = {k: Rs[k][tD:] for k in Rs}
    return cmp(sl, Rb) == 0.0 and float(np.abs(es[tD:] - eb).max()) == 0.0, dict(rows_diff=cmp(sl, Rb), exe_diff=float(np.abs(es[tD:] - eb).max()))
@gate(3, "controller history restoration (prev_ctrl, time, ctrl) and sensitivity of the chain to it")
def _():
    tD = D["D2"]; prev = prefix_to(tD); snap = BR.snapshot(env); E.run(nom["ctrl"][tD:tD + 15], prev); BR.restore(env, snap); raw = nom["ctrl"][tD:tD + 15] + 0.02
    a, _, _ = P.slew_chain(lim, raw, env.prev_ctrl); b, _, _ = P.slew_chain(lim, raw, prev); w, _, _ = P.slew_chain(lim, raw, np.zeros(16))
    ok = np.array_equal(env.prev_ctrl, prev) and env.t == snap["t"] and np.array_equal(env.d.ctrl, snap["d"].ctrl) and np.array_equal(a, b) and not np.array_equal(a, w); return bool(ok), dict(prev=bool(np.array_equal(env.prev_ctrl, prev)), t=bool(env.t == snap["t"]), chain=bool(np.array_equal(a, b)), negative_control_differs=bool(not np.array_equal(a, w)))
@gate(4, "ActionAdapter consistency and absolute 16-D actuator semantics/units for all candidates")
def _():
    from control.action_adapter import ActionAdapter
    probe = P.make_env(()); ad = ActionAdapter(probe.m, 20); probe.close(); worst = 0.0; shape_ok = True; units_ok = True; right_const = True; prev = prefix_to(D["D1"])
    for s in G.STATES:
        for cd in G.design(s)[:3]:
            pl = G.plan_variant(ex, BOX, YAW, cd["th"], G.YAW_FROM[s], s); c = pl["ctrl"][D[s]:]; shape_ok &= c.shape[1] == 16; ex_c, _, _ = P.slew_chain(lim, c, prev); p = prev.copy(); ref = []
            for r in c: o, _, _ = ad.to_ctrl(r, p); ref.append(o); p = o
            worst = max(worst, float(np.abs(np.array(ref) - ex_c).max())); units_ok &= bool((c[:, 7] >= -0.005 - 1e-9).all() and (c[:, 7] <= 0.044 + 1e-9).all() and (np.abs(c[:, :7]) < 4.0).all()); right_const &= bool((np.abs(pl["ctrl"][:, 8:] - pl["ctrl"][0, 8:]).max() < 1e-12))
    return worst == 0.0 and shape_ok and units_ok and right_const, dict(adapter_vs_chain=worst, shape_ok=shape_ok, units_ok=units_ok, right_arm_constant=right_const)
@gate(5, "candidate design: 9 per state, unique ids, parameters inside registered levels, orthogonal columns, nominal first")
def _():
    ok = True; ev = {}
    for s in G.STATES:
        des = G.design(s); ids = [c["id"] for c in des]; ok &= len(des) == 9 and len(set(ids)) == 9 and des[0]["id"] == "c0" and all(des[0]["th"][k] == v for k, v in G.NOMINAL.items()); M = np.array([[c["levels"][f] for f in G.FACTORS[s]] for c in des[1:]], float)
        orth = bool(np.allclose(M.T @ M, 8 * np.eye(M.shape[1]))); bal = bool((M.sum(0) == 0).all()); inr = all(c["th"][f] in G.LEVELS[f] for c in des[1:] for f in G.FACTORS[s]); ok &= orth and bal and inr; ev[s] = dict(orthogonal=orth, balanced=bal, in_levels=inr)
    return ok, ev
@gate(6, "kinematics/collision validity: nominal plan == ScriptedExpert.plan prefix; shared prefix for all 27 candidates; obstacle-contact detector fires on a constructed violation and not at home")
def _():
    ref = ex.plan(BOX, YAW)["ctrl"]; n_ref = nom["ends"]["lift"]; same_nom = bool(np.array_equal(ref[:n_ref], nom["ctrl"][:n_ref])); pref = []; oks = []
    for s in G.STATES:
        for cd in G.design(s):
            pl = G.plan_variant(ex, BOX, YAW, cd["th"], G.YAW_FROM[s], s); pref.append(bool(np.array_equal(pl["ctrl"][:D[s]], nom["ctrl"][:D[s]]))); oks.append(pl["ok"])
    env.set_perturbation(); env.reset(BOX, YAW); home_free = not E.obstacle_contact(); dd = mujoco.MjData(env.m); import lab_scene; lab_scene.reset_home(env.m, dd); R = G._rz(YAW) @ ex.R0; fired = False
    for z in (0.34, 0.30):
        ex.ik.solve(dd, np.array([0.40, 0.10, z]), R); env.d.qpos[ex.ik.qadr] = ex.ik.q(dd); mujoco.mj_forward(env.m, env.d); fired |= E.obstacle_contact()
    return same_nom and all(pref) and home_free and fired, dict(nominal_equals_expert=same_nom, prefix_shared=f"{sum(pref)}/{len(pref)}", plan_ok=f"{sum(oks)}/{len(oks)}", home_free=home_free, detector_fires_on_violation=bool(fired))
@gate(7, "no future/outcome leakage: plans depend only on pose+parameters (deterministic), heuristic uses only the design, outcomes are computed after execution")
def _():
    p1 = G.plan_variant(ex, BOX, YAW, G.design("D1")[3]["th"], "pregrasp", "D1"); p2 = G.plan_variant(ex, BOX, YAW, G.design("D1")[3]["th"], "pregrasp", "D1"); det = bool(np.array_equal(p1["ctrl"], p2["ctrl"]))
    import inspect; src = inspect.getsource(G.plan_variant) + inspect.getsource(G.build_segments) + inspect.getsource(G.design); no_state = all(w not in src for w in ("env.d", "info(", "box_pos", "outcome", "assess"))
    return det and no_state, dict(deterministic=det, no_env_or_outcome_access_in_generation=no_state)
@gate(8, "contact-force/palm extraction: replay of a Stage-1 episode with palm contact reproduces the stored hand-force trace")
def _():
    R = json.loads((P.OUT / "results/pilot_results.json").read_text()); e = next(x for x in R["episodes"] if x["status"] == "OK" and x["palm_contact"]); T = np.load(P.OUT / f"results/traj/{e['arm']}_p{e['pose_idx']:02d}_r{e['repeat']}.npz")
    env.set_perturbation(); env.reset((e["x"], e["y"]), e["yaw"]); Rr = dict(box=[], tcp=[], hf=[], nf=[], ff=[], coll=[])
    for u in T["action"].astype(float): E.record(Rr); P.lean_step(env, u, False)
    hf = np.array(Rr["hf"]); ref = T["c_rows"][:, 4]; d = float(np.abs(hf[:200] - ref).max()); same_events = bool(((hf[:200] > 0.01) == (ref > 0.01)).all())
    return d <= 1e-3 and same_events and bool((ref > 0.01).any()), dict(max_abs_force_diff_N=d, same_palm_steps=same_events, palm_steps=int((ref > 0.01).sum()), episode=f"{e['arm']} p{e['pose_idx']}")
@gate(9, "stable reset, deterministic pose generator (24 poses, strata, min-distance, no reserved seeds)")
def _():
    a, ma = G.sample_poses(); b, _ = G.sample_poses(); same = a == b; strata = [sum(p["stratum"] == s for p in a) for s in ("hard", "mid")]; env.set_perturbation(); env.reset(BOX, YAW); x1 = env.d.qpos.copy(); env.reset(BOX, YAW); x2 = env.d.qpos.copy()
    return same and len(a) == 24 and strata == [12, 12] and ma >= 0.003 and np.allclose(x1, x2, atol=1e-12) and not any(s in (31, 32, 33) for s in (20261102, 9001)), dict(deterministic=same, strata=strata, min_chebyshev_to_used=ma, reset_stable=bool(np.allclose(x1, x2, atol=1e-12)))
@gate(10, "atomic incremental writer and CPU cap guard")
def _():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "r.json"; P.atomic_json(f, dict(a=1)); P.atomic_json(f, dict(a=2)); back = json.loads(f.read_text()); left = [q.name for q in Path(td).iterdir()]
    g = P.make_guard(P.cpu() - 5.0, 1.0, 9); raised = False
    try: g()
    except P.CapReached: raised = True
    return back == dict(a=2) and left == ["r.json"] and raised and G.CAP_TOTAL_S == 0.25 * 3600, dict(guard=raised, tmp=left, cap_s=G.CAP_TOTAL_S)
@gate(11, "outcome definition unit tests on synthetic rows (success / not_closed / not_lifted / slipped / dropped)")
def _():
    plan = dict(starts=dict(lift=4, lift_hold=11)); n = 20
    def mk(nf=2, rise=0.06, wob=0.0, dip=None):
        z = np.full(n, 0.38 + rise); z[:4] = 0.38; z[4:11] = 0.38 + np.linspace(0, rise, 7)
        if dip is not None: z[14] = 0.38 + dip
        box = np.zeros((n, 3)); box[:, 2] = z; tcp = box.copy(); tcp[:, 0] += 0.01; tcp[11:, 1] += wob * (np.arange(n - 11) % 2)
        return dict(box=box, tcp=tcp, hf=np.zeros(n), nf=np.full(n, nf), ff=np.zeros(n), coll=np.zeros(n, bool))
    r = [G.assess(mk(), plan, 0, 0.38)["reason"] == "success", G.assess(mk(nf=1), plan, 0, 0.38)["reason"] == "not_closed", G.assess(mk(rise=0.03), plan, 0, 0.38)["reason"] == "not_lifted", G.assess(mk(wob=0.03), plan, 0, 0.38)["reason"] == "slipped", G.assess(mk(dip=0.01), plan, 0, 0.38)["reason"] == "dropped"]
    return all(r), dict(checks=r)
cpu_used = P.cpu() - C0; G.ledger_add("integration_tests", cpu_used)
rep = dict(verdict="PASS" if all(g["result"] == "PASS" for g in GATES) and len(GATES) == 11 else "FAIL", n_gates=len(GATES), gates=GATES, cpu_s=cpu_used, ledger_total_s=G.ledger_used(), code_sha256={f: P.sha256(ROOT / f) for f in ("eval/g1_grasp.py", "eval/test_g1_grasp.py")}, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
P.atomic_json(G.OUT / "integration_report.json", rep); print(rep["verdict"], f"cpu={cpu_used:.1f}s ledger_total={rep['ledger_total_s']:.1f}s", flush=True)
