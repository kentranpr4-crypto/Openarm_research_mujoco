#!/usr/bin/env python3
"""Stage 1 INTEGRATION GATES (mandatory before the pilot).  Run once (ACT venv): python eval/test_closed_loop_pilot.py
Writes outputs/closed_loop_pilot/integration_report.json (verdict PASS only if every gate passes) and adds its CPU to the Stage-1 ledger.  Does not run any pilot episode (episodes here are 40-60-step plumbing checks on the registered poses, never counted as results)."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, traceback, tempfile, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); import closed_loop_pilot as P
torch.set_num_threads(1); C0 = P.cpu(); GATES = []; CAP_TEST = 0.05 * 3600
NEEDED_SETUP = {1}; ONLY = set(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else None
def gate(n, name):
    def deco(fn):
        if ONLY is not None and n not in ONLY and n not in NEEDED_SETUP: return fn
        t0 = time.perf_counter(); c0 = P.cpu()
        try:
            if P.cpu() - C0 > CAP_TEST: raise P.CapReached("test cap")
            ok, ev = fn()
        except Exception as e: ok, ev = False, f"EXCEPTION {type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
        GATES.append(dict(gate=n, name=name, result="PASS" if ok else "FAIL", evidence=ev, cpu_s=round(P.cpu() - c0, 2))); print(f"G{n:02d} {'PASS' if ok else 'FAIL'}  {name}  {ev if not ok else ''}", flush=True); return fn
    return deco
policies = P.load_chunk_policies(torch); cams = policies[P.DEFAULT].cams; w3 = P.load_wm(torch)
import decision_utility_lib as L
from control.safety_limits import SafetyLimits
from control.action_adapter import ActionAdapter
env = P.make_env(cams); obs_ = P.LiveObserver(env); lim = SafetyLimits.from_model(env.m); POSES = {p["idx"]: p for p in P.pilot_poses()}; pose = POSES[4]       # x=0.38, block 0
# ---- expert trajectory (shared by G1/G2/G4)
import wm1_sim as S
from policies.baseline import ScriptedExpert
expert = ScriptedExpert(env.m, control_hz=20); T_EXP = 80
plan = expert.plan((0.40, 0.10), 0.0); A = np.array(plan["ctrl"], float)[:T_EXP]; Ex = np.zeros_like(A)
env.set_perturbation(); env.reset((0.40, 0.10), 0.0); prev = env.prev_ctrl.copy()
for t in range(T_EXP): Ex[t], _, _ = lim.clip_and_slew(A[t], prev, 0.05); prev = Ex[t]
LIVE = {}
@gate(1, "live privileged observer == wm1_sim at the SAME simulation state (state, b, c rows)")
def _():
    env0 = P.make_env(()); obs0 = P.LiveObserver(env0); m = env0.m
    S._ENV.update(env=env0, gb=m.geom("blue_box_collision").id, gL=m.geom("openarm_left_left_finger_collision").id, gR=m.geom("openarm_left_right_finger_collision").id, gH=m.geom("openarm_left_hand_collision").id, jq=m.joint("openarm_left_finger_joint1").id)
    rows = []; orig = env0.step
    def hooked(ctrl): b, c = obs0.row(); rows.append((env0.adapter.state(env0.d).astype(np.float32), b, c)); return orig(ctrl)
    env0.step = hooked; Rr = S.simulate(Ex.astype(np.float32), (0.40, 0.10), 0.0); env0.step = orig; b_ref, c_ref = S.contact_features(Rr)
    Sr = np.array([r[0] for r in rows]); Br = np.array([r[1] for r in rows]); Cr = np.array([r[2] for r in rows]); n = len(rows)
    LIVE.update(S=Sr, B=Br, C=Cr); d = dict(n_rows=n, state=float(np.abs(Sr - Rr["state"][:n]).max()), b=float(np.abs(Br - b_ref[:n]).max()), c=float(np.abs(Cr - c_ref[:n]).max()))
    return n == T_EXP and max(d["state"], d["b"], d["c"]) <= 1e-6 and bool(np.isfinite(Br).all() and np.isfinite(Cr).all()), d
@gate(2, "no future leakage into WM inputs (window identical if rows after t are removed or poisoned; allowed keys only)")
def _():
    t = 40; ce = np.repeat(Ex[t:t + 20][None], 2, 0); W1 = P.build_window(LIVE["S"], LIVE["B"], LIVE["C"], Ex, t, ce, [0.4, 0.1, 0.0])
    Sp, Bp, Cp, Ap = LIVE["S"].copy(), LIVE["B"].copy(), LIVE["C"].copy(), Ex.copy(); Sp[t + 1:] = np.nan; Bp[t + 1:] = np.nan; Cp[t + 1:] = np.nan; Ap[t:] = np.nan
    W2 = P.build_window(Sp, Bp, Cp, Ap, t, ce, [0.4, 0.1, 0.0]); same = all(np.array_equal(W1[k], W2[k], equal_nan=False) for k in W1)
    keys_ok = set(W1) == {"S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "t", "Y_e", "pose"} and not W1["Y_e"].any()
    return same and keys_ok, dict(identical=same, keys=sorted(W1))
@gate(3, "ACT chunk runner (A0) reproduces ACTRunner.select_action + preprocessing in-process and the stored xsweep c0_seed0 rollout")
def _():
    from policies.act_infer import ACTRunner
    H = 60; r = P.run_episode("A0", pose, 0, env, obs_, policies, w3, lim, horizon=H, keep_arrays=True); mine = r["_arrays"]["action"]
    probe = P.make_env(()); ad = ActionAdapter(probe.m, 20); probe.close(); runner = ACTRunner(ROOT / "outputs/checkpoints/e6b_c0_seed0/last", ad, n_action_steps=None, threads=1)
    env.set_perturbation(); o = env.reset((pose["x"], pose["y"]), pose["yaw"]); runner.reset(env.prev_ctrl); ref = []
    for t in range(H): a = runner.act(o["state"], o["images"]); ref.append(np.asarray(a, np.float32)); o = env.step(a)
    ref = np.array(ref); d_in = float(np.abs(mine - ref).max()); hist = np.load(P.XSWEEP_C0S0 / "rollouts" / f"ep{pose['idx']:04d}.npz")["action"][:H]; d_hist = float(np.abs(mine - hist).max())
    return d_in <= 1e-5 and d_hist <= 1e-4, dict(max_abs_vs_ACTRunner=d_in, max_abs_vs_stored_xsweep_rollout=d_hist, steps=H)
@gate(4, "time indexing of S/B/C history (rows t-3..t), A_hist (actions t-4..t-1), A_fut (executed t..t+19)")
def _():
    n = 60; Sx = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 16)); Bx = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 7)); Cx = Bx.copy(); Ax = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 16)) + 1000; t = 30
    ce = np.stack([np.tile((np.arange(20, dtype=np.float32) + 2000)[:, None], (1, 16))]); W = P.build_window(Sx, Bx, Cx, Ax, t, ce, [0, 0, 0])
    ok = (W["S_hist"][0][:, 0] == [27, 28, 29, 30]).all() and (W["B_hist"][0][:, 0] == [27, 28, 29, 30]).all() and (W["A_hist"][0][:, 0] == [1026, 1027, 1028, 1029]).all() and (W["A_fut"][0][:, 0] == np.arange(20) + 2000).all() and W["S_hist"].shape == (1, 4, 8) and W["A_fut"].shape == (1, 20, 8)
    return bool(ok), dict(S=W["S_hist"][0][:, 0].tolist(), A_hist=W["A_hist"][0][:, 0].tolist())
@gate(5, "executed actions: clip THEN slew from the last executed command, equals ActionAdapter.to_ctrl step by step, legal, NaN rejected")
def _():
    rng = np.random.default_rng(0); raw = rng.normal(0, 3.0, (20, 16)); prev = env.prev_ctrl.copy(); ex, nr, ns = P.slew_chain(lim, raw, prev)
    probe = P.make_env(()); ad = ActionAdapter(probe.m, 20); probe.close(); p = prev.copy(); ref = []
    for r in raw: c, _, _ = ad.to_ctrl(r, p); ref.append(c); p = c
    same = bool(np.array_equal(ex, np.array(ref))); legal = P.check_legal(lim, ex, prev)
    c1 = np.clip(raw[0], lim.lo, lim.hi); order_a = prev + np.clip(c1 - prev, -lim.max_speed * 0.05, lim.max_speed * 0.05); order_b = np.clip(prev + np.clip(raw[0] - prev, -lim.max_speed * 0.05, lim.max_speed * 0.05), lim.lo, lim.hi)
    try: ad.to_ctrl(np.full(16, np.nan), prev); nan_rejected = False
    except ValueError: nan_rejected = True
    return same and legal and nan_rejected and nr > 0 and ns > 0, dict(equal_to_adapter=same, legal=legal, nan_rejected=nan_rejected, clipped=nr, slewed=ns, order_matters_here=bool(not np.allclose(order_a, order_b)))
@gate(6, "no state/cache leakage between checkpoints and episodes")
def _():
    env.set_perturbation(); o = env.reset((pose["x"], pose["y"]), pose["yaw"]); a1 = policies["c0_s0"].chunk(o["state"], o["images"]); b1 = policies["c0_s1"].chunk(o["state"], o["images"]); a2 = policies["c0_s0"].chunk(o["state"], o["images"]); c1 = policies["c0_s2"].chunk(o["state"], o["images"])
    r_a2 = P.run_episode("A2", pose, 0, env, obs_, policies, w3, lim, horizon=40, keep_arrays=True); r_a0 = P.run_episode("A0", pose, 0, env, obs_, policies, w3, lim, horizon=40, keep_arrays=True); r_a0b = P.run_episode("A0", pose, 0, env, obs_, policies, w3, lim, horizon=40, keep_arrays=True)
    ok = np.array_equal(a1, a2) and not np.array_equal(a1, b1) and not np.array_equal(b1, c1) and np.array_equal(r_a0["_arrays"]["action"], r_a0b["_arrays"]["action"]) and np.array_equal(r_a2["_arrays"]["action"][:20], r_a0["_arrays"]["action"][:20])
    return bool(ok), dict(ckpt_deterministic=bool(np.array_equal(a1, a2)), ckpts_differ=bool(not np.array_equal(a1, b1)), A0_repeat_identical=bool(np.array_equal(r_a0["_arrays"]["action"], r_a0b["_arrays"]["action"])), first_chunk_same_A0_A2=bool(np.array_equal(r_a2["_arrays"]["action"][:20], r_a0["_arrays"]["action"][:20])), a2_decision=r_a2["decisions"][1].get("J"))
@gate(7, "no stale/batched prediction coupling in WM scoring (batch == individual, order-equivariant, candidates differ)")
def _():
    env.set_perturbation(); o = env.reset((pose["x"], pose["y"]), pose["yaw"]); ces = np.stack([P.slew_chain(lim, policies[k].chunk(o["state"], o["images"]), env.prev_ctrl)[0] for k in P.CK_NAMES])
    t = 40; Wb = P.build_window(LIVE["S"], LIVE["B"], LIVE["C"], Ex, t, ces, [0.4, 0.1, 0.0]); Pb = w3.predict(Wb)["Y_b"]; singles = []
    for i in range(3): singles.append(w3.predict(P.build_window(LIVE["S"], LIVE["B"], LIVE["C"], Ex, t, ces[i:i + 1], [0.4, 0.1, 0.0]))["Y_b"][0])
    d_batch = float(np.abs(Pb - np.array(singles)).max()); perm = [2, 0, 1]; Pp = w3.predict(P.build_window(LIVE["S"], LIVE["B"], LIVE["C"], Ex, t, ces[perm], [0.4, 0.1, 0.0]))["Y_b"]; d_perm = float(np.abs(Pp - Pb[perm]).max()); differ = float(np.abs(Pb[0] - Pb[1]).max())
    return d_batch < 1e-4 and d_perm < 1e-4 and differ > 1e-6, dict(batch_vs_single=d_batch, permutation=d_perm, candidates_differ=differ)
@gate(8, "malformed shapes/dimensions rejected (window, chunk, action)")
def _():
    errs = 0
    for bad in (np.zeros((1, 19, 16)), np.zeros((1, 20, 15))):
        try: P.build_window(LIVE["S"], LIVE["B"], LIVE["C"], Ex, 40, bad, [0, 0, 0]); errs += 0
        except ValueError: errs += 1
    try: P.build_window(LIVE["S"], LIVE["B"], LIVE["C"], Ex, 2, np.zeros((1, 20, 16)), [0, 0, 0])
    except ValueError: errs += 1
    try: P.build_window(LIVE["S"][:, :7], LIVE["B"], LIVE["C"], Ex, 40, np.zeros((1, 20, 16)), [0, 0, 0])
    except ValueError: errs += 1
    probe = P.make_env(()); ad = ActionAdapter(probe.m, 20); probe.close()
    try: ad.to_ctrl(np.zeros(15), env.prev_ctrl); errs += 0
    except ValueError: errs += 1
    return errs == 5, dict(rejected=errs, of=5)
@gate(9, "stable reset / seed behaviour (A0 repeats identical; A1 RNG reproducible per (pose, repeat), differs across repeats; no reserved seeds)")
def _():
    g = lambda pid, rep: [int(np.random.default_rng(P.RNG_KEY + [pid, rep]).integers(3)) for _ in range(9)]; same = g(4, 0) == g(4, 0); diff = g(4, 0) != g(4, 1) or g(4, 1) != g(4, 2); r1 = [int(np.random.default_rng(P.RNG_KEY + [4, 0]).integers(3)) for _ in range(9)]
    seeds_ok = not any(s in (31, 32, 33) for s in P.RNG_KEY) and P.RNG_KEY == [20261101, 8001]
    # sequential draws from ONE generator per episode, as in run_episode
    def seq(pid, rep): r = np.random.default_rng(P.RNG_KEY + [pid, rep]); return [int(r.integers(3)) for _ in range(9)]
    return same and diff and seeds_ok and seq(4, 0) == seq(4, 0) and seq(4, 0) != seq(4, 1), dict(a1_reproducible=same, differs_across_repeats=diff, rng_key=P.RNG_KEY, seq_p4=[seq(4, r) for r in range(3)])
@gate(10, "CPU cap guard and atomic incremental writer")
def _():
    g = P.make_guard(P.cpu() - 5.0, 1.0, 99); raised = False
    try: g()
    except P.CapReached: raised = True
    ok_margin = False
    g2 = P.make_guard(P.cpu(), 1000.0, 98)
    try: g2(margin=2000.0)
    except P.CapReached: ok_margin = True
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.json"; P.atomic_json(f, dict(a=1)); P.atomic_json(f, dict(a=2, b=[1, 2])); back = json.loads(f.read_text()); left = [q.name for q in Path(td).iterdir()]
    ledger_ok = P.CAP_TOTAL_S == 0.90 * 3600
    return raised and ok_margin and back == dict(a=2, b=[1, 2]) and left == ["x.json"] and ledger_ok, dict(guard_raises=raised, margin_guard=ok_margin, atomic_ok=back == dict(a=2, b=[1, 2]), stray_tmp_files=left, cap_total_s=P.CAP_TOTAL_S)
@gate(11, "orchestrator task wrapper: OK / CAP / ERROR statuses, JSON-serialisable, no exception across the pool boundary")
def _():
    P.init_worker(1e9, objs=(policies, w3, env)); ok = P.run_task((4, "A2", 0, 40)); json.dumps(ok, default=float)
    err = P.run_task((999, "A2", 0, 40)); P.G["cap_s"] = 0.0; cap = P.run_task((4, "A0", 0, 40)); P.G["cap_s"] = 1e9
    return ok["status"] == "OK" and err["status"] == "ERROR" and cap["status"] == "CAP" and ok["action_invalid_chunks"] == 0 and ok["finite"], dict(ok=ok["status"], err=err["status"], cap=cap["status"], outcome_after_40_steps=ok["outcome"])
code_files = ["eval/closed_loop_pilot.py", "eval/test_closed_loop_pilot.py"]
cpu_used = P.cpu() - C0; P.ledger_add("integration_tests", cpu_used)
rep = dict(verdict="PASS" if all(g["result"] == "PASS" for g in GATES) and len(GATES) == 11 else "FAIL", n_gates=len(GATES), gates=GATES, cpu_s=cpu_used, ledger_total_s=P.ledger_used(), code_sha256={f: P.sha256(ROOT / f) for f in code_files},
           at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), note="plumbing episodes (40-60 steps) are not pilot results")
P.atomic_json(P.OUT / "integration_report.json", rep); print(rep["verdict"], f"cpu={cpu_used:.1f}s", flush=True)
