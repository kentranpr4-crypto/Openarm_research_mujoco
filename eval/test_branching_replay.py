#!/usr/bin/env python3
"""INTEGRATION GATES for the offline branching replay (mandatory before the replay).  Run once (ACT venv): python eval/test_branching_replay.py
Writes outputs/branching_replay/integration_report.json (verdict PASS only if every gate passes) and adds its CPU to the experiment ledger.  Plumbing runs use pose 4 at decision states t=40,100 only; they are never counted as results."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import sys, json, time, tempfile, traceback, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); import closed_loop_pilot as P, branching_replay as BR, branching_metrics as BM
import decision_utility_lib as L
torch.set_num_threads(1); C0 = P.cpu(); GATES = []; CAP_TEST = 0.07 * 3600
def gate(n, name):
    def deco(fn):
        t0 = time.perf_counter(); c0 = P.cpu()
        try:
            if P.cpu() - C0 > CAP_TEST: raise P.CapReached("test cap")
            ok, ev = fn()
        except Exception as e: ok, ev = False, f"EXCEPTION {type(e).__name__}: {e}\n{traceback.format_exc()[-700:]}"
        GATES.append(dict(gate=n, name=name, result="PASS" if ok else "FAIL", evidence=ev, cpu_s=round(P.cpu() - c0, 2))); print(f"G{n:02d} {'PASS' if ok else 'FAIL'}  {name}  {ev if not ok else ''}", flush=True); return fn
    return deco
BR.init_G(); G = BR.G; env, obs_, lim, w3, policies = G["env"], G["obs"], G["lim"], G["w3"], G["policies"]; POSE = G["poses"][4]; X, Y, YAW = POSE["x"], POSE["y"], POSE["yaw"]
A0T = dict(np.load(P.OUT / "results/traj/A0_p04_r0.npz")); ACT0 = A0T["action"].astype(float)          # Stage-1 A0 executed actions of the same pose (reference only)
def to_state(n):
    """reset and execute the first n Stage-1 A0 actions; returns histories"""
    env.set_perturbation(); o = env.reset((X, Y), YAW); rows, _ = BR.exec_chunk(env, obs_, ACT0[:n]); return dict(S=list(rows["S"]), B=list(rows["B"]), C=list(rows["C"]), A=list(ACT0[:n].astype(np.float32)))
def cmp_rows(a, b): return max(float(np.abs(a[k] - b[k]).max()) for k in ("S", "B", "C", "box", "tcp"))
@gate(1, "snapshot/restore determinism: same env repeat, after another branch, and a fresh env instance (bit-exact)")
def _():
    hist = to_state(60); snap = BR.snapshot(env); seqX = ACT0[60:80]; seqY = ACT0[60:80] + 0.02
    BR.restore(env, snap); r1, _ = BR.exec_chunk(env, obs_, seqX); BR.restore(env, snap); r2, _ = BR.exec_chunk(env, obs_, seqX); BR.restore(env, snap); BR.exec_chunk(env, obs_, seqY); BR.restore(env, snap); r3, _ = BR.exec_chunk(env, obs_, seqX)
    env2 = P.make_env(()); obs2 = P.LiveObserver(env2); env2.set_perturbation(); env2.reset((0.5, 0.2), 0.3); BR.restore(env2, snap); r4, _ = BR.exec_chunk(env2, obs2, seqX); env2.close()
    d = dict(repeat=cmp_rows(r1, r2), after_other_branch=cmp_rows(r1, r3), fresh_env=cmp_rows(r1, r4)); return max(d.values()) == 0.0, d
@gate(2, "clone/restore of internal control history (prev_ctrl, time, ctrl) and sensitivity of the chain to it")
def _():
    hist = to_state(60); snap = BR.snapshot(env); prev = np.asarray(env.prev_ctrl, float).copy(); raw = np.random.default_rng(1).normal(0, 1.0, (20, 16)) * 0.3 + prev; ex0, _, _ = P.slew_chain(lim, raw, prev)
    BR.exec_chunk(env, obs_, ACT0[60:100] + 0.05); BR.restore(env, snap); ex1, _, _ = P.slew_chain(lim, raw, env.prev_ctrl); ex_wrong, _, _ = P.slew_chain(lim, raw, np.zeros(16))
    ok = np.array_equal(env.prev_ctrl, prev) and env.t == snap["t"] and np.array_equal(env.d.ctrl, snap["d"].ctrl) and np.array_equal(ex0, ex1) and not np.array_equal(ex0, ex_wrong)
    return bool(ok), dict(prev_equal=bool(np.array_equal(env.prev_ctrl, prev)), t_equal=bool(env.t == snap["t"]), ctrl_equal=bool(np.array_equal(env.d.ctrl, snap["d"].ctrl)), chain_equal=bool(np.array_equal(ex0, ex1)), negative_control_differs=bool(not np.array_equal(ex0, ex_wrong)))
tmp = tempfile.TemporaryDirectory(); KEEP = {}; RECS = []
t_run = time.perf_counter(); RECS, st = BR.process_pose(4, G, lambda margin=0.0: None, states=[40, 100], do_j2=True, keep_clones=KEEP, out_arrays=Path(tmp.name)); BR.run_j2(KEEP, G, lambda margin=0.0: None)
@gate(3, "same-state equivalence: branch c0_s0 == Stage-1 A0 continuation (box rows, executed actions), and replay actions == Stage-1 A0")
def _():
    out = {}
    for rec in RECS:
        t = rec["t"]; Z = np.load(Path(tmp.name) / f"{rec['state_id']}.npz"); out[t] = dict(box=float(np.abs(Z["box"][0] - A0T["box_pos"][t:t + 21]).max()), act=float(np.abs(Z["exe"][0] - ACT0[t:t + 20]).max()), tcp=float(np.abs(Z["tcp"][0] - A0T["tcp_pos"][t:t + 21]).max()))
    return max(max(v.values()) for v in out.values()) <= 1e-6, out
@gate(4, "no future leakage: predictions identical before/after executing branches, and when rows after t are poisoned")
def _():
    hist = to_state(60); t = 60; exe = np.stack([P.slew_chain(lim, ACT0[60:80] + 0.01 * k, np.asarray(hist["A"][-1], float))[0] for k in range(3)]); fp = BR.hist_fingerprint(hist)
    p1, W1 = BR.score_state(w3, hist, t, exe, [X, Y, YAW]); snap = BR.snapshot(env)
    for k in range(3): BR.restore(env, snap); BR.exec_chunk(env, obs_, exe[k])
    BR.restore(env, snap); p2, _ = BR.score_state(w3, hist, t, exe, [X, Y, YAW]); same_after = (p1["J20"] == p2["J20"]) and (p1["J2"] == p2["J2"]) and fp == BR.hist_fingerprint(hist)
    pois = {k: list(v) for k, v in hist.items()}
    for k in ("S", "B", "C"): pois[k] += [np.full_like(np.asarray(hist[k][0]), np.nan) for _ in range(25)]
    pois["A"] += [np.full(16, np.nan, np.float32) for _ in range(25)]; p3, _ = BR.score_state(w3, pois, t, exe, [X, Y, YAW]); same_poison = (p1["J20"] == p3["J20"]) and (p1["J2"] == p3["J2"])
    keys_ok = set(W1) == {"S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "t", "Y_e", "pose"} and not W1["Y_e"].any()
    return bool(same_after and same_poison and keys_ok), dict(unchanged_after_branch_execution=bool(same_after), unchanged_with_poisoned_future=bool(same_poison), keys_ok=bool(keys_ok))
@gate(5, "exact actuator semantics: executed == ActionAdapter chain from raw outputs, legal, applied ctrl == executed, A_fut == executed[:, :, :8]")
def _():
    from control.action_adapter import ActionAdapter
    probe = P.make_env(()); ad = ActionAdapter(probe.m, 20); probe.close(); worst = 0.0; legal = True; app = 0.0
    for rec in RECS:
        Z = np.load(Path(tmp.name) / f"{rec['state_id']}.npz"); prev = Z["prev"].astype(float)
        for k in range(3):
            p = prev.copy(); ref = []
            for r in Z["raw"][k].astype(float): c, _, _ = ad.to_ctrl(r, p); ref.append(c); p = c
            worst = max(worst, float(np.abs(np.array(ref, np.float32) - Z["exe"][k]).max())); legal &= bool(P.check_legal(lim, Z["exe"][k].astype(float), prev)); app = max(app, rec["branches"][k]["applied_ctrl_maxdiff"])
    hist = to_state(60); exe = np.stack([P.slew_chain(lim, ACT0[60:80] + 0.01 * k, np.asarray(hist["A"][-1], float))[0] for k in range(3)]); _, W = BR.score_state(w3, hist, 60, exe, [X, Y, YAW]); afut = float(np.abs(W["A_fut"] - exe[:, :, :8].astype(np.float32)).max())
    return worst <= 1e-6 and legal and app == 0.0 and afut == 0.0, dict(adapter_vs_stored_exe_maxdiff=worst, legal=legal, applied_ctrl_maxdiff=app, A_fut_vs_exe=afut)
@gate(6, "J2 pipeline consistency: rollout first block == J20 prediction path; hold continuation; 41 rows; realised J2 monotone in palm")
def _():
    hist = to_state(60); t = 60; exe = np.stack([P.slew_chain(lim, ACT0[60:80] + 0.01 * k, np.asarray(hist["A"][-1], float))[0] for k in range(3)]); W = P.build_window(hist["S"], hist["B"], hist["C"], hist["A"], t, exe, [X, Y, YAW]); Pp = w3.predict(W); bb = np.asarray(hist["B"][t])[None] + np.cumsum(Pp["Y_b"], 1)
    hold = np.repeat(exe[:, -1:, :8], 20, axis=1); A = np.concatenate([W["A_hist"], exe[:, :, :8].astype(np.float32), hold.astype(np.float32)], 1); R = L.rollout(w3.predict, dict(S=W["S_hist"].copy(), B=W["B_hist"].copy(), C=W["C_hist"].copy()), A, np.full(3, A.shape[1] + 1), [X, Y, YAW], t0=4)
    d_first = max(float(np.abs(np.asarray(R[i]["b"][1:21]) - bb[i]).max()) for i in range(3)); rows = [len(r["b"]) for r in R]
    mono = all((not b["terms_real"]["palm"]) or b["terms2_real"]["palm"] for rec in RECS if rec["t"] in BR.J2_STATES for b in rec["branches"]); have = all("J2_real" in b and b["hold_applied_ctrl_maxdiff"] == 0.0 for rec in RECS for b in rec["branches"]); pj = all(len(rec["pred"]["J2"]) == 3 for rec in RECS)
    return d_first < 1e-5 and rows == [41, 41, 41] and mono and have and pj, dict(first_block_diff=d_first, rows=rows, palm_monotone=bool(mono), realised_J2_present=bool(have), predicted_J2_present=bool(pj))
@gate(7, "stable identifiers, unique ids, atomic JSON writer, no stray tmp, CPU guard")
def _():
    ids = [b["branch_id"] for r in RECS for b in r["branches"]] + [r["state_id"] for r in RECS]; uniq = len(ids) == len(set(ids)); fmt = all(i.startswith("p04_t") for i in ids)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "r.json"; P.atomic_json(f, dict(states=[{k: v for k, v in r.items()} for r in RECS])); back = json.loads(f.read_text()); left = [q.name for q in Path(td).iterdir()]
    g = P.make_guard(P.cpu() - 5.0, 1.0, 9); raised = False
    try: g()
    except P.CapReached: raised = True
    return uniq and fmt and len(back["states"]) == len(RECS) and left == ["r.json"] and raised and BR.CAP_TOTAL_S == 0.40 * 3600, dict(unique=uniq, n_ids=len(ids), tmp_left=left, guard=raised, cap_s=BR.CAP_TOTAL_S)
@gate(8, "metrics unit tests on synthetic cases (pairwise accuracy, regret, headroom, gating, pose bootstrap)")
def _():
    r = []
    c, n = BM.pair_counts([0.3, 0.1, 0.2], [0.3, 0.1, 0.2]); r.append((c, n) == (3.0, 3))
    c, n = BM.pair_counts([0.1, 0.3, 0.2], [0.3, 0.1, 0.2]); r.append((c, n) == (0.0, 3))
    c, n = BM.pair_counts([0.2, 0.2, 0.2], [0.3, 0.1, 0.2]); r.append((c, n) == (1.5, 3))
    c, n = BM.pair_counts([0.3, 0.1, 0.2], [0.30, 0.295, 0.298]); r.append(n == 0)
    m = BM.state_metrics([0.2, 0.1, 0.3], [0.3, 0.1, 0.2], 0); r.append(abs(m["headroom"] - 0.2) < 1e-12 and m["selected"] == 1 and m["regret_wm"] == 0.0 and abs(m["regret_random"] - 0.1) < 1e-12 and m["has_headroom"])
    m = BM.state_metrics([0.205, 0.2, 0.3], [0.3, 0.1, 0.2], 0); r.append(m["gated_selected"] == 0 and m["selected"] == 1 and abs(m["regret_gated"] - 0.2) < 1e-12)
    m = BM.state_metrics([0.2, 0.1, 0.3], [0.3, 0.1, 0.2], 0); r.append(abs(m["norm_regret_default"] - 2.0) < 1e-9 and abs(m["norm_den"] - 0.1) < 1e-12)
    est = BM.boot_ratio({1: 1.0, 2: 1.0, 3: 1.0}, {1: 2.0, 2: 2.0, 3: 2.0}); r.append(abs(est[0] - 0.5) < 1e-12 and abs(est[1] - 0.5) < 1e-12 and BM.boot_ratio({1: 1.0, 2: 0.0}, {1: 1.0, 2: 1.0}) == BM.boot_ratio({1: 1.0, 2: 0.0}, {1: 1.0, 2: 1.0}))
    mm = BM.boot_mean_by_pose({1: [1.0, 3.0], 2: [2.0]}); r.append(abs(mm[0] - 2.0) < 1e-12)
    return all(r), dict(checks=r)
code_files = ["eval/branching_replay.py", "eval/branching_metrics.py", "eval/test_branching_replay.py"]
cpu_used = P.cpu() - C0; BR.ledger_add("integration_tests", cpu_used); tmp.cleanup()
rep = dict(verdict="PASS" if all(g["result"] == "PASS" for g in GATES) and len(GATES) == 8 else "FAIL", n_gates=len(GATES), gates=GATES, cpu_s=cpu_used, ledger_total_s=BR.ledger_used(), code_sha256={f: P.sha256(ROOT / f) for f in code_files}, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), note="plumbing replays (pose 4, t=40/100) are not results")
P.atomic_json(BR.OUT / "integration_report.json", rep); print(rep["verdict"], f"cpu={cpu_used:.1f}s ledger_total={rep['ledger_total_s']:.1f}s", flush=True)
