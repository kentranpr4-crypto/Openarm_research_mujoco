#!/usr/bin/env python3
"""G2b LIMITED INTERVENTION CPU BENCHMARK - HARDENED (r2 of the harness; NOT executed against the simulator yet; NOT an evaluation).
Design (review 2026-11-05, items A1-A10):
  LAUNCHER (this file, stdlib only, started before any heavy import) -> CHILD (same file, `--child`, own session/process group) -> real or mock workload.
  * CPU accounting starts at process start: the launcher spawns the child in a NEW SESSION (process group = child pid) and every --poll-s samples the WHOLE LIVE TREE (the child, every live descendant found by parent links and by process-group membership): sum over live members of utime+stime (all threads) + cutime+cstime
    (children already reaped by that member).  A reaped child's CPU lives only in its parent's c-fields and a live child's only in its own u-fields, so nothing is counted twice.  The tree is killed (killpg SIGKILL) when the sampled aggregate reaches --cap-s minus --kill-margin-s; afterwards the launcher verifies that no member of the group remains.
    Final aggregate = max(authoritative getrusage(RUSAGE_CHILDREN) delta, running maximum of the sampled tree) + the launcher's own CPU (max, not sum: the two measure the same CPU).  Overshoot of the cap is reported explicitly.  A start-up sub-limit (--startup-limit-s, default 45) stops the run (PARTIAL) if that much CPU is spent before the --startup-milestone (default `ik_initialized`) is logged.  CPU time and wall-clock time are reported separately; there is NO wall-clock timeout and no budget decision is ever taken from wall time.
  * The child also runs a cooperative guard (cap * --guard-frac) at milestones and every chunk, so that it normally stops itself cleanly (status PARTIAL) before the launcher has to kill it.
  * Incremental atomic JSON: the launcher writes <report> at launch (before any import in the child), whenever the child's own report changes and at the end; the child writes <report>.child.json at every milestone (process start, imports, ACT checkpoint, MuJoCo env + camera/renderer, IK, prefix, decision query, every branch start/end).
  * Status vocabulary: COMPLETE | PARTIAL (cooperative guard stop OR launcher hard-cap kill) | STOPPED_ON_FAIL (an integration/safety check failed) | BLOCKED (no /proc, crash, accounting impossible).  A cap/guard stop is never reported as PASS/COMPLETE.  No retry logic exists anywhere.
  * Limitations: the polling resolution (--poll-s) bounds the overshoot of a hard kill to about poll_s x (number of busy threads); a descendant that leaves the process group AND is re-parented to init (double fork + setsid) escapes the tree walk (none exists in this workload); if /proc cannot be read the run is BLOCKED.
Benchmark scope when run for real (needs separate authorization; fresh cap): ONE known development pose (G2a idx 3000), snapshot at t=40, horizon 160, branches c0, O3, D1, D2, S0; frozen ACT checkpoint c0_s0; frozen G2a correction code imported READ-ONLY after its sha256 is checked against the G2a freeze.
Only costs, validity and action semantics are reported; no outcome comparison and no treatment effect.
Usage: python3 eval/g2b_intervention_benchmark.py --cap-s 90 --report outputs/g2b_integration/intervention_benchmark_v2.json     (real; use the ACT venv python)
       python3 eval/g2b_intervention_benchmark.py --cap-s 2 --report /tmp/x.json --mock spin_nocheck --mock-arg 5                 (mock workloads, no simulator)"""
import os, sys, json, time, resource, signal, subprocess, argparse, traceback, threading
from pathlib import Path
T0_WALL = time.time(); ROOT = Path(__file__).resolve().parent.parent; TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
def cpu_total():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
CPU_AT_START = cpu_total()
def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(obj, indent=1, default=float)); os.replace(tmp, path)
class Stop(Exception): pass
class Guard:
    def __init__(self, cap_s, frac, startup_limit=0.0, startup_milestone="ik_initialized"): self.cap, self.limit = float(cap_s), float(cap_s) * float(frac); self.n_checks = 0; self.startup_limit = float(startup_limit or 0.0); self.startup_milestone = startup_milestone; self.startup_done = False
    def check(self, margin=0.0):
        self.n_checks += 1
        if self.startup_limit and not self.startup_done and cpu_total() >= self.startup_limit: raise Stop(f"start-up sub-limit: {cpu_total():.2f} s CPU >= {self.startup_limit} s before the milestone `{self.startup_milestone}`")
        if cpu_total() + margin > self.limit: raise Stop(f"cooperative CPU guard: aggregate {cpu_total():.2f} s + margin {margin:.2f} s > guard {self.limit:.2f} s (cap {self.cap:.2f} s)")
class Report:
    """child-side incremental report with milestones (CPU and wall kept separate)"""
    def __init__(self, path, role="child"):
        self.path = Path(path); self.d = dict(role=role, status="RUNNING", pid=os.getpid(), cpu_at_script_start_s=round(CPU_AT_START, 3), milestones=[], branches=[], shared={}); self._last = (T0_WALL, CPU_AT_START)
    guard = None
    def milestone(self, name, **extra):
        if self.guard is not None and name == self.guard.startup_milestone: self.guard.startup_done = True
        w, c = time.time(), cpu_total(); m = dict(name=name, wall_s=round(w - T0_WALL, 3), cpu_s=round(c, 3), dwall_s=round(w - self._last[0], 3), dcpu_s=round(c - self._last[1], 3), **extra); self._last = (w, c); self.d["milestones"].append(m); self.write(); return m
    def write(self): self.d["cpu_now_s"] = round(cpu_total(), 3); self.d["wall_now_s"] = round(time.time() - T0_WALL, 3); atomic_json(self.path, self.d)
# ----------------------------------------------------------------------------------------------------------------------------- mock workloads (no simulator)
def _burn(sec):
    t = time.process_time()
    while time.process_time() - t < sec: sum(i * i for i in range(2000))
def mock_spin(rep, g, arg):                         # cooperative: checks the guard every ~0.05 s of CPU
    rep.milestone("mock_start"); end = time.process_time() + arg
    while time.process_time() < end: _burn(0.05); g.check()
    rep.milestone("mock_end"); rep.d["status"] = "COMPLETE"
def mock_spin_nocheck(rep, g, arg): rep.milestone("mock_start"); _burn(arg); rep.milestone("mock_end"); rep.d["status"] = "COMPLETE"       # no cooperative check: only the launcher can stop it
def mock_sleep(rep, g, arg): rep.milestone("mock_start"); time.sleep(arg); rep.milestone("mock_end"); rep.d["status"] = "COMPLETE"            # wall time without CPU
def mock_child(rep, g, arg):                        # CPU burned in a child process that is waited for -> must appear in the aggregate
    rep.milestone("mock_start"); subprocess.run([sys.executable, "-c", f"import time\nt=time.process_time()\nwhile time.process_time()-t<{arg}: sum(i*i for i in range(2000))"], check=True); rep.milestone("mock_end"); rep.d["status"] = "COMPLETE"
def mock_threads(rep, g, arg):                      # 3 busy threads for `arg` wall seconds: CPU > wall
    rep.milestone("mock_start"); end = time.time() + arg
    def work():
        while time.time() < end: sum(i * i for i in range(2000))
    ts = [threading.Thread(target=work) for _ in range(3)]; [t.start() for t in ts]; [t.join() for t in ts]; rep.milestone("mock_end"); rep.d["status"] = "COMPLETE"
def mock_milestone_then_wait(rep, g, arg): rep.milestone("mock_start"); rep.milestone("mock_second_milestone"); time.sleep(arg); rep.d["status"] = "COMPLETE"                  # incremental-report smoke test
def mock_crash(rep, g, arg): rep.milestone("mock_start"); raise RuntimeError("mock crash")
def mock_live_grandchild(rep, g, arg):              # a LIVE (never waited for) grandchild burns CPU while the child sleeps: only the live-tree sampling can see it
    gp = subprocess.Popen([sys.executable, "-c", f"import time,os\nopen(os.environ['GC_PID_FILE'],'w').write(str(os.getpid()))\nt=time.process_time()\nwhile time.process_time()-t<{arg}: sum(i*i for i in range(2000))"]); rep.milestone("mock_grandchild_started", grandchild_pid=gp.pid); time.sleep(arg + 5); rep.d["status"] = "COMPLETE"
def mock_startup_slow(rep, g, arg): rep.milestone("mock_start"); _burn(arg); rep.milestone("mock_end"); rep.d["status"] = "COMPLETE"      # never logs mock_ready
def mock_startup_ok(rep, g, arg): rep.milestone("mock_start"); rep.milestone("mock_ready"); _burn(arg); rep.milestone("mock_end"); rep.d["status"] = "COMPLETE"
MOCKS = dict(live_grandchild=mock_live_grandchild, startup_slow=mock_startup_slow, startup_ok=mock_startup_ok, spin=mock_spin, spin_nocheck=mock_spin_nocheck, sleep=mock_sleep, child=mock_child, threads=mock_threads, wait=mock_milestone_then_wait, crash=mock_crash)
# ----------------------------------------------------------------------------------------------------------------------------- real workload (lazy heavy imports; every stage is a milestone)
def real(rep, guard, args):
    sys.path[:0] = [str(ROOT), str(ROOT / "sim"), str(ROOT / "eval"), str(ROOT / "data")]; os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import numpy as np; rep.milestone("import_numpy"); guard.check(10.0)
    import torch; torch.set_num_threads(1); rep.milestone("import_torch"); guard.check(10.0)
    import closed_loop_pilot as P; import g2a_act_corrections as G; import g2b_s0 as S0, g2b_trace as TR, g2b_labeller as LB, branching_replay as BR
    from policies.baseline import ScriptedExpert
    from control.safety_limits import SafetyLimits
    rep.milestone("import_project_modules"); guard.check(10.0)
    T_DEC, HORIZON, CHUNK = 40, 160, 20; d = rep.d; d.update(task="G2b limited intervention CPU benchmark (hardened)", pose=3000, t_dec=T_DEC, horizon=HORIZON, cap_s=guard.cap, guard_s=guard.limit)
    F = json.loads((ROOT / "outputs/g2a_act_corrections/protocol_freeze.json").read_text()); bad = [f for f, h in F["files_sha256"].items() if P.sha256(ROOT / f) != h]
    if bad: raise RuntimeError(f"frozen G2a files changed: {bad}")
    d["g2a_freeze_files_verified"] = sorted(F["files_sha256"]); pose = [q for q in F["poses"] if q["idx"] == 3000][0]; box_xy = (pose["x"], pose["y"]); yaw = pose["yaw"]; rep.milestone("g2a_freeze_verified")
    pol = P.load_chunk_policies(torch, ["c0_s0"])["c0_s0"]; rep.milestone("act_checkpoint_loaded"); guard.check(10.0)
    env = P.make_env(pol.cams); m = env.m; rep.milestone("mujoco_env_and_camera_renderer_initialized", note="LabEnv builds the model, the camera rig and the offscreen renderer together; they cannot be separated without touching frozen code"); guard.check(10.0)
    ex = ScriptedExpert(m, control_hz=20); kin = G.Kin(ex); lim = SafetyLimits.from_model(m); rep.milestone("ik_initialized"); guard.check(10.0)
    env.set_perturbation(); env.reset(box_xy, yaw); prev = env.prev_ctrl.copy(); g_prev0 = float(prev[7]); meta = dict(id="bench", pose=3000, t0=0); rec0 = TR.TraceRecorder(m, meta, refresh=True); box_z0 = float(env.d.xpos[rec0.cmap.box_body][2]); exe_pre = []; c1 = cpu_total()
    for t in range(0, T_DEC):
        if t % CHUNK == 0: o = BR.obs_full(env); raw = pol.chunk(o["state"], o["images"]); exe, _, _ = P.slew_chain(lim, raw, prev)
        u = exe[t % CHUNK]; rec0.record(env.d, u[7], u); P.lean_step(env, u, False); exe_pre.append(u.copy()); prev = u
    snap = BR.snapshot(env); prev40 = prev.copy(); d["shared"]["prefix_0_40_cpu_s"] = round(cpu_total() - c1, 3); rep.milestone("prefix_0_40_done"); guard.check(8.0)
    c2 = cpu_total(); o40 = BR.obs_full(env); raw40 = pol.chunk(o40["state"], o40["images"]); d["shared"]["decision_query_cpu_s"] = round(cpu_total() - c2, 3)
    err = G.oracle_error(kin, raw40, box_xy, yaw); applicable = bool(err is not None and exe_pre[-1][7] >= G.CLOSE_TRIG); d["shared"].update(applicable=applicable, oracle_error=err); rep.milestone("decision_state_query_done")
    if not applicable: raise Stop("pose 3000 is NOT_APPLICABLE for the correction candidates")
    cmap = {c["id"]: c for c in G.CANDS}; PLAN = [("c0", "c0"), ("O3", "c3"), ("D1", "c5"), ("D2", "c4"), ("S0", None)]; ref = {}; costs = []
    class Spy:
        def __init__(s, p, T): s.p, s.T, s.n = p, T, 0; s.policy = p.policy; s._o = p.policy.reset; s.resets = 0; p.policy.reset = s._r
        def _r(s): s.resets += 1; s._o()
        def chunk(s, st, im): c = cpu_total(); r = s.p.chunk(st, im); s.T["act_s"] += cpu_total() - c; s.n += 1; return r
    for name, gid in PLAN:
        guard.check(1.4 * max(costs) if costs else 8.0); rep.milestone(f"branch_start_{name}")
        T = dict(corr_s=0.0, act_s=0.0, step_s=0.0, rec_s=0.0, label_s=0.0); c_br = cpu_total(); w_br = time.time(); BR.restore(env, snap); rec = TR.TraceRecorder(m, meta, refresh=True); rec.rows = {k: list(v) for k, v in rec0.rows.items()}; rec._last = rec0._last; exe_all = []; nq = 0; resets = None; ik_fail = 0; n_ik = 0; hold_ok = None
        def do(u):
            c = cpu_total(); rec.record(env.d, u[7], u); T["rec_s"] += cpu_total() - c; c = cpu_total(); P.lean_step(env, u, False); T["step_s"] += cpu_total() - c; exe_all.append(np.asarray(u, float).copy())
            if len(exe_all) % CHUNK == 0: guard.check()
        if gid is not None:
            c = cpu_total(); raw_mod, dq_off, res = G.correct_chunk(kin, raw40, box_xy, yaw, cmap[gid], err); T["corr_s"] = cpu_total() - c; ik_fail, n_ik = int(res["ik_fail"]), int(res["n_ik"]); act = kin.ik.act; exe, _, _ = P.slew_chain(lim, raw_mod, prev40.copy())
            for u in exe: do(u)
            p_ = exe[-1]
            for t in range(T_DEC + CHUNK, HORIZON, CHUNK):
                c = cpu_total(); o = BR.obs_full(env); raw = pol.chunk(o["state"], o["images"]); T["act_s"] += cpu_total() - c; nq += 1
                if dq_off is not None: raw = raw.copy(); raw[:, act] += dq_off
                exe, _, _ = P.slew_chain(lim, raw, p_)
                for u in exe: do(u)
                p_ = exe[-1]
        else:
            spy = Spy(pol, T)
            class Ad:
                def step(s, u): c = cpu_total(); P.lean_step(env, u, False); T["step_s"] += cpu_total() - c
                def observe(s): c = cpu_total(); r = BR.obs_full(env); T["act_s"] += cpu_total() - c; return r
            def on_step(t, u, role): c = cpu_total(); rec.record(env.d, u[7], u); T["rec_s"] += cpu_total() - c
            ex_s0, lg = S0.run_branch(Ad(), spy, lim, prev40, T_DEC, HORIZON, k=S0.K_S0, on_step=on_step); exe_all = [np.asarray(u, float) for u in ex_s0]; nq = spy.n; resets = spy.resets; pol.policy.reset = spy._o
            hold_ok = bool(len(exe_all) == HORIZON - T_DEC and all(np.array_equal(exe_all[i], prev40) for i in range(S0.K_S0)))
        c = cpu_total(); rec.record_final(env.d); A = rec.arrays(); T["rec_s"] += cpu_total() - c
        c = cpu_total(); lab = LB.label(TR.to_labeller_trace(A, dict(t0=0, box_z0=box_z0, g_prev0=g_prev0))); T["label_s"] = cpu_total() - c; exe_arr = np.array(exe_all); total = cpu_total() - c_br; costs.append(total)
        legal = bool(P.check_legal(lim, exe_arr, prev40)); finite = bool(np.isfinite(exe_arr).all() and all(np.isfinite(v).all() for v in A.values())); pen = float(A["pen_obst"].max()); tcp = A["tcp_pos"][T_DEC:T_DEC + CHUNK + 1]; ex20 = exe_arr[:CHUNK]
        if name == "c0": ref.update(ex20=ex20.copy(), tcp=tcp.copy())
        sem = dict(arm_linf_vs_c0_first_chunk_rad=float(np.abs(ex20[:, :7] - ref["ex20"][:, :7]).max()), gripper_linf_vs_c0_first_chunk_m=float(np.abs(ex20[:, 7] - ref["ex20"][:, 7]).max()), tcp_dev_vs_c0_first_chunk_m=float(np.linalg.norm(tcp - ref["tcp"], axis=1).max()))
        valid = bool(legal and finite and pen <= G.PEN_MAX and ik_fail == 0 and lab["terminal"] != "INVALID" and len(exe_arr) == HORIZON - T_DEC and len(A["g_exec"]) == HORIZON + 1)
        distinct = bool(sem["arm_linf_vs_c0_first_chunk_rad"] >= G.DISTINCT["arm"] or sem["tcp_dev_vs_c0_first_chunk_m"] >= G.DISTINCT["tcp"])
        sem_ok = {"c0": True, "O3": distinct, "D1": distinct, "D2": bool(sem["arm_linf_vs_c0_first_chunk_rad"] == 0.0 and sem["gripper_linf_vs_c0_first_chunk_m"] >= 0.002), "S0": bool(hold_ok and resets == 1 and nq == len(S0.s0_query_steps(T_DEC, HORIZON)))}[name]
        row = dict(branch=name, g2a_id=gid, total_cpu_s=round(total, 3), wall_s=round(time.time() - w_br, 3), execution_status=("OK" if (valid and sem_ok) else "FAIL"), trace_and_label_cpu_s=round(T["rec_s"] + T["label_s"], 3), setup_ik_cpu_s=round(T["corr_s"], 3), act_cpu_s=round(T["act_s"], 3), n_act_queries_in_branch=nq, mj_step_cpu_s=round(T["step_s"], 3), recorder_cpu_s=round(T["rec_s"], 3), labelling_cpu_s=round(T["label_s"], 3),
                   other_restore_overhead_cpu_s=round(total - sum(T.values()), 3), n_ik=n_ik, ik_fail=ik_fail, executed_steps=int(len(exe_arr)), trace_rows=int(len(A["g_exec"])), legal_clip_slew=legal, finite=finite, obstacle_penetration_max_m=pen, label_computed_not_invalid=bool(lab["terminal"] != "INVALID"), valid=valid,
                   action_semantics=sem, action_semantics_as_expected=sem_ok, s0=dict(hold_8_equals_last_target=hold_ok, policy_reset_count=resets) if gid is None else None)
        d["branches"].append(row); rep.milestone(f"branch_end_{name}", total_cpu_s=row["total_cpu_s"], valid=valid, semantics_ok=sem_ok)
        if not (valid and sem_ok): d["status"] = "STOPPED_ON_FAIL"; d["stop_reason"] = f"{name}: valid={valid}, semantics_as_expected={sem_ok}"; return
    d["status"] = "COMPLETE"
# ----------------------------------------------------------------------------------------------------------------------------- child / launcher
def child_main(args):
    rep = Report(args.report + ".child.json"); rep.milestone("child_process_start_stdlib_only"); guard = Guard(args.cap_s, args.guard_frac, args.startup_limit_s, args.startup_milestone); rep.guard = guard; rep.d.update(cap_s=args.cap_s, guard_s=guard.limit, startup_limit_s=guard.startup_limit)
    try:
        if args.mock: MOCKS[args.mock](rep, guard, args.mock_arg)
        else: real(rep, guard, args)
    except Stop as e: rep.d["status"] = "PARTIAL"; rep.d["stop_reason"] = str(e)
    except Exception as e: rep.d["status"] = "BLOCKED"; rep.d["stop_reason"] = f"{type(e).__name__}: {e}"; rep.d["traceback"] = traceback.format_exc()[-1500:]
    rep.milestone("child_exit", status=rep.d["status"]); return 0
def proc_table():
    t = {}
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try: f = Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split(); t[int(d)] = dict(ppid=int(f[1]), pgrp=int(f[2]), own=(int(f[11]) + int(f[12])) / TICK, reaped=(int(f[13]) + int(f[14])) / TICK)
        except (FileNotFoundError, ProcessLookupError, IndexError, PermissionError, ValueError): continue
    return t
def tree_members(table, root, pgid):
    m = {root} if root in table else set(); m |= {p for p, v in table.items() if v["pgrp"] == pgid}; grew = True
    while grew:
        grew = False
        for p, v in table.items():
            if p not in m and v["ppid"] in m: m.add(p); grew = True
    return m
def tree_cpu(root, pgid):
    """-> (aggregate CPU of the live tree in seconds, sorted live member pids); (None, []) when nothing is left / unreadable.  Each member contributes utime+stime (all its threads) + cutime+cstime (its already-reaped children): no CPU is counted twice."""
    try: tb = proc_table()
    except OSError: return None, []
    m = tree_members(tb, root, pgid)
    if not m: return None, []
    return sum(tb[p]["own"] + tb[p]["reaped"] for p in m), sorted(m)
def read_json(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def launcher_main(args):
    rep = dict(task="G2b intervention CPU benchmark launcher", status="LAUNCHED", cap_s=args.cap_s, guard_frac=args.guard_frac, poll_s=args.poll_s, kill_margin_s=args.kill_margin_s, startup_limit_s=args.startup_limit_s, startup_milestone=args.startup_milestone, mock=args.mock, wall_clock_timeout_used_for_budget=False, accounting={})
    report = Path(args.report)
    if not Path("/proc/self/stat").exists() or tree_cpu(os.getpid(), os.getpgid(0))[0] is None: rep.update(status="BLOCKED", stop_reason="aggregate CPU of live descendants cannot be read (/proc unavailable): the cap would not be enforced accurately, so the benchmark is BLOCKED"); atomic_json(report, rep); print(json.dumps(dict(status="BLOCKED"))); return 2
    atomic_json(report, rep); ru0 = resource.getrusage(resource.RUSAGE_CHILDREN); cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--cap-s", str(args.cap_s), "--guard-frac", str(args.guard_frac), "--report", args.report, "--startup-limit-s", str(args.startup_limit_s or 0), "--startup-milestone", args.startup_milestone]
    if args.mock: cmd += ["--mock", args.mock, "--mock-arg", str(args.mock_arg)]
    t_launch = time.time(); p = subprocess.Popen(cmd, start_new_session=True); pgid = p.pid; peak = 0.0; members_seen = set(); reason = None; kill_sample = None; unreadable = 0; last_write = 0.0; last_chk = 0.0; killed = False
    def kill_group():
        try: os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError): pass
    for sg in (signal.SIGTERM, signal.SIGINT): signal.signal(sg, lambda *_: (kill_group(), sys.exit(130)))
    while p.poll() is None:
        c, mem = tree_cpu(p.pid, pgid)
        if c is None: unreadable += 1
        else:
            unreadable = 0; peak = max(peak, c); members_seen |= set(mem)
            if c >= args.cap_s - args.kill_margin_s: kill_sample = c; reason = f"launcher hard cap: sampled tree CPU {c:.2f} s reached cap {args.cap_s} s minus kill margin {args.kill_margin_s} s; whole process group killed (SIGKILL)"; kill_group(); killed = True; break
            if args.startup_limit_s and c >= args.startup_limit_s and time.time() - last_chk > 0.3:
                last_chk = time.time(); chj = read_json(args.report + ".child.json"); names = [m["name"] for m in (chj or {}).get("milestones", [])]
                if args.startup_milestone not in names: kill_sample = c; reason = f"start-up sub-limit: {c:.2f} s CPU >= {args.startup_limit_s} s before the milestone `{args.startup_milestone}`; whole process group killed"; kill_group(); killed = True; break
        if unreadable >= 5: reason = "BLOCKED: the live process tree became unreadable while the child was running"; kill_group(); killed = True; break
        if time.time() - last_write > 1.0: last_write = time.time(); ch = read_json(args.report + ".child.json"); rep.update(status="RUNNING", sampled_tree_cpu_s=round(peak, 3), live_members=len(mem) if c is not None else None, child_report_snapshot=ch); atomic_json(report, rep)
        time.sleep(args.poll_s)
    t_detect = time.time(); rc = p.wait(); kill_group(); remaining = []
    for _ in range(30):                                                                                                      # verify that nothing of the workload is left running (up to 3 s)
        tb = proc_table(); remaining = sorted(q for q in tree_members(tb, -1, pgid) if q != os.getpid()); remaining = [q for q in remaining if tb.get(q, {}).get("pgrp") == pgid]
        if not remaining: break
        kill_group(); time.sleep(0.1)
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN); own = resource.getrusage(resource.RUSAGE_SELF); rus = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime); own_s = own.ru_utime + own.ru_stime; agg = max(rus, peak) + own_s
    ch = read_json(args.report + ".child.json"); cs = (ch or {}).get("status")
    if reason is None and unreadable == 0: reason = (ch or {}).get("stop_reason")
    status = "BLOCKED" if (reason or "").startswith("BLOCKED") else ("PARTIAL" if killed else ("BLOCKED" if (rc != 0 or ch is None) else cs if cs in ("COMPLETE", "PARTIAL", "STOPPED_ON_FAIL", "BLOCKED") else "BLOCKED"))
    rep.update(status=status, stop_reason=reason, child_exit_code=rc, child_report=ch, accounting=dict(authoritative_rusage_children_cpu_s=round(rus, 3), sampled_tree_peak_cpu_s=round(peak, 3), launcher_cpu_s=round(own_s, 3), aggregate_cpu_s=round(agg, 3), child_cpu_total_s=round(rus, 3), cap_s=args.cap_s, kill_threshold_s=args.cap_s - args.kill_margin_s,
               killed_by_launcher=killed, kill_triggering_sample_cpu_s=kill_sample, cap_overshoot_s=round(max(0.0, agg - args.cap_s), 3), cap_respected=bool(agg <= args.cap_s), wall_s=round(time.time() - t_launch, 3), kill_to_exit_s=round(time.time() - t_detect, 3), poll_s=args.poll_s, members_seen=len(members_seen),
               descendants_remaining_after_run=remaining, no_workload_process_left=bool(not remaining), basis="aggregate = max(getrusage(RUSAGE_CHILDREN) delta, running max of the sampled live tree: sum over members of utime+stime+cutime+cstime) + launcher RUSAGE_SELF; the two child figures measure the same CPU, so they are never added"))
    rep["accounting"]["killed_by_cap"] = killed; atomic_json(report, rep); print(json.dumps(dict(status=status, aggregate_cpu_s=rep["accounting"]["aggregate_cpu_s"], wall_s=rep["accounting"]["wall_s"], cap_s=args.cap_s, overshoot_s=rep["accounting"]["cap_overshoot_s"], leftovers=remaining))); return 0
def parse(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--cap-s", type=float, required=True, help="hard aggregate CPU cap in CPU-seconds (no default on purpose)"); ap.add_argument("--report", required=True); ap.add_argument("--guard-frac", type=float, default=0.92); ap.add_argument("--poll-s", type=float, default=0.1); ap.add_argument("--kill-margin-s", type=float, default=0.5); ap.add_argument("--startup-limit-s", type=float, default=45.0, help="0 disables"); ap.add_argument("--startup-milestone", default="ik_initialized")
    ap.add_argument("--child", action="store_true"); ap.add_argument("--mock", choices=sorted(MOCKS)); ap.add_argument("--mock-arg", type=float, default=1.0); return ap.parse_args(argv)
if __name__ == "__main__":
    a = parse(); sys.exit(child_main(a) if a.child else launcher_main(a))
