#!/usr/bin/env python3
"""MOCK tests of the hardened benchmark launcher: CPU accounting, hard-cap termination, cooperative guard, incremental report, failure statuses.  Pure Python workloads (busy loops, sleeps, a CPU-burning grandchild); NO ACT, NO MuJoCo, NO torch.
Run: python3 eval/test_g2b_benchmark_accounting.py -> outputs/g2b_impl/benchmark_accounting_report.json (a failing run is additionally kept as ..._FAIL_<time>.json)"""
import sys, os, json, time, subprocess, tempfile, traceback, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; SCRIPT = ROOT / "eval/g2b_intervention_benchmark.py"; OUT = ROOT / "outputs/g2b_impl"; TMP = OUT / "bench_mock"; TMP.mkdir(parents=True, exist_ok=True)
def run(name, cap, mock, arg, guard_frac=0.92, extra=(), env=None):
    rp = TMP / f"{name}.json"
    for f in (rp, Path(str(rp) + ".child.json")):
        if f.exists(): f.unlink()
    t = time.time(); r = subprocess.run([sys.executable, str(SCRIPT), "--cap-s", str(cap), "--report", str(rp), "--mock", mock, "--mock-arg", str(arg), "--guard-frac", str(guard_frac), *extra], capture_output=True, text=True, env=env)
    return json.loads(rp.read_text()), time.time() - t, r
RES = []
def test(name):
    def deco(fn):
        try: ok, ev = fn()
        except Exception as e: ok, ev = False, dict(exception=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-600:])
        RES.append(dict(test=name, result="PASS" if ok else "FAIL", evidence=ev)); print(("PASS " if ok else "FAIL ") + name, "" if ok else json.dumps(ev, default=str)[:400], flush=True)
    return deco
@test("cooperative CPU workload completes: status COMPLETE, aggregate CPU ~ requested CPU, accounting from process start")
def _():
    r, w, p = run("spin", 20, "spin", 1.0); a = r["accounting"]; ch = r["child_report"]
    return r["status"] == "COMPLETE" and 0.9 <= a["child_cpu_total_s"] <= 1.8 and abs(ch["cpu_now_s"] - a["child_cpu_total_s"]) < 0.5 and ch["cpu_at_script_start_s"] < 0.5 and r["wall_clock_timeout_used_for_budget"] is False, dict(accounting=a)
@test("CPU time and wall time are separate: a 2 s sleep costs ~0 CPU and >= 2 s wall")
def _():
    r, w, p = run("sleep", 20, "sleep", 2.0); a = r["accounting"]; return r["status"] == "COMPLETE" and a["wall_s"] >= 2.0 and a["child_cpu_total_s"] < 0.8, dict(accounting=a)
@test("CPU of a waited-for grandchild process is included in the aggregate")
def _():
    r, w, p = run("child", 20, "child", 1.0); a = r["accounting"]; return r["status"] == "COMPLETE" and a["child_cpu_total_s"] >= 0.9, dict(accounting=a)
@test("multi-threaded workload: all threads are accounted (aggregate >= the CPU actually burned)")
def _():
    r, w, p = run("threads", 20, "threads", 1.5); a = r["accounting"]; return r["status"] == "COMPLETE" and a["child_cpu_total_s"] >= 1.3, dict(accounting=a)
@test("HARD CAP: a workload with no cooperative checks is killed by the launcher; status PARTIAL (never COMPLETE); CPU stays within cap + poll overshoot")
def _():
    r, w, p = run("hardcap", 1.5, "spin_nocheck", 8.0); a = r["accounting"]
    return r["status"] == "PARTIAL" and a["killed_by_cap"] and 0.9 <= a["child_cpu_total_s"] <= 1.5 + 0.3 and a["aggregate_cpu_s"] <= 1.5 + 0.3 and a["no_workload_process_left"] and "launcher hard cap" in (r["stop_reason"] or "") and "cap_overshoot_s" in a, dict(accounting=a, status=r["status"])
@test("LIVE DESCENDANT: a live, never-waited grandchild burning CPU while the child sleeps is counted by the tree sampling; the cap kills the WHOLE process group; the grandchild is dead afterwards and nothing is left running")
def _():
    gf = TMP / "gc.pid"; gf.exists() and gf.unlink(); e = dict(os.environ, GC_PID_FILE=str(gf)); r, w, p = run("livegc", 1.5, "live_grandchild", 20.0, env=e); a = r["accounting"]; time.sleep(0.3); gpid = int(gf.read_text()) if gf.exists() else None
    def alive(pid):
        try: os.kill(pid, 0); return True
        except ProcessLookupError: return False
        except PermissionError: return True
    zombie_free = gpid is not None and not Path(f"/proc/{gpid}").exists()
    return r["status"] == "PARTIAL" and a["killed_by_launcher"] and a["sampled_tree_peak_cpu_s"] >= 0.9 and a["aggregate_cpu_s"] <= 1.5 + 0.4 and a["no_workload_process_left"] and a["members_seen"] >= 2 and zombie_free and w < 10, dict(accounting=a, grandchild_pid=gpid, grandchild_gone=zombie_free, wall=w)
@test("NO DOUBLE COUNTING: 1.0 s of CPU burned in a reaped grandchild gives an aggregate of about 1.0 s (not 2), and the sampled tree agrees with getrusage within 0.4 s")
def _():
    r, w, p = run("nodouble", 20, "child", 1.0); a = r["accounting"]
    return r["status"] == "COMPLETE" and a["aggregate_cpu_s"] < 1.8 and a["aggregate_cpu_s"] >= 0.95 and abs(a["sampled_tree_peak_cpu_s"] - a["authoritative_rusage_children_cpu_s"]) < 0.4, dict(accounting=a)
@test("START-UP SUB-LIMIT: CPU >= limit before the required milestone stops the run as PARTIAL (process group killed); the same load after the milestone is allowed")
def _():
    r1, w1, _p = run("startup_slow", 20, "startup_slow", 5.0, extra=("--startup-limit-s", "1.0", "--startup-milestone", "mock_ready")); r2, w2, _q = run("startup_ok", 20, "startup_ok", 1.5, extra=("--startup-limit-s", "1.0", "--startup-milestone", "mock_ready"))
    a1, a2 = r1["accounting"], r2["accounting"]
    return r1["status"] == "PARTIAL" and "start-up sub-limit" in (r1["stop_reason"] or "") and a1["aggregate_cpu_s"] < 2.2 and a1["no_workload_process_left"] and r2["status"] == "COMPLETE", dict(slow=dict(status=r1["status"], reason=r1["stop_reason"], acc=a1), ok=dict(status=r2["status"], acc=a2))
@test("overshoot is reported explicitly and the cap is respected on every capped mock run (cap_respected, cap_overshoot_s present)")
def _():
    rs = [json.loads((TMP / f"{n}.json").read_text())["accounting"] for n in ("hardcap", "livegc", "startup_slow")]; return all("cap_overshoot_s" in a and a["cap_respected"] and a["cap_overshoot_s"] == 0.0 for a in rs), rs
@test("COOPERATIVE GUARD: the child stops itself at guard_frac * cap with status PARTIAL; the launcher does not need to kill it")
def _():
    r, w, p = run("guard", 3.0, "spin", 9.0, guard_frac=0.5); a = r["accounting"]
    return r["status"] == "PARTIAL" and not a["killed_by_cap"] and a["child_cpu_total_s"] < 2.2 and "cooperative" in (r["stop_reason"] or ""), dict(accounting=a, stop=r["stop_reason"])
@test("a crash in the workload is reported as BLOCKED (never PASS/COMPLETE) with the traceback")
def _():
    r, w, p = run("crash", 10, "crash", 1.0); return r["status"] == "BLOCKED" and "mock crash" in (r["stop_reason"] or ""), dict(status=r["status"], stop=r["stop_reason"])
@test("incremental atomic report: the launcher report exists before the child finishes and the child report already holds milestones while it runs")
def _():
    rp = TMP / "incremental.json"; cp = Path(str(rp) + ".child.json")
    for f in (rp, cp):
        if f.exists(): f.unlink()
    pr = subprocess.Popen([sys.executable, str(SCRIPT), "--cap-s", "30", "--report", str(rp), "--mock", "wait", "--mock-arg", "3"]); seen = None; t = time.time()
    while time.time() - t < 2.5 and seen is None:
        if cp.exists() and rp.exists():
            try: c = json.loads(cp.read_text()); l = json.loads(rp.read_text())
            except Exception: continue
            if len(c["milestones"]) >= 3 and c["status"] == "RUNNING": seen = (len(c["milestones"]), c["status"], l["status"])
        time.sleep(0.05)
    pr.wait(); fin = json.loads(rp.read_text()); return seen is not None and pr.returncode == 0 and fin["status"] == "COMPLETE", dict(seen_while_running=seen, final=fin["status"])
@test("real-mode guard path without the simulator: with a tiny cap the real workload stops after the numpy import with status PARTIAL and a milestone log (no torch / ACT / MuJoCo import)")
def _():
    rp = TMP / "real_tiny_cap.json"; r = subprocess.run([sys.executable, str(SCRIPT), "--cap-s", "1.5", "--kill-margin-s", "0.2", "--report", str(rp)], capture_output=True, text=True); rep = json.loads(rp.read_text()); names = [m["name"] for m in rep["child_report"]["milestones"]] if rep.get("child_report") else []
    return rep["status"] == "PARTIAL" and "import_numpy" in names and "import_torch" not in names and "act_checkpoint_loaded" not in names, dict(status=rep["status"], milestones=names, stop=rep["stop_reason"], acc=rep["accounting"])
@test("no /proc -> BLOCKED with an explicit limitation message (simulated)")
def _():
    spec = importlib.util.spec_from_file_location("bench_mod", SCRIPT); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); orig = Path.exists
    Path.exists = lambda self: False if str(self) == "/proc/self/stat" else orig(self)
    try: rc = mod.launcher_main(mod.parse(["--cap-s", "5", "--report", str(TMP / "noproc.json")]))
    finally: Path.exists = orig
    rep = json.loads((TMP / "noproc.json").read_text()); return rc == 2 and rep["status"] == "BLOCKED" and "cannot be read" in rep["stop_reason"] and "BLOCKED" in rep["stop_reason"], dict(status=rep["status"])
@test("static: no wall-clock timeout is used to enforce the budget; no retry logic; cap has no default; status vocabulary never contains PASS")
def _():
    src = SCRIPT.read_text(); code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("#", '"""')))
    return ("timeout=" not in code) and ("retry" not in code.lower().replace("no retry", "")) and ('"--cap-s", type=float, required=True' in src) and ('status="PASS"' not in src), dict()
rep = dict(verdict="PASS" if all(r["result"] == "PASS" for r in RES) else "FAIL", n=len(RES), results=RES); (OUT / "benchmark_accounting_report.json").write_text(json.dumps(rep, indent=1, default=str))
if rep["verdict"] != "PASS": (OUT / f"benchmark_accounting_report_FAIL_{time.strftime('%H%M%S')}.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(dict(verdict=rep["verdict"], n=rep["n"])))
