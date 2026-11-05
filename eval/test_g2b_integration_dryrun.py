#!/usr/bin/env python3
"""FULL-HARNESS DRY-RUN (no ACT, no torch, no MuJoCo): the real orchestrator, launcher, child, report writer, S0 runner, labeller and the REAL gate code of I2-I4 against a deterministic mock environment (eval/g2b_integration_dryrun.py).
Scenarios: normal completion; gate FAIL (I2, I3, I4); exception inside a gate; cooperative cap; hard cap (kill); missing report; truncated report; child crash; I1 reuse refused (tampered manifest); gate subset (final status must not be COMPLETE); static AST checks of every rep.milestone(...) call site.
Run: python3 eval/test_g2b_integration_dryrun.py -> outputs/g2b_impl/integration_dryrun_report.json (failing run also kept as ..._FAIL_<time>.json)"""
import sys, os, json, ast, time, subprocess, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "outputs/g2b_impl"; TMP = OUT / "dryrun"; TMP.mkdir(parents=True, exist_ok=True); SCRIPT = ROOT / "eval/g2b_integration_v2.py"
MAN = ROOT / "outputs/g2b_integration/I1_integrity_manifest.json"; I1REP = ROOT / "outputs/g2b_integration/integration_v3_run_BLOCKED_after_I1.json"; MAN23 = ROOT / "outputs/g2b_integration/I2I3_integrity_manifest.json"; V4REP = ROOT / "outputs/g2b_integration/integration_v4_I2_I4.json"
def reuse_json(i1=True, i23=False, man1=MAN, man23=MAN23):
    r = []
    if i1: r.append(dict(kind="I1", gates=["I1"], manifest=str(man1), report=str(I1REP)))
    if i23: r.append(dict(kind="I23", gates=["I2", "I3"], manifest=str(man23), report=str(V4REP)))
    return json.dumps(r)
RES = []; ALLOWED = {"COMPLETE", "PARTIAL", "STOPPED_ON_FAIL", "BLOCKED"}
def t(name, ok, ev=None): RES.append(dict(test=name, result="PASS" if ok else "FAIL", evidence=ev)); print(("PASS " if ok else "FAIL ") + name, "" if ok else json.dumps(ev, default=str)[:400], flush=True)
def validate(rep, expect_complete=None):
    """schema + consistency of a launcher report; returns a list of problems"""
    p = []; req = {"task", "status", "cap_s", "accounting", "stop_reason", "child_report"}
    if not req <= set(rep): p.append(f"missing top-level keys {req - set(rep)}")
    if rep.get("status") not in ALLOWED: p.append(f"status {rep.get('status')}")
    a = rep.get("accounting", {})
    for k in ("aggregate_cpu_s", "authoritative_rusage_children_cpu_s", "sampled_tree_peak_cpu_s", "launcher_cpu_s", "wall_s", "cap_overshoot_s", "cap_respected", "no_workload_process_left", "killed_by_launcher"):
        if k not in a: p.append(f"accounting.{k} missing")
    ch = rep.get("child_report")
    if ch:
        for m in ch.get("milestones", []):
            if not {"name", "wall_s", "cpu_s", "dwall_s", "dcpu_s"} <= set(m): p.append(f"milestone schema {m.get('name')}")
        names = [g["gate"] for g in ch.get("gates", [])]
        if len(names) != len(set(names)): p.append("duplicate gate rows")
        for g in ch.get("gates", []):
            if not {"gate", "source", "result", "cpu_s", "wall_s", "evidence"} <= set(g): p.append(f"gate row schema {g.get('gate')}")
            if g["result"] not in ("PASS", "FAIL", "PARTIAL"): p.append(f"gate result {g['result']}")
        for pr_ in ch.get("i4_poses", []):
            if not {"pose", "status", "completed_physics_steps", "act_queries", "setup_cpu_s", "act_cpu_s", "physics_cpu_s", "recorder_cpu_s", "labeller_cpu_s", "aggregate_cpu_s", "wall_s", "validity", "code_sha256"} <= set(pr_): p.append(f"i4 pose record schema {pr_.get('pose')}")
        passed = {g["gate"] for g in ch.get("gates", []) if g["result"] == "PASS"}
        if rep.get("status") == "COMPLETE" and passed != {"I1", "I2", "I3", "I4"}: p.append(f"COMPLETE without PASS evidence for all four gates: {sorted(passed)}")
        ends = [m for m in ch.get("milestones", []) if m["name"].startswith("gate_") and m["name"].endswith("_end")]
        if any("gate_cpu_s" not in m for m in ends): p.append("gate end milestone lacks gate_cpu_s")
    try: json.loads(json.dumps(rep))
    except Exception as e: p.append(f"json round trip {e}")
    if list(Path(rep.get("_dir", TMP)).glob("*.tmp")): p.append("leftover .tmp files")
    return p
want = lambda n: (not os.environ.get("DRY_ONLY")) or n in os.environ["DRY_ONLY"].split(",")
def run(name, mode, cap=30.0, gates="I2,I3,I4", manifest=MAN, extra=(), guard_frac="0.92", scale="1.0", reuse=None):
    rp = TMP / f"{name}.json"
    for f in (rp, Path(str(rp) + ".child.json")):
        if f.exists(): f.unlink()
    env = dict(os.environ, G2B_INT_DRY=mode, G2B_INT_GATES=gates, G2B_INT_REUSE=(reuse if reuse is not None else reuse_json(True, False, man1=manifest)), G2B_INT_NEED_SCALE=scale); t0 = time.time()
    r = subprocess.run([sys.executable, str(SCRIPT), "--cap-s", str(cap), "--report", str(rp), "--startup-milestone", "act_setup_done", "--startup-limit-s", "0", "--guard-frac", guard_frac, *extra], capture_output=True, text=True, env=env)
    rep = json.loads(rp.read_text()) if rp.exists() else {}; return rep, time.time() - t0, r
def gates_of(rep): return {g["gate"]: g for g in (rep.get("child_report") or {}).get("gates", [])}
# ---- static
src = {f: (ROOT / "eval" / f).read_text() for f in ("g2b_integration_v2.py", "g2b_integration_dryrun.py", "g2b_intervention_benchmark.py")}; RESERVED = {"name", "wall_s", "cpu_s", "dwall_s", "dcpu_s", "self"}; bad = []; sites = 0
for f, text in src.items():
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "milestone":
            sites += 1; kws = [k.arg for k in node.keywords]
            if any(k is None for k in kws): bad.append((f, node.lineno, "** expansion"))
            if set(kws) & RESERVED: bad.append((f, node.lineno, sorted(set(kws) & RESERVED)))
            if len(kws) != len(set(kws)): bad.append((f, node.lineno, "duplicate keyword"))
t(f"static AST: all {sites} rep.milestone(...) call sites avoid the reserved fields {sorted(RESERVED)} (so gate_cpu_s can never collide with cpu_s), use no ** expansion and no duplicate keyword", not bad and sites >= 20, dict(sites=sites, bad=bad))
for f in src: compile(src[f], f, "exec")
t("static: syntax of the harness, the dry-run mock and the launcher compiles", True)
# ---- scenarios
rep, w, r = run("normal", "normal"); g = gates_of(rep); pr = validate(rep)
t("NORMAL: I1 reused (verified) + stub-free I2, I3, I4 gate code PASS on the mock -> COMPLETE; schema valid; every gate wrote an incremental milestone; gate_cpu_s present", rep.get("status") == "COMPLETE" and set(g) == {"I1", "I2", "I3", "I4"} and g["I1"]["source"] == "REUSED_FROM_SAVED_REPORT" and all(g[k]["source"] == "THIS_RUN" and g[k]["result"] == "PASS" for k in ("I2", "I3", "I4")) and not pr, dict(status=rep.get("status"), problems=pr, stop=rep.get("stop_reason"), gate_results={k: v["result"] for k, v in g.items()}, stderr=r.stderr[-300:]))
for nm, mode, failing, notrun in (("fail_i2", "bad_restore", "I2", ["I3", "I4"]), ("fail_i3", "stateful_policy", "I3", ["I4"]), ("fail_i4", "two_poses", "I4", [])):
    rep, w, r = run(nm, mode); g = gates_of(rep); pr = validate(rep)
    t(f"GATE FAIL ({failing}): status STOPPED_ON_FAIL, {failing} FAIL with evidence, later gates not executed, never COMPLETE", rep.get("status") == "STOPPED_ON_FAIL" and g.get(failing, {}).get("result") == "FAIL" and not any(k in g for k in notrun) and not pr, dict(status=rep.get("status"), gates={k: v["result"] for k, v in g.items()}, problems=pr, stop=rep.get("stop_reason")))
rep, w, r = run("exc_i3", "raise_in_i3"); g = gates_of(rep); pr = validate(rep)
t("EXCEPTION inside a gate (I3): BLOCKED with the traceback, the already finished I2 PASS evidence is preserved, no I3/I4 rows, never COMPLETE", rep.get("status") == "BLOCKED" and "mock policy exception" in (rep.get("stop_reason") or "") and g.get("I2", {}).get("result") == "PASS" and "I3" not in g and not pr, dict(status=rep.get("status"), stop=rep.get("stop_reason"), gates=list(g), problems=pr))
rep, w, r = run("coop_cap", "coop_cap", cap=3.0); a = rep.get("accounting", {}); pr = validate(rep)
t("COOPERATIVE CAP: the child stops itself (PARTIAL) before the cap, no kill needed, cap respected, nothing left running", rep.get("status") == "PARTIAL" and a.get("killed_by_launcher") is False and a.get("cap_respected") and a.get("no_workload_process_left") and not pr, dict(status=rep.get("status"), stop=rep.get("stop_reason"), acc=a, problems=pr))
rep, w, r = run("hard_cap", "hard_spin", cap=3.0, scale="0"); a = rep.get("accounting", {}); pr = validate(rep)
t("HARD CAP: a workload that ignores the guard is killed with its whole process group (PARTIAL), CPU stays within cap + 0.3 s, overshoot reported, nothing left running", rep.get("status") == "PARTIAL" and a.get("killed_by_launcher") and a.get("aggregate_cpu_s", 99) <= 3.0 + 0.3 and a.get("no_workload_process_left") and "cap_overshoot_s" in a and not pr, dict(status=rep.get("status"), acc=a, problems=pr))
rep, w, r = run("no_report", "no_report"); t("MISSING child report: BLOCKED (never COMPLETE)", rep.get("status") == "BLOCKED", dict(status=rep.get("status"), stop=rep.get("stop_reason")))
rep, w, r = run("truncated", "truncated_report"); t("TRUNCATED child report: BLOCKED (never COMPLETE)", rep.get("status") == "BLOCKED", dict(status=rep.get("status"), stop=rep.get("stop_reason")))
rep, w, r = run("crash", "child_crash_hard"); t("CHILD CRASH (exit code 7): not COMPLETE", rep.get("status") in ("BLOCKED", "PARTIAL"), dict(status=rep.get("status"), rc=rep.get("child_exit_code")))
tm = TMP / "tampered_manifest.json"; man = json.loads(MAN.read_text()) if MAN.exists() else {}; man2 = copy.deepcopy(man); k0 = next(iter(man2.get("code", {"x": 0})), None)
if k0 and k0 in man2["code"]: man2["code"][k0]["sha256"] = "0" * 64
tm.write_text(json.dumps(man2)); rep, w, r = run("reuse_refused", "normal", manifest=tm); g = gates_of(rep)
t("I1 REUSE REFUSED when an input hash differs from the manifest: BLOCKED with 'I1 reuse is no longer justified', no gate executed", rep.get("status") == "BLOCKED" and "no longer justified" in (rep.get("stop_reason") or "") and not any(x in g for x in ("I2", "I3", "I4")), dict(status=rep.get("status"), stop=rep.get("stop_reason")))
rep, w, r = run("subset", "normal", gates="I2,I3"); pr = validate(rep)
t("FINAL STATUS RULE: with I4 not executed the final status is PARTIAL (required gate without PASS evidence), never COMPLETE", rep.get("status") == "PARTIAL" and "I4" in (rep.get("stop_reason") or "") and not pr, dict(status=rep.get("status"), stop=rep.get("stop_reason"), problems=pr))
# ---- I4 incremental reporting (final version of the code)
RJ = reuse_json(True, True); POSE_KEYS = ("pose", "status", "completed_physics_steps", "act_queries", "setup_cpu_s", "act_cpu_s", "physics_cpu_s", "recorder_cpu_s", "labeller_cpu_s", "aggregate_cpu_s", "wall_s", "validity", "code_sha256")
rep, w, r = run("i4_only_normal", "normal", gates="I4", reuse=RJ); g = gates_of(rep); ch = rep.get("child_report") or {}; pr = validate(rep); poses = ch.get("i4_poses", [])
t("I4-ONLY NORMAL: I1 + I2/I3 evidence reused (verified, code regions identical), I4 executed with 3 poses -> COMPLETE; every pose record has all required fields and valid traces", rep.get("status") == "COMPLETE" and set(g) == {"I1", "I2", "I3", "I4"} and [g[k]["source"] for k in ("I1", "I2", "I3", "I4")] == ["REUSED_FROM_SAVED_REPORT"] * 3 + ["THIS_RUN"] and len(poses) == 3 and all(set(POSE_KEYS) <= set(x) and x["status"] == "COMPLETE" and x["validity"]["all_valid"] for x in poses) and not pr, dict(status=rep.get("status"), stop=rep.get("stop_reason"), n_poses=len(poses), problems=pr, stderr=r.stderr[-300:]))
rp = TMP / "i4_incremental.json"; cp = Path(str(rp) + ".child.json")
for f in (rp, cp):
    if f.exists(): f.unlink()
env = dict(os.environ, G2B_INT_DRY="i4_slow", G2B_INT_GATES="I4", G2B_INT_REUSE=RJ, G2B_INT_NEED_SCALE="1.0"); pr_ = subprocess.Popen([sys.executable, str(SCRIPT), "--cap-s", "30", "--report", str(rp), "--startup-milestone", "act_setup_done", "--startup-limit-s", "0"], env=env); seen = None; t_ = time.time()
while time.time() - t_ < 20 and pr_.poll() is None and seen is None:
    if cp.exists():
        try: c = json.loads(cp.read_text())
        except Exception: continue
        if c.get("status") == "RUNNING" and len(c.get("i4_poses", [])) >= 1 and (c.get("i4_in_progress") is not None or len(c["i4_poses"]) < 3): seen = (len(c["i4_poses"]), c["status"], (c.get("i4_in_progress") or {}).get("pose"), (c.get("i4_in_progress") or {}).get("completed_physics_steps"))
    time.sleep(0.02)
pr_.wait(); fin = json.loads(rp.read_text()); t("INCREMENTAL I4 REPORT: while the run is still RUNNING the child report already holds the completed pose record(s) and the in-progress pose with its completed physics steps; no .tmp files remain", seen is not None and fin["status"] == "COMPLETE" and not list(TMP.glob("*.tmp")), dict(seen_while_running=seen, final=fin["status"]))
rep, w, r = run("i4_between", "i4_between", cap=3.0, guard_frac="0.5", gates="I4", scale="0.03", reuse=RJ); g = gates_of(rep); ch = rep.get("child_report") or {}; pr = validate(rep); poses = ch.get("i4_poses", [])
t("INTERRUPTION BETWEEN POSES (cooperative guard before pose 3): status PARTIAL, gate I4 labelled PARTIAL, poses 1-2 preserved as COMPLETE with full records, no pose 3 record, never COMPLETE", rep.get("status") == "PARTIAL" and g.get("I4", {}).get("result") == "PARTIAL" and [x["pose"] for x in poses] == [3000, 3001] and all(x["status"] == "COMPLETE" for x in poses) and g["I4"]["evidence"]["completed_poses"] == 2 and g["I4"]["evidence"]["interrupted_pose"] is None and not pr, dict(status=rep.get("status"), poses=[x["pose"] for x in poses], gate=g.get("I4", {}).get("result"), stop=rep.get("stop_reason"), problems=pr))
rep, w, r = run("i4_mid", "i4_mid", cap=3.0, guard_frac="0.5", gates="I4", scale="0.03", reuse=RJ); g = gates_of(rep); ch = rep.get("child_report") or {}; pr = validate(rep); poses = ch.get("i4_poses", []); ip = (g.get("I4", {}).get("evidence") or {}).get("interrupted_pose") or {}
t("INTERRUPTION MID-POSE: PARTIAL, pose 1 preserved, the interrupted pose 2 is reported IN_PROGRESS with its completed physics steps and CPU so far", rep.get("status") == "PARTIAL" and g.get("I4", {}).get("result") == "PARTIAL" and [x["pose"] for x in poses] == [3000] and ip.get("pose") == 3001 and ip.get("status") == "IN_PROGRESS" and "completed_physics_steps" in ip and not pr, dict(status=rep.get("status"), poses=[x["pose"] for x in poses], interrupted=ip, problems=pr))
rep, w, r = run("i4_hardkill", "hard_spin", cap=3.0, gates="I4", scale="0", reuse=RJ); ch = rep.get("child_report") or {}; a = rep.get("accounting", {}); poses = ch.get("i4_poses", [])
t("HARD KILL DURING I4: the launcher kills the whole group (PARTIAL); the completed pose-1 record and the in-progress pose-2 record are preserved in the report; nothing left running; overshoot reported", rep.get("status") == "PARTIAL" and a.get("killed_by_launcher") and a.get("no_workload_process_left") and "cap_overshoot_s" in a and [x["pose"] for x in poses] == [3000] and (ch.get("i4_in_progress") or {}).get("pose") == 3001, dict(status=rep.get("status"), acc=a, poses=[x["pose"] for x in poses], in_progress=ch.get("i4_in_progress")))
rep, w, r = run("i4_two_poses", "two_poses", gates="I4", reuse=RJ); g = gates_of(rep)
t("COMPLETE REQUIRES ALL PLANNED I4 POSES: with 2 of the 3 planned poses the gate FAILs (STOPPED_ON_FAIL), never COMPLETE", rep.get("status") == "STOPPED_ON_FAIL" and g.get("I4", {}).get("result") == "FAIL", dict(status=rep.get("status"), gates={k: v["result"] for k, v in g.items()}))
tm2 = TMP / "tampered_manifest23.json"; m23 = json.loads(MAN23.read_text()); m23["harness_regions_sha256"]["I3"] = "0" * 64; tm2.write_text(json.dumps(m23)); rep, w, r = run("reuse23_refused", "normal", gates="I4", reuse=reuse_json(True, True, man23=tm2)); g = gates_of(rep)
t("I2/I3 REUSE REFUSED when the I3 gate-code region differs from the code that produced the saved evidence: BLOCKED, I4 not executed", rep.get("status") == "BLOCKED" and "I23 reuse is no longer justified" in (rep.get("stop_reason") or "") and "I4" not in g, dict(status=rep.get("status"), stop=rep.get("stop_reason")))
rep, w, r = run("i4_no_reuse", "normal", gates="I4", reuse="[]"); pr = validate(rep)
t("FINAL STATUS RULE: I4 alone without reusing verified I1-I3 evidence is PARTIAL (required gates without PASS evidence), never COMPLETE", rep.get("status") == "PARTIAL" and "I1" in (rep.get("stop_reason") or "") and not pr, dict(status=rep.get("status"), stop=rep.get("stop_reason"), problems=pr))
import hashlib
t("code hashes of the final tested files are recorded in this report", True, {f: hashlib.sha256((ROOT / "eval" / f).read_bytes()).hexdigest() for f in ("g2b_integration_v2.py", "g2b_integration_dryrun.py", "test_g2b_integration_dryrun.py", "test_g2b_trace_alignment.py", "g2b_intervention_benchmark.py", "g2b_labeller.py")})
ok_all = all(x["result"] == "PASS" for x in RES); rep_out = dict(verdict="PASS" if ok_all else "FAIL", n=len(RES), results=RES); (OUT / "integration_dryrun_report.json").write_text(json.dumps(rep_out, indent=1, default=str))
if not ok_all: (OUT / f"integration_dryrun_report_FAIL_{time.strftime('%H%M%S')}.json").write_text(json.dumps(rep_out, indent=1, default=str))
print(json.dumps(dict(verdict=rep_out["verdict"], n=len(RES))))
