#!/usr/bin/env python3
"""Runs ON the Colab VM, detached (nohup/setsid).  Supervises ONE B0 pilot (variant b0 == the original train.py), 1 seed, with the REGISTERED config and NO overrides.
Total wall-clock budget (training + gate pauses) is a hard cap (default 7200 s).  At every 400 iterations the supervisor PAUSES the training process group (SIGSTOP), copies the saved `latest.pt` of that iteration (sha256 checked against the checkpoint manifest), runs the registered gate evaluation on CPU
(p1_wm_eval --actor-episodes 32 --grad-states 24, p1_actor_eval --episodes 64, p1_gate_check) and resumes training only if the gate check exits 0 (CONTINUE).  STOP (1), INCOMPLETE/INCONSISTENT (2), an evaluation crash, a training crash or the deadline end the run.  After the run ends a final gate evaluation (with the full
exploitation diagnostics) is run on the last saved checkpoint; its cost is NOT counted in the training budget.  Final held-out goals (28,000,000+) are never touched.  B1-B3 are never started.
Status for the remote monitor: <out-dir>/status.json (rewritten every loop); history: <out-dir>/gate_log.jsonl.
Usage on the VM:  python3 research/supermimic_openarm/scripts/colab_t4_pilot.py --uuid GPU-... --benchmark-json outputs/p1_t4/benchmark.json --dataset /content/p1_inputs/wm_eval_dataset_v1.npz [--budget-s 7200]"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
CFG = "research/supermimic_openarm/configs/pushing_p1_pilot.yaml"; GATE_EVERY = 400; BUDGET_ITERATIONS = 8000
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def tail_row(mp):
    try: lines = [l for l in Path(mp).read_text().splitlines() if l.strip()]; return json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError): return None
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--uuid", default=""); ap.add_argument("--benchmark-json", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--budget-s", type=float, default=7200.0); ap.add_argument("--run-name", default="p1_pilot_b0_t4"); ap.add_argument("--out-dir", default="outputs/p1_t4_pilot"); a = ap.parse_args(argv)
    test = os.environ.get("P1_ALLOW_CPU_TEST") == "1"; out = Path(a.out_dir); py = sys.executable
    if out.exists(): sys.exit(f"refusing to reuse {out}")
    from research.supermimic_openarm.config import load_config
    from research.supermimic_openarm import runs
    cfg = load_config(CFG, []); rd = runs.run_dir_for(cfg, a.run_name)
    if rd.exists(): sys.exit(f"refusing to reuse run directory {rd}")
    out.mkdir(parents=True); (out / "gates").mkdir(); st = dict(phase="starting", iteration=0, elapsed_s=0.0, budget_s=a.budget_s, gates_done=[], reason=None, finished=False, test_mode=test, run_dir=str(rd), config_sha256=sha(CFG)); logf = out / "gate_log.jsonl"
    def write(): (out / "status.json").write_text(json.dumps(st, indent=1, default=float))
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()), MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    if a.uuid and not test: env["CUDA_VISIBLE_DEVICES"] = a.uuid
    cmd = os.environ["P1_PILOT_TRAIN_CMD"].split() if test and os.environ.get("P1_PILOT_TRAIN_CMD") else [py, "-m", "research.supermimic_openarm.p1_train_variant", "--variant", "b0", "--config", CFG, "--run-name", a.run_name]
    t0 = time.time(); trainlog = open(out / "train.log", "wb"); proc = subprocess.Popen(cmd, env=env, stdout=trainlog, stderr=subprocess.STDOUT, preexec_fn=os.setsid); pgid = os.getpgid(proc.pid); st.update(phase="training", pid=proc.pid, train_cmd=cmd); write(); last_gate = 0; stopped = [False]
    def kill_training(reason):
        st["reason"] = reason; stopped[0] = True
        for sig in (signal.SIGCONT, signal.SIGTERM):
            try: os.killpg(pgid, sig)
            except ProcessLookupError: break
        for _ in range(60):
            if proc.poll() is not None: break
            time.sleep(1)
        if proc.poll() is None:
            try: os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError: pass
        proc.wait()
    signal.signal(signal.SIGTERM, lambda *x: (kill_training("supervisor_terminated"), sys.exit(0)))
    def gate(it, final=False):
        """evaluate the saved checkpoint of iteration `it`; returns (exit_code, record)"""
        t = time.time(); gdir = out / "gates"; man = json.loads((rd / "checkpoints" / "manifest.json").read_text()).get("latest.pt", {}); rec = dict(iteration=it, final=final)
        if man.get("iteration") != it: rec["error"] = f"manifest iteration {man.get('iteration')} != {it}"; return 2, rec
        copy = gdir / f"ckpt_it{it:06d}.pt"; (copy.write_bytes((rd / "checkpoints" / "latest.pt").read_bytes()))
        if sha(copy) != man.get("sha256"): rec["error"] = "checkpoint copy sha256 differs from the manifest (INCONSISTENT)"; return 2, rec
        rec["checkpoint_sha256"] = man["sha256"]; genv = dict(env, CUDA_VISIBLE_DEVICES=""); wm = gdir / f"wm_it{it:06d}.json"; ac = gdir / f"actor_it{it:06d}.json"; gj = gdir / f"gate_it{it:06d}.json"
        extra = ["--opt-states", "24", "--fd-states", "24"] if final else ["--opt-states", "0", "--fd-states", "0"]
        r1 = subprocess.run([py, "-m", "research.supermimic_openarm.p1_wm_eval", "--out", str(wm), "--checkpoint", str(copy), "--dataset", a.dataset, "--actor-episodes", "32", "--grad-states", "24", *extra], env=genv, capture_output=True, text=True); rec["wm_eval_s"] = time.time() - t; rec["wm_eval_rc"] = r1.returncode
        if r1.returncode: rec["error"] = "p1_wm_eval crashed: " + r1.stdout[-400:] + r1.stderr[-600:]; return 2, rec
        t2 = time.time(); r2 = subprocess.run([py, "-m", "research.supermimic_openarm.p1_actor_eval", "--checkpoint", str(copy), "--episodes", "64", "--out", str(ac)], env=genv, capture_output=True, text=True); rec["actor_eval_s"] = time.time() - t2; rec["actor_eval_rc"] = r2.returncode
        if r2.returncode: rec["error"] = "p1_actor_eval crashed: " + r2.stdout[-400:] + r2.stderr[-600:]; return 2, rec
        wms = sorted(str(p) for p in gdir.glob("wm_it*.json"))
        r3 = subprocess.run([py, "-m", "research.supermimic_openarm.p1_gate_check", "--wm-eval", *wms, "--actor-eval", str(ac), "--benchmark", a.benchmark_json, "--iteration", str(it), "--budget-iterations", str(BUDGET_ITERATIONS), "--out", str(gj)], env=genv, capture_output=True, text=True)
        code = r3.returncode
        if code == 1 and not gj.exists(): code = 2; rec["error"] = "p1_gate_check exited 1 without writing its result file: treated as a crash (INCOMPLETE), not as a STOP decision"          # a Python traceback also exits 1
        rec.update(gate_check_rc=r3.returncode, gate_check_output=r3.stdout[-2500:], gate_check_stderr=r3.stderr[-600:], total_s=time.time() - t); return code, rec
    while True:
        time.sleep(float(os.environ.get("P1_PILOT_POLL_S", "15"))); el = time.time() - t0; row = tail_row(rd / "metrics.jsonl"); it = int(row["iteration"]) if row else 0; st.update(elapsed_s=el, iteration=it, last_row=row)
        rc = proc.poll()
        if rc is not None:
            st["reason"] = "completed_all_iterations" if (rc == 0 and it >= cfg.train.iterations) else f"training_process_exited_rc{rc}"; break
        if el >= a.budget_s: kill_training("time_cap_reached"); break
        man = {}
        try: man = json.loads((rd / "checkpoints" / "manifest.json").read_text()).get("latest.pt", {})
        except (OSError, json.JSONDecodeError): pass
        mit = man.get("iteration") or 0
        if mit and mit % GATE_EVERY == 0 and mit > last_gate:
            st["phase"] = "gate"; write(); os.killpg(pgid, signal.SIGSTOP)
            try: code, rec = gate(mit)
            except Exception as e: code, rec = 2, dict(iteration=mit, error=f"{type(e).__name__}: {e}")
            last_gate = mit; rec["elapsed_at_check_s"] = el; st["gates_done"].append(dict(iteration=mit, exit_code=code)); logf.open("a").write(json.dumps(rec) + "\n")
            if code != 0: kill_training({1: "gate_STOP", 2: "gate_INCOMPLETE_or_INCONSISTENT"}.get(code, f"gate_exit_{code}") + f"_at_{mit}"); break
            os.killpg(pgid, signal.SIGCONT); st["phase"] = "training"
        write()
    st["phase"] = "final_evaluation"; st["training_wall_s"] = time.time() - t0; write()
    try:
        man = json.loads((rd / "checkpoints" / "manifest.json").read_text()).get("latest.pt", {}); fit = man.get("iteration") or 0
        if fit and fit not in [g["iteration"] for g in st["gates_done"]]:                                                                      # the last saved checkpoint has not been gated yet
            code, rec = gate(fit, final=True); rec["final_after_stop"] = True; st["final_gate_exit"] = code; logf.open("a").write(json.dumps(rec) + "\n")
    except Exception as e: st["final_evaluation_error"] = f"{type(e).__name__}: {e}"
    st.update(phase="finished", finished=True, final_iteration=(tail_row(rd / "metrics.jsonl") or {}).get("iteration"), total_s=time.time() - t0); write(); print(json.dumps({k: st[k] for k in ("reason", "final_iteration", "training_wall_s", "gates_done")}, default=float)); return 0
if __name__ == "__main__": sys.exit(main())
