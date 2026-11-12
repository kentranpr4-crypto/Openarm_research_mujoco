#!/usr/bin/env python3
"""Local driver, runs on the machine where the Colab CLI is installed and authenticated (the owner's workstation).  One command: verify -> (session) -> upload -> setup -> smoke -> benchmark -> decision -> [pilot B0, supervised] -> sync -> verify -> release.
Authorisation scope (from the owner): upload the P1 bundle + the validation dataset to the named Colab session; run the benchmark; run ONE B0 pilot, one seed, <= 2 h, ONLY if the benchmark conditions pass.  No commit, no push, no issue, no B1-B3, no final-goal (28,000,000+) evaluation.
Pilot conditions (all must hold): smoke OK; hashes equal the registered values; peak GPU memory (nvidia-smi, incl. CUDA context) <= 80% of the card's memory; projected iterations >= 4000 inside the 2 h wall-clock cap, counting checkpoints, trainer evaluations and the registered gate evaluations.  Otherwise it stops after the benchmark and writes the numbers.  The registered config / seed / batch sizes are never changed.
Usage (package directory = this script's directory's parent, see make_t4_package.py):
  python3 colab_t4_run.py --verify-only                 # local checks only, touches nothing remote
  python3 colab_t4_run.py                               # full run incl. pilot if the conditions pass   (--benchmark-only to stop after the decision)"""
import argparse, base64, hashlib, json, re, subprocess, sys, tarfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REMOTE = "/content/p1"; INPUTS = "/content/p1_inputs"; CFG_REMOTE = "research/supermimic_openarm/configs/pushing_p1_pilot.yaml"
FORBIDDEN = re.compile(r"(\.venv|/venv/|\.git/|__pycache__|\.pt$|\.pth$|\.npz$|\.ckpt$|\.log$|outputs/|openarm_wam|openarm_mujoco/|credential|oauth|secret|token|id_rsa|\.env$|\.pem$|\.key$)", re.I); ALLOWED_TOP = {"sim", "control", "research", "BUNDLE_MANIFEST.json"}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
STATE = {}
def log(*x): print(time.strftime("%H:%M:%S"), *x, flush=True)
class Colab:
    def __init__(self, exe, session): self.exe, self.s = exe, session
    def run(self, args, timeout=None, check=True):
        p = subprocess.run([self.exe, *args], capture_output=True, text=True, timeout=timeout)
        if check and p.returncode: raise RuntimeError(f"`colab {' '.join(args[:3])}...` failed (rc {p.returncode}): {(p.stdout + p.stderr)[-600:]}")
        return p
    def ssh(self, line, timeout=3600, check=True): return self.run(["ssh", "-s", self.s, "--", "bash", "-lc", line], timeout=timeout, check=check)
def verify(pkg, res):
    """local checks: registered full hashes, bundle integrity + content policy; returns (ok, info)"""
    exp = json.loads((pkg / "EXPECTED_HASHES.json").read_text()); bundle = pkg / "p1_bundle.tar.gz"; ds = pkg / "wm_eval_dataset_v1.npz"; info = dict(expected=exp); problems = []
    if not exp.get("gates_sha256") or not exp.get("protocol_sha256") or not exp.get("dataset_sha256"): problems.append("a registered reference hash is missing -> stop before training")
    with tarfile.open(bundle) as t:
        names = t.getnames(); bad = [n for n in names if FORBIDDEN.search(n) or n.split("/")[0] not in ALLOWED_TOP]; man = json.loads(t.extractfile("BUNDLE_MANIFEST.json").read()); mism = []
        for f, h in man["files"].items():
            if hashlib.sha256(t.extractfile(f).read()).hexdigest() != h: mism.append(f)
        got = {"gates_sha256": hashlib.sha256(t.extractfile("research/supermimic_openarm/gates/p1_pilot_gates.yaml").read()).hexdigest(), "protocol_sha256": hashlib.sha256(t.extractfile("research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml").read()).hexdigest()}
        (res / "bundle_listing.txt").write_text("\n".join(names) + "\n"); (res / "colab_p1_bootstrap.py").write_bytes(t.extractfile("research/supermimic_openarm/scripts/colab_p1_bootstrap.py").read())
    got["dataset_sha256"] = sha(ds); info.update(got=got, bundle_sha256=sha(bundle), bundle_files=len(names), bundle_git_commit=man["git_commit"], uncommitted_paths_in_bundle=len(man["dirty_paths_in_bundle"]), forbidden_or_unexpected_members=bad, manifest_mismatches=mism)
    for k in ("gates_sha256", "protocol_sha256", "dataset_sha256"):
        if got[k] != exp.get(k): problems.append(f"{k} differs from the registered value: {got[k]} != {exp.get(k)}")
    if bad: problems.append(f"bundle contains forbidden / unexpected members: {bad[:5]}")
    if mism: problems.append(f"bundle integrity failed for {mism[:3]}")
    info["problems"] = problems; return not problems, info
def decide(smoke, bench, budget_s, total_mib, memprobe=None):
    c = {}; c["smoke_ok"] = smoke.get("verdict") == "SMOKE_OK"; c["benchmark_ok"] = bench.get("verdict") == "BENCHMARK_OK"
    peak = max([x for x in (smoke.get("nvidia_smi_peak_used_mib_incl_context"), (bench.get("gpu") or {}).get("mem_used_mib_peak_nvidia_smi")) if x is not None] or [None]) if (smoke or bench) else None
    c["vram_headroom_ok"] = peak is not None and peak <= 0.80 * total_mib
    s2 = (smoke.get("steps") or {}); t_it = max([x for x in ((bench.get("training_step_latency") or {}).get("iteration_estimate_s"), (s2.get("2_training") or {}).get("seconds_per_iteration_steady")) if x] or [None]) if smoke or bench else None
    t_ck = (s2.get("3_checkpoint") or {}).get("save_s"); t_ev = (s2.get("4_rollout_eval") or {}).get("projected_trainer_eval_32_episodes_s"); t_g = (s2.get("5_gate_evaluation_cost") or {}).get("total_gate_evaluation_s"); start = max(0.0, ((s2.get("2_training") or {}).get("wall_s_incl_warmup") or 0) - 3 * (t_it or 0))
    N = None
    if None not in (t_it, t_ck, t_ev, t_g):
        N = 0
        while (N + 1) * t_it + ((N + 1) // 200) * t_ck + ((N + 1) // 400) * (t_ev + t_g) + start <= budget_s and N < 8000: N += 1
    c["host_ram_predicted_ok"] = bool(memprobe and memprobe.get("fits_with_15pct_headroom")) if memprobe is not None else True                              # the 2-worker probe predicts the registered 24-worker config fits in host RAM
    hm = bench.get("host_memory") or {}; c["host_ram_headroom_ok"] = hm.get("available_min_gib") is not None and hm["available_min_gib"] >= 1.5                    # real host memory (MemAvailable) never below 1.5 GiB while 24 env processes + training run
    c["projected_iterations_ge_4000"] = N is not None and N >= 4000
    return dict(conditions=c, go=all(c.values()), peak_vram_mib=peak, host_memory=hm, memprobe=memprobe, vram_total_mib=total_mib, vram_limit_mib=0.80 * total_mib, seconds_per_iteration=t_it, checkpoint_save_s=t_ck, trainer_eval_32_s=t_ev, gate_evaluation_s=t_g, startup_s=start, projected_iterations_in_budget=N, budget_s=budget_s,
                model="N*t_iter + floor(N/200)*t_ckpt + floor(N/400)*(t_trainer_eval + t_gate_eval) + startup <= budget; t_iter = max(benchmark estimate, smoke steady)")
def detached(C, name, cmd, pre, poll=20, timeout=5400):
    """run `cmd` on the VM detached (setsid + nohup) and poll a return-code file, so a dropped SSH connection cannot kill it; returns (rc, log tail)"""
    d = f"{REMOTE}/outputs/p1_t4"; script = f"mkdir -p {d}\ncd {REMOTE}\nexport PYTHONPATH={REMOTE}\n{('export P1_ALLOW_CPU_TEST=1' if pre else '')}\n{cmd}\necho $? > {d}/{name}.rc\n"
    C.ssh(f"mkdir -p {d}; rm -f {d}/{name}.rc; echo {base64.b64encode(script.encode()).decode()} | base64 -d > {d}/{name}.sh; ( setsid nohup bash {d}/{name}.sh > {d}/{name}.log 2>&1 < /dev/null & ); sleep 1", timeout=120); t0 = time.time(); fails = 0; rc = None
    while time.time() - t0 < timeout:
        time.sleep(poll)
        try: r = C.ssh(f"cat {d}/{name}.rc 2>/dev/null || true", timeout=120); fails = 0
        except Exception as e:
            fails += 1
            if fails >= 10: raise RuntimeError(f"lost the VM while waiting for {name}: {e}")
            continue
        if r.stdout.strip().lstrip("-").isdigit(): rc = int(r.stdout.strip()); break
    tail = C.ssh(f"tail -c 3000 {d}/{name}.log", timeout=120, check=False).stdout; return rc, tail
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--package-dir", default=str(HERE.parent)); ap.add_argument("--session", default="openarm-test"); ap.add_argument("--colab", default="colab"); ap.add_argument("--results-dir"); ap.add_argument("--verify-only", action="store_true"); ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--budget-s", type=float, default=7200.0); ap.add_argument("--poll-s", type=float, default=60.0); ap.add_argument("--sync-every-s", type=float, default=600.0); ap.add_argument("--no-release", action="store_true"); ap.add_argument("--gpu", default="T4"); ap.add_argument("--test-env", action="store_true", help="self-test mode: P1_ALLOW_CPU_TEST=1 on the remote side (CPU results are never GPU evidence)"); ap.add_argument("--test-force-go", action="store_true", help="self-test only (needs --test-env): take the pilot branch even though the GPU conditions are not met"); a = ap.parse_args(argv)
    pkg = Path(a.package_dir); res = Path(a.results_dir or pkg / f"results_{time.strftime('%Y%m%dT%H%M%S')}"); res.mkdir(parents=True, exist_ok=False); summary = dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), session=a.session, stage="verify"); STATE.update(summary=summary, res=res)
    def save(): (res / "run_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    ok, info = verify(pkg, res); summary["verification"] = info; save(); log("verification:", "OK" if ok else "FAILED", json.dumps(info["got"]), f"bundle files {info['bundle_files']}, git {info['bundle_git_commit'][:10]}, uncommitted paths in bundle {info['uncommitted_paths_in_bundle']}")
    if not ok: log("STOP before any remote action:", "; ".join(info["problems"])); summary["stage"] = "stopped_verification"; save(); return 3
    if a.verify_only: summary["stage"] = "verified_only"; save(); log("verify-only: done"); return 0
    C = Colab(a.colab, a.session); testenv = ["--env", "P1_ALLOW_CPU_TEST=1"] if a.test_env else []; pre = "P1_ALLOW_CPU_TEST=1 " if a.test_env else ""
    # ---- session
    summary["stage"] = "session"; sess = C.run(["sessions"], check=False)
    if sess.returncode: log("cannot list sessions (authentication?):", (sess.stdout + sess.stderr)[-300:]); summary["stage"] = "stopped_no_auth"; save(); return 4
    if a.session not in sess.stdout:
        log(f"session {a.session} not found -> creating it (colab new --gpu {a.gpu})"); C.run(["new", "-s", a.session, "--gpu", a.gpu], timeout=900)
    smi = C.ssh("nvidia-smi", timeout=120, check=False); summary["nvidia_smi_head"] = smi.stdout[:1200]; log("nvidia-smi:", " | ".join(l.strip() for l in smi.stdout.splitlines() if "MiB" in l or "Tesla" in l)[:300])
    if not a.test_env and "T4" not in smi.stdout: log("STOP: the session has no Tesla T4:", smi.stdout[:200]); summary["stage"] = "stopped_not_t4"; save(); return 5
    m = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", smi.stdout); total_mib = float(m.group(2)) if m else 15360.0
    # ---- upload + remote hash check
    summary["stage"] = "upload"; C.ssh(f"mkdir -p {INPUTS}", timeout=60); C.run(["upload", "-s", a.session, str(pkg / "p1_bundle.tar.gz"), "/content/p1_bundle.tar.gz"], timeout=1200); C.run(["upload", "-s", a.session, str(pkg / "wm_eval_dataset_v1.npz"), f"{INPUTS}/wm_eval_dataset_v1.npz"], timeout=1200)
    r = C.ssh(f"sha256sum /content/p1_bundle.tar.gz {INPUTS}/wm_eval_dataset_v1.npz").stdout; rh = {l.split()[1]: l.split()[0] for l in r.splitlines() if l.strip()}
    if rh.get("/content/p1_bundle.tar.gz") != info["bundle_sha256"] or rh.get(f"{INPUTS}/wm_eval_dataset_v1.npz") != info["got"]["dataset_sha256"]: log("STOP: uploaded file hashes differ", rh); summary["stage"] = "stopped_upload_hash"; save(); return 6
    log("uploaded and hash-verified on the VM")
    # ---- setup
    summary["stage"] = "setup"; C.run(["exec", "-s", a.session, "-f", str(res / "colab_p1_bootstrap.py"), "--timeout", "0", "--env", "P1_MODE=setup", "--env", f"P1_ROOT={REMOTE}", "--env", "P1_BUNDLE=/content/p1_bundle.tar.gz", *testenv], timeout=3600)
    r = C.ssh(f"cd {REMOTE} && sha256sum research/supermimic_openarm/gates/p1_pilot_gates.yaml research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml").stdout; rh = {l.split()[1].split("/")[-1]: l.split()[0] for l in r.splitlines() if l.strip()}
    if rh.get("p1_pilot_gates.yaml") != info["expected"]["gates_sha256"] or rh.get("p1_actor_ab_protocol.yaml") != info["expected"]["protocol_sha256"]: log("STOP: remote gate/protocol hash mismatch", rh); summary["stage"] = "stopped_remote_hash"; save(); return 6
    log("remote gate + protocol hashes equal the registered values")
    run = lambda cmd, t=3600: C.ssh(f"cd {REMOTE} && PYTHONPATH={REMOTE} {pre}{cmd}", timeout=t, check=False)
    # ---- VM resources + SAFE memory probe (2 workers) before anything starts 24 env processes
    summary["stage"] = "memprobe"; res_txt = C.ssh("free -m; nproc; df -h /content | tail -1; python3 -c 'import torch,sys;print(sys.version.split()[0], torch.__version__, torch.version.cuda)'", timeout=120, check=False).stdout; (res / "vm_resources.txt").write_text(res_txt); log("VM resources:", " | ".join(res_txt.splitlines()[:3]))
    rc, tail = detached(C, "memprobe", "python3 research/supermimic_openarm/scripts/colab_t4_memprobe.py --out outputs/p1_t4/memprobe.json", testenv, timeout=900); C.run(["download", "-s", a.session, f"{REMOTE}/outputs/p1_t4/memprobe.json", str(res / "memprobe.json")], check=False)
    memprobe = json.loads((res / "memprobe.json").read_text()) if (res / "memprobe.json").exists() else None; summary["memprobe"] = memprobe; ram_ok = bool(memprobe and memprobe["fits_with_15pct_headroom"])
    log("memory probe:", json.dumps({k: memprobe[k] for k in ("per_worker_pss_mib", "mem_available_before_gib", "predicted_host_gib_for_registered_config", "fits_with_15pct_headroom")}) if memprobe else f"FAILED rc={rc}: {tail[-300:]}")
    # ---- smoke
    summary["stage"] = "smoke"; small = "" if ram_ok else " --small-env"; log("smoke test (CUDA, real training steps, checkpoint round trip, rollout eval, one gate evaluation)" + (" -- DIAGNOSTIC small-env mode: the registered 24-env config is predicted NOT to fit in host RAM" if small else "") + " ...")
    rc, tail = detached(C, "smoke", f"python3 research/supermimic_openarm/scripts/colab_t4_smoke.py --dataset {INPUTS}/wm_eval_dataset_v1.npz --out outputs/p1_t4/smoke.json{small}", testenv, timeout=5400); (res / "smoke_stdout.txt").write_text(f"rc={rc}\n{tail}")
    C.run(["download", "-s", a.session, f"{REMOTE}/outputs/p1_t4/smoke.json", str(res / "smoke.json")], check=False); smoke = json.loads((res / "smoke.json").read_text()) if (res / "smoke.json").exists() else {}; summary["smoke"] = {k: smoke.get(k) for k in ("verdict", "all_steps_ok", "nvidia_smi_peak_used_mib_incl_context")}
    # ---- benchmark (explicit UUID; cap 12 GiB so an OOM is a clean NO-GO instead of an exhausted card)
    summary["stage"] = "benchmark"; r = run("python3 -m research.supermimic_openarm.p1_benchmark --list", 300); (res / "gpu_list.txt").write_text(r.stdout + r.stderr); r = run("python3 -m research.supermimic_openarm.p1_benchmark --pick-idle ''", 300); uuids = [l.split("\t")[0] for l in r.stdout.splitlines() if l.startswith("GPU-")]
    uuid = uuids[0] if len(uuids) == 1 else None; summary["gpu_uuid"] = uuid; bench = {}
    if not ram_ok: log("benchmark with 24 env processes SKIPPED: the memory probe predicts the registered config does not fit in host RAM (running it could OOM-kill the Colab runtime)")
    elif uuid or a.test_env:
        sel = f"--gpu-uuid {uuid}" if uuid else "--cpu-selftest"; rc, tail = detached(C, "benchmark", f"python3 -m research.supermimic_openarm.p1_benchmark {sel} --budget-s 300 --num-envs 24 --steps 480 --updates 5 --max-vram-gb 12 --min-spare-cores 0 --out outputs/p1_t4/benchmark.json", testenv, timeout=900); (res / "benchmark_stdout.txt").write_text(f"rc={rc}\n{tail}")
        C.run(["download", "-s", a.session, f"{REMOTE}/outputs/p1_t4/benchmark.json", str(res / "benchmark.json")], check=False); bench = json.loads((res / "benchmark.json").read_text()) if (res / "benchmark.json").exists() else {}
    else: log("no single idle GPU found on the VM:", uuids)
    dec = decide(smoke, bench, a.budget_s, total_mib, memprobe); dec["hashes_equal_registered"] = True; dec["benchmark_skipped_for_ram"] = not ram_ok; dec["go"] = bool(dec["go"] and (uuid is not None)); summary["decision"] = dec; (res / "decision.json").write_text(json.dumps(dec, indent=1, default=float)); save()
    log("DECISION", "GO" if dec["go"] else "NO-GO", json.dumps({k: dec[k] for k in ("conditions", "peak_vram_mib", "seconds_per_iteration", "projected_iterations_in_budget")}, default=float))
    if a.test_env and a.test_force_go: dec["go"] = True; summary["decision"]["forced_for_self_test"] = True
    if a.benchmark_only or not dec["go"]:
        summary["stage"] = "finished_at_benchmark"; summary["pilot_started"] = False; save(); log("stopped at the benchmark (pilot conditions not met or --benchmark-only); the numbers are in", res / "decision.json"); return release(C, a, res, summary, ok=True)
    # ---- pilot
    summary["stage"] = "pilot"; summary["pilot_started"] = True; save(); log(f"launching the B0 pilot (<= {a.budget_s:.0f} s wall clock, gate every 400 iterations)")
    C.ssh(f"cd {REMOTE} && ( PYTHONPATH={REMOTE} {pre}setsid nohup python3 research/supermimic_openarm/scripts/colab_t4_pilot.py --uuid '{uuid or ''}' --benchmark-json outputs/p1_t4/benchmark.json --dataset {INPUTS}/wm_eval_dataset_v1.npz --budget-s {a.budget_s} > outputs/p1_t4_pilot_launch.log 2>&1 < /dev/null & ) ; sleep 1", timeout=120)      # subshell: only the pilot is detached, the SSH call returns at once
    tgz = "/content/p1_sync.tgz"; paths = "outputs/p1_t4 outputs/p1_t4_pilot outputs/supermimic_openarm/p1_pilot_b0_t4"; last_sync = 0.0; last_gates = 0; fails = 0; stat = {}
    def sync(full=False):
        ex = "" if full else "--exclude='gates/ckpt_it*.pt'"; C.ssh(f"cd {REMOTE} && tar czf {tgz} {ex} {paths} 2>/dev/null; ls -l {tgz}", timeout=900); dst = res / f"sync_{time.strftime('%H%M%S')}{'_FINAL' if full else ''}.tgz"; C.run(["download", "-s", a.session, tgz, str(dst)], timeout=3600); return dst
    while True:
        time.sleep(a.poll_s)
        try:
            r = C.ssh(f"cat {REMOTE}/outputs/p1_t4_pilot/status.json", timeout=120); stat = json.loads(r.stdout); fails = 0
        except Exception as e:
            fails += 1; log(f"status poll failed ({fails}/10): {str(e)[:160]}")
            if fails >= 10: summary["stage"] = "lost_session"; save(); log("STOP monitoring: Colab session unreachable; last synced artifacts are in", res); return 7
            continue
        log(f"pilot: phase={stat.get('phase')} iteration={stat.get('iteration')} elapsed={stat.get('elapsed_s', 0):.0f}s gates={[(g['iteration'], g['exit_code']) for g in stat.get('gates_done', [])]} reason={stat.get('reason')}")
        if time.time() - last_sync > a.sync_every_s or len(stat.get("gates_done", [])) > last_gates:
            try: sync(); last_sync = time.time(); last_gates = len(stat.get("gates_done", []))
            except Exception as e: log("periodic sync failed:", str(e)[:200])
        if stat.get("finished"): break
    summary["pilot_status"] = stat; final = sync(full=True); summary["final_sync"] = str(final); summary["stage"] = "pilot_finished"; save(); return release(C, a, res, summary, ok=True, final=final)
def release(C, a, res, summary, ok, final=None):
    """verify the downloaded artifacts, then free the Colab session"""
    good = True; notes = []
    if final is not None:
        ex = res / "final_extract"; ex.mkdir(exist_ok=True)
        try:
            with tarfile.open(final) as t: t.extractall(ex, filter="data")
            rdir = ex / "outputs/supermimic_openarm/p1_pilot_b0_t4/seed1000"; need = [ex / "outputs/p1_t4_pilot/status.json", ex / "outputs/p1_t4_pilot/gate_log.jsonl", rdir / "metrics.jsonl", rdir / "checkpoints/manifest.json", rdir / "checkpoints/latest.pt"]
            miss = [str(p.relative_to(ex)) for p in need if not p.exists() and p.name != "gate_log.jsonl"]; notes += [f"missing: {miss}"] if miss else []
            man = json.loads((rdir / "checkpoints/manifest.json").read_text()) if (rdir / "checkpoints/manifest.json").exists() else {}
            for nm, mm in man.items():
                f = rdir / "checkpoints" / nm
                if f.exists() and sha(f) != mm["sha256"]: notes.append(f"checkpoint {nm} sha256 differs from the manifest"); 
            rows = [json.loads(l) for l in (rdir / "metrics.jsonl").read_text().splitlines() if l.strip()] if (rdir / "metrics.jsonl").exists() else []; summary["downloaded_iterations"] = rows[-1]["iteration"] if rows else None
            (res / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(res)}\n" for p in sorted(ex.rglob("*")) if p.is_file())); good = not notes
        except Exception as e: good = False; notes.append(f"{type(e).__name__}: {e}")
    if final is None:                                                                                                          # benchmark stage: the downloaded artifacts are the smoke and benchmark json files
        for nm in ("smoke.json", "benchmark.json", "decision.json", "memprobe.json"):
            if nm == "benchmark.json" and summary.get("decision", {}).get("benchmark_skipped_for_ram"): continue
            try: json.loads((res / nm).read_text())
            except Exception as e: good = False; notes.append(f"{nm}: {type(e).__name__}")
    summary["artifact_verification"] = dict(ok=good, notes=notes)
    if good and not a.no_release:
        try: C.run(["stop", "-s", a.session], timeout=300); summary["session_released"] = True; log("artifacts verified; Colab session released")
        except Exception as e: summary["session_released"] = False; log("release failed:", str(e)[:200])
    else: summary["session_released"] = False; log("session NOT released:", "artifact verification failed " + str(notes) if not good else "--no-release")
    (res / "run_summary.json").write_text(json.dumps(summary, indent=1, default=float)); log("results:", res); return 0 if good else 8
if __name__ == "__main__":
    try: sys.exit(main())
    except SystemExit: raise
    except Exception as e:                                                                                                      # never leave the owner without a record; the Colab session is NOT released on an error
        import traceback; msg = f"{type(e).__name__}: {e}"; print("ERROR:", msg, file=sys.stderr); traceback.print_exc()
        if STATE.get("summary") is not None: STATE["summary"].update(error=msg, session_released=False, note="driver error; the Colab session was NOT released; check the VM (outputs/p1_t4_pilot/status.json) before anything else"); (STATE["res"] / "run_summary.json").write_text(json.dumps(STATE["summary"], indent=1, default=float))
        sys.exit(9)
