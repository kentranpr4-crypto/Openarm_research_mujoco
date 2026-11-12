"""Bounded, reproducible P1 BENCHMARK on ONE explicitly chosen GPU (selected by UUID).  It MEASURES; it never trains, never writes a checkpoint, never launches the pilot, never kills or modifies another process.
What it measures: (a) simulator-only stepping speed (1 env, scripted pusher, CPU), (b) vectorised env transitions/s INCLUDING policy inference, (c) world-model update latency and actor/critic update latency (the training step), (d) peak VRAM (torch allocator AND nvidia-smi incl. CUDA context),
(e) GPU utilisation (nvidia-smi, 1 Hz), (f) CPU consumption (cores busy, from /proc ticks of the benchmark process tree) and peak RSS, (g) total wall-clock.
Safety: read-only preflight (REFUSES if the GPU already runs another compute process or is busy, unless --allow-shared, which also forces a 4 GiB VRAM cap), hard wall-clock budget (SIGTERM, then SIGKILL of ONLY its own process group), `nice`, per-process VRAM cap, output written to a NEW json.
Usage (repo root):
  python -m research.supermimic_openarm.p1_benchmark --list                        # topology + workloads, nothing else
  python -m research.supermimic_openarm.p1_benchmark --pick-idle "RTX 5090"        # prints the UUID(s) of IDLE GPUs whose name contains the text (exit 3 if none); starts nothing
  python -m research.supermimic_openarm.p1_benchmark --gpu-uuid GPU-xxxxxxxx-... [--budget-s 300 --num-envs 8 --steps 800 --updates 5 --max-vram-gb 8] [--out outputs/p1_validation/benchmark_<name>.json]
  python -m research.supermimic_openarm.p1_benchmark --cpu-selftest              # exercises the same measuring code on CPU (no GPU needed); results are NOT GPU evidence"""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path
CFG = "research/supermimic_openarm/configs/pushing_p1_pilot.yaml"
def smi(args, timeout=20):
    try: r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout); return r.stdout if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired): return None
def topology():
    """read-only snapshot: GPUs (index, uuid, name, memory, utilisation), compute processes per GPU (pid, command line), PCIe/NVLink topology matrix, CPU count"""
    out = dict(nvidia_smi_available=smi(["-L"]) is not None, cpu_count=os.cpu_count(), loadavg1=os.getloadavg()[0], gpus=[], processes=[], topo_matrix=None)
    g = smi(["--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pci.bus_id", "--format=csv,noheader,nounits"])
    if g:
        for line in g.strip().splitlines():
            i, u, n, mt, mu, ut, bus = [x.strip() for x in line.split(",")]; out["gpus"].append(dict(index=int(i), uuid=u, name=n, mem_total_mib=int(mt), mem_used_mib=int(mu), util_pct=int(ut), pci_bus=bus))
        p = smi(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        for line in (p or "").strip().splitlines():
            u, pid, nm, um = [x.strip() for x in line.split(",")]
            try: cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")[:160]
            except OSError: cmd = "?"
            out["processes"].append(dict(gpu_uuid=u, pid=int(pid), name=nm, used_mib=int(um), cmdline=cmd))
        out["topo_matrix"] = smi(["topo", "-m"])
    return out
def print_topology(t):
    if not t["nvidia_smi_available"]: print("nvidia-smi not available: no NVIDIA driver visible to this shell (nothing can be selected or benchmarked on a GPU here)"); return
    for g in t["gpus"]:
        procs = [p for p in t["processes"] if p["gpu_uuid"] == g["uuid"]]
        print(f"GPU {g['index']}  {g['uuid']}  {g['name']}  {g['mem_used_mib']}/{g['mem_total_mib']} MiB  util {g['util_pct']}%  pci {g['pci_bus']}  compute processes: {len(procs)}")
        for p in procs: print(f"    pid {p['pid']}  {p['used_mib']} MiB  {p['cmdline'] or p['name']}")
    print(t["topo_matrix"] or "")
def meminfo():
    """(MemTotal, MemAvailable) in GiB from /proc/meminfo (the REAL host memory, unlike the per-process RSS sum which double-counts shared pages)"""
    d = {}
    try:
        for l in open("/proc/meminfo"):
            k, v = l.split(":"); d[k] = int(v.split()[0]) / 2 ** 20
    except OSError: return None, None
    return d.get("MemTotal"), d.get("MemAvailable")
def tree_pids(root):
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try: kids.setdefault(int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1]), []).append(int(d))
            except (OSError, IndexError, ValueError): pass
    out, stack = [], [root]
    while stack: p = stack.pop(); out.append(p); stack += kids.get(p, [])
    return out
def proc_ticks_rss(pid):
    try: f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split(); return int(f[11]) + int(f[12]), int(f[21]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError): return 0, 0
def provenance(gpu):
    """what produced this file: code version, inputs, library and driver versions (no host name, no user paths)"""
    import hashlib, platform
    def sha(p):
        try: return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except OSError: return None
    def git(*a):
        try: return subprocess.run(["git", *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception: return None
    pv = dict(python=sys.version.split()[0], platform=platform.platform(), git_commit=git("rev-parse", "HEAD"), git_dirty_files=len([l for l in (git("status", "--porcelain") or "").splitlines() if "openarm_wam" not in l]), sha256=dict(config=sha(CFG), gates=sha("research/supermimic_openarm/gates/p1_pilot_gates.yaml"), benchmark_tool=sha(__file__), trainer=sha("research/supermimic_openarm/trainer.py")))
    for mod in ("torch", "mujoco", "numpy"):
        try: m = __import__(mod); pv[mod] = m.__version__
        except Exception: pv[mod] = None
    try: import torch; pv["torch_cuda_build"] = torch.version.cuda
    except Exception: pass
    if gpu:
        d = smi(["--query-gpu=driver_version,name,uuid,compute_cap", "--format=csv,noheader", "-i", gpu["uuid"]]); pv["gpu_query"] = d.strip() if d else None
    return pv
def sim_only_speed(n=300):
    """one env, scripted pusher actions, no policy network: physics + adapter + state packing only"""
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    from .baselines.push_expert import PushExpert, MAX_STEP
    env = OpenArmTaskEnv("pushing", None, 150, 20); ex = PushExpert(); env.reset(26_000_000); k = 0; t = time.perf_counter(); done = False
    while k < n:
        if done: env.reset(26_000_000 + k); done = False
        o = env.step(env.state_vec()[L.PREV] + ex.act(env) * MAX_STEP); done = o["terminated"] or o["truncated"]; k += 1
    dt = time.perf_counter() - t; return dict(control_steps=k, seconds=dt, control_steps_per_s=k / dt, physics_steps_per_s=k * env.substeps / dt, ms_per_control_step=1e3 * dt / k, substeps_per_control_step=env.substeps)
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--gpu-uuid"); ap.add_argument("--cpu-selftest", action="store_true"); ap.add_argument("--budget-s", type=int, default=300); ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800); ap.add_argument("--updates", type=int, default=5); ap.add_argument("--max-vram-gb", type=float, default=8.0); ap.add_argument("--allow-shared", action="store_true"); ap.add_argument("--out"); ap.add_argument("--pick-idle", metavar="NAME", help="print the UUID(s) of idle GPUs whose name contains NAME (case-insensitive; empty = any) and exit; starts nothing"); ap.add_argument("--min-spare-cores", type=float, default=None, help="default num_envs + 1; lower it ONLY on a small VM")
    a = ap.parse_args(argv)
    topo = topology()
    if a.list: print_topology(topo); return 0
    if a.pick_idle is not None:
        ok = [g for g in topo["gpus"] if a.pick_idle.lower() in g["name"].lower() and not [p for p in topo["processes"] if p["gpu_uuid"] == g["uuid"]] and g["util_pct"] <= 30 and (g["mem_total_mib"] - g["mem_used_mib"]) / 1024 >= a.max_vram_gb + 1]
        if not ok: print(f"NO_IDLE_GPU: no GPU matching {a.pick_idle!r} is idle (no compute process, util <= 30%, >= {a.max_vram_gb + 1} GiB free); nvidia-smi available: {topo['nvidia_smi_available']}", file=sys.stderr); return 3
        for g in ok: print(g["uuid"], g["name"], f"{g['mem_total_mib']} MiB", sep="\t")
        return 0
    if not a.gpu_uuid and not a.cpu_selftest: print("ERROR: --gpu-uuid is required (see --list). No device is chosen implicitly.", file=sys.stderr); return 2
    if a.budget_s > 900: print("ERROR: --budget-s is capped at 900 s for a benchmark", file=sys.stderr); return 2
    rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), mode="cpu_selftest" if a.cpu_selftest else "gpu", topology_before=topo, budget_s=a.budget_s, requested=dict(num_envs=a.num_envs, steps=a.steps, updates=a.updates, max_vram_gb=a.max_vram_gb), starts_training=False)
    env = dict(os.environ); overrides = [f"env.num_envs={a.num_envs}"]; vram_cap = a.max_vram_gb
    if a.cpu_selftest: overrides += ["system.device=cpu", "imagination.batch_size=256", "imagination.chunk_size=128", "world_model.batch_size=256", "world_model.hidden=128", "actor.hidden=128", "critic.hidden=64"]; gpu = None
    else:
        gpu = next((g for g in topo["gpus"] if g["uuid"] == a.gpu_uuid), None)
        if gpu is None: print(f"RESOURCE_BLOCKED: GPU UUID {a.gpu_uuid} not found (nvidia-smi available: {topo['nvidia_smi_available']}). Nothing was started.", file=sys.stderr); return 3
        others = [p for p in topo["processes"] if p["gpu_uuid"] == gpu["uuid"]]; free_gib = (gpu["mem_total_mib"] - gpu["mem_used_mib"]) / 1024; blockers = []
        if others and not a.allow_shared: blockers.append(f"{len(others)} other compute process(es) on this GPU (e.g. {others[0]['cmdline'][:60]!r}); use a different UUID, or --allow-shared (4 GiB cap, same budget)")
        if gpu["util_pct"] > 30 and not a.allow_shared: blockers.append(f"utilisation {gpu['util_pct']}% > 30%")
        if free_gib < vram_cap + 1: blockers.append(f"free VRAM {free_gib:.1f} GiB < cap {vram_cap} GiB + 1")
        need = a.min_spare_cores if a.min_spare_cores is not None else a.num_envs + 1
        if (os.cpu_count() or 1) - os.getloadavg()[0] < need: blockers.append(f"spare CPU cores {(os.cpu_count() or 1) - os.getloadavg()[0]:.1f} < required {need}")
        if blockers: rep["blockers"] = blockers; print("RESOURCE_BLOCKED:", "; ".join(blockers), file=sys.stderr); return 3
        if a.allow_shared: vram_cap = min(vram_cap, 4.0)
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]; overrides += ["system.device=cuda:0", f"system.max_vram_gb={vram_cap}"]
    out = Path(a.out or f"outputs/p1_validation/benchmark_{'cpu_selftest' if a.cpu_selftest else gpu['index']}_{time.strftime('%Y%m%dT%H%M%S')}.json")
    if out.exists(): print(f"ERROR: {out} exists (refusing to overwrite)", file=sys.stderr); return 2
    print("simulator-only stepping speed ...", flush=True); rep["sim_only"] = sim_only_speed(); print(json.dumps(rep["sim_only"]), flush=True)
    cmd = ["nice", "-n", "10", sys.executable, "-m", "research.supermimic_openarm.benchmark", "--config", CFG, "--steps", str(a.steps), "--updates", str(a.updates)] + [x for o in overrides for x in ("--override", o)]
    t0 = time.time(); p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid); samples = []; last = {}; peak_rss = 0; mem_total, mem_avail0 = meminfo(); mem_min = mem_avail0; cores = []; timed_out = False; hz = os.sysconf("SC_CLK_TCK"); lines = []
    import threading
    def reader():
        for ln in p.stdout: lines.append(ln)
    th = threading.Thread(target=reader, daemon=True); th.start()
    while p.poll() is None:
        now = time.time()
        if now - t0 > a.budget_s: timed_out = True; os.killpg(p.pid, signal.SIGTERM); time.sleep(5); (p.poll() is None) and os.killpg(p.pid, signal.SIGKILL); break          # only OUR process group
        pids = tree_pids(p.pid); ticks = 0; rss = 0
        for q in pids: tk, rs = proc_ticks_rss(q); ticks += tk; rss += rs
        _, ma = meminfo(); mem_min = ma if (mem_min is None or (ma is not None and ma < mem_min)) else mem_min
        if last: cores.append((ticks - last["ticks"]) / hz / max(now - last["t"], 1e-6))
        last = dict(ticks=ticks, t=now); peak_rss = max(peak_rss, rss)
        if gpu:
            r = smi(["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", gpu["uuid"]])
            if r: u, m = [int(x) for x in r.strip().split(",")]; samples.append((u, m))
        time.sleep(1.0)
    p.wait(); th.join(timeout=5); text = "".join(lines); rep["wall_clock_s"] = time.time() - t0; rep["timed_out_budget"] = timed_out; rep["exit_code"] = p.returncode
    m = re.search(r"written:\s*(\S+\.json)", text); bench = json.load(open(m.group(1))) if m and Path(m.group(1)).exists() else None
    if bench: rep["benchmark"] = {k: v for k, v in bench.items() if k != "metadata"}; rep["training_step_latency"] = dict(world_model_update_s=bench["world_model_update_s"], actor_critic_update_s=bench["actor_critic_update_s"], collect_s_per_iteration_at_4_steps_per_iter=4 * a.num_envs / max(bench["env_transitions_per_s"], 1e-9), iteration_estimate_s=bench["world_model_update_s"] * 1 + bench["actor_critic_update_s"] + 4 * a.num_envs / max(bench["env_transitions_per_s"], 1e-9))
    rep["host_memory"] = dict(total_gib=mem_total, available_at_start_gib=mem_avail0, available_min_gib=mem_min)
    rep["cpu"] = dict(mean_cores_busy=sum(cores) / max(len(cores), 1), peak_cores_busy=max(cores or [0]), peak_rss_gib=peak_rss / 2 ** 30, samples=len(cores))
    if gpu and samples: rep["gpu"] = dict(uuid=gpu["uuid"], name=gpu["name"], util_mean_pct=sum(s[0] for s in samples) / len(samples), util_max_pct=max(s[0] for s in samples), mem_used_mib_peak_nvidia_smi=max(s[1] for s in samples), mem_used_mib_before=gpu["mem_used_mib"], samples=len(samples), vram_cap_gb=vram_cap)
    rep["provenance"] = provenance(gpu); rep["topology_after"] = topology(); rep["log_tail"] = text[-1500:]; rep["verdict"] = "NOT_GPU_EVIDENCE_SELFTEST" if a.cpu_selftest else ("BUDGET_EXCEEDED" if timed_out else ("BENCHMARK_OK" if p.returncode == 0 else f"FAILED_EXIT_{p.returncode}"))
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, indent=1, default=float)); print(json.dumps({k: rep.get(k) for k in ("verdict", "wall_clock_s", "sim_only", "training_step_latency", "cpu", "gpu")}, indent=1, default=float)); print("written:", out); return 0 if rep["verdict"] in ("BENCHMARK_OK", "NOT_GPU_EVIDENCE_SELFTEST") else 1
if __name__ == "__main__": sys.exit(main())
