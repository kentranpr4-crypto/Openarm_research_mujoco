#!/usr/bin/env python3
"""Runs ON the Colab VM (via `colab ssh -s NAME -- python3 ...`).  Smoke + cost measurement for the P1 pilot on the VM's GPU.  Starts NO pilot.  Writes ONE new json.
Checks and measures: (1) PyTorch sees CUDA and a real CUDA op runs; GPU UUID / name / PyTorch + CUDA versions / VRAM total + free; (2) a few real training iterations with the REGISTERED config (batch sizes, num_envs, horizon unchanged) in a throw-away run dir, with peak VRAM (torch allocator and nvidia-smi incl. the CUDA context) and
seconds per phase; (3) checkpoint save + reload round trip (equal weights) and its cost; (4) one MuJoCo rollout evaluation (cost per episode -> cost of the trainer's 32-goal evaluation); (5) the cost of ONE registered gate evaluation (p1_wm_eval + p1_actor_eval + p1_gate_check) on that checkpoint.
`P1_ALLOW_CPU_TEST=1` lets the control flow run on a machine without CUDA for testing; the result is then labelled NOT_GPU_EVIDENCE and can never satisfy the pilot conditions.
Usage on the VM (cwd = repo root of the unpacked bundle):  python3 research/supermimic_openarm/scripts/colab_t4_smoke.py --dataset /content/p1_inputs/wm_eval_dataset_v1.npz --out outputs/p1_t4/smoke.json [--iterations 3]"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path
CFG = "research/supermimic_openarm/configs/pushing_p1_pilot.yaml"
def smi(args):
    try: r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=20); return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired): return None
def run(cmd, env=None):
    t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, env=env); return p.returncode, time.time() - t, (p.stdout + p.stderr)[-1500:]
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True); ap.add_argument("--iterations", type=int, default=3); ap.add_argument("--run-name", default="p1_t4_smoke"); ap.add_argument("--small-env", action="store_true", help="DIAGNOSTIC ONLY: 2 sync envs instead of the registered 24 subprocess envs (used when the host RAM cannot hold the registered config); batch sizes, networks and horizon stay registered; the verdict can never satisfy the pilot conditions"); a = ap.parse_args(argv); out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    allow_cpu = os.environ.get("P1_ALLOW_CPU_TEST") == "1"; rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), starts_pilot=False, steps={}); ok = True
    import torch
    cuda = torch.cuda.is_available(); dev = "cuda:0" if cuda else "cpu"; env_info = dict(torch=torch.__version__, torch_cuda_build=torch.version.cuda, cuda_available=cuda, cpu_count=os.cpu_count())
    if cuda:
        p = torch.cuda.get_device_properties(0); free, total = torch.cuda.mem_get_info(); x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t = time.time(); y = x @ x; torch.cuda.synchronize(); mm = time.time() - t
        q = smi(["--query-gpu=uuid,name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]); env_info.update(gpu_name=p.name, compute_capability=f"{p.major}.{p.minor}", vram_total_mib=total / 2 ** 20, vram_free_mib=free / 2 ** 20, matmul_4096_s=mm, matmul_finite=bool(torch.isfinite(y).all()), nvidia_smi_query=q, gpu_uuid=(q or "").split(",")[0].strip() if q else None)
        ok &= bool(env_info["matmul_finite"]) and bool(env_info["gpu_uuid"])
    else: ok &= allow_cpu; env_info["note"] = "CUDA NOT available" + (" (allowed: P1_ALLOW_CPU_TEST)" if allow_cpu else " -> FAIL")
    rep["steps"]["1_cuda"] = dict(ok=bool(ok), **env_info)
    from research.supermimic_openarm.config import load_config
    from research.supermimic_openarm.trainer import Trainer
    from research.supermimic_openarm import runs
    overrides = ([] if cuda else ["system.device=cpu", "env.num_envs=2", "env.backend=sync"]) + (["env.num_envs=2", "env.backend=sync"] if a.small_env and cuda else []) + [x for x in os.environ.get("P1_TEST_OVERRIDES", "").split(",") if x]                                                                   # CPU test mode ONLY; on a GPU VM the registered config is used unchanged
    cfg = load_config(CFG, overrides); rd = runs.run_dir_for(cfg, a.run_name); stop = threading.Event(); peak = [0]
    def sampler():
        while not stop.is_set():
            r = smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]) if cuda else None
            if r: peak[0] = max(peak[0], int(r.split()[0]))
            time.sleep(0.5)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    try:
        runs.prepare_run_dir(cfg, rd, resume=False); tr = Trainer(cfg, rd, log=lambda *x: None); runs.write_run_files(cfg, rd, runs.build_env_manifest(cfg, tr.model_info), resume=False)
        if cuda: torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); res = tr.train(max_iterations=a.iterations); wall = time.time() - t0; rows = [json.loads(l) for l in (rd / "metrics.jsonl").read_text().splitlines() if l.strip()]
        per = {k: [r[k] for r in rows] for k in ("t_collect", "t_wm", "t_actor")}; steady = rows[1:] if len(rows) > 1 else rows
        rep["steps"]["2_training"] = dict(ok=bool(res["iteration"] == a.iterations and all(r.get("bad_grads", 0) == 0 for r in rows)), iterations=a.iterations, wall_s_incl_warmup=wall, per_iteration=per, seconds_per_iteration_steady=sum(r["t_collect"] + r["t_wm"] + r["t_actor"] for r in steady) / len(steady),
                                          torch_peak_allocated_mib=(torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None), torch_peak_reserved_mib=(torch.cuda.max_memory_reserved() / 2 ** 20 if cuda else None), batch=dict(num_envs=cfg.env.num_envs, wm_batch=cfg.world_model.batch_size, imag_batch=cfg.imagination.batch_size, horizon=cfg.imagination.horizon), run_dir=str(rd))
        ck = rd / "checkpoints" / "latest.pt"; t = time.time(); tr.save("smoke_extra.pt"); t_save = time.time() - t; t = time.time(); tr2 = Trainer(cfg, rd, resume=True, log=lambda *x: None); t_load = time.time() - t
        same = all(torch.equal(v.cpu(), tr2.actor.state_dict()[k].cpu()) for k, v in tr.actor.state_dict().items()) and all(torch.equal(v.cpu(), tr2.wm.state_dict()[k].cpu()) for k, v in tr.wm.state_dict().items())
        rep["steps"]["3_checkpoint"] = dict(ok=bool(same and tr2.iteration == a.iterations), save_s=t_save, load_s=t_load, bytes=ck.stat().st_size, weights_equal_after_reload=bool(same), iteration_after_reload=tr2.iteration)
        tr2._make_vec(); t = time.time(); r_eval, _ = tr2.evaluate(n_episodes=4, tag="smoke"); t_eval4 = time.time() - t; tr2.vec.close()
        rep["steps"]["4_rollout_eval"] = dict(ok=bool(r_eval["success_rate"] >= 0.0), episodes=4, seconds_for_4=t_eval4, seconds_per_episode=t_eval4 / 4, projected_trainer_eval_32_episodes_s=8 * t_eval4)
    except Exception as e:
        rep["steps"].setdefault("exception", dict(ok=False, error=f"{type(e).__name__}: {e}")); ok = False
    stop.set(); th.join(timeout=2); rep["nvidia_smi_peak_used_mib_incl_context"] = peak[0] if cuda else None
    # ---- (5) one registered gate evaluation on the smoke checkpoint
    if "exception" not in rep["steps"]:
        ckpt = str(rd / "checkpoints" / "latest.pt"); g = Path(a.out).parent / "smoke_gate"; g.mkdir(parents=True, exist_ok=True); envc = dict(os.environ, PYTHONPATH=str(Path.cwd()), CUDA_VISIBLE_DEVICES="")
        py = [sys.executable, "-m"]
        rc1, t1, o1 = run(py + ["research.supermimic_openarm.p1_wm_eval", "--out", str(g / "wm.json"), "--checkpoint", ckpt, "--dataset", a.dataset, "--actor-episodes", "32", "--grad-states", "24", "--opt-states", "0", "--fd-states", "0"], envc)
        rc2, t2, o2 = run([sys.executable, "-m", "research.supermimic_openarm.p1_actor_eval", "--checkpoint", ckpt, "--episodes", "64", "--out", str(g / "actor.json")], envc)
        rc3, t3, o3 = run([sys.executable, "-m", "research.supermimic_openarm.p1_gate_check", "--wm-eval", str(g / "wm.json"), "--actor-eval", str(g / "actor.json"), "--iteration", str(a.iterations), "--budget-iterations", "8000"], envc)
        rep["steps"]["5_gate_evaluation_cost"] = dict(ok=bool(rc1 == 0 and rc2 == 0 and rc3 in (0, 1, 2)), wm_eval_s=t1, actor_eval_64_episodes_s=t2, gate_check_s=t3, gate_check_exit=rc3, gate_check_tail=o3[-600:], total_gate_evaluation_s=t1 + t2 + t3, wm_eval_tail=("" if rc1 == 0 else o1), actor_eval_tail=("" if rc2 == 0 else o2),
                                                       note="registered gate command; --opt-states 0 --fd-states 0 only skips the exploitation / finite-difference diagnostics that no gate uses")
    all_ok = all(v.get("ok") for v in rep["steps"].values()) and ok; rep["all_steps_ok"] = bool(all_ok); rep["small_env_diagnostic"] = bool(a.small_env); rep["verdict"] = ("NOT_GPU_EVIDENCE" if not cuda else (("SMOKE_DIAGNOSTIC_SMALL_ENV" if a.small_env else "SMOKE_OK") if all_ok else "SMOKE_FAILED"))
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, indent=1, default=float)); print(json.dumps({k: rep[k] for k in ("verdict", "all_steps_ok", "nvidia_smi_peak_used_mib_incl_context")}, default=float)); print("written", out); return 0 if all_ok else 1
if __name__ == "__main__": sys.exit(main())
