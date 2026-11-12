#!/usr/bin/env python3
"""Runs ON the VM.  SAFE host-memory probe (2 env workers) BEFORE anything starts the registered 24-worker configuration: measures the real (PSS) memory of one env worker and of the parent, reads MemTotal/MemAvailable, and predicts the host memory the registered pilot config needs.
Why: the env backend is `subproc` with the `spawn` start method, so every worker is a fresh interpreter that imports torch; workers do not share memory.  Measured in the development sandbox: ~1.7 GiB per worker -> ~40 GiB for 24 workers, which would OOM-kill a 12-15 GB VM (and could take the Colab runtime down with it).
Writes ONE new json; starts no training.  Usage:  python3 research/supermimic_openarm/scripts/colab_t4_memprobe.py --out outputs/p1_t4/memprobe.json"""
import argparse, json, os, sys, time
from pathlib import Path
CFG = "research/supermimic_openarm/configs/pushing_p1_pilot.yaml"
def pss(pid):
    try: return sum(int(l.split()[1]) for l in open(f"/proc/{pid}/smaps_rollup") if l.startswith("Pss:")) / 1024.0
    except OSError: return 0.0
def mem():
    d = {}
    for l in open("/proc/meminfo"): k, v = l.split(":"); d[k] = int(v.split()[0]) / 2 ** 20
    return d["MemTotal"], d["MemAvailable"]
def children(pid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1]) == pid: out.append(int(d))
            except (OSError, IndexError, ValueError): pass
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args(); out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    from research.supermimic_openarm.config import load_config
    from research.supermimic_openarm.envs.vector import make_vec_env
    cfg = load_config(CFG, []); total, avail0 = mem(); import torch
    if torch.cuda.is_available(): torch.cuda.init(); torch.zeros(1, device="cuda")
    parent = pss(os.getpid()); vec = make_vec_env("subproc", a.workers, cfg.env.task, cfg.env.task_params, 150, cfg.env.control_hz, 1); vec.reset()
    for _ in range(5): vec.step(vec.reset()[0][:, 16:24].astype("float64"))
    time.sleep(2); kids = [pss(k) for k in children(os.getpid())]; workers = sorted(v for v in kids if v > 100)
    if not workers: vec.close(); sys.exit("could not measure the env workers")
    per = sum(workers) / len(workers); _, avail1 = mem(); vec.close(); n = cfg.env.num_envs
    replay_gib = n * cfg.replay.capacity_per_env * 100 * 4 / 2 ** 30; predicted = (n * per + parent) / 1024 + replay_gib + 0.5
    rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), registered_num_envs=n, backend=cfg.env.backend, probe_workers=a.workers, per_worker_pss_mib=per, workers_pss_mib=workers, parent_pss_mib_with_cuda_context=parent, mem_total_gib=total, mem_available_before_gib=avail0, mem_available_with_probe_gib=avail1,
               replay_buffer_gib_estimate=replay_gib, predicted_host_gib_for_registered_config=predicted, fits_with_15pct_headroom=bool(predicted <= 0.85 * avail0), note="prediction = n_envs x per-worker PSS + parent PSS (with CUDA context) + replay buffer + 0.5 GiB; a prediction, not a measurement of the full run")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("per_worker_pss_mib", "mem_available_before_gib", "predicted_host_gib_for_registered_config", "fits_with_15pct_headroom")})); return 0
if __name__ == "__main__": sys.exit(main())
