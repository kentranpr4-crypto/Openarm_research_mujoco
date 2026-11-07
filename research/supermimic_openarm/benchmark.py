"""CLI: python -m research.supermimic_openarm.benchmark --config CONFIG --steps 1000
Bounded profiling: env transitions/s, world-model and actor/critic update times, peak GPU memory (hard-capped by system.max_vram_gb). Writes outputs/supermimic_openarm/benchmarks/*.json.
No checkpoint is created; exits non-zero if the VRAM cap is exceeded (NO-GO)."""
import argparse, json, sys, time, math, tempfile
from pathlib import Path
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m research.supermimic_openarm.benchmark"); ap.add_argument("--config", required=True); ap.add_argument("--steps", type=int, default=1000, help="environment transitions to collect"); ap.add_argument("--override", action="append", default=[]); ap.add_argument("--updates", type=int, default=3)
    a = ap.parse_args(argv)
    from .config import load_config
    from . import utils
    try:
        cfg = load_config(a.config, a.override); print("resolved configuration:\n" + json.dumps(cfg.to_dict(), indent=1))
        import torch
        from .trainer import Trainer
        out_dir = utils.REPO / cfg.experiment.output_dir / "benchmarks"; out_dir.mkdir(parents=True, exist_ok=True); tr = Trainer(cfg, out_dir / "_scratch", resume=False, log=lambda *x: None); tr.t0 = time.time(); dev = tr.device; cuda = dev.type == "cuda"
        if cuda: torch.cuda.reset_peak_memory_stats(dev)
        tr._make_vec(); n_vec = max(1, math.ceil(a.steps / cfg.env.num_envs)); t = time.perf_counter(); tr.collect(max(n_vec, cfg.world_model.window + 2), random=True); t_env = time.perf_counter() - t; transitions = max(n_vec, cfg.world_model.window + 2) * cfg.env.num_envs
        tr.collect(2); times_wm, times_ac = [], []
        for _ in range(a.updates):
            if cuda: torch.cuda.synchronize()
            t = time.perf_counter(); tr.update_world_model(); 
            if cuda: torch.cuda.synchronize()
            times_wm.append(time.perf_counter() - t); t = time.perf_counter(); r = tr.update_actor_critic()
            if cuda: torch.cuda.synchronize()
            times_ac.append(time.perf_counter() - t)
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if cuda else None; tr.vec.close()
        res = dict(config=a.config, device=str(dev), precision=cfg.system.precision, env_backend=cfg.env.backend, num_envs=cfg.env.num_envs, horizon=cfg.imagination.horizon, imagination_batch=cfg.imagination.batch_size, env_transitions=int(transitions), env_seconds=t_env, env_transitions_per_s=transitions / t_env,
                   world_model_update_s=float(sum(times_wm) / len(times_wm)), actor_critic_update_s=float(sum(times_ac) / len(times_ac)), ms_per_env_step_incl_policy=1e3 * t_env / transitions, peak_vram_gb=peak, vram_cap_gb=cfg.system.max_vram_gb, actor_bad_grads=r["bad_grads"], actor_grad_norm=r["actor_grad_norm"],
                   gpu=(torch.cuda.get_device_name(dev) if cuda else None), verdict=("NOT_TESTED_NO_GPU" if not cuda else ("GO" if peak <= cfg.system.max_vram_gb and r["bad_grads"] == 0 else "NO-GO")), measured_not_estimated=True, metadata=utils.repro_metadata())
        p = out_dir / f"{Path(a.config).stem}_{time.strftime('%Y%m%dT%H%M%S')}.json"; utils.atomic_write_json(p, res); print(json.dumps({k: v for k, v in res.items() if k != "metadata"}, indent=1)); print("written:", p)
        return 0 if res["verdict"] in ("GO", "NOT_TESTED_NO_GPU") else 1
    except torch.cuda.OutOfMemoryError as e: print(f"NO-GO: GPU out of memory under the cap: {e}", file=sys.stderr); return 1
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e: print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 2
if __name__ == "__main__": sys.exit(main())
