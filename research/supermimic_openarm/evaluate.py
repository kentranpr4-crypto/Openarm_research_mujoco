"""CLI: python -m research.supermimic_openarm.evaluate --checkpoint PATH --task reaching [--episodes N] [--seed-base S]
Evaluates the checkpoint's deterministic actor on held-out goals (seeds disjoint from training, never 31-33) and writes a NEW json next to the run (never overwrites)."""
import argparse, json, sys, time
from pathlib import Path
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m research.supermimic_openarm.evaluate"); ap.add_argument("--checkpoint", required=True); ap.add_argument("--task", required=True); ap.add_argument("--episodes", type=int, default=0); ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--device", default=None); ap.add_argument("--save-traj", type=int, default=2); a = ap.parse_args(argv)
    from .config import from_dict
    from .checkpoint import load_checkpoint, CheckpointError
    try:
        ck = load_checkpoint(a.checkpoint, "cpu"); cfg = from_dict(ck["config"])
        if cfg.env.task != a.task: print(f"ERROR: checkpoint was trained on task {cfg.env.task!r}, not {a.task!r}", file=sys.stderr); return 2
        if cfg.experiment.algo != "supermimic": print("ERROR: evaluate supports supermimic checkpoints; PPO/BC baselines are evaluated by their trainers", file=sys.stderr); return 2
        if a.device: cfg.system.device = a.device
        if a.seed_base: cfg.train.eval_seed_base = a.seed_base
        cfg.validate()
        import torch
        from .trainer import Trainer
        run_dir = Path(a.checkpoint).resolve().parents[1]; tr = Trainer(cfg, run_dir / "_eval_tmp", resume=False, log=lambda *x: None)
        tr.wm.load_state_dict(ck["wm"]); tr.actor.load_state_dict(ck["actor"]); tr.critic.load_state_dict(ck["critic"]); n = a.episodes or cfg.train.eval_episodes
        res, trajs = tr.evaluate(n_episodes=n, save_traj=a.save_traj, tag=f"evaluate_{Path(a.checkpoint).stem}")
        obs = torch.zeros(1, tr.task.actor_obs_dim, device=tr.device)
        with torch.no_grad():
            for _ in range(20): tr.actor(obs)
            if tr.device.type == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(200): tr.actor(obs)
            if tr.device.type == "cuda": torch.cuda.synchronize()
        res["actor_inference_latency_ms"] = (time.perf_counter() - t) / 200 * 1e3; res["checkpoint"] = str(a.checkpoint); res["checkpoint_iteration"] = ck["iteration"]
        from . import utils
        out = run_dir / "eval" / f"evaluate_{Path(a.checkpoint).stem}_{time.strftime('%Y%m%dT%H%M%S')}.json"; utils.atomic_write_json(out, res)
        import numpy as np
        for k, t_ in enumerate(trajs): np.savez_compressed(out.with_suffix(f".ep{k}.npz"), **{kk: np.asarray(v) for kk, v in t_.items()})
        print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=1)); print("written:", out); return 0
    except (CheckpointError, FileNotFoundError, ImportError, RuntimeError, ValueError) as e: print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 2
if __name__ == "__main__": sys.exit(main())
