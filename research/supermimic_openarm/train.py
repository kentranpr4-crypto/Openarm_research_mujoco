"""CLI: python -m research.supermimic_openarm.train --config research/supermimic_openarm/configs/reaching_4090.yaml [--override k.v=x ...] [--resume RUN_DIR] [--run-name N] [--dry-run]"""
import argparse, json, sys
from pathlib import Path
def parse(argv=None):
    ap = argparse.ArgumentParser(prog="python -m research.supermimic_openarm.train", description="Train (SuperMimic-style world-model actor, PPO or BC baseline) on an OpenArm task. Never overwrites a completed experiment.")
    ap.add_argument("--config", help="YAML config (required unless --resume)"); ap.add_argument("--override", action="append", default=[], metavar="KEY.PATH=VALUE", help="override a config value, e.g. imagination.horizon=16")
    ap.add_argument("--resume", metavar="RUN_DIR", help="resume an interrupted run directory (uses its saved config)"); ap.add_argument("--run-name", help="experiment directory name (default: experiment.name)")
    ap.add_argument("--max-iterations", type=int, help="stop after this many additional iterations (checkpoint is written)"); ap.add_argument("--dry-run", action="store_true", help="validate and print the resolved configuration, then exit")
    return ap, ap.parse_args(argv)
def main(argv=None):
    ap, a = parse(argv)
    from .config import load_config, from_dict
    from . import runs
    if a.resume:
        rd = Path(a.resume); cf = rd / "config.resolved.json"
        if not cf.is_file(): ap.error(f"{cf} not found: not a run directory")
        cfg = from_dict(json.loads(cf.read_text()))
    else:
        if not a.config: ap.error("--config is required")
        try: cfg = load_config(a.config, a.override)
        except (FileNotFoundError, ValueError) as e: print(f"ERROR: {e}", file=sys.stderr); return 2
        rd = runs.run_dir_for(cfg, a.run_name)
    print("resolved configuration:\n" + json.dumps(cfg.to_dict(), indent=1)); print(f"run directory: {rd}")
    if a.dry_run: return 0
    try:
        runs.prepare_run_dir(cfg, rd, resume=bool(a.resume))
        if cfg.experiment.algo == "supermimic":
            from .trainer import Trainer
            tr = Trainer(cfg, rd, resume=bool(a.resume)); runs.write_run_files(cfg, rd, runs.build_env_manifest(cfg, tr.model_info), resume=bool(a.resume)); out = tr.train(a.max_iterations)
        elif cfg.experiment.algo == "ppo":
            from .baselines.ppo import train_ppo; runs.write_run_files(cfg, rd, runs.build_env_manifest(cfg), resume=False); out = train_ppo(cfg, rd, a.max_iterations)
        else:
            from .baselines.bc import train_bc; runs.write_run_files(cfg, rd, runs.build_env_manifest(cfg), resume=False); out = train_bc(cfg, rd)
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as e: print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 2
    print("finished:", json.dumps(out)); return 0
if __name__ == "__main__": sys.exit(main())
