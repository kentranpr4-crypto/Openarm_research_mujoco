"""Entry point for the P1 actor-experiment variants.  `--variant b0` delegates, unchanged, to the ORIGINAL `train.py` (baseline B0 is never re-implemented); b1/b2/b3 use `p1_variants.P1VariantTrainer`.
Everything else (config, --override, --run-name, --resume, --max-iterations, --dry-run) is exactly train.py's CLI, so the four variants share one config file and differ ONLY by --variant / --variant-arg.  A new run directory is required (train.py's rule), and `variant.json` is written into it.
Usage (repo root), DRY-RUN shown; do not remove --dry-run without separate approval:
  python -m research.supermimic_openarm.p1_train_variant --variant b1 --variant-arg rho=0.1 --config research/supermimic_openarm/configs/pushing_p1_pilot.yaml --run-name p1ab_b1_s1000 --dry-run"""
import argparse, json, sys
from pathlib import Path
def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False); ap.add_argument("--variant", required=True, choices=["b0", "b1", "b2", "b3"]); ap.add_argument("--variant-arg", action="append", default=[], metavar="KEY=VALUE"); a, rest = ap.parse_known_args(argv)
    if a.variant == "b0":
        if a.variant_arg: print("ERROR: B0 takes no --variant-arg", file=sys.stderr); return 2
        from .train import main as base_main; return base_main(rest)
    from .train import parse
    from .p1_variants import P1VariantTrainer, DEFAULTS
    t_ap, t = parse(rest)
    try: vargs = dict(kv.split("=", 1) for kv in a.variant_arg)
    except ValueError: print("ERROR: --variant-arg must be KEY=VALUE", file=sys.stderr); return 2
    bad = sorted(set(vargs) - set(DEFAULTS))
    if bad: print(f"ERROR: unknown --variant-arg {bad}; known: {sorted(DEFAULTS)}", file=sys.stderr); return 2
    from .config import load_config, from_dict
    from . import runs, utils
    if t.resume:
        rd = Path(t.resume); cf = rd / "config.resolved.json"; vf = rd / "variant.json"
        if not cf.is_file() or not vf.is_file(): t_ap.error(f"{rd} is not a variant run directory (config.resolved.json / variant.json missing)")
        saved = json.loads(vf.read_text())
        if saved["variant"] != a.variant: print(f"ERROR: run directory was created by variant {saved['variant']}, not {a.variant}", file=sys.stderr); return 2
        cfg = from_dict(json.loads(cf.read_text())); vargs = {**saved["variant_args"]}
    else:
        if not t.config: t_ap.error("--config is required")
        try: cfg = load_config(t.config, t.override)
        except (FileNotFoundError, ValueError) as e: print(f"ERROR: {e}", file=sys.stderr); return 2
        rd = runs.run_dir_for(cfg, t.run_name)
    print("resolved configuration:\n" + json.dumps(cfg.to_dict(), indent=1)); print(f"variant: {a.variant} {vargs}"); print(f"run directory: {rd}")
    if t.dry_run: return 0
    if cfg.experiment.algo != "supermimic": print("ERROR: variants apply to algo=supermimic only", file=sys.stderr); return 2
    try:
        runs.prepare_run_dir(cfg, rd, resume=bool(t.resume)); tr = P1VariantTrainer(cfg, rd, a.variant, resume=bool(t.resume), **vargs); runs.write_run_files(cfg, rd, runs.build_env_manifest(cfg, tr.model_info), resume=bool(t.resume))
        if not t.resume: utils.atomic_write_json(rd / "variant.json", dict(variant=a.variant, variant_args=tr.v, git=utils.git_info()))
        out = tr.train(t.max_iterations)
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as e: print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 2
    print("finished:", json.dumps(out)); return 0
if __name__ == "__main__": sys.exit(main())
