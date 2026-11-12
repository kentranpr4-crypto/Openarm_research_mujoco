"""READ-ONLY audit of whether several SuperMimic runs (e.g. the three P0 reaching seeds) are statistically independent.  Reads each run's config.resolved.json only; writes one NEW json.
What it establishes from the CODE + the stored configs: the training episode seeds are episode_seed(train_seed_base, env_idx, episode_idx) (envs/vector.py), i.e. they do NOT depend on experiment.seed; experiment.seed only drives network initialisation, exploration noise,
warm-up random actions and replay sampling.  So runs with the same train_seed_base see exactly the same sequence of training goals per environment.
Usage (repo root):  python -m research.supermimic_openarm.p0_seed_audit --run-dir RUN1/seed1000 --run-dir RUN2/seed1001 --run-dir RUN3/seed1002 --out outputs/p0_seed_audit.json [--episodes-per-env 200]"""
import argparse, json, sys, time
from pathlib import Path
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", action="append", required=True); ap.add_argument("--out", required=True); ap.add_argument("--episodes-per-env", type=int, default=200); a = ap.parse_args(argv); out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    from .envs.vector import episode_seed
    runs = []
    for rd in a.run_dir:
        c = json.loads((Path(rd) / "config.resolved.json").read_text()); runs.append(dict(run_dir=rd, experiment_seed=c["experiment"]["seed"], train_seed_base=c["train"]["train_seed_base"], eval_seed_base=c["train"]["eval_seed_base"], num_envs=c["env"]["num_envs"], warmup_steps=c["train"]["warmup_steps"], task=c["env"]["task"],
                         replay_rng_seed=c["experiment"]["seed"], config_sha=__import__("hashlib").sha256(json.dumps({k: v for k, v in c.items() if k != "experiment"}, sort_keys=True).encode()).hexdigest()[:16]))
    def seeds(r): return {(e, k): episode_seed(r["train_seed_base"], e, k) for e in range(r["num_envs"]) for k in range(a.episodes_per_env)}
    S = [seeds(r) for r in runs]; pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            common = sum(1 for k in S[i] if k in S[j] and S[i][k] == S[j][k]); pairs.append(dict(a=runs[i]["run_dir"], b=runs[j]["run_dir"], identical_training_episode_seeds=common, of=len(S[i]), fraction=common / max(len(S[i]), 1), same_eval_seed_base=runs[i]["eval_seed_base"] == runs[j]["eval_seed_base"], same_train_seed_base=runs[i]["train_seed_base"] == runs[j]["train_seed_base"], different_experiment_seed=runs[i]["experiment_seed"] != runs[j]["experiment_seed"]))
    shared = all(p["same_train_seed_base"] for p in pairs) if pairs else None
    rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=runs, pairs=pairs, training_goal_sequence_shared=shared,
               independent_across_runs=["network initialisation", "actor exploration noise", "warm-up random actions (their actions; the start states/goals they begin from are shared)", "replay sampling order"] if pairs and all(p["different_experiment_seed"] for p in pairs) else "experiment seeds identical: nothing is independent",
               shared_across_runs=(["training start states and goals per (env, episode) index"] if shared else []) + (["best-checkpoint selection goals (eval_seed_base)"] if pairs and all(p["same_eval_seed_base"] for p in pairs) else []),
               verdict=("NOT_INDEPENDENT: at least two runs share experiment.seed" if any(not p["different_experiment_seed"] for p in pairs) else "PARTIALLY_INDEPENDENT: different network/optimisation randomness but identical training goal sequence" if shared else ("INDEPENDENT_GOAL_SEQUENCES" if pairs else "SINGLE_RUN")),
               note="Replications on a shared training-goal sequence estimate optimisation variance only; they do not test sensitivity to the training distribution.  Generalisation claims must come from held-out goals that were never used for training or checkpoint selection.")
    out.write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("verdict", "independent_across_runs", "shared_across_runs", "pairs")}, indent=1)); print("written", out); return 0
if __name__ == "__main__": sys.exit(main())
