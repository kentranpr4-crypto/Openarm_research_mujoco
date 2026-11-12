"""Evaluate (or tune) the scripted pushing expert on a seed range with the STRICT success criterion (unchanged: box within 3 cm, speed < 0.03 m/s, never lifted > 1 cm, held 5 steps, 150 steps).
Reports exact numerator/denominator, Wilson 95% interval and failure categories.  Writes a NEW json (refuses to overwrite).  CPU only, process-parallel.
Usage (repository root):
  python -m research.supermimic_openarm.p1_expert_eval --seed-base 24000000 --episodes 48 --out outputs/p1_validation/expert_tune_A.json [--param v_push=0.08 --param z_push=0.380] [--workers 4]
Seed policy: 24,000,000-24,000,999 TUNING (may be used to choose parameters); 24,100,000+ DEVELOPMENT EVALUATION (used once for the reported number; never tuned on)."""
import argparse, math, sys, time
import numpy as np
from pathlib import Path
def wilson(k, n, z=1.959964):
    if n == 0: return (0.0, 1.0)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return (max(0.0, c - h), min(1.0, c + h))
def run_one(args):
    seed, params = args
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    from .baselines.push_expert import PushExpert, MAX_STEP
    global _ENV
    if "_ENV" not in globals(): _ENV = OpenArmTaskEnv("pushing", None, 150, 20)
    env = _ENV; d = env.d; ex = PushExpert(**params); env.reset(seed); g = env.goal.copy(); inf = dict(env.episode_info); done = False; fell = False; touched = False; D = []; SPD = []; RISE = []; vmax = 0.0
    while not done:
        o = env.step(env.state_vec()[L.PREV] + ex.act(env) * MAX_STEP); done = o["terminated"] or o["truncated"]; fell |= o["terminated"]; bp = d.qpos[env.box_adr:env.box_adr + 3]; D.append(float(np.linalg.norm(bp[:2] - g))); SPD.append(float(o["info"]["box_speed"])); RISE.append(float(env.box_max_rise)); vmax = max(vmax, SPD[-1]); touched |= o["info"]["finger_contacts"] > 0
    ln = (g - np.array(inf["box_xy"])); ln = ln / np.linalg.norm(ln); bf = d.qpos[env.box_adr:env.box_adr + 2] - g; signed = float(bf @ ln); lateral = float(abs(bf[0] * ln[1] - bf[1] * ln[0]))    # signed > 0: overshoot past the goal along the push line
    placed = D[-1] < 0.03 and SPD[-1] < 0.03; why = "success" if env.ever_success else ("box off table" if fell else ("no contact" if not touched else ("lifted > 1 cm (final distance ok)" if (placed and RISE[-1] > 0.01) else ("lifted > 1 cm (not placed)" if RISE[-1] > 0.01 else (("overshoot (> 3 cm past the goal)" if signed > 0.02 and D[-1] >= 0.03 else ("stopped short (> 3 cm)" if signed < -0.02 and D[-1] >= 0.03 else ("off the line (lateral > 3 cm)" if D[-1] >= 0.03 else "not at rest"))))))))
    return dict(seed=seed, success=bool(env.ever_success), placed=bool(placed), why=why, final_dist_m=D[-1], max_rise_m=RISE[-1], max_box_speed_m_s=vmax, signed_along_goal_m=signed, lateral_m=lateral, steps=len(D), mass_kg=inf["mass_kg"], friction_scale=inf["friction_scale"], goal_distance_m=float(np.linalg.norm(g - np.array(inf["box_xy"]))))
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed-base", type=int, required=True); ap.add_argument("--episodes", type=int, default=48); ap.add_argument("--out", required=True); ap.add_argument("--param", action="append", default=[]); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists(): print(f"ERROR: {out} exists (refusing to overwrite)", file=sys.stderr); return 2
    from .utils import assert_seeds_allowed, atomic_write_json, repro_metadata
    seeds = [a.seed_base + i for i in range(a.episodes)]; assert_seeds_allowed(*seeds[:1], *seeds[-1:]); params = {k: float(v) for k, v in (x.split("=") for x in a.param)}; t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool: eps = pool.map(run_one, [(s, params) for s in seeds], chunksize=2)
    n = len(eps); k = sum(e["success"] for e in eps); lo, hi = wilson(k, n); causes = {}
    for e in eps: causes[e["why"]] = causes.get(e["why"], 0) + 1
    rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed_range=[seeds[0], seeds[-1]], params_overrides=params, strict_success=k, episodes=n, strict_success_rate=k / n, wilson95=[lo, hi], placed=sum(e["placed"] for e in eps), failure_causes=causes, seconds=round(time.time() - t0, 1),
               failing_seeds=[e["seed"] for e in eps if not e["success"]], per_episode=eps, metadata=repro_metadata(dict(no_training=True)))
    atomic_write_json(out, rep); print(f"strict success {k}/{n} = {k / n:.3f} (Wilson95 {lo:.3f}-{hi:.3f}) | placed {rep['placed']}/{n} | causes {causes} | {rep['seconds']} s | params {params}"); return 0
if __name__ == "__main__": sys.exit(main())
