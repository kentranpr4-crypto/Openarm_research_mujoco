#!/usr/bin/env python3
"""Evaluate the scripted expert and/or a trained ACT policy on paired random box configurations.
Run in the ACT venv:  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa /home/user/.venvs/act/bin/python eval/evaluate.py --agent act --ckpt outputs/checkpoints/x/last ...
Both agents see the SAME seeded configurations (pairing). Ground truth (box/TCP pose) is read from the simulator ONLY to compute
labels/metrics; the ACT agent receives state + images only.  Writes <out>/<agent>/<condition>/{episodes.jsonl,summary.json,videos/*.mp4,rollouts/*.npz}."""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import sys, json, time, argparse, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)

CONDITIONS = {   # name -> (region, env perturbation kwargs)
    "nominal": ("train", {}), "unseen_pose": ("unseen", {}),
    "light_dim": ("train", dict(light_scale=0.55)), "light_bright": ("train", dict(light_scale=1.6)), "light_shift": ("train", dict(light_shift=(-0.6, 0.8, 0.0))),
    "cam_shift": ("train", dict(cam_shift=(0.02, 0.01, 0.01), cam_tilt_deg=3.0)),
}

def wilson(k, n, z=1.96):
    if n == 0: return [float("nan")] * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [float(c - h), float(c + h)]

def classify(lab, rec):
    if lab["task_success"]: return "success"
    if not lab["pick_success"]: return "no_pick"          # never held the box >=5 cm up
    return "dropped_or_misplaced"                        # picked, but not resting in the tray

def run_agent_episode(agent, ckpt, n_act_steps, env, expert, idx, seed, region, pert, horizon, video_dir, rollout_dir, save_video):
    from policies.baseline import sample_box_pose
    from eval.metrics import task_labels, tracking_error, smoothness
    import lab_scene
    rng = np.random.default_rng(seed * 100000 + idx); x, y, yaw = sample_box_pose(rng, region)
    env.set_perturbation(**pert); o = env.reset((x, y), yaw)
    plan = expert.plan((x, y), yaw) if agent == "expert" else None
    T = min(horizon, len(plan["ctrl"])) if agent == "expert" else horizon
    S, A, B, TCP, BS, frames = [], [], [], [], [], []
    runner = env.runner if agent == "act" else None
    if runner: runner.reset(env.prev_ctrl); runner.latency_ms, runner.query_ms, runner.clipped, runner.slewed = [], [], 0, 0
    for t in range(T):
        i = env.info(); S.append(o["state"]); B.append(i["box_pos"]); TCP.append(i["tcp_pos"]); BS.append(i["box_speed"])
        if agent == "expert": a = plan["ctrl"][t]
        else: a = runner.act(o["state"], o["images"])                       # <- state + images only
        A.append(np.asarray(a, np.float32))
        if save_video and "images" in o and o["images"]: frames.append(np.concatenate([o["images"][c] for c in sorted(o["images"])], 1))
        o = env.step(a)
    i = env.info(); B.append(i["box_pos"]); TCP.append(i["tcp_pos"]); BS.append(i["box_speed"])
    lab = task_labels(B, TCP, BS, lab_scene.TRAY_C, lab_scene.TABLE_TOP_Z + lab_scene.TRAY_BASE_T)
    S, A = np.array(S), np.array(A)
    rec = dict(episode_index=idx, seed=seed, agent=agent, region=region, box_xy=[x, y], box_yaw=yaw, steps=T, outcome=classify(lab, None), **lab,
               tracking=tracking_error(S, A), smoothness=smoothness(A, env.hz))
    if agent == "act":
        rec.update(latency_ms_all=runner.latency_ms, latency_ms_queries=runner.query_ms, clipped=runner.clipped, slewed=runner.slewed)
    if rollout_dir is not None:
        rollout_dir.mkdir(parents=True, exist_ok=True); np.savez_compressed(rollout_dir / f"ep{idx:04d}.npz", state=S, action=A, box_pos=np.array(B), tcp_pos=np.array(TCP))
    if frames and (save_video == "all" or not lab["task_success"] or idx < 2):
        import imageio.v2 as iio; video_dir.mkdir(parents=True, exist_ok=True)
        path = video_dir / f"ep{idx:04d}_{rec['outcome']}.mp4"
        try:
            try: iio.mimsave(path, frames, fps=env.hz, macro_block_size=1)          # imageio-ffmpeg backend (sim venv)
            except TypeError: iio.mimsave(path, frames, fps=env.hz)                  # PyAV backend (ACT venv) has no macro_block_size
        except Exception as e: print(f"[warn] video not written for ep {idx}: {type(e).__name__}: {e}", flush=True)
    return rec

def worker(job):
    ids, a, cond = job
    os.environ.setdefault("LP_NUM_THREADS", "1")
    from lab_env import LabEnv
    from policies.baseline import ScriptedExpert
    region, pert = CONDITIONS[cond]
    cams = () if a["agent"] == "expert" else None
    runner = None
    if a["agent"] == "act":
        from control.action_adapter import ActionAdapter
        from policies.act_infer import ACTRunner
        import lab_scene
        probe = LabEnv(cameras=()); ad = ActionAdapter(probe.m, a["hz"]); probe.close()
        runner = ACTRunner(a["ckpt"], ad, n_action_steps=a["n_action_steps"], threads=1); cams = tuple(runner.cams)
    env = LabEnv(image_hw=tuple(a["image_hw"]), control_hz=a["hz"], render_profile=a["render_profile"], cameras=cams)
    env.runner = runner; expert = ScriptedExpert(env.m, control_hz=a["hz"])
    out = Path(a["out"]) / a["agent"] / cond; res = []
    for idx in ids:
        r = run_agent_episode(a["agent"], a["ckpt"], a["n_action_steps"], env, expert, idx, a["seed"], region, pert, a["horizon"], out / "videos",
                              out / "rollouts" if a["save_rollouts"] else None, bool(cams) and a["video"])
        print(f"[{a['agent']}/{cond}] ep {idx}: {r['outcome']}", flush=True); res.append(r)
    env.close(); return res

def summarise(recs, cond, agent):
    from eval.metrics import latency_percentiles
    n = len(recs); k = lambda key: sum(r[key] for r in recs)
    s = dict(agent=agent, condition=cond, n=n, **{f"{m}_rate": k(f"{m}_success") / n for m in ("pick", "place", "task")},
             **{f"{m}_ci95": wilson(k(f"{m}_success"), n) for m in ("pick", "place", "task")}, outcomes={o: sum(r["outcome"] == o for r in recs) for o in sorted({r["outcome"] for r in recs})},
             tracking_mean_arm_rad=float(np.mean([r["tracking"]["mean_arm"] for r in recs])), tracking_max_arm_rad=float(np.max([r["tracking"]["max_arm"] for r in recs])),
             mean_abs_jerk=float(np.mean([r["smoothness"]["mean_abs_jerk"] for r in recs])), mean_abs_step=float(np.mean([r["smoothness"]["mean_abs_step"] for r in recs])))
    if agent == "act":
        allms = sum((r["latency_ms_all"] for r in recs), []); qms = sum((r["latency_ms_queries"] for r in recs), [])
        s.update(latency_ms_per_step=latency_percentiles(allms), latency_ms_per_query=latency_percentiles(qms) if qms else None,
                 clipped_total=sum(r["clipped"] for r in recs), slewed_total=sum(r["slewed"] for r in recs), device="cpu (1 thread)")
    return s

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--agent", choices=["expert", "act"], required=True); ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--conditions", nargs="+", default=["nominal"]); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--horizon", type=int, default=200); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-action-steps", type=int); ap.add_argument("--out", type=Path, default=ROOT / "outputs/eval/run"); ap.add_argument("--hz", type=int, default=20)
    ap.add_argument("--image-hw", type=int, nargs=2, default=[120, 160]); ap.add_argument("--render-profile", default="fast"); ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--save-rollouts", action="store_true"); ap.add_argument("--ids", type=int, nargs="+", help="explicit episode indices (e.g. the raw ids of training episodes); overrides --start/--n")
    a = ap.parse_args(); cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    if a.agent == "act" and not a.ckpt: sys.exit("--ckpt required for --agent act")
    summaries = {}
    for cond in (list(CONDITIONS) if a.conditions == ["all"] else a.conditions):
        ids = list(a.ids) if a.ids else list(range(a.start, a.start + a.n)); chunks = [ids[i::a.workers] for i in range(a.workers)]
        if a.workers == 1: recs = worker((ids, cfg, cond))
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers) as pool: recs = sum(pool.map(worker, [(c, cfg, cond) for c in chunks if c]), [])
        recs.sort(key=lambda r: r["episode_index"]); d = a.out / a.agent / cond; d.mkdir(parents=True, exist_ok=True)
        (d / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in recs)); s = summarise(recs, cond, a.agent); (d / "summary.json").write_text(json.dumps(s, indent=1)); summaries[cond] = s
        print(f"== {a.agent}/{cond}: task {s['task_rate']:.2f} {s['task_ci95']} pick {s['pick_rate']:.2f} place {s['place_rate']:.2f} outcomes {s['outcomes']}", flush=True)
    (a.out / f"summary_{a.agent}.json").write_text(json.dumps(dict(config=cfg, conditions=summaries), indent=1))

if __name__ == "__main__":
    main()
