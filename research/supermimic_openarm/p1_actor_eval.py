"""Held-out evaluation of a P1 (pushing) checkpoint's ACTOR: strict success (Wilson 95%), final goal error, box lift, action-safety statistics.  READ-ONLY for the checkpoint; writes one NEW json.  CPU only (the actor is small).
Seeds default to 27,000,000+ (never used for training 1,100,000+, best-checkpoint selection 9,000,000+, expert tuning 24,000,000+ or WM data 25,000,000+).
Safety statistics (per control step, from the executed controls of the ORIGINAL clip->slew adapter): executed command outside the ctrl range, slew above the limit, non-finite state, box leaving the table (terminated), minimum box-robot contact distance, right arm moved.
Usage (repo root):  python -m research.supermimic_openarm.p1_actor_eval --checkpoint RUN/seed1000/checkpoints/best.pt --episodes 64 --out outputs/p1_validation/actor_eval_<name>.json [--seed-base 27000000]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--episodes", type=int, default=64); ap.add_argument("--seed-base", type=int, default=27_000_000); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); a = ap.parse_args(argv); out = Path(a.out)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    import torch
    from .config import from_dict
    from .p1_variants import load_actor
    from .envs.tasks import make_task
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    from .utils import assert_seeds_allowed, atomic_write_json, repro_metadata, sha256_file
    from .p1_expert_eval import wilson
    dev = torch.device(a.device); ck = torch.load(a.checkpoint, map_location=dev, weights_only=False); cfg = from_dict(ck["config"]); seeds = [a.seed_base + i for i in range(a.episodes)]; assert_seeds_allowed(*seeds[:1], *seeds[-1:])
    task = make_task(cfg.env.task, **cfg.env.task_params); task.max_steps = cfg.env.max_steps or task.max_steps; actor, adapter = load_actor(cfg, ck, dev, task); env = OpenArmTaskEnv(cfg.env.task, cfg.env.task_params, task.max_steps, cfg.env.control_hz); lim = env.adapter.limits; rows = []; viol_range = viol_slew = nonfinite = fell = right_moved = n_term = n_trunc = n_sticky = 0; steps = 0; min_pen = 0.0
    for sd in seeds:
        s, g, tcp = env.reset(sd); right0 = env.hold_right.copy(); prev = env.prev_ctrl[:8].copy(); done = False; ret = 0.0; pen = 0.0; goal_d = None
        while not done:
            with torch.no_grad(): act = actor(task.actor_obs(torch.as_tensor(s, device=dev)[None], torch.as_tensor(g, device=dev)[None], torch.as_tensor(tcp, device=dev)[None])).float(); tgt = adapter.target_from_action(act, torch.as_tensor(s[L.PREV], device=dev)[None])[0].cpu().numpy().astype(np.float64)
            o = env.step(tgt); ex = o["executed"]; s, tcp = o["state"], o["tcp"]; ret += o["reward"]; steps += 1; done = o["terminated"] or o["truncated"]
            viol_range += int(np.any(ex < lim.lo[:8] - 1e-6) or np.any(ex > lim.hi[:8] + 1e-6)); viol_slew += int(np.any(np.abs(ex - prev) > lim.max_speed[:8] / cfg.env.control_hz * 1.0000001 + 1e-6)); prev = ex.copy(); nonfinite += int(not o["info"]["finite"]); right_moved += int(not np.array_equal(env.d.ctrl[8:], right0))
            for c in env.d.contact[:env.d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in env.box_geoms) != (g2 in env.box_geoms) and "openarm" in env.m.body(env.m.geom_bodyid[g2 if g1 in env.box_geoms else g1]).name: pen = min(pen, float(c.dist))
        fell += int(o["terminated"]); n_term += int(o["terminated"]); n_trunc += int(o["truncated"]); n_sticky += int(bool(env.ever_success) and not bool(o["info"]["success"])); min_pen = min(min_pen, pen); gd = float(np.linalg.norm(env.d.qpos[env.box_adr:env.box_adr + 2] - g)); rows.append(dict(seed=sd, success=bool(env.ever_success), final_goal_error_m=gd, box_max_rise_m=float(env.box_max_rise), steps=int(env.t), ret=float(ret)))
    k = sum(r["success"] for r in rows); n = len(rows); lo, hi = wilson(k, n)
    rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), checkpoint=str(a.checkpoint), checkpoint_sha256=sha256_file(a.checkpoint), checkpoint_iteration=ck.get("iteration"), seed_range=[seeds[0], seeds[-1]], task=cfg.env.task, success=k, episodes=n, success_rate=k / n, wilson95=[lo, hi], median_final_goal_error_m=float(np.median([r["final_goal_error_m"] for r in rows])),
               fraction_lifted_over_1cm=float(np.mean([r["box_max_rise_m"] > 0.01 for r in rows])), episode_ends=dict(terminated_failure_or_nonfinite=n_term, truncated_at_horizon=n_trunc, note="terminated = task.failed (box left the table) or a non-finite state; truncated = reached max_steps.  Success is the environment's sticky 'held 5 consecutive steps at some time' flag; success_reached_but_not_held_at_the_end counts episodes that reached it and later left the goal", success_reached_but_not_held_at_the_end=n_sticky), safety=dict(control_steps=steps, executed_outside_ctrl_range=viol_range, executed_slew_over_limit=viol_slew, non_finite_states=nonfinite, episodes_box_left_table=fell, fraction_box_left_table=fell / n, right_arm_command_changed_steps=right_moved, min_box_robot_contact_distance_m=min_pen), per_episode=rows,
               note="privileged-state simulation; held-out simulator goals; not a real-robot result", metadata=repro_metadata(dict(read_only_checkpoint=True)))
    atomic_write_json(out, rep); print(json.dumps({k_: rep[k_] for k_ in ("checkpoint_iteration", "success", "episodes", "wilson95", "median_final_goal_error_m", "fraction_lifted_over_1cm", "episode_ends", "safety")}, indent=1)); print("written", out); return 0
if __name__ == "__main__": sys.exit(main())
