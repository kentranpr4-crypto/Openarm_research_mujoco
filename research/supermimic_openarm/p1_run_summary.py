"""READ-ONLY summary of ONE P1 training run directory (any variant), for the B0-B3 comparison.  Writes one NEW json.
Reports: iterations, environment transitions, wall-clock and GPU-reserved hours (wall-clock x 1 GPU; NOT utilisation-weighted: pair it with the p1_benchmark GPU utilisation), time split per iteration, environment throughput, actor gradient statistics (update magnitude proxy),
variant-specific metrics (b1_*, b2_*), the trainer's own periodic evaluation (SELECTION data: goals 9,000,000+, used to pick best.pt -- never a final result), and the actor / world-model parameter step between consecutive checkpoint COPIES (--ckpt-copies, oldest first).
Final held-out results come from p1_actor_eval (goals 27,000,000+) and p1_wm_eval, not from this file.
Usage (repo root):  python -m research.supermimic_openarm.p1_run_summary --run-dir outputs/supermimic_openarm/RUN/seed1000 [--ckpt-copies c400.pt c800.pt ...] --out outputs/p1_validation/run_summary_<name>.json"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--ckpt-copies", nargs="*", default=[]); ap.add_argument("--out", required=True); a = ap.parse_args(argv); out = Path(a.out); rd = Path(a.run_dir)
    if out.exists(): sys.exit(f"refusing to overwrite {out}")
    mp = rd / "metrics.jsonl"
    if not mp.is_file(): sys.exit(f"{mp} not found")
    rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]; last = rows[-1]; col = lambda k: np.array([r[k] for r in rows if k in r and r[k] is not None], float)
    tc, tw, ta = col("t_collect"), col("t_wm"), col("t_actor"); g = col("actor_grad_norm"); wall = float(last.get("time_s", 0.0)); var = json.loads((rd / "variant.json").read_text()) if (rd / "variant.json").is_file() else dict(variant="b0", variant_args={})
    ev = [dict(iteration=r["iteration"], eval_success=r["eval_success"], eval_return=r["eval_return"], eval_goal_err=r.get("eval_goal_err"), eval_wm_rmse=r.get("eval_wm_rmse")) for r in rows if "eval_success" in r]
    vm = {k: dict(mean=float(col(k).mean()), last=float(col(k)[-1])) for k in sorted({k for r in rows for k in r if k.startswith(("b1_", "b2_", "wm_loss_member"))}) if len(col(k))}
    rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run_dir=str(rd), variant=var["variant"], variant_args=var["variant_args"], iterations=int(last["iteration"]), env_transitions=int(last["env_steps"]), wall_clock_s_last_segment=wall, gpu_reserved_hours_upper_bound=wall / 3600.0,
               note_hours="time_s restarts at each resume; sum the segments of a resumed run; GPU hours here = wall-clock x 1 reserved GPU", seconds_per_iteration=dict(collect=float(tc.mean()) if len(tc) else None, world_model=float(tw.mean()) if len(tw) else None, actor_critic=float(ta.mean()) if len(ta) else None),
               env_transitions_per_s_collect=float(rows[-1]["env_steps"] / tc.sum()) if len(tc) and tc.sum() > 0 else None, actor_grad_norm=dict(mean=float(g.mean()), p95=float(np.percentile(g, 95)), max=float(g.max())) if len(g) else None, iterations_with_bad_grads=int(sum(1 for r in rows if r.get("bad_grads", 0))),
               imag_objective_last20_mean=float(col("imag_objective")[-20:].mean()) if len(col("imag_objective")) else None, train_ep_return_last=float(col("train_ep_return")[-1]) if len(col("train_ep_return")) else None, variant_metrics=vm, trainer_eval_selection_data=ev, selection_note="goals 9,000,000+ (best.pt selection) - NOT a final result")
    if a.ckpt_copies:
        import torch
        steps = []; prev = None
        for p in a.ckpt_copies:
            ck = torch.load(p, map_location="cpu", weights_only=False); cur = dict(it=ck["iteration"], actor=torch.cat([v.flatten().float() for v in ck["actor"].values()]), wm=torch.cat([v.flatten().float() for v in ck["wm"].values()]))
            if prev is not None: steps.append(dict(from_iteration=prev["it"], to_iteration=cur["it"], actor_l2=float((cur["actor"] - prev["actor"]).norm()), actor_relative=float((cur["actor"] - prev["actor"]).norm() / max(float(prev["actor"].norm()), 1e-12)), wm_l2=float((cur["wm"] - prev["wm"]).norm())))
            prev = cur
        rep["parameter_steps_between_checkpoint_copies"] = steps
    out.write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("variant", "iterations", "env_transitions", "gpu_reserved_hours_upper_bound", "seconds_per_iteration", "actor_grad_norm", "variant_metrics")}, indent=1, default=float)); print("written", out); return 0
if __name__ == "__main__": sys.exit(main())
