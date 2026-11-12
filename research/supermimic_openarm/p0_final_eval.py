"""P0 (reaching) FINAL independent evaluation: best.pt AND latest.pt of every seed on the same N held-out goals, with success + Wilson 95% interval, TCP error, TCP velocity / acceleration / jerk (finite differences of the MEASURED TCP at the control rate),
dwell metrics, slew-limit saturation, world-model error, and the paired best-vs-latest comparison (exact McNemar test).  READ-ONLY: weights only (eval, no_grad), no training, checkpoints and run directories are never modified; --out must be NEW.
Run it on COPIES of the run directories if the originals must stay byte-for-byte untouched (this script itself writes nothing into them).
Usage (repo root):  python -m research.supermimic_openarm.p0_final_eval --run-root ROOT --seeds 1000 1001 1002 --episodes 100 --seed-base 9900000 --out outputs/p0_final/<new>"""
import argparse, json, math, os, statistics, sys, time
from pathlib import Path
import numpy as np
T = dict(per_seed_min_success=0.95, pooled_min_success=0.97, median_final_error_m_max=0.005, min_frac_longest_dwell_ge_50=0.95, min_mean_stay_fraction=0.95, latest_flag_below=0.90)      # PRE-REGISTERED (same as the P0 plan); accel/jerk are DESCRIPTIVE only
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [round(max(0, c - h), 4), round(min(1, c + h), 4)]
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def pct(x, q): return float(np.percentile(x, q)) if len(x) else None
def parse(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run-root"); ap.add_argument("--seeds", nargs="*", type=int, default=[1000, 1001, 1002]); ap.add_argument("--run-dir", action="append", default=[]); ap.add_argument("--labels", nargs="*"); ap.add_argument("--run-pattern", default="reaching_long_s{seed}/seed{seed}")
    ap.add_argument("--ckpts", nargs="*", default=["best.pt", "latest.pt"]); ap.add_argument("--episodes", type=int, default=100); ap.add_argument("--seed-base", type=int, default=9_900_000); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--allow-commit-mismatch", action="store_true"); ap.add_argument("--save-traj", action="store_true"); return ap.parse_args(argv)
def run_checkpoint(label, ck_name, rd, a, device, torch, log):
    from .config import from_dict
    from .visualize import provenance, build_models
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    pv, cfg_d = provenance(label, rd, ck_name, a.allow_commit_mismatch); cfg = from_dict(cfg_d); ck = torch.load(pv["checkpoint"], map_location=device, weights_only=False); task, adapter, wm, actor = build_models(cfg, ck, device); hz = cfg.env.control_hz; dt = 1.0 / hz
    env = OpenArmTaskEnv(cfg.env.task, cfg.env.task_params, task.max_steps, hz); tol, vtol, hold = task.p["tol"], task.p["speed_tol"], task.hold_steps; ms = adapter.max_step.cpu().numpy(); eps = []; trajs = []
    for i in range(a.episodes):
        sd = a.seed_base + i; s, g, tcp = env.reset(sd); S, TCP, EX, SPD = [s], [tcp.copy()], [], [0.0]; done = False
        while not done:
            with torch.no_grad(): act = actor(task.actor_obs(torch.as_tensor(s, device=device)[None], torch.as_tensor(g, device=device)[None], torch.as_tensor(tcp, device=device)[None])).float(); tgt = adapter.target_from_action(act, torch.as_tensor(s[L.PREV], device=device)[None])[0].cpu().numpy().astype(np.float64)
            o = env.step(tgt); s, tcp = o["state"], o["tcp"]; S.append(s); TCP.append(tcp.copy()); EX.append(o["executed"]); SPD.append(float(o["info"]["tcp_speed"])); done = o["terminated"] or o["truncated"]
        S, EX, TCP = np.stack(S), np.stack(EX), np.stack(TCP); gd = np.linalg.norm(TCP - g, axis=1); spd = np.array(SPD); v = np.diff(TCP, axis=0) / dt; ac = np.diff(v, axis=0) / dt; jk = np.diff(ac, axis=0) / dt
        inst = (gd < tol) & (spd < vtol); run = best = 0
        for x in inst: run = run + 1 if x else 0; best = max(best, run)
        first = next((k for k in range(len(inst)) if inst[k:k + hold].all() and len(inst[k:k + hold]) == hold), None); settle = (first + hold) if first is not None else None
        with torch.no_grad():
            St, Et = torch.as_tensor(S, device=device), torch.as_tensor(EX, device=device); one = wm(St[:-1], Et); w8 = wm.loss_w; e1 = (((wm.state_norm(one) - wm.state_norm(St[1:])) ** 2) * w8).sum(-1).div(w8.sum()); ro = [St[0:1]]
            for k in range(len(EX)): ro.append(wm(ro[-1], Et[k:k + 1]))
            ro = torch.cat(ro[1:]); er = (((wm.state_norm(ro) - wm.state_norm(St[1:])) ** 2) * w8).sum(-1).div(w8.sum())
        dex = np.abs(np.diff(np.vstack([env.hold_right[:0].reshape(0, 8) if False else S[0, L.PREV][None], EX]), axis=0))
        eps.append(dict(seed=int(sd), success=bool(env.ever_success), final_error_m=float(gd[-1]), min_error_m=float(gd.min()), first_hold_s=(None if first is None else first * dt), longest_dwell_steps=int(best), stay_fraction_after_first=(None if first is None else float(inst[first:].mean())),
                        peak_tcp_speed_m_s=float(np.linalg.norm(v, axis=1).max()), peak_tcp_acc_m_s2=float(np.linalg.norm(ac, axis=1).max()), rms_tcp_acc_m_s2=float(np.sqrt((np.linalg.norm(ac, axis=1) ** 2).mean())), peak_tcp_jerk_m_s3=float(np.linalg.norm(jk, axis=1).max()), rms_tcp_jerk_m_s3=float(np.sqrt((np.linalg.norm(jk, axis=1) ** 2).mean())),
                        max_tcp_speed_after_settle_m_s=(float(spd[settle:].max()) if settle is not None and settle < len(spd) else None), peak_tcp_acc_after_settle_m_s2=(float(np.linalg.norm(ac[settle:], axis=1).max()) if settle is not None and settle < len(ac) else None),
                        frac_ctrl_steps_at_slew_limit=float((dex >= 0.98 * ms[None]).mean()), wm_one_step_rmse_norm=float(e1.mean().sqrt()), wm_rollout_rmse_norm=float(er.mean().sqrt())))
        if a.save_traj: trajs.append(dict(seed=sd, goal=g, tcp=TCP, exec=EX))
        if (i + 1) % 20 == 0: log(f"  {label} {ck_name}: {i + 1}/{a.episodes} episodes, success so far {sum(e['success'] for e in eps)}")
    return pv, eps, trajs, dict(tol_m=tol, speed_tol_m_s=vtol, hold_steps=hold, hz=hz)
def summarize(eps):
    n = len(eps); k = sum(e["success"] for e in eps); f = lambda key: [e[key] for e in eps if e[key] is not None]; q = lambda key: dict(median=pct(f(key), 50), p95=pct(f(key), 95), max=pct(f(key), 100))
    return dict(n=n, n_success=k, success_rate=k / n, wilson95=wilson(k, n), failing_seeds=[e["seed"] for e in eps if not e["success"]], final_error_m=dict(median=pct(f("final_error_m"), 50), p95=pct(f("final_error_m"), 95), max=pct(f("final_error_m"), 100)), first_hold_s=dict(median=pct(f("first_hold_s"), 50), p95=pct(f("first_hold_s"), 95)),
                frac_longest_dwell_ge_50=sum(e["longest_dwell_steps"] >= 50 for e in eps) / n, mean_stay_fraction_after_first=(float(np.mean(f("stay_fraction_after_first"))) if f("stay_fraction_after_first") else None), peak_tcp_speed_m_s=q("peak_tcp_speed_m_s"), peak_tcp_acc_m_s2=q("peak_tcp_acc_m_s2"), rms_tcp_acc_m_s2=q("rms_tcp_acc_m_s2"),
                peak_tcp_jerk_m_s3=q("peak_tcp_jerk_m_s3"), rms_tcp_jerk_m_s3=q("rms_tcp_jerk_m_s3"), max_tcp_speed_after_settle_m_s=q("max_tcp_speed_after_settle_m_s"), peak_tcp_acc_after_settle_m_s2=q("peak_tcp_acc_after_settle_m_s2"), frac_ctrl_steps_at_slew_limit=float(np.mean(f("frac_ctrl_steps_at_slew_limit"))),
                wm_rollout_rmse_norm=float(np.mean(f("wm_rollout_rmse_norm"))), wm_one_step_rmse_norm=float(np.mean(f("wm_one_step_rmse_norm"))))
def main(argv=None):
    a = parse(argv); sys.path.insert(0, str(Path(__file__).resolve().parents[2])); out = Path(a.out)
    if out.exists() and any(out.iterdir()): sys.exit(f"ERROR: --out exists and is not empty (nothing is overwritten): {out}")
    try:
        import torch
        from .utils import assert_seeds_allowed, repro_metadata, atomic_write_json
        from .visualize import locate_runs
        assert_seeds_allowed(*[a.seed_base + i for i in range(a.episodes)]); runs = locate_runs(a); out.mkdir(parents=True, exist_ok=True); device = torch.device(a.device); res = {}; prov = []; crit = None; allep = {}
        for label, rd in runs:
            res[label] = {}
            for ckn in a.ckpts:
                print(f"== {label} / {ckn}", flush=True); pv, eps, trajs, crit = run_checkpoint(label, ckn, rd, a, device, torch, print); prov.append(pv); res[label][ckn] = summarize(eps); allep[(label, ckn)] = eps
                if a.save_traj: np.savez_compressed(out / f"traj_{label.replace(' ', '_')}_{ckn.replace('.pt', '')}.npz", **{f"goal_{t['seed']}": t["goal"] for t in trajs}, **{f"tcp_{t['seed']}": t["tcp"] for t in trajs})
        cmp = {}
        for label in res:
            if "best.pt" in res[label] and "latest.pt" in res[label]:
                A_, B_ = allep[(label, "best.pt")], allep[(label, "latest.pt")]; b = sum(x["success"] and not y["success"] for x, y in zip(A_, B_)); c = sum(y["success"] and not x["success"] for x, y in zip(A_, B_))
                cmp[label] = dict(best_only_success=b, latest_only_success=c, mcnemar_exact_p=mcnemar_exact(b, c), median_final_error_diff_m_latest_minus_best=float(np.median([y["final_error_m"] - x["final_error_m"] for x, y in zip(A_, B_)])))
        best = [r["best.pt"] for r in res.values() if "best.pt" in r]; K = sum(r["n_success"] for r in best); N = sum(r["n"] for r in best); checks = {}
        checks["P1 each best.pt >= 95/100"] = all(r["success_rate"] >= T["per_seed_min_success"] for r in best) and len(best) == len(res); checks["P2 pooled best.pt >= 0.97"] = bool(N) and K / N >= T["pooled_min_success"]
        checks["S1 median final error <= 5 mm (each best.pt)"] = all(r["final_error_m"]["median"] <= T["median_final_error_m_max"] for r in best); checks["S2 >= 95% episodes with longest dwell >= 50 steps (each best.pt)"] = all(r["frac_longest_dwell_ge_50"] >= T["min_frac_longest_dwell_ge_50"] for r in best)
        checks["S3 mean stay fraction after first hold >= 0.95 (each best.pt)"] = all(r["mean_stay_fraction_after_first"] is not None and r["mean_stay_fraction_after_first"] >= T["min_mean_stay_fraction"] for r in best)
        prim = checks["P1 each best.pt >= 95/100"] and checks["P2 pooled best.pt >= 0.97"]; sec = all(checks[k] for k in checks if k.startswith("S")); verdict = "PASS" if prim and sec else ("PARTIAL" if prim else "FAIL")
        flags = [f"{l}: latest.pt success {r['latest.pt']['success_rate']:.2f} < {T['latest_flag_below']}" for l, r in res.items() if "latest.pt" in r and r["latest.pt"]["success_rate"] < T["latest_flag_below"]]
        rep = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), expected=dict(n=a.episodes, seed_base=a.seed_base, goals="deterministic from the seed; identical for every checkpoint"), success_criterion=crit, thresholds_preregistered=T, per_checkpoint=res, best_vs_latest=cmp,
                   pooled_best=dict(n_success=K, n=N, success_rate=(K / N if N else None), wilson95=(wilson(K, N) if N else None), caveat="the runs share the same training goal sequence (train_seed_base), so pooling does not add independent replicates of the training-data variance"), checks=checks, latest_flags=flags, verdict=verdict, provenance=prov,
                   metadata=repro_metadata(dict(no_training=True, no_parameter_updates=True, checkpoints_modified=False, protected_seeds_used=False)), derivative_definitions="velocity/acceleration/jerk = first/second/third finite differences of the measured TCP position at the control rate (20 Hz); descriptive, no pass threshold")
        atomic_write_json(out / "results.json", rep); lines = [f"# P0 final evaluation ({a.episodes} goals per checkpoint, seed base {a.seed_base})", f"VERDICT: {verdict}", ""]
        for l, r in res.items():
            for ckn, s in r.items(): lines.append(f"- {l} {ckn}: {s['n_success']}/{s['n']} (95% {s['wilson95'][0]:.3f}-{s['wilson95'][1]:.3f}), median error {1000 * s['final_error_m']['median']:.2f} mm, p95 {1000 * s['final_error_m']['p95']:.2f} mm, peak acc median {s['peak_tcp_acc_m_s2']['median']:.1f} m/s2, peak jerk median {s['peak_tcp_jerk_m_s3']['median']:.0f} m/s3, steps at slew limit {s['frac_ctrl_steps_at_slew_limit']:.2f}")
        lines += ["", "checks: " + json.dumps(checks), "best vs latest: " + json.dumps(cmp)]; (out / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print("written:", out / "results.json"); return 0
    except SystemExit as e: print(e, file=sys.stderr); return 2
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as e: print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 2
if __name__ == "__main__": sys.exit(main())
