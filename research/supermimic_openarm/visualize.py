"""READ-ONLY visual evaluation of trained SuperMimic-OpenArm reaching checkpoints.
Loads best.pt (weights only; model.eval(), torch.no_grad(): NO training, NO parameter update, checkpoints are never modified), runs ONE fixed list of held-out reaching goals on every model in the real OpenArm v1 MuJoCo environment,
renders the ACTUAL simulator state of every control step (no synthesised motion), overlays the goal / success sphere / TCP trail / measured error / TCP speed / success status, writes per-goal MP4s, side-by-side comparison MP4s,
comparison plots, per-episode metrics (success, dwell time, world-model error), provenance, and a minimal local HTML viewer.
Usage (repository root):  python -m research.supermimic_openarm.visualize --run-root ROOT --seeds 1000 1001 1002 --episodes 10 --gl egl
Extra packages (not needed for training):  pip install matplotlib imageio-ffmpeg
The output directory is NEW (timestamped); a non-empty --out is refused."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
EVAL_SEED_BASE = 9_800_000                       # fixed list: 9,800,000 + i  (disjoint from selection 9.0e6, earlier evals 9.5e6 / 9.7e6 and from 31-33)
SRC_COMMIT = "3ee510b595"                         # commit the checkpoints were trained at
def parse(argv=None):
    ap = argparse.ArgumentParser(prog="python -m research.supermimic_openarm.visualize", description=__doc__.split("\n")[0])
    ap.add_argument("--run-root", help="directory containing reaching_long_s<SEED>/seed<SEED>"); ap.add_argument("--seeds", nargs="*", type=int, default=[1000, 1001, 1002]); ap.add_argument("--run-dir", action="append", default=[], help="explicit run directory (repeatable) instead of --run-root/--seeds")
    ap.add_argument("--labels", nargs="*", help="labels for --run-dir entries"); ap.add_argument("--run-pattern", default="reaching_long_s{seed}/seed{seed}"); ap.add_argument("--ckpt", default="best.pt"); ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", help="NEW output directory (default outputs/supermimic_viz/<timestamp>)"); ap.add_argument("--gl", default=os.environ.get("MUJOCO_GL", "egl"), choices=["egl", "osmesa", "glfw"], help="MuJoCo OpenGL backend (headless: egl or osmesa)")
    ap.add_argument("--width", type=int, default=480); ap.add_argument("--height", type=int, default=360); ap.add_argument("--fps", type=int, default=20); ap.add_argument("--device", default="cpu"); ap.add_argument("--no-render", action="store_true", help="metrics and plots only")
    ap.add_argument("--allow-commit-mismatch", action="store_true"); return ap, ap.parse_args(argv)
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def wilson(k, n, z=1.96):
    import math
    if not n: return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [max(0.0, c - h), min(1.0, c + h)]
def locate_runs(a):
    runs = []
    if a.run_dir:
        for i, d in enumerate(a.run_dir): runs.append((a.labels[i] if a.labels and i < len(a.labels) else Path(d).name, Path(d).expanduser()))
    else:
        if not a.run_root: raise SystemExit("ERROR: give --run-root (and --seeds) or --run-dir")
        for s in a.seeds: runs.append((f"seed {s}", Path(a.run_root).expanduser() / a.run_pattern.format(seed=s)))
    return runs
def provenance(label, rd, ckname, allow_mismatch):
    """audit: files exist, SHA-256 vs manifest, config, source commit; raises SystemExit on integrity problems"""
    ck = rd / "checkpoints" / ckname
    for p in (ck, rd / "checkpoints" / "manifest.json", rd / "config.resolved.json", rd / "metadata.json"):
        if not p.is_file(): raise SystemExit(f"ERROR: missing {p}")
    man = json.loads((rd / "checkpoints" / "manifest.json").read_text()).get(ckname); sh = sha256(ck)
    if not man or man["sha256"] != sh: raise SystemExit(f"ERROR: checkpoint SHA-256 does not match the manifest: {ck}")
    md = json.loads((rd / "metadata.json").read_text()); commit = ((md.get("git") or {}).get("commit") or "")
    if not commit.startswith(SRC_COMMIT) and not allow_mismatch: raise SystemExit(f"ERROR: {label} was trained at commit {commit[:10]}, expected {SRC_COMMIT} (use --allow-commit-mismatch to override)")
    cfg = json.loads((rd / "config.resolved.json").read_text())
    if cfg["env"]["task"] != "reaching" or cfg["experiment"]["algo"] != "supermimic": raise SystemExit(f"ERROR: {label} is not a SuperMimic reaching run")
    return dict(label=label, run_dir=str(rd), checkpoint=str(ck), checkpoint_sha256=sh, manifest_sha256_ok=True, checkpoint_bytes=ck.stat().st_size, checkpoint_iteration=man.get("iteration"), trained_commit=commit, trained_torch=md.get("torch"), trained_gpu=md.get("gpu"), seed=cfg["experiment"]["seed"], completed=(rd / "COMPLETED").is_file()), cfg
def build_models(cfg, ckpt, device):
    import torch
    from .models import WorldModel, Actor
    from .envs.tasks import make_task
    from .trainer import build_fk_and_adapter
    task = make_task(cfg.env.task, **cfg.env.task_params); task.max_steps = cfg.env.max_steps or task.max_steps; _, adapter, _ = build_fk_and_adapter(cfg, device); w, a = cfg.world_model, cfg.actor
    wm = WorldModel(task.goal_dim, w.hidden, w.layers, w.contact_head, w.reward_head).to(device); actor = Actor(task.actor_obs_dim, a.hidden, a.layers).to(device); wm.load_state_dict(ckpt["wm"]); actor.load_state_dict(ckpt["actor"]); wm.eval(); actor.eval()
    for p in list(wm.parameters()) + list(actor.parameters()): p.requires_grad_(False)
    return task, adapter, wm, actor
class Renderer:
    """renders the ACTUAL running simulation (env.m / env.d) from a free camera; overlays are drawn into the scene (goal sphere = 2 cm success radius, TCP trail) and onto the frame (text)."""
    def __init__(self, env, w, h):
        import mujoco; self.mj = mujoco; m = env.m; m.light_castshadow[:] = 0; m.mat_reflectance[:] = 0.0; m.vis.quality.offsamples = 0; m.vis.global_.offwidth = max(w, int(m.vis.global_.offwidth)); m.vis.global_.offheight = max(h, int(m.vis.global_.offheight))
        self.r = mujoco.Renderer(m, h, w, max_geom=4000); self.cam = mujoco.MjvCamera(); self.cam.azimuth, self.cam.elevation, self.cam.distance = 150.0, -18.0, 0.80; self.cam.lookat[:] = [0.40, 0.12, 0.47]; self.env, self.w, self.h = env, w, h
    def _sphere(self, pos, r, rgba):
        s = self.r.scene
        if s.ngeom >= s.maxgeom: return
        self.mj.mjv_initGeom(s.geoms[s.ngeom], self.mj.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, np.float32)); s.ngeom += 1
    def frame(self, goal, trail, lines, color):
        self.r.update_scene(self.env.d, camera=self.cam)
        for p in trail[::2]: self._sphere(p, 0.004, [0.1, 0.9, 1.0, 1.0])
        self._sphere(trail[-1], 0.009, [1.0, 1.0, 0.0, 1.0]); self._sphere(goal, 0.020, [1.0, 0.1, 0.1, 0.35]); self._sphere(goal, 0.004, [1.0, 0.1, 0.1, 1.0])
        img = self.r.render().copy(); return draw_text(img, lines, color)
def draw_text(img, lines, color):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    try: f = ImageFont.load_default(size=14)
    except TypeError: f = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 18 * len(lines) + 6], fill=(0, 0, 0)); 
    for i, (t, c) in enumerate(lines): d.text((6, 4 + 18 * i), t, fill=c or (255, 255, 255), font=f)
    d.rectangle([0, im.height - 5, im.width, im.height], fill=color); return np.asarray(im)
def run_model(label, cfg, ckpt, device, seeds, renderer_factory, frames_dir, log):
    import torch
    from .envs.openarm_env import OpenArmTaskEnv
    from .envs import layout as L
    task, adapter, wm, actor = build_models(cfg, ckpt, device); env = OpenArmTaskEnv(cfg.env.task, cfg.env.task_params, task.max_steps, cfg.env.control_hz); rend = renderer_factory(env) if renderer_factory else None; tol, vtol, hold = task.p["tol"], task.p["speed_tol"], task.hold_steps; eps = []
    for k, sd in enumerate(seeds):
        s, g, tcp = env.reset(sd); S, TCP, EX = [s], [tcp.copy()], []; speed, dist, frames, trail, held, ever = [0.0], [float(np.linalg.norm(tcp - g))], [], [tcp.copy()], 0, False; done = False; t = 0
        def lines(held, ever):
            ok = dist[-1] < tol and speed[-1] < vtol; st = ("SUCCESS (held %d/%d)" % (min(held, hold), hold) if ever else ("IN GOAL: holding %d/%d" % (held, hold) if ok else "moving to goal")); col = (40, 200, 60) if ever else ((240, 200, 40) if ok else (200, 60, 60))
            return [(f"{label} | goal {k + 1}/{len(seeds)} (seed {sd}) | t = {t / cfg.env.control_hz:5.2f} s", None), (f"TCP error {1000 * dist[-1]:7.1f} mm   (success < {1000 * tol:.0f} mm)", None), (f"TCP speed {speed[-1]:6.3f} m/s   (success < {vtol:.2f} m/s)", None), (st, col)], col
        if rend: l, c = lines(0, False); frames.append(rend.frame(g, np.array(trail), l, c))
        while not done:
            with torch.no_grad(): a = actor(task.actor_obs(torch.as_tensor(s, device=device)[None], torch.as_tensor(g, device=device)[None], torch.as_tensor(tcp, device=device)[None])).float(); tgt = adapter.target_from_action(a, torch.as_tensor(s[L.PREV], device=device)[None])[0].cpu().numpy().astype(np.float64)
            o = env.step(tgt); s, tcp = o["state"], o["tcp"]; t += 1; S.append(s); TCP.append(tcp.copy()); EX.append(o["executed"]); trail.append(tcp.copy()); dist.append(float(np.linalg.norm(tcp - g))); speed.append(float(o["info"]["tcp_speed"]))
            held = o["info"]["held"]; ever = o["info"]["ever_success"]; done = o["terminated"] or o["truncated"]
            if rend: l, c = lines(held, ever); frames.append(rend.frame(g, np.array(trail), l, c))
        S, EX = np.stack(S), np.stack(EX)
        with torch.no_grad():                                                                              # world-model error where measurable: same definition as Trainer.evaluate
            St, Et = torch.as_tensor(S, device=device), torch.as_tensor(EX, device=device); one = wm(St[:-1], Et); w8 = wm.loss_w
            e1 = (((wm.state_norm(one) - wm.state_norm(St[1:])) ** 2) * w8).sum(-1).div(w8.sum()); ro = [St[0:1]]
            for j in range(len(EX)): ro.append(wm(ro[-1], Et[j:j + 1]))
            ro = torch.cat(ro[1:]); er = (((wm.state_norm(ro) - wm.state_norm(St[1:])) ** 2) * w8).sum(-1).div(w8.sum())
        d_, v_ = np.array(dist), np.array(speed); inst = (d_ < tol) & (v_ < vtol); run = best = 0
        for x in inst: run = run + 1 if x else 0; best = max(best, run)
        first = next((i for i in range(len(inst)) if inst[i:i + hold].all() and len(inst[i:i + hold]) == hold), None); recomputed = first is not None                  # success recomputed from the recorded arrays
        eps.append(dict(k=k, seed=int(sd), goal=g.tolist(), success=bool(env.ever_success), success_recomputed=bool(recomputed), consistent=bool(env.ever_success == recomputed), first_hold_step=first, first_hold_time_s=(None if first is None else first / cfg.env.control_hz),
                        longest_dwell_steps=int(best), steps_in_criterion=int(inst.sum()), stay_fraction_after_first=(None if first is None else float(inst[first:].mean())), final_error_m=float(d_[-1]), min_error_m=float(d_.min()), max_speed_after_5s=float(v_[min(100, len(v_) - 1):].max()) if len(v_) > 1 else 0.0,
                        wm_one_step_rmse_norm=float(e1.mean().sqrt()), wm_rollout_rmse_norm=float(er.mean().sqrt()), dist=d_.tolist(), speed=v_.tolist(), tcp=np.stack(TCP).tolist(), executed_ctrl_abs_step_mean=float(np.abs(np.diff(EX, axis=0)).mean()) if len(EX) > 1 else 0.0))
        if rend: p = frames_dir / label.replace(" ", "_"); p.mkdir(parents=True, exist_ok=True); np.save(p / f"goal{k:02d}.npy", np.stack(frames))
        log(f"  {label} goal {k + 1}/{len(seeds)}: {'SUCCESS' if eps[-1]['success'] else 'fail   '} final error {1000 * eps[-1]['final_error_m']:.1f} mm, longest dwell {best} steps")
    return eps, dict(tol_m=tol, speed_tol_m_s=vtol, hold_steps=hold, max_steps=task.max_steps, hz=cfg.env.control_hz)
import numpy as np
def write_video(frames, path, fps):
    import imageio.v2 as iio
    try:
        wr = iio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1, pixelformat="yuv420p")
        for f in frames: wr.append_data(f)
        wr.close(); return str(path)
    except Exception as e:
        if shutil.which("ffmpeg"):                                                                         # fallback: pipe raw frames to the ffmpeg binary
            h, w = frames[0].shape[:2]; p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
            for f in frames: p.stdin.write(f.tobytes())
            p.stdin.close(); p.wait(); return str(path)
        g = path.with_suffix(".gif"); iio.mimsave(str(g), frames, duration=1.0 / fps); print(f"WARNING: MP4 encoder unavailable ({e}); wrote GIF {g}. Install it with: pip install imageio-ffmpeg", file=sys.stderr); return str(g)
def make_plots(res, crit, out, hz):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError: print("WARNING: matplotlib missing, plots skipped (pip install matplotlib)", file=sys.stderr); return []
    labels = list(res); cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]; files = []; t = None
    def curves(key, ylabel, thr, fname, log=False):
        fig, ax = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 3.6), sharey=True, squeeze=False)
        for i, lb in enumerate(labels):
            for e in res[lb]["episodes"]: y = np.array(e[key]); ax[0, i].plot(np.arange(len(y)) / hz, y, lw=1, alpha=0.7, color=cols[i % 5])
            ax[0, i].axhline(thr, color="k", ls="--", lw=1); ax[0, i].set_title(f"{lb}: {len(res[lb]['episodes'])} held-out goals"); ax[0, i].set_xlabel("time (s)"); ax[0, i].grid(alpha=0.3)
            if log: ax[0, i].set_yscale("log")
        ax[0, 0].set_ylabel(ylabel); fig.tight_layout(); p = out / fname; fig.savefig(p, dpi=120); plt.close(fig); files.append(p.name)
    curves("dist", "TCP distance to goal (m)", crit["tol_m"], "plot_distance_to_goal.png", log=True); curves("speed", "TCP speed (m/s)", crit["speed_tol_m_s"], "plot_tcp_speed.png", log=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); x = np.arange(len(labels)); sr = [res[l]["success_rate"] for l in labels]; ci = np.array([res[l]["success_wilson95"] for l in labels]); ax[0].bar(x, sr, color=cols[:len(labels)]); ax[0].errorbar(x, sr, yerr=[np.array(sr) - ci[:, 0], ci[:, 1] - np.array(sr)], fmt="none", ecolor="k", capsize=4)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylim(0, 1.05); ax[0].set_title("success rate (95% Wilson)"); w = 0.8 / len(labels)
    for i, l in enumerate(labels):
        ax[1].bar(np.arange(len(res[l]["episodes"])) + i * w, [e["longest_dwell_steps"] for e in res[l]["episodes"]], w, label=l, color=cols[i % 5]); ax[2].bar(np.arange(len(res[l]["episodes"])) + i * w, [e["wm_rollout_rmse_norm"] for e in res[l]["episodes"]], w, label=l, color=cols[i % 5])
    ax[1].axhline(crit["hold_steps"], color="k", ls="--", lw=1); ng = len(res[labels[0]]["episodes"]); 
    for a_ in (ax[1], ax[2]): a_.set_xticks(np.arange(ng) + 0.4 - w / 2); a_.set_xticklabels([str(i + 1) for i in range(ng)])
    ax[1].set_title(f"longest consecutive steps inside criterion (need {crit['hold_steps']})"); ax[1].set_xlabel("goal"); ax[1].legend(fontsize=8); ax[2].set_title("world-model open-loop RMSE (normalised)"); ax[2].set_xlabel("goal"); fig.tight_layout(); p = out / "plot_success_dwell_wm.png"; fig.savefig(p, dpi=120); plt.close(fig); files.append(p.name); return files
def write_viewer(out, res, crit, prov, files):
    rows = "".join(f"<tr><td>{l}</td><td>{r['n_success']}/{r['n']}</td><td>{r['success_rate']:.2f} ({r['success_wilson95'][0]:.2f}-{r['success_wilson95'][1]:.2f})</td><td>{1000*r['mean_final_error_m']:.1f}</td><td>{r['mean_first_hold_time_s'] if r['mean_first_hold_time_s'] is None else round(r['mean_first_hold_time_s'],2)}</td><td>{r['mean_longest_dwell_steps']:.1f}</td><td>{r['mean_wm_rollout_rmse_norm']:.3f}</td><td>{r['inconsistent_success_flags']}</td></tr>" for l, r in res.items())
    cmp = "".join(f"<h3>goal {k + 1}</h3><video src='{f}' controls loop width='100%'></video>" for k, f in enumerate(files.get("compare", [])))
    pl = "".join(f"<img src='{f}' style='max-width:100%'><br>" for f in files.get("plots", [])); pv = json.dumps(prov, indent=1)
    (out / "viewer.html").write_text(f"""<!doctype html><meta charset=utf-8><title>SuperMimic reaching - visual evaluation</title><body style='font-family:sans-serif;max-width:1500px;margin:20px auto;padding:0 16px'>
<h1>SuperMimic-OpenArm reaching: ACTUAL MuJoCo rollouts of trained checkpoints</h1><p>Read-only evaluation: weights are loaded, never updated. Success = TCP within {1000*crit['tol_m']:.0f} mm of the goal AND TCP speed &lt; {crit['speed_tol_m_s']} m/s for {crit['hold_steps']} consecutive control steps ({crit['hold_steps']/crit['hz']:.2f} s at {crit['hz']} Hz), within {crit['max_steps']} steps. Simulation only (privileged-state training; not Sim2Real). Red sphere = goal with the {1000*crit['tol_m']:.0f} mm success radius; cyan = TCP trail.</p>
<table border=1 cellpadding=5><tr><th>model</th><th>success</th><th>rate (95% CI)</th><th>final error mm</th><th>mean time to first hold (s)</th><th>mean longest dwell (steps)</th><th>mean WM rollout RMSE</th><th>flag mismatches</th></tr>{rows}</table>
<h2>Plots</h2>{pl}<h2>Side-by-side comparison (same goals, left to right: {', '.join(res)})</h2>{cmp}<h2>Provenance</h2><pre>{pv}</pre></body>""")
def main(argv=None):
    ap, a = parse(argv); os.environ["MUJOCO_GL"] = a.gl
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        import torch
        from .config import from_dict
        from .utils import assert_seeds_allowed, repro_metadata, atomic_write_json, REPO
        runs = locate_runs(a); seeds = [EVAL_SEED_BASE + i for i in range(a.episodes)]; assert_seeds_allowed(*seeds)
        out = Path(a.out) if a.out else REPO / "outputs" / "supermimic_viz" / time.strftime("%Y%m%dT%H%M%S")
        if out.exists() and any(out.iterdir()): raise SystemExit(f"ERROR: output directory is not empty (nothing is overwritten): {out}")
        out.mkdir(parents=True, exist_ok=True); log = print; prov, res, allrows, crit = [], {}, {}, None; device = torch.device(a.device); frames_dir = out / "_frames"; goals = None
        for label, rd in runs:
            pv, cfg_d = provenance(label, rd, a.ckpt, a.allow_commit_mismatch); cfg = from_dict(cfg_d); cfg.system.device = a.device; ck = torch.load(pv["checkpoint"], map_location=device, weights_only=False); prov.append(pv); log(f"== {label}: {rd} (iteration {ck['iteration']}, sha256 {pv['checkpoint_sha256'][:12]}…)")
            def factory(env): return Renderer(env, a.width, a.height)
            eps, crit = run_model(label, cfg, ck, device, seeds, None if a.no_render else factory, frames_dir, log); n = len(eps); k = sum(e["success"] for e in eps); fh = [e["first_hold_time_s"] for e in eps if e["first_hold_time_s"] is not None]
            res[label] = dict(n=n, n_success=k, success_rate=k / n, success_wilson95=wilson(k, n), mean_final_error_m=float(np.mean([e["final_error_m"] for e in eps])), mean_first_hold_time_s=(float(np.mean(fh)) if fh else None), mean_longest_dwell_steps=float(np.mean([e["longest_dwell_steps"] for e in eps])), mean_wm_rollout_rmse_norm=float(np.mean([e["wm_rollout_rmse_norm"] for e in eps])),
                              mean_wm_one_step_rmse_norm=float(np.mean([e["wm_one_step_rmse_norm"] for e in eps])), inconsistent_success_flags=int(sum(not e["consistent"] for e in eps)), episodes=eps)
            g = [e["goal"] for e in eps]
            if goals is None: goals = g
            elif not np.allclose(goals, g): raise SystemExit("ERROR: models were evaluated on different goals")
        atomic_write_json(out / "results.json", dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), eval_seeds=seeds, goals=goals, success_criterion=crit, models=res, provenance=prov, metadata=repro_metadata(dict(source_commit_expected=SRC_COMMIT, no_training=True, no_parameter_updates=True, checkpoints_modified=False, protected_seeds_used=False, renderer=("none" if a.no_render else a.gl)))))
        files = dict(videos=[], compare=[], plots=make_plots(res, crit, out, crit["hz"]))
        if not a.no_render:
            vd = out / "videos"; vd.mkdir(); cd = out / "comparison_videos"; cd.mkdir(); labels = list(res)
            for k in range(a.episodes):
                fr = {l: np.load(frames_dir / l.replace(" ", "_") / f"goal{k:02d}.npy") for l in labels}
                for l in labels: files["videos"].append(write_video(list(fr[l]), vd / f"{l.replace(' ', '_')}_goal{k + 1:02d}.mp4", a.fps))
                T = min(len(v) for v in fr.values()); files["compare"].append(Path(write_video([np.concatenate([fr[l][t] for l in labels], 1) for t in range(T)], cd / f"compare_goal{k + 1:02d}.mp4", a.fps)).relative_to(out).as_posix())
            shutil.rmtree(frames_dir)
        write_viewer(out, res, crit, prov, files); print("\n" + json.dumps({l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("episodes", "success_wilson95")} for l, r in res.items()}, indent=1)); print("output directory:", out); print("open:", out / "viewer.html"); return 0
    except SystemExit as e: print(e, file=sys.stderr); return 2
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as e: print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 2
if __name__ == "__main__": sys.exit(main())
