"""Watch a trained SuperMimic-OpenArm reaching policy LIVE in the native MuJoCo viewer (needs a display).  READ-ONLY: loads best.pt weights only (eval, no_grad), never trains, never writes a checkpoint, writes nothing.
The robot you see is the real simulation (physics steps at 1 kHz, display ~100 Hz); the red sphere is the goal (2 cm success radius), cyan dots the measured TCP trail.  Close the window to quit.
Usage (repository root):  python -m research.supermimic_openarm.view_live --run-dir <RUN>/seed1000 [--ckpt best.pt] [--episodes 10] [--loop] [--realtime 1.0]
Viewer: drag = rotate, right-drag = pan, scroll = zoom, Space = pause.   --no-gui runs the same loop without a window (tests)."""
import argparse, os, sys, time
os.environ.setdefault("MUJOCO_GL", "glfw")
from pathlib import Path
def parse(argv=None):
    ap = argparse.ArgumentParser(prog="python -m research.supermimic_openarm.view_live", description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True); ap.add_argument("--ckpt", default="best.pt"); ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--seed-base", type=int, default=9_800_000, help="goal seeds seed-base + i (default: the fixed visual-check list)")
    ap.add_argument("--loop", action="store_true", help="repeat the goal list until the window is closed"); ap.add_argument("--realtime", type=float, default=1.0, help="1.0 = real time, 0.25 = slow motion"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-gui", action="store_true"); ap.add_argument("--no-hold", action="store_true", help="exit after the last episode instead of waiting for the window to be closed"); ap.add_argument("--allow-commit-mismatch", action="store_true"); a = ap.parse_args(argv)
    if a.realtime <= 0: ap.error("--realtime must be > 0")
    if a.episodes < 1: ap.error("--episodes must be >= 1")
    return a
class SyncMj:
    """proxy for the mujoco module used by ONE env instance: forwards everything, and syncs the viewer every 10th physics step so motion is smooth"""
    def __init__(self, mj, viewer, dt, realtime): self._mj, self._v, self._n, self._dt, self._rt = mj, viewer, 0, dt, realtime
    def __getattr__(self, k): return getattr(self._mj, k)
    def mj_step(self, m, d):
        self._mj.mj_step(m, d); self._n += 1
        if self._v is not None and self._n % 10 == 0: self._v.sync(); time.sleep(10 * self._dt / self._rt)
def main(argv=None):
    a = parse(argv); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        import numpy as np, torch
        from .config import from_dict
        from .visualize import provenance, build_models
        from .envs.openarm_env import OpenArmTaskEnv
        from .envs import layout as L
        from .utils import assert_seeds_allowed
        rd = Path(a.run_dir).expanduser(); pv, cfg_d = provenance(rd.name, rd, a.ckpt, a.allow_commit_mismatch); cfg = from_dict(cfg_d); device = torch.device(a.device); ck = torch.load(pv["checkpoint"], map_location=device, weights_only=False)
        seeds = [a.seed_base + i for i in range(a.episodes)]; assert_seeds_allowed(*seeds); task, adapter, wm, actor = build_models(cfg, ck, device); env = OpenArmTaskEnv(cfg.env.task, cfg.env.task_params, task.max_steps, cfg.env.control_hz)
        print(f"loaded {pv['checkpoint']} (iteration {pv['checkpoint_iteration']}, sha256 {pv['checkpoint_sha256'][:12]}..., trained at {pv['trained_commit'][:10]}); success = TCP < {1000 * task.p['tol']:.0f} mm and < {task.p['speed_tol']} m/s for {task.hold_steps} steps", flush=True)
        viewer = None; ctx = None
        if not a.no_gui:
            import mujoco, mujoco.viewer
            ctx = mujoco.viewer.launch_passive(env.m, env.d); viewer = ctx.__enter__(); viewer.cam.lookat[:] = [0.40, 0.12, 0.47]; viewer.cam.distance = 0.9; viewer.cam.azimuth = 150; viewer.cam.elevation = -18
        env.mj = SyncMj(env.mj, viewer, env.m.opt.timestep, a.realtime)
        def alive(): return True if viewer is None else viewer.is_running()
        def draw(goal, trail):
            if viewer is None: return
            import mujoco
            with viewer.lock():
                s = viewer.user_scn; s.ngeom = 0
                def sph(p, r, rgba):
                    if s.ngeom < s.maxgeom: mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, float), np.eye(3).flatten(), np.asarray(rgba, np.float32)); s.ngeom += 1
                sph(goal, 0.020, [1, 0.1, 0.1, 0.35]); sph(goal, 0.004, [1, 0.1, 0.1, 1])
                for p in trail[::3][-60:]: sph(p, 0.004, [0.1, 0.9, 1, 1])
            viewer.sync()
        try:
            rounds = 0
            while alive():
                for k, sd in enumerate(seeds):
                    if not alive(): break
                    s, g, tcp = env.reset(sd); trail = [tcp.copy()]; draw(g, trail)
                    if viewer is not None: time.sleep(1.0 / a.realtime)
                    done = False; t = 0
                    while not done and alive():
                        with torch.no_grad(): act = actor(task.actor_obs(torch.as_tensor(s, device=device)[None], torch.as_tensor(g, device=device)[None], torch.as_tensor(tcp, device=device)[None])).float(); tgt = adapter.target_from_action(act, torch.as_tensor(s[L.PREV], device=device)[None])[0].cpu().numpy().astype(np.float64)
                        o = env.step(tgt); s, tcp = o["state"], o["tcp"]; t += 1; trail.append(tcp.copy()); done = o["terminated"] or o["truncated"]; draw(g, trail)
                    err = float(np.linalg.norm(tcp - g))
                    if not done: print(f"goal {k + 1}/{len(seeds)} (seed {sd}): interrupted after {t} steps (window closed)", flush=True); break
                    print(f"goal {k + 1}/{len(seeds)} (seed {sd}): {'SUCCESS' if env.ever_success else 'fail   '}  final TCP error {1000 * err:6.1f} mm  steps {t}", flush=True)
                    if viewer is not None: time.sleep(1.5 / a.realtime)
                rounds += 1
                if not a.loop: break
            if viewer is not None and not a.no_hold and alive():
                print("episodes finished: close the viewer window to quit", flush=True)
                while alive(): viewer.sync(); time.sleep(0.05)
        finally:
            if ctx is not None: ctx.__exit__(None, None, None)
        if viewer is not None: sys.stdout.flush(); sys.stderr.flush(); os._exit(0)          # skip native GL teardown at interpreter exit (it can segfault when torch and GLFW are both loaded); nothing is written by this script
        return 0
    except SystemExit as e: print(e, file=sys.stderr); return 2
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as e: print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr); return 2
if __name__ == "__main__": sys.exit(main())
