#!/usr/bin/env python3
"""WM-1 intervention PILOT (simulation only): 10 initial poses x 25 open-loop action variants of the scripted expert = 250 episodes (cap).  The variants are applied to a scripted plan
(feed-forward, independent of the state) by a subclass/wrapper; policies/baseline.py, ACT and the controller are NOT modified.  Executed actions pass through the SAME clip + slew limits
as the ACT runner (control/safety_limits.SafetyLimits.clip_and_slew: ctrlrange, 2.0 rad/s arm, 0.2 m/s gripper).  Writes outputs/wm1/pilot/{episodes/*.npz, pilot_index.json}.
Run (ACT venv): python data/wm1_intervene.py [--workers 4]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, argparse, hashlib, time, numpy as np
from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
import wm1_sim as S
OUT = ROOT / "outputs/wm1/pilot"
EVAL_SEED_LABEL = 60                          # synthetic poses; never 31-33
# 10 pilot poses (non-round values so that no group coincides with an existing initial state). split = explicit, same for all 25 variants of a pose.
POSES = [dict(pid=0, x=0.4237, y=0.1013, yaw=0.0071, split="train", region="TRAIN"), dict(pid=1, x=0.4011, y=0.0817, yaw=-0.0953, split="train", region="TRAIN"), dict(pid=2, x=0.4392, y=0.1189, yaw=0.0987, split="train", region="TRAIN"),
         dict(pid=3, x=0.3913, y=0.0911, yaw=0.0462, split="val", region="TRAIN"), dict(pid=4, x=0.4312, y=0.1106, yaw=-0.0531, split="test_id", region="TRAIN"),
         dict(pid=5, x=0.3659, y=0.1007, yaw=0.0113, split="train", region="LOW_A"), dict(pid=6, x=0.3761, y=0.0793, yaw=0.1021, split="train", region="LOW_A"), dict(pid=7, x=0.3702, y=0.1196, yaw=-0.0977, split="test_low_heldout", region="LOW_A"),
         dict(pid=8, x=0.3453, y=0.0999, yaw=0.0049, split="test_ood_low_strict", region="LOW_B_strict"), dict(pid=9, x=0.4711, y=0.1003, yaw=0.0087, split="test_ood_high", region="HIGH")]
# 25 variants = control + 24: depth (5) + close-timing (6) + joint noise (6) + speed (4) + release height (3)
VARIANTS = [dict(vid=0, family="control", level=0)]
VARIANTS += [dict(vid=1 + i, family="depth", level=d) for i, d in enumerate((-0.01, 0.01, 0.02, 0.03, 0.04))]           # extra TCP-forward grasp shift [m] relative to the expert's +0.01 m
VARIANTS += [dict(vid=6 + i, family="timing", level=n) for i, n in enumerate((-10, -5, -2, 2, 5, 10))]                    # close command shifted by n steps (+ = later)
VARIANTS += [dict(vid=12 + i, family="noise", level=sg, seed=sd) for i, (sg, sd) in enumerate([(0.005, 0), (0.005, 1), (0.01, 0), (0.01, 1), (0.02, 0), (0.02, 1)])]   # AR(1) noise on left-arm joint targets [rad]
VARIANTS += [dict(vid=18 + i, family="speed", level=s) for i, s in enumerate((0.8, 0.9, 1.1, 1.25))]                      # time warp of lift..lower
VARIANTS += [dict(vid=22 + i, family="release_height", level=dz) for i, dz in enumerate((-0.01, 0.01, 0.02))]             # release height offset [m]
assert len(VARIANTS) == 25 and len(POSES) == 10

class InterventionExpert:
    """Wraps ScriptedExpert: re-plans with a grasp-depth / release-height offset (subclass-style override of segments()), then applies feed-forward time/noise edits to the ctrl array."""
    def __init__(self, model):
        from policies.baseline import ScriptedExpert
        class _E(ScriptedExpert):
            depth = 0.0; dz = 0.0
            def segments(self, box_xy, yaw):
                f = np.array([np.cos(yaw), np.sin(yaw), 0.0]); out = []
                for name, pos, g, dur in super().segments(box_xy, yaw):
                    pos = np.array(pos, float)
                    if name in ("approach", "close", "grip_hold"): pos = pos + self.depth * f
                    if name == "lift": pos[:2] = pos[:2] + self.depth * f[:2]
                    if name in ("lower", "release", "release_hold"): pos[2] += self.dz
                    out.append((name, pos, g, dur))
                return out
        self.e = _E(model, control_hz=20)
    def plan(self, pose, v):
        self.e.depth = v["level"] if v["family"] == "depth" else 0.0; self.e.dz = v["level"] if v["family"] == "release_height" else 0.0
        pl = self.e.plan((pose["x"], pose["y"]), pose["yaw"]); A = np.array(pl["ctrl"], float); names = pl["phase_names"]; ph = pl["phase"]
        if v["family"] == "timing" and v["level"] != 0:                                    # shift the gripper command (column 7) in time
            n = v["level"]; g = A[:, 7].copy(); idx = np.clip(np.arange(len(g)) - n, 0, len(g) - 1); A[:, 7] = g[idx]
        if v["family"] == "speed":                                                           # time-warp the rows between the start of 'lift' and the start of 'release'
            i0 = int(np.argmax(ph == names.index("lift"))); i1 = int(np.argmax(ph == names.index("release"))); seg = A[i0:i1]; n_new = max(2, int(round(len(seg) / v["level"])))
            tt = np.linspace(0, len(seg) - 1, n_new); seg2 = np.stack([np.interp(tt, np.arange(len(seg)), seg[:, j]) for j in range(16)], 1); A = np.concatenate([A[:i0], seg2, A[i1:]])
        if v["family"] == "noise":
            rng = np.random.default_rng(1000 * v["seed"] + int(round(v["level"] * 1e4))); rho = 0.9; n = np.zeros((len(A), 7)); e = rng.standard_normal((len(A), 7))
            for t in range(1, len(A)): n[t] = rho * n[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
            A[:, :7] += v["level"] * n
        return A, pl

_W = {}
def run_one(job):
    pose, v = job["pose"], job["variant"]; name = f"p{pose['pid']:02d}_v{v['vid']:02d}"; f = OUT / "episodes" / f"{name}.npz"
    if f.exists() and not job.get("force"): return dict(name=name, status="cached")
    E = S.get_env(); env = E["env"]
    if "ie" not in _W: _W["ie"] = InterventionExpert(env.m)
    from control.safety_limits import SafetyLimits
    lim = SafetyLimits.from_model(env.m); A, pl = _W["ie"].plan(pose, v)
    env.set_perturbation(); env.reset((pose["x"], pose["y"]), pose["yaw"]); prev = env.prev_ctrl.copy(); Ex = np.zeros_like(A); n_rng = n_slew = 0
    for t in range(len(A)): Ex[t], r_, s_ = lim.clip_and_slew(A[t], prev, 1 / 20); prev = Ex[t]; n_rng += r_; n_slew += s_
    R = S.simulate(Ex.astype(np.float32), (pose["x"], pose["y"]), pose["yaw"]); R["requested_action"] = A.astype(np.float32); OUT.joinpath("episodes").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **R)
    from eval.metrics import task_labels
    import lab_scene
    lab = task_labels(R["box_pos"], R["tcp_pos"], R["box_speed"], lab_scene.TRAY_C, lab_scene.TABLE_TOP_Z + lab_scene.TRAY_BASE_T)
    return dict(name=name, status="new", pid=pose["pid"], vid=v["vid"], family=v["family"], level=v["level"], seed=v.get("seed"), split=pose["split"], region=pose["region"], box_xy=[pose["x"], pose["y"]], box_yaw=pose["yaw"], T=len(Ex), plan_ok=bool(pl["ok"]), ik_fail_steps=int(pl["ik_fail_steps"]),
                clipped_range=n_rng, slewed=n_slew, h_action=hashlib.sha1(np.ascontiguousarray(Ex.astype(np.float32)).tobytes()).hexdigest()[:16], max_abs_requested_minus_executed=float(np.abs(A - Ex).max()), **{k: (bool(v_) if isinstance(v_, (bool, np.bool_)) else float(v_)) for k, v_ in lab.items()},
                finite=bool(all(np.isfinite(R[k]).all() for k in R if k not in ("finger_contact_a",))))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    jobs = [dict(pose=p, variant=v) for p in POSES for v in VARIANTS]; assert len(jobs) == 250
    t0 = time.time()
    with Pool(a.workers) as pool: R = pool.map(run_one, jobs, chunksize=2)
    cfg = dict(poses=POSES, variants=VARIANTS, eval_seed_label=EVAL_SEED_LABEL, forbidden_seeds=[31, 32, 33], limits="control/safety_limits.SafetyLimits (ctrlrange; 2.0 rad/s arm; 0.2 m/s gripper), same clip_and_slew as the ACT runner", cap_episodes=250)
    prev = json.loads((OUT / "pilot_index.json").read_text())["episodes"] if (OUT / "pilot_index.json").exists() else []
    have = {r["name"]: r for r in prev}; [have.__setitem__(r["name"], r) for r in R if r["status"] == "new"]
    (OUT / "pilot_index.json").write_text(json.dumps(dict(config=cfg, episodes=list(have.values())), indent=1, default=str)); print(f"{len(jobs)} episodes in {time.time()-t0:.0f}s; new {sum(r['status']=='new' for r in R)}")

if __name__ == "__main__": main()
