#!/usr/bin/env python3
"""Generate demonstrations from the scripted IK expert with a randomised blue-box pose.
One .npz per episode (readable with plain numpy) + meta.json (schema/config/versions) + episodes.jsonl (per-episode metadata).

Per-step arrays (frame t = observation BEFORE the action is applied; action_t is applied for 1/hz s):
  observation.state (T,16) f32      joint positions of the actuated joints, actuator order (rad; gripper finger slide in metres)
  observation.velocity (T,16) f32   matching joint velocities
  observation.images.<cam> (T,H,W,3) u8     cams = front, wrist_left, wrist_right (policy inputs)
  action (T,16) f32                 executed ctrl command (absolute joint-position targets, actuator order) - see control/action_adapter.py
  timestamp (T,) f32                t/hz seconds
  gripper_state (T,2) f32           [left, right] total opening width = 2*finger1 position (metres)
  phase (T,) i16                    expert phase index (names in episodes.jsonl)
  eval.box_pos (T,3), eval.box_quat (T,4), eval.tcp_pos (T,3), eval.box_speed (T,)   GROUND TRUTH, evaluation/labels only
Usage: python data/collect_demos.py --n 20 --seed 0 --out outputs/datasets/demo_v1 [--region train|unseen] [--workers 4]"""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import sys, json, time, argparse, subprocess, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)

CAMS = ("front", "wrist_left", "wrist_right")

def git(*a, cwd=ROOT):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True); return r.stdout.strip() if r.returncode == 0 else None

def run_episode(env, expert, idx, base_seed, region, light_jitter, out_dir):
    from policies.baseline import sample_box_pose
    from eval.metrics import task_labels
    import lab_scene
    seed = base_seed * 100000 + idx; rng = np.random.default_rng(seed)
    x, y, yaw = sample_box_pose(rng, region)
    light = 1.0 + (rng.uniform(-light_jitter, light_jitter) if light_jitter > 0 else 0.0)
    env.set_perturbation(light_scale=light)
    t0 = time.time(); plan = expert.plan((x, y), yaw); o = env.reset((x, y), yaw)
    rec = {k: [] for k in ("state", "vel", "action", "ts", "phase", "bp", "bq", "tcp", "bs")}; imgs = {c: [] for c in CAMS}
    for t, a in enumerate(plan["ctrl"]):
        i = env.info()
        rec["state"].append(o["state"]); rec["vel"].append(o["velocity"]); rec["action"].append(a.astype(np.float32)); rec["ts"].append(t / env.hz)
        rec["phase"].append(plan["phase"][t]); rec["bp"].append(i["box_pos"]); rec["bq"].append(i["box_quat"]); rec["tcp"].append(i["tcp_pos"]); rec["bs"].append(i["box_speed"])
        for c in CAMS: imgs[c].append(o["images"][c])
        o = env.step(a)
    i = env.info(); fin = (list(rec["bp"]) + [i["box_pos"]], list(rec["tcp"]) + [i["tcp_pos"]], list(rec["bs"]) + [i["box_speed"]])
    lab = task_labels(*fin, lab_scene.TRAY_C, lab_scene.TABLE_TOP_Z + lab_scene.TRAY_BASE_T)
    st = np.array(rec["state"], np.float32)
    arrays = {"observation.state": st, "observation.velocity": np.array(rec["vel"], np.float32), "action": np.array(rec["action"], np.float32),
              "timestamp": np.array(rec["ts"], np.float32), "gripper_state": 2 * st[:, [7, 15]], "phase": np.array(rec["phase"], np.int16),
              "eval.box_pos": np.array(rec["bp"], np.float32), "eval.box_quat": np.array(rec["bq"], np.float32), "eval.tcp_pos": np.array(rec["tcp"], np.float32),
              "eval.box_speed": np.array(rec["bs"], np.float32)}
    arrays.update({f"observation.images.{c}": np.stack(imgs[c]) for c in CAMS})
    np.savez_compressed(out_dir / f"episode_{idx:06d}.npz", **arrays)
    v = np.abs(np.diff(arrays["action"], axis=0)).max(axis=0) * env.hz
    return dict(episode_index=idx, seed=seed, region=region, box_xy=[x, y], box_yaw=yaw, light_scale=light, length=len(st), **lab,
                plan_ok=bool(plan["ok"]), ik_pos_err_max_mm=float(plan["ik_pos_err"].max() * 1e3), ik_rot_err_max=float(plan["ik_rot_err"].max()),
                max_arm_cmd_speed_rad_s=float(np.delete(v, [7, 15]).max()), phase_names=plan["phase_names"], phase_starts=[int(np.argmax(plan["phase"] == k)) for k in range(len(plan["phase_names"]))],
                wall_s=time.time() - t0, file=f"episode_{idx:06d}.npz")

def worker(args):
    idxs, a = args
    os.environ.setdefault("LP_NUM_THREADS", "1")          # one llvmpipe thread per worker; parallelism comes from --workers
    from lab_env import LabEnv
    from policies.baseline import ScriptedExpert
    env = LabEnv(image_hw=tuple(a["image_hw"]), control_hz=a["hz"], render_profile=a["render_profile"], cameras=CAMS); expert = ScriptedExpert(env.m, control_hz=a["hz"])
    out = Path(a["out"]); res = []
    for idx in idxs:
        if a["resume"] and (out / f"episode_{idx:06d}.npz").exists() and (out / f"episode_{idx:06d}.json").exists():
            res.append(json.loads((out / f"episode_{idx:06d}.json").read_text())); continue
        r = run_episode(env, expert, idx, a["seed"], a["region"], a["light_jitter"], out)
        (out / f"episode_{idx:06d}.json").write_text(json.dumps(r)); res.append(r)
        print(f"[ep {idx}] T={r['length']} task_success={r['task_success']} plan_ok={r['plan_ok']} {r['wall_s']:.1f}s", flush=True)
    env.close(); return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/datasets/demo_v1"); ap.add_argument("--region", choices=["train", "unseen"], default="train")
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--image-hw", type=int, nargs=2, default=[120, 160])
    ap.add_argument("--hz", type=int, default=20); ap.add_argument("--light-jitter", type=float, default=0.0); ap.add_argument("--resume", action="store_true"); ap.add_argument("--render-profile", choices=["fast", "studio"], default="fast")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    idxs = list(range(a.start, a.start + a.n)); chunks = [idxs[i::a.workers] for i in range(a.workers)]
    t0 = time.time()
    if a.workers == 1: results = worker((idxs, cfg))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool: results = sum(pool.map(worker, [(c, cfg) for c in chunks if c]), [])
    results.sort(key=lambda r: r["episode_index"])
    with open(a.out / "episodes.jsonl", "a" if a.resume and (a.out / "episodes.jsonl").exists() else "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    import mujoco
    from control.action_adapter import ActionAdapter
    from lab_env import LabEnv
    env = LabEnv(cameras=()); ad = ActionAdapter(env.m, a.hz); env.close()
    meta = dict(schema_version=1, created=time.strftime("%Y-%m-%dT%H:%M:%S"), control_hz=a.hz, sim_timestep=float(env.m.opt.timestep), image_hw=a.image_hw, cameras=list(CAMS),
                features={"observation.state": dict(shape=[16], dtype="float32", names=ad.joint_names, units="rad (arm), m (gripper finger slide, v1)"),
                          "observation.velocity": dict(shape=[16], dtype="float32", names=ad.joint_names),
                          "action": dict(shape=[16], dtype="float32", names=ad.names, semantics="absolute joint-position targets written to data.ctrl, actuator order; includes model-based gravity-comp offset"),
                          **{f"observation.images.{c}": dict(shape=[a.image_hw[0], a.image_hw[1], 3], dtype="uint8") for c in CAMS},
                          "timestamp": dict(shape=[], dtype="float32", units="s"), "gripper_state": dict(shape=[2], dtype="float32", units="m total opening [left,right]")},
                action_range=dict(lo=ad.limits.lo.tolist(), hi=ad.limits.hi.tolist(), max_speed=ad.limits.max_speed.tolist()),
                eval_only=["eval.box_pos", "eval.box_quat", "eval.tcp_pos", "eval.box_speed", "phase"], config=cfg,
                versions=dict(mujoco=mujoco.__version__, numpy=np.__version__, repo_commit=git("rev-parse", "HEAD"), repo_dirty=bool(git("status", "--porcelain")),
                              upstream_commit=git("rev-parse", "HEAD", cwd=ROOT / "sim/openarm_mujoco")),
                n_episodes=len(results), n_task_success=int(sum(r["task_success"] for r in results)), wall_s=time.time() - t0)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1)); print(f"done: {len(results)} episodes, {meta['n_task_success']} task-success, {meta['wall_s']:.0f}s -> {a.out}")

if __name__ == "__main__":
    main()
