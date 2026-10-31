#!/usr/bin/env python3
"""Expert demonstrations in a custom box-x region (coverage experiment). Identical to data/collect_demos.py (same expert, cameras, render profile, light jitter, rng order,
labels, file format, meta.json) - only the pose sampler is replaced by a uniform draw from --x/--y/--yaw ranges (same draw order x, y, yaw as policies.baseline.sample_box_pose).
Usage: python data/collect_demos_region.py --n 30 --seed 40 --start 100 --x 0.34 0.38 --out outputs/datasets/demo_ext [--workers 4]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, time, argparse, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)

def worker(args):
    idxs, a = args
    import policies.baseline as PB
    r = dict(x=tuple(a["x"]), y=tuple(a["y"]), yaw=tuple(a["yaw"]))
    PB.sample_box_pose = lambda rng, region="train": tuple(float(rng.uniform(*r[k])) for k in ("x", "y", "yaw"))
    import collect_demos as CD
    return CD.worker((idxs, a))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--x", type=float, nargs=2, required=True); ap.add_argument("--y", type=float, nargs=2, default=[0.07, 0.13]); ap.add_argument("--yaw", type=float, nargs=2, default=[-0.20, 0.20])
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--image-hw", type=int, nargs=2, default=[120, 160]); ap.add_argument("--hz", type=int, default=20)
    ap.add_argument("--light-jitter", type=float, default=0.15); ap.add_argument("--render-profile", default="fast"); ap.add_argument("--resume", action="store_true"); a = ap.parse_args()
    if a.seed in (31, 32, 33): sys.exit("forbidden seed")
    a.out.mkdir(parents=True, exist_ok=True); cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}; cfg["region"] = "custom"
    idxs = list(range(a.start, a.start + a.n)); chunks = [idxs[i::a.workers] for i in range(a.workers)]; t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool: results = sum(pool.map(worker, [(c, cfg) for c in chunks if c]), [])
    results.sort(key=lambda r: r["episode_index"])
    with open(a.out / "episodes.jsonl", "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    import mujoco, subprocess
    import collect_demos as CD
    from control.action_adapter import ActionAdapter
    from lab_env import LabEnv
    env = LabEnv(cameras=()); ad = ActionAdapter(env.m, a.hz); env.close()
    meta = json.loads((ROOT / "outputs/datasets/demo_train_clipped/meta.json").read_text())      # schema/features identical to the original demos
    meta.update(created=time.strftime("%Y-%m-%dT%H:%M:%S"), config=cfg, n_episodes=len(results), n_task_success=int(sum(r["task_success"] for r in results)), wall_s=time.time() - t0)
    meta.pop("post_processing", None); meta.pop("source_dataset", None)
    meta["versions"] = dict(mujoco=mujoco.__version__, numpy=np.__version__, repo_commit=CD.git("rev-parse", "HEAD"), repo_dirty=bool(CD.git("status", "--porcelain")), upstream_commit=CD.git("rev-parse", "HEAD", cwd=ROOT / "sim/openarm_mujoco"))
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1)); print(f"done: {len(results)} episodes, {meta['n_task_success']} task-success, {meta['wall_s']:.0f}s -> {a.out}")

if __name__ == "__main__": main()
