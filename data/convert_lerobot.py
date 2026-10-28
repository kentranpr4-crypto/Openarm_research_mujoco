#!/usr/bin/env python3
"""Raw .npz demos -> LeRobotDataset (v0.6.1 API). Run inside the ACT venv (/home/user/.venvs/act).
Policy-visible features only: observation.state, observation.velocity, observation.images.<cam>, action. Ground-truth
`eval.*` arrays (box/TCP pose) are deliberately NOT converted, so a policy trained on this dataset cannot see them.
Episode-level train/val split is written to <out>/openarm_splits.json (seeded). By default only task-successful,
cleanly-planned episodes are converted.
Usage: python data/convert_lerobot.py outputs/datasets/demo_train outputs/lerobot/openarm_lab_pickplace [--val-frac 0.1]"""
import sys, json, shutil, argparse, numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "pick the blue box with the left arm and place it in the tray"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("raw", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--val-frac", type=float, default=0.1); ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--include-failed", action="store_true"); ap.add_argument("--max-episodes", type=int); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    meta = json.loads((a.raw / "meta.json").read_text()); eps = [json.loads(l) for l in (a.raw / "episodes.jsonl").read_text().splitlines() if l.strip()]
    keep = [e for e in eps if a.include_failed or (e["task_success"] and e["plan_ok"])][: a.max_episodes]
    if a.out.exists():
        if not a.overwrite: sys.exit(f"{a.out} exists (use --overwrite; nothing is deleted otherwise)")
        shutil.rmtree(a.out)
    H, W = meta["image_hw"]; F = meta["features"]
    features = {"observation.state": dict(dtype="float32", shape=(16,), names=F["observation.state"]["names"]),
                "observation.velocity": dict(dtype="float32", shape=(16,), names=F["observation.velocity"]["names"]),
                "action": dict(dtype="float32", shape=(16,), names=F["action"]["names"]),
                **{f"observation.images.{c}": dict(dtype="image", shape=(H, W, 3), names=["height", "width", "channels"]) for c in meta["cameras"]}}
    ds = LeRobotDataset.create(repo_id="local/openarm_lab_pickplace", fps=meta["control_hz"], features=features, root=a.out,
                               robot_type="openarm_v1_bimanual", use_videos=False)
    for e in keep:
        with np.load(a.raw / e["file"]) as z:
            for t in range(e["length"]):
                fr = {"observation.state": z["observation.state"][t], "observation.velocity": z["observation.velocity"][t], "action": z["action"][t], "task": TASK}
                for c in meta["cameras"]: fr[f"observation.images.{c}"] = z[f"observation.images.{c}"][t]
                ds.add_frame(fr)
        ds.save_episode(); print(f"converted episode {e['episode_index']} ({e['length']} frames)", flush=True)
    ds.finalize()
    rng = np.random.default_rng(a.split_seed); idx = np.arange(len(keep)); rng.shuffle(idx)
    nval = max(1, int(round(len(keep) * a.val_frac))) if len(keep) > 1 and a.val_frac > 0 else 0
    split = dict(val=sorted(int(i) for i in idx[:nval]), train=sorted(int(i) for i in idx[nval:]), lerobot_to_raw_episode={i: e["episode_index"] for i, e in enumerate(keep)},
                 source=str(a.raw), n_dropped=len(eps) - len(keep), split_seed=a.split_seed, note="indices are LeRobot episode indices; split is by EPISODE")
    (a.out / "openarm_splits.json").write_text(json.dumps(split, indent=1)); print("splits:", len(split["train"]), "train /", len(split["val"]), "val ->", a.out)

if __name__ == "__main__":
    main()
