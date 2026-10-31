#!/usr/bin/env python3
"""Build a merged coverage-experiment dataset WITHOUT touching any existing data:
  raw merge  : <out_raw>/ = symlinks to the 60 original demos (demo_train_clipped, first, same order) + the 30 new demos, merged episodes.jsonl, meta.json
  LeRobot    : <out_lr>/  via data/convert_lerobot.py (unchanged), then openarm_splits.json is REPLACED (in the new dir only) by the controlled split:
               original 60 keep the V1 split (outputs/lerobot/pickplace_v1/openarm_splits.json); of the new demos 27/30 train, 3 val (default_rng(0) over sorted new indices).
Usage (ACT venv): python data/build_cov_dataset.py --new outputs/datasets/demo_ext --out-raw outputs/datasets/cov_ext_raw --out-lr outputs/lerobot/cov_ext"""
import sys, json, argparse, subprocess, os, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; ORIG = ROOT / "outputs/datasets/demo_train_clipped"; V1 = ROOT / "outputs/lerobot/pickplace_v1"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", type=Path, required=True); ap.add_argument("--out-raw", type=Path, required=True); ap.add_argument("--out-lr", type=Path, required=True); ap.add_argument("--n-val-new", type=int, default=3); a = ap.parse_args()
    for d in (a.out_raw, a.out_lr):
        if d.exists(): sys.exit(f"{d} exists; refusing to overwrite")
    a.out_raw.mkdir(parents=True)
    orig = [json.loads(l) for l in (ORIG / "episodes.jsonl").read_text().splitlines() if l.strip()]; new = [json.loads(l) for l in (a.new / "episodes.jsonl").read_text().splitlines() if l.strip()]
    for e in orig:
        for ext in ("npz", "json"): os.symlink((ORIG / e["file"]).with_suffix("." + ext).resolve(), a.out_raw / Path(e["file"]).with_suffix("." + ext).name)
    for e in new:
        for ext in ("npz", "json"): os.symlink((a.new / e["file"]).with_suffix("." + ext).resolve(), a.out_raw / Path(e["file"]).with_suffix("." + ext).name)
    (a.out_raw / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in orig + new) + "\n")
    meta = json.loads((ORIG / "meta.json").read_text()); meta.update(n_episodes=len(orig) + len(new), n_task_success=sum(e["task_success"] for e in orig + new), merged_from=[str(ORIG), str(a.new)], note="symlink merge; originals first")
    (a.out_raw / "meta.json").write_text(json.dumps(meta, indent=1))
    subprocess.run([sys.executable, str(ROOT / "data/convert_lerobot.py"), str(a.out_raw), str(a.out_lr)], check=True)
    sp = json.loads((a.out_lr / "openarm_splits.json").read_text()); v1 = json.loads((V1 / "openarm_splits.json").read_text())
    keep = [e for e in orig + new if e["task_success"] and e["plan_ok"]]; n_orig = len(orig)
    assert [e["episode_index"] for e in keep[:n_orig]] == [e["episode_index"] for e in orig], "originals must come first and unchanged"
    new_idx = sorted(i for i in range(n_orig, len(keep))); rng = np.random.default_rng(0); val_new = sorted(int(i) for i in rng.choice(new_idx, size=a.n_val_new, replace=False))
    train = sorted(v1["train"] + [i for i in new_idx if i not in val_new]); val = sorted(v1["val"] + val_new)
    split = dict(val=val, train=train, lerobot_to_raw_episode={i: e["episode_index"] for i, e in enumerate(keep)}, source=str(a.out_raw), n_original=n_orig, n_new=len(new_idx), val_new=val_new,
                 rule="originals keep V1 split; new demos: n_val_new chosen with default_rng(0) over sorted new indices", split_seed=0, note="indices are LeRobot episode indices; split is by EPISODE")
    (a.out_lr / "openarm_splits.json").write_text(json.dumps(split, indent=1)); print("split:", len(train), "train /", len(val), "val; new val", val_new)

if __name__ == "__main__": main()
