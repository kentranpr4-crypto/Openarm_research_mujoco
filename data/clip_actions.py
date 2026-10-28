#!/usr/bin/env python3
"""Make a copy of a raw demo dataset whose `action` is clipped to the actuator ctrlrange (meta.action_range).
Why: early demos recorded gravity-compensated targets slightly outside ctrlrange (left_joint2, <=0.011 rad); MuJoCo clamps ctrl
itself, so the executed command was already the clipped one - this makes the stored action equal the executed command.
The source dataset is left untouched; lineage is written to meta.json['post_processing'].  Usage: python data/clip_actions.py SRC DST"""
import sys, json, shutil, numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists(): sys.exit(f"{dst} exists; refusing to overwrite")
dst.mkdir(parents=True)
meta = json.loads((src / "meta.json").read_text()); lo, hi = np.array(meta["action_range"]["lo"]), np.array(meta["action_range"]["hi"])
stats = dict(episodes_modified=0, values_clipped=0, max_abs_change=0.0)
for f in sorted(src.glob("episode_*.npz")):
    with np.load(f) as z: d = {k: z[k] for k in z.files}
    a = np.clip(d["action"], lo, hi).astype(np.float32); ch = np.abs(a - d["action"]); n = int((ch > 1e-9).sum())
    if n: stats["episodes_modified"] += 1; stats["values_clipped"] += n; stats["max_abs_change"] = max(stats["max_abs_change"], float(ch.max()))
    d["action"] = a; np.savez_compressed(dst / f.name, **d)
for f in list(src.glob("episode_*.json")) + [src / "episodes.jsonl"]: shutil.copy(f, dst / f.name)
meta["post_processing"] = dict(step="clip_actions.py", source=str(src), clipped_to="meta.action_range", **stats); meta["source_dataset"] = str(src)
(dst / "meta.json").write_text(json.dumps(meta, indent=1)); print(stats)
