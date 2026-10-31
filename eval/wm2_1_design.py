#!/usr/bin/env python3
"""WM-2.1 design artefact (no training): leave-one-action-family-out (LOFO) role manifest derived from the FROZEN wm2 dataset without modifying it.
For a held-out family F: every interventional episode of family F (all poses, all splits) gets role 'heldout_family'; all other episodes keep their original split role.  The frozen splits and files are untouched.
Outputs outputs/wm2_1/lofo_manifest.json (rule + window counts + sha256 of the frozen index it was derived from).  Run (ACT venv): python eval/wm2_1_design.py"""
import json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; W = ROOT / "outputs/wm2/windows"; FAM = {"observational": -1, "control": 0, "depth": 1, "timing": 2, "noise": 3, "speed": 4, "release_height": 5}
SPL = ["train", "val", "test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]
cnt = {sp: {f: int((np.load(W / f"{sp}.npz")["family"] == c).sum()) for f, c in FAM.items()} for sp in SPL}
M = dict(derived_from=dict(index_sha256=hashlib.sha256((ROOT / "outputs/wm2/dataset/index.json").read_bytes()).hexdigest(), frozen_freeze="outputs/wm2/freeze.json"), rule="role(episode) = 'heldout_family' if episode.family == F else episode.split; frozen splits are NOT modified",
         candidates={}, recommended_primary="release_height")
for F in ("depth", "timing", "noise", "speed", "release_height"):
    tr = sum(cnt["train"][f] for f in FAM if f != F); va = sum(cnt["val"][f] for f in FAM if f != F); held = {sp: cnt[sp][F] for sp in SPL}
    M["candidates"][F] = dict(train_windows_without_family=tr, val_windows_without_family=va, train_windows_removed=cnt["train"][F], heldout_family_windows_by_original_split=held, heldout_family_windows_total=int(sum(held.values())),
                             evaluation_pairs={"pose_ID + action OOD": held["test_id"], "pose OOD-low (heldout+strict) + action OOD": held["test_low_heldout"] + held["test_ood_low_strict"], "pose OOD-high + action OOD": held["test_ood_high"], "train/val poses + action OOD": held["train"] + held["val"]})
M["window_counts_by_split_and_family"] = cnt
(ROOT / "outputs/wm2_1").mkdir(exist_ok=True); (ROOT / "outputs/wm2_1/lofo_manifest.json").write_text(json.dumps(M, indent=1)); print(json.dumps({k: (v["train_windows_without_family"], v["heldout_family_windows_total"]) for k, v in M["candidates"].items()}))
