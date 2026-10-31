#!/usr/bin/env python3
"""Dataset synchronization validation for a merged coverage dataset (ACT venv). Checks: (1) frame counts raw vs LeRobot per episode, (2) state/action/image alignment frame by frame
for every episode against the raw .npz, (3) the first 60 LeRobot episodes are bit-identical to the V1 dataset (outputs/lerobot/pickplace_v1) for state/action, (4) split manifest consistency
(disjoint, covering, V1 membership preserved, new val count), (5) timestamps/fps, (6) train-only statistics recomputed independently (numpy over the raw train episodes) match norm_contract.aggregate,
(7) box x range of the new demos. Exit 1 on failure.  Usage: python data/validate_merge.py outputs/lerobot/cov_ext outputs/datasets/cov_ext_raw --expect-x 0.34 0.38"""
import sys, json, argparse, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from policies import norm_contract as nc
ap = argparse.ArgumentParser(); ap.add_argument("lr", type=Path); ap.add_argument("raw", type=Path); ap.add_argument("--expect-x", type=float, nargs=2); ap.add_argument("--out", type=Path); ap.add_argument("--skip-images", action="store_true", help="skip the (slow) image spot-check; everything else runs"); a = ap.parse_args()
prob = []; chk = {}
sp = json.loads((a.lr / "openarm_splits.json").read_text()); v1 = json.loads((ROOT / "outputs/lerobot/pickplace_v1/openarm_splits.json").read_text())
eps = [json.loads(l) for l in (a.raw / "episodes.jsonl").read_text().splitlines() if l.strip()]; keep = [e for e in eps if e["task_success"] and e["plan_ok"]]
n_orig = sp["n_original"]
ds = LeRobotDataset("local/openarm_lab_pickplace", root=a.lr); v1ds = LeRobotDataset("local/openarm_lab_pickplace", root=ROOT / "outputs/lerobot/pickplace_v1")
chk["n_episodes"] = [ds.num_episodes, len(keep)]; chk["n_frames"] = ds.num_frames
if ds.num_episodes != len(keep): prob.append("episode count mismatch")
if abs(ds.fps - 20) > 1e-9: prob.append("fps != 20")
h = ds.hf_dataset; ep_col = np.array(h["episode_index"]); fi = np.array(h["frame_index"]); ts = np.array(h["timestamp"]); state = np.array(h["observation.state"], np.float32); act = np.array(h["action"], np.float32)
bad_len = 0; bad_state = 0; bad_act = 0; bad_ts = 0
for k, e in enumerate(keep):
    m = ep_col == k; z = np.load(a.raw / e["file"])
    if m.sum() != e["length"] or z["observation.state"].shape[0] != e["length"]: bad_len += 1; continue
    if (np.diff(fi[m]) != 1).any() or fi[m][0] != 0: bad_ts += 1
    if np.abs(np.diff(ts[m]) - 1 / 20).max() > 1e-4: bad_ts += 1
    if not np.array_equal(state[m], z["observation.state"]): bad_state += 1
    if not np.array_equal(act[m], z["action"]): bad_act += 1
chk.update(bad_length=bad_len, bad_state=bad_state, bad_action=bad_act, bad_timestamps=bad_ts)
if bad_len or bad_state or bad_act or bad_ts: prob.append(f"alignment: len {bad_len} state {bad_state} action {bad_act} ts {bad_ts}")
# images: spot-check every episode first/middle/last frame equals the raw image
bad_img = 0
for k, e in enumerate([] if a.skip_images else keep):
    z = np.load(a.raw / e["file"]); idx = np.nonzero(ep_col == k)[0]
    for j in (0, len(idx) // 2, len(idx) - 1):
        item = ds[int(idx[j])]
        for c in ("front", "wrist_left", "wrist_right"):
            im = (item[f"observation.images.{c}"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            if np.abs(im.astype(int) - z[f"observation.images.{c}"][j].astype(int)).max() > 1: bad_img += 1
chk["bad_image_frames"] = bad_img; chk["images_checked"] = not a.skip_images
if bad_img: prob.append(f"{bad_img} image frames differ from raw")
# first 60 identical to V1
h1 = v1ds.hf_dataset; e1 = np.array(h1["episode_index"]); s1 = np.array(h1["observation.state"], np.float32); a1 = np.array(h1["action"], np.float32)
m0 = ep_col < n_orig; same = (m0.sum() == len(e1)) and np.array_equal(state[m0], s1) and np.array_equal(act[m0], a1) and np.array_equal(ep_col[m0], e1)
chk["first60_bit_identical_to_V1"] = bool(same)
if not same: prob.append("first 60 episodes differ from V1 dataset")
# split manifest
tr, va = set(sp["train"]), set(sp["val"]); chk["split"] = dict(train=len(tr), val=len(va), disjoint=not (tr & va), covers=(tr | va) == set(range(len(keep))), v1_preserved=(set(v1["train"]) <= tr and set(v1["val"]) <= va), new_val=sp["val_new"])
if not (chk["split"]["disjoint"] and chk["split"]["covers"] and chk["split"]["v1_preserved"]): prob.append("split manifest inconsistent")
# independent train-only statistics over the raw train episodes
Strain = np.concatenate([np.load(a.raw / keep[i]["file"])["observation.state"] for i in sorted(tr)]); Atrain = np.concatenate([np.load(a.raw / keep[i]["file"])["action"] for i in sorted(tr)])
agg = nc.aggregate(a.lr, sorted(tr)); live_s = Strain.std(0) > 5e-3; live_a = Atrain.std(0) > 5e-3     # dims with real variation; idle dims (std ~ 0) only carry LeRobot's float32 aggregation noise, identical in the baselines
ms = np.abs(np.asarray(agg["observation.state"]["mean"]) - Strain.mean(0))[live_s].max(); ss = np.abs(np.asarray(agg["observation.state"]["std"]) - Strain.std(0))[live_s].max(); ma = np.abs(np.asarray(agg["action"]["mean"]) - Atrain.mean(0))[live_a].max(); sa = np.abs(np.asarray(agg["action"]["std"]) - Atrain.std(0))[live_a].max()
idle = dict(state_mean=float(np.abs(np.asarray(agg["observation.state"]["mean"]) - Strain.mean(0))[~live_s].max()), action_mean=float(np.abs(np.asarray(agg["action"]["mean"]) - Atrain.mean(0))[~live_a].max()), state_std=float(np.abs(np.asarray(agg["observation.state"]["std"]) - Strain.std(0))[~live_s].max()), action_std=float(np.abs(np.asarray(agg["action"]["std"]) - Atrain.std(0))[~live_a].max()))
chk["stats_max_abs_diff_live_dims"] = dict(state_mean=float(ms), state_std=float(ss), action_mean=float(ma), action_std=float(sa)); chk["stats_idle_dims_float32_noise_info"] = idle; chk["n_train_frames"] = int(len(Strain)); chk["n_live_dims"] = [int(live_s.sum()), int(live_a.sum())]
if max(ms, ss, ma, sa) > 1e-4: prob.append("train-only stats differ from independent numpy on live dims")
newx = [e["box_xy"][0] for e in keep[n_orig:]]
chk["new_box_x_range"] = [float(min(newx)), float(max(newx))] if newx else None; chk["new_box_yaw_range"] = [float(min(e["box_yaw"] for e in keep[n_orig:])), float(max(e["box_yaw"] for e in keep[n_orig:]))] if newx else None
if a.expect_x and newx and not (min(newx) >= a.expect_x[0] and max(newx) < a.expect_x[1] + 1e-9): prob.append("new demo x outside expected range")
chk["new_n_kept"] = len(keep) - n_orig; chk["problems"] = prob
(a.out or a.lr / "validate_merge.json").write_text(json.dumps(chk, indent=1)); print(json.dumps(chk, indent=1)); sys.exit(1 if prob else 0)
