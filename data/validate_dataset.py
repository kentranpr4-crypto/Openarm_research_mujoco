#!/usr/bin/env python3
"""Independent dataset reader/validator + visualiser. Depends ONLY on numpy, Pillow and imageio (no MuJoCo, no LeRobot, no
repo imports), so a dataset can be inspected on any machine. Exit code 1 if any schema check fails.
Usage: python data/validate_dataset.py outputs/datasets/demo_train [--viz-episode 0] [--viz-dir DIR]"""
import sys, json, argparse, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import imageio.v2 as iio

def load_episode(path):
    with np.load(path) as z: return {k: z[k] for k in z.files}

def validate(root):
    root = Path(root); meta = json.loads((root / "meta.json").read_text())
    eps = [json.loads(l) for l in (root / "episodes.jsonl").read_text().splitlines() if l.strip()]
    hz, feats = meta["control_hz"], meta["features"]; lo, hi = np.array(meta["action_range"]["lo"]), np.array(meta["action_range"]["hi"])
    expect = set(feats) - {"timestamp"} | {"timestamp"} | set(meta["eval_only"])
    problems, stats = [], dict(n_episodes=len(eps), lengths=[], track_err=[], labels=dict(pick=0, place=0, task=0))
    for e in eps:
        f = root / e["file"]
        if not f.exists(): problems.append((e["episode_index"], "missing file")); continue
        z = load_episode(f); T = e["length"]; tag = e["episode_index"]
        if set(z) != expect: problems.append((tag, f"keys differ: missing {sorted(expect - set(z))} extra {sorted(set(z) - expect)}"))
        for k, spec in feats.items():
            if k not in z: continue
            want = tuple([T] + spec["shape"])
            if z[k].shape != want: problems.append((tag, f"{k} shape {z[k].shape} != {want}"))
            if str(z[k].dtype) != spec["dtype"]: problems.append((tag, f"{k} dtype {z[k].dtype} != {spec['dtype']}"))
        for k, v in z.items():
            if v.dtype.kind == "f" and not np.isfinite(v).all(): problems.append((tag, f"{k} has non-finite values"))
        ts = z["timestamp"].astype(np.float64)
        if abs(ts[0]) > 1e-6 or np.abs(np.diff(ts) - 1.0 / hz).max() > 1e-4: problems.append((tag, "timestamps not 0, 1/hz, 2/hz ..."))
        a = z["action"]
        if (a < lo - 1e-6).any() or (a > hi + 1e-6).any(): problems.append((tag, "action outside ctrlrange"))
        for c in meta["cameras"]:
            im = z[f"observation.images.{c}"]
            if im.std() < 1.0: problems.append((tag, f"camera {c} looks constant/black"))
        if e["task_success"] and not (e["pick_success"] and e["place_success"]): problems.append((tag, "label inconsistency"))
        # state follows the previous command: |a[t-1]-s[t]| (a sanity measure, not a pass/fail)
        stats["track_err"].append(float(np.abs(a[:-1, :7] - z["observation.state"][1:, :7]).mean()))
        stats["lengths"].append(T)
        for k, n in (("pick_success", "pick"), ("place_success", "place"), ("task_success", "task")): stats["labels"][n] += int(e[k])
    npz = sorted(p.name for p in root.glob("episode_*.npz"))
    if len(npz) != len(eps): problems.append(("dataset", f"{len(npz)} npz files vs {len(eps)} episodes.jsonl rows"))
    rep = dict(root=str(root), ok=not problems, n_problems=len(problems), problems=problems[:30], n_episodes=len(eps), length_min_mean_max=[min(stats["lengths"]), float(np.mean(stats["lengths"])), max(stats["lengths"])] if eps else None,
               mean_abs_cmd_vs_next_state_arm_rad=float(np.mean(stats["track_err"])) if eps else None, labels=stats["labels"],
               box_xy_range=[[min(e["box_xy"][i] for e in eps), max(e["box_xy"][i] for e in eps)] for i in (0, 1)] if eps else None,
               versions=meta["versions"], control_hz=hz, cameras=meta["cameras"])
    (root / "validation.json").write_text(json.dumps(rep, indent=1)); return rep

def visualise(root, ep_idx=0, out=None):
    root = Path(root); out = Path(out or root / "viz"); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "meta.json").read_text()); e = [json.loads(l) for l in (root / "episodes.jsonl").read_text().splitlines()][ep_idx]
    z = load_episode(root / e["file"]); cams = meta["cameras"]; T = e["length"]
    # contact sheet: one column per phase start, one row per camera
    starts = e["phase_starts"]; names = e["phase_names"]; h, w = z[f"observation.images.{cams[0]}"].shape[1:3]
    sheet = Image.new("RGB", (w * len(starts), (h + 14) * len(cams)))
    d = ImageDraw.Draw(sheet)
    for r, c in enumerate(cams):
        for k, s in enumerate(starts):
            sheet.paste(Image.fromarray(z[f"observation.images.{c}"][s]), (k * w, r * (h + 14) + 14)); d.text((k * w + 2, r * (h + 14) + 1), f"{c}:{names[k]}", fill=(255, 255, 255))
    sheet.save(out / f"episode{ep_idx}_phases.png")
    frames = [np.concatenate([z[f"observation.images.{c}"][t] for c in cams], 1) for t in range(T)]
    iio.mimsave(out / f"episode{ep_idx}.mp4", frames, fps=meta["control_hz"], macro_block_size=1)
    # action trajectory plot (each of the 16 dims scaled to its ctrlrange)
    lo, hi = np.array(meta["action_range"]["lo"]), np.array(meta["action_range"]["hi"]); a = (z["action"] - lo) / (hi - lo + 1e-9)
    s = (z["observation.state"] - lo) / (hi - lo + 1e-9); W, H = 640, 360; img = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(img)
    for j in range(16):
        col = tuple(int(x) for x in np.array([(j * 53) % 200, (j * 97) % 200, (j * 29) % 200]))
        dr.line([(t / (T - 1) * (W - 1), H - 10 - np.clip(a[t, j], 0, 1) * (H - 20)) for t in range(T)], fill=col, width=1)
    for st in starts: dr.line([(st / (T - 1) * (W - 1), 0), (st / (T - 1) * (W - 1), H)], fill=(200, 200, 200))
    img.save(out / f"episode{ep_idx}_actions.png")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--viz-episode", type=int, default=0); ap.add_argument("--viz-dir"); ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args(); r = validate(a.root)
    print(json.dumps({k: r[k] for k in r if k not in ("versions",)}, indent=1)[:2500])
    if not a.no_viz and r["n_episodes"]: print("viz ->", visualise(a.root, a.viz_episode, a.viz_dir))
    sys.exit(0 if r["ok"] else 1)
