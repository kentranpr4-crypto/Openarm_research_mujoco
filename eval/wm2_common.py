#!/usr/bin/env python3
"""WM-2 shared data layer: builds per-split window arrays (contract v1, H=4, K=20, stride 5) from the frozen dataset outputs/wm2/dataset, plus the paired-intervention evaluation windows.
Only replay_ok episodes produce windows.  Arrays are float32; metadata arrays identify group (initial state), episode, t, kind, family, level, pose.  Run: python eval/wm2_common.py  (builds outputs/wm2/windows/*.npz)"""
import sys, json, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "data"))
import wm1_windows as W
DS = ROOT / "outputs/wm2/dataset"; WIN = ROOT / "outputs/wm2/windows"; H, K = W.H, W.K
FAMS = {"control": 0, "depth": 1, "timing": 2, "noise": 3, "speed": 4, "release_height": 5}
SPLITS = ["train", "val", "test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]
def load_index(): return json.loads((DS / "index.json").read_text())
def group_ids(idx):
    keys = sorted({tuple(e["group"]) for e in idx}); return {k: i for i, k in enumerate(keys)}
def _ep(e): z = np.load(DS / "episodes" / f"{e['eid']}.npz"); return {k: z[k] for k in z.files}
def window_dict(ep, t): X, Y = W.get_window(ep, t); return X, Y
def build_split(split, idx=None, gid=None):
    idx = idx or load_index(); gid = gid or group_ids(idx); E = [e for e in idx if e["split"] == split and e["replay_ok"]]; cols = {k: [] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e", "group", "episode", "t", "active", "kind", "family", "level", "pid", "pose")}
    for ei, e in enumerate(E):
        ep = _ep(e); T = len(ep["a"])
        for t in W.window_starts(T):
            X, Y = window_dict(ep, t)
            for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut"): cols[k].append(X[k])
            for k in ("Y_s", "Y_b", "Y_c", "Y_e"): cols[k].append(Y[k])
            act = bool(np.linalg.norm(ep["b"][t + K, :3] - ep["b"][t, :3]) > 0.005 or (ep["a"][t:t + K, 7] < 0.028).any())
            cols["group"].append(gid[tuple(e["group"])]); cols["episode"].append(idx.index(e) if False else ei); cols["t"].append(t); cols["active"].append(act); cols["kind"].append(1 if e["kind"] == "interventional" else 0)
            cols["family"].append(FAMS[e["family"]] if e["family"] else -1); cols["level"].append(e["level"] if e["level"] is not None else np.nan); cols["pid"].append(e["pid"] if e["pid"] is not None else -1); cols["pose"].append([e["box_xy"][0], e["box_xy"][1], e["box_yaw"]])
    out = {k: np.asarray(v, np.float32 if k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e", "level", "pose") else (bool if k == "active" else np.int64)) for k, v in cols.items()}
    out["eid"] = np.array([E[i]["eid"] for i in out["episode"]]); return out
def build_pairs(idx=None, gid=None):
    """Paired-intervention windows: for every held-out interventional pose and every variant, the window at t_pair = control close step - 10 (same absolute t for all 25 variants)."""
    idx = idx or load_index(); gid = gid or group_ids(idx); I = [e for e in idx if e["kind"] == "interventional" and e["split"].startswith("test")]; byp = {}
    for e in I: byp.setdefault(e["pid"], {})[e["idx"]] = e
    cols = {k: [] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e", "group", "pid", "vid", "family", "level", "t", "split_code")}; names = {}
    for pid, vs in byp.items():
        ctl = vs[0]; cl = ctl["outcomes"]["close_step"]
        if cl is None or not ctl["replay_ok"]: continue
        t = int(min(max(cl - 10, W.H), len(_ep(ctl)["a"]) - 1 - W.K))
        for vid, e in vs.items():
            if not e["replay_ok"]: continue
            ep = _ep(e)
            if t > len(ep["a"]) - 1 - W.K: continue
            X, Y = window_dict(ep, t)
            for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut"): cols[k].append(X[k])
            for k in ("Y_s", "Y_b", "Y_c", "Y_e"): cols[k].append(Y[k])
            cols["group"].append(gid[tuple(e["group"])]); cols["pid"].append(pid); cols["vid"].append(vid); cols["family"].append(FAMS[e["family"]]); cols["level"].append(e["level"]); cols["t"].append(t); cols["split_code"].append(SPLITS.index(e["split"])); names[pid] = e["split"]
    return {k: np.asarray(v, np.float32 if k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e", "level") else np.int64) for k, v in cols.items()}
if __name__ == "__main__":
    WIN.mkdir(parents=True, exist_ok=True); idx = load_index(); gid = group_ids(idx)
    for sp in SPLITS: d = build_split(sp, idx, gid); np.savez_compressed(WIN / f"{sp}.npz", **d); print(sp, len(d["t"]), "active", int(d["active"].sum()), "groups", len(set(d["group"].tolist())), "interventional windows", int(d["kind"].sum()))
    p = build_pairs(idx, gid); np.savez_compressed(WIN / "pairs.npz", **p); print("pairs", len(p["t"]), "poses", len(set(p["pid"].tolist())))
    json.dump({str(k): v for k, v in gid.items()}, open(WIN / "group_ids.json", "w"))
