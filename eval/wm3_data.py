#!/usr/bin/env python3
"""WM-3 data additions (no change to frozen wm2 data): phase-aligned family windows for Gate v2 G4 (held-out interventional poses of the FROZEN wm2 test splits) and train-only contact scales.
family windows: depth t=close-10 | speed t=(close+release)//2-10 | timing t=close-25 | release_height t=release-10 | noise t=close+10 (t clipped to the valid range); control (vid 0) included for every family.
Writes outputs/wm3/family_windows.npz and outputs/wm3/contact_scales.json.  Run (ACT venv): python eval/wm3_data.py"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "data"))
import wm2_common as C, wm1_windows as W1
FAM = {"depth": (1, lambda c, r: c - 10), "speed": (4, lambda c, r: (c + r) // 2 - 10), "timing": (2, lambda c, r: c - 25), "release_height": (5, lambda c, r: r - 10), "noise": (3, lambda c, r: c + 10)}
def build():
    idx = C.load_index(); I = [e for e in idx if e["kind"] == "interventional" and e["split"].startswith("test") and e["replay_ok"]]; byp = {}
    for e in I: byp.setdefault(e["pid"], {})[e["idx"]] = e
    out = {}
    for name, (fid, tf) in FAM.items():
        rows = {k: [] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e", "pid", "vid", "level", "t", "split_code")}
        for pid, vs in byp.items():
            c = vs[0]; cl, rl = c["outcomes"]["close_step"], c["outcomes"]["release_step"]
            if cl is None or rl is None: continue
            t = tf(cl, rl)
            for vid, e in vs.items():
                if e["family"] not in ("control", name): continue
                ep = C._ep(e); tt = int(min(max(t, W1.H), len(ep["a"]) - 1 - W1.K)); X, Y = W1.get_window(ep, tt)
                for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut"): rows[k].append(X[k])
                for k in ("Y_s", "Y_b", "Y_c", "Y_e"): rows[k].append(Y[k])
                rows["pid"].append(pid); rows["vid"].append(vid); rows["level"].append(e["level"] if e["level"] is not None else 0.0); rows["t"].append(tt); rows["split_code"].append(C.SPLITS.index(e["split"]))
        for k, v in rows.items(): out[f"{name}/{k}"] = np.asarray(v, np.float32 if k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e", "level") else np.int64)
    return out
if __name__ == "__main__":
    fw = build(); np.savez_compressed(ROOT / "outputs/wm3/family_windows.npz", **fw); print({n: int(len(fw[f"{n}/pid"])) for n in FAM})
    tr = np.load(ROOT / "outputs/wm2/windows/train.npz"); Yc = tr["Y_c"].reshape(-1, 7); sc = dict(channels=["finger_force", "n_contacts", "a_contact", "a_valid", "palm_force", "table_force", "tray_force"], std=Yc.std(0).tolist(), used_for_gate=[0, 1, 4, 5, 6], source="train windows, all steps")
    (ROOT / "outputs/wm3/contact_scales.json").write_text(json.dumps(sc, indent=1)); print(sc["std"])
