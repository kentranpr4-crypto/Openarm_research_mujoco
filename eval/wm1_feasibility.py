#!/usr/bin/env python3
"""WM-1 baseline feasibility (NO learned model): non-learned baselines (persistence, constant-velocity) on the WM-1 windows per split and horizon, label positives per split/group,
and sample-size feasibility for the planned gates.  Run (ACT venv): python eval/wm1_feasibility.py  -> outputs/wm1/feasibility.json"""
import sys, json, numpy as np
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "data"))
import wm1_windows as W
DS = ROOT / "outputs/wm1/dataset"; idx = [e for e in json.loads((DS / "index.json").read_text()) if e["replay_ok"]]; HZ = (1, 5, 10, 20)
acc = defaultdict(lambda: defaultdict(list)); pos = defaultdict(lambda: defaultdict(set)); posw = defaultdict(lambda: defaultdict(int)); nw = defaultdict(int); grp_err = defaultdict(lambda: defaultdict(list))
for e in idx:
    z = np.load(DS / "episodes" / f"{e['eid']}.npz"); s, b, a, c = z["s"], z["b"], z["a"], z["c"]; sp = e["split"]; g = tuple(e["group"])
    for t in W.window_starts(len(a)):
        X, Y = W.get_window(dict(s=s, b=b, a=a, c=c), t); nw[sp] += 1; active = bool(np.linalg.norm(b[t + W.K, :3] - b[t, :3]) > 0.005 or (a[t:t + W.K, 7] < 0.028).any()); v = s[t] - s[t - 1]
        for k in HZ:
            d_s = np.abs(s[t + k, :7] - s[t, :7]).mean(); d_g = abs(s[t + k, 7] - s[t, 7]); d_b = np.linalg.norm(b[t + k, :3] - b[t, :3]) * 1000; d_a = abs(b[t + k, 6] - b[t, 6]) * 100
            cv = np.abs(s[t + k, :7] - (s[t, :7] + k * v[:7])).mean(); cvb = np.linalg.norm(b[t + k, :3] - (b[t, :3] + k * (b[t, :3] - b[t - 1, :3]))) * 1000
            for tag, flag in (("all", True), ("active", active)):
                if not flag: continue
                for name, val in (("persist_joint_mae_rad", d_s), ("persist_gripper_mae_m", d_g), ("persist_box_pos_mm", d_b), ("persist_aoffset_cm", d_a), ("constvel_joint_mae_rad", cv), ("constvel_box_pos_mm", cvb)): acc[(sp, tag, k)][name].append(val)
            if k == 20 and active: grp_err[sp][g].append(d_b)
        for j, nm in enumerate(("close", "palm", "lifted", "placed")):
            if Y["Y_e"][j] > 0: pos[sp][nm].add(g); posw[sp][nm] += 1
R = dict(windows_per_split=dict(nw), baselines={}, positives={}, groups_with_positives={}, persistence_box_pos_mm_k20_group_sd={})
for (sp, tag, k), d in sorted(acc.items()): R["baselines"].setdefault(sp, {}).setdefault(tag, {})[f"K={k}"] = {n: dict(mean=float(np.mean(v)), p50=float(np.median(v)), n=len(v)) for n, v in d.items()}
for sp in nw: R["positives"][sp] = dict(posw[sp]); R["groups_with_positives"][sp] = {k: len(v) for k, v in pos[sp].items()}; gm = [np.mean(v) for v in grp_err[sp].values()]; R["persistence_box_pos_mm_k20_group_sd"][sp] = dict(mean=float(np.mean(gm)) if gm else None, sd=float(np.std(gm, ddof=1)) if len(gm) > 1 else None, n_groups=len(gm))
# groups per split (for bootstrap feasibility)
gs = defaultdict(set)
for e in idx: gs[e["split"]].add(tuple(e["group"]))
R["usable_groups_per_split"] = {k: len(v) for k, v in gs.items()}
(DS.parent / "feasibility.json").write_text(json.dumps(R, indent=1)); print(json.dumps({k: R[k] for k in ("windows_per_split", "positives", "groups_with_positives", "usable_groups_per_split")}, indent=1))
for sp in ("train", "val", "test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"):
    b = R["baselines"][sp]["active"]; print(sp, {k: (round(b[k]["persist_box_pos_mm"]["mean"], 1), round(b[k]["constvel_box_pos_mm"]["mean"], 1), round(b[k]["persist_joint_mae_rad"]["mean"], 4), round(b[k]["constvel_joint_mae_rad"]["mean"], 4), round(b[k]["persist_aoffset_cm"]["mean"], 2)) for k in ("K=1", "K=5", "K=10", "K=20")})
