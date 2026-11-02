#!/usr/bin/env python3
"""READ-ONLY post-hoc diagnostic: re-rank the stored branching-replay predictions with the 0.05*palm term removed from the stored predicted J20 (frozen J untouched; ONE variant only; development data).
Run: python3 eval/palm_term_removed_diagnostic.py -> outputs/branching_replay/palm_term_removed_diagnostic.json"""
import json, glob, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); import branching_metrics as BM
S = [s for f in sorted(glob.glob(str(ROOT / "outputs/branching_replay/results/worker*.json"))) for s in json.loads(Path(f).read_text())["states"]]
def run(drop_palm):
    M = []
    for s in S:
        Jp = np.array(s["pred"]["J20"])
        if drop_palm: Jp = Jp - 0.05 * np.array([q["palm"] for q in s["pred"]["terms"]])
        m = BM.state_metrics(Jp, [b["J20_real"] for b in s["branches"]], s["pose_idx"]); m["pose"] = s["pose_idx"]; m["phase"] = s["phase"]; M.append(m)
    num = {}; den = {}
    for m in M: num[m["pose"]] = num.get(m["pose"], 0) + m["pair"][0]; den[m["pose"]] = den.get(m["pose"], 0) + m["pair"][1]
    pp = lambda f: {p: [f(m) for m in M if m["pose"] == p] for p in {m["pose"] for m in M}}
    out = dict(acc=BM.boot_ratio(num, den), pairs=int(sum(den.values())), regret_wm=BM.boot_mean_by_pose(pp(lambda m: m["regret_wm"])), paired_vs_default=BM.boot_paired_by_pose(pp(lambda m: m["regret_wm"]), pp(lambda m: m["regret_default"])))
    for ph in ("approach", "grasp", "carry", "place"):
        Mp = [m for m in M if m["phase"] == ph]; n = {}; d = {}
        for m in Mp: n[m["pose"]] = n.get(m["pose"], 0) + m["pair"][0]; d[m["pose"]] = d.get(m["pose"], 0) + m["pair"][1]
        out[ph] = dict(acc=BM.boot_ratio(n, d), pairs=int(sum(d.values())), regret_minus_default=float(np.mean([m["regret_wm"] - m["regret_default"] for m in Mp])))
    return out
r = {"with_palm_term(frozen J20)": run(False), "palm_term_removed(diagnostic only)": run(True)}
(ROOT / "outputs/branching_replay/palm_term_removed_diagnostic.json").write_text(json.dumps(r, indent=1, default=float)); print(json.dumps({k: v["acc"] for k, v in r.items()}))
