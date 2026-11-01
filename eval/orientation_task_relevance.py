#!/usr/bin/env python3
"""Is box ORIENTATION relevant to grasp / placement task semantics?  Descriptive analysis on the 2 484 usable interventional episodes of the contract-v2 dataset (quaternions from replay; DEVELOPMENT data, true trajectories only — no model prediction involved).
Features: max D2-symmetric rotation of the box relative to its orientation at the close command during the hold phase (grasp slip / in-hand rotation), max sym-tilt over the episode, initial yaw.  Outcomes: task / pick / place success.   Run: python3 eval/orientation_task_relevance.py"""
import sys, json, numpy as np, time, resource
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); import geometry_v2 as Gm, decision_utility_lib as L
t0 = resource.getrusage(resource.RUSAGE_SELF); idx = [e for e in json.loads((ROOT / "outputs/wm2_v2/dataset/index.json").read_text()) if e["kind"] == "interventional" and e.get("v2_ok")]; rows = []
for e in idx:
    o = e["outcomes"]; c, r = o.get("close_step"), o.get("release_step")
    if c is None: continue
    q = np.load(ROOT / "outputs/wm2_v2/dataset/episodes" / f"{e['eid']}.npz")["q"]; T = len(q); r = min(r if r is not None else T - 1, T - 1); hold = q[c:r + 1]
    rows.append(dict(success=e["task_success"], pick=e["pick"], place=e["place"], hold_rot=float(Gm.geodesic_sym(hold[0], hold).max()), max_tilt=float(Gm.tilt_sym(q).max()), yaw0=abs(float(e["box_yaw"])), family=e["family"]))
def auc(feat, y): return L.auroc(1 - np.array(y, int), np.array(feat, float))          # AUROC of the feature for predicting FAILURE
res = dict(n_episodes=len(rows), success_rate=float(np.mean([r["success"] for r in rows])), failure_count=int(sum(not r["success"] for r in rows)))
for k in ("hold_rot", "max_tilt", "yaw0"): res[f"AUROC_{k}_for_failure"] = auc([r[k] for r in rows], [r["success"] for r in rows])
bins = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 10]; tab = []
for lo, hi in zip(bins[:-1], bins[1:]):
    s = [r["success"] for r in rows if lo <= r["hold_rot"] < hi]; tab.append(dict(hold_rotation_rad=[lo, hi], n=len(s), success_rate=float(np.mean(s)) if s else None))
res["success_by_in_hold_rotation_bin"] = tab; ctl = [r for r in rows if r["family"] == "control"]; res["control_variant"] = dict(n=len(ctl), success_rate=float(np.mean([r["success"] for r in ctl])), max_initial_abs_yaw_rad=float(max(r["yaw0"] for r in ctl)))
res["place_success_definition"] = "box centre within 0.03 m xy and 0.006 m z of the tray base at rest — no orientation term (eval/metrics.py)"; ru = resource.getrusage(resource.RUSAGE_SELF); res["cpu_seconds"] = ru.ru_utime + ru.ru_stime - t0.ru_utime - t0.ru_stime
(ROOT / "outputs/decision_utility/orientation_task_relevance.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
