#!/usr/bin/env python3
"""Reconcile the 314 'yaw spikes' of the first audit (window-step elements of the TRAIN windows with |Y_b yaw| > pi/2) with the 122 episode-level events of the root-cause analysis.
Element (window w, step j) <-> episode transition tau = t_w + j (row tau -> tau+1).  Run: python eval/yaw_event_reconcile.py"""
import json, numpy as np, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; WIN = ROOT / "outputs/wm2/windows"
ev = json.load(open(ROOT / "outputs/yaw_audit/yaw_events.json")); evd = {(e["eid"], e["t"]): e for e in ev}; res = {}
for sp in ["train", "val", "test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]:
    z = np.load(WIN / f"{sp}.npz"); y = z["Y_b"][:, :, 3]; w, j = np.nonzero(np.abs(y) > np.pi / 2); eid = z["eid"]; t = z["t"]; keys = [(str(eid[a]), int(t[a] + b)) for a, b in zip(w, j)]; cnt = collections.Counter(keys)
    in_set = [k for k in cnt if k in evd]; cats = collections.Counter(evd[k]["cat"][0] for k in cnt if k in evd); elem_cat = collections.Counter()
    for k, c in cnt.items(): elem_cat[evd[k]["cat"][0] if k in evd else "?"] += c
    sp_events = [e for e in ev if e["split"] == sp]; covered = {k for k in cnt}
    res[sp] = dict(elements=int(len(keys)), unique_events_in_windows=len(cnt), all_in_122=all(k in evd for k in cnt), multiplicity_hist=dict(collections.Counter(cnt.values())), events_in_split=len(sp_events), events_not_covered_by_any_window=sum(1 for e in sp_events if (e["eid"], e["t"]) not in covered),
                   events_by_category=dict(cats), elements_by_category=dict(elem_cat), windows_with_spike=int(len(set(w))), episodes_with_spike=len({k[0] for k in cnt}))
tot = sum(r["events_in_split"] for r in res.values()); res["_total_events_all_splits"] = tot; res["_note"] = "category letters: S=Euler singularity (rest), S2=singular while moving (both start with 'S'), P=physical fast rotation, W=+-pi crossing"
(ROOT / "outputs/contract_v2/yaw_event_reconcile.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["train"], indent=1)); print({k: (v["elements"], v["unique_events_in_windows"], v["events_in_split"], v["events_not_covered_by_any_window"]) for k, v in res.items() if not k.startswith("_")}, tot)
