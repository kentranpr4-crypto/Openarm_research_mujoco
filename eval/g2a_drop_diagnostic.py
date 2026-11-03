#!/usr/bin/env python3
"""POST-HOC DIAGNOSTIC (not part of the frozen protocol): are the nominal-ACT 'dropped' labels of G2a real drops or successful carries that reach the tray before t=100?  Re-executes ONLY the nominal (c0) branch of the states labelled 'dropped'/'not_lifted' (bit-exact reproducible, ~1 s each), stores box/TCP traces.
Run (ACT venv): python eval/g2a_drop_diagnostic.py   (uses the remaining G2a CPU budget; ledger entry added)"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import closed_loop_pilot as P, branching_replay as BR, g2a_act_corrections as G, lab_scene
c0 = P.cpu(); R = json.loads((G.OUT / "results/g2a_results.json").read_text()); sel = [p for p in R["poses"] if p["applicable"] and next(x for x in p["candidates"] if x["cand"] == "c0")["outcome"]["reason"] in ("dropped", "not_lifted")][:16]
guard = P.make_guard(c0, 120.0, 1); Rn = G.Runner(); out = []; tray = np.array(lab_scene.TRAY_C)
for p in sel:
    guard(margin=5.0); pose = p["pose"]; o, prev, Rp, exe_pre = Rn.prefix(pose); z0 = float(Rp["box"][0][2]); snap = BR.snapshot(Rn.env); raw40 = Rn.policy.chunk(o["state"], o["images"]); Rb, exe, _, _ = Rn.branch(snap, prev, raw40, None)
    rise = Rb["box"][:, 2] - z0; up = np.where(rise >= 0.05)[0]; kl = int(up[0]) if len(up) else None; kd = None
    if kl is not None:
        d = np.where(rise[kl:] < 0.02)[0]; kd = kl + int(d[0]) if len(d) else None
    rec = dict(pose=pose["idx"], x=pose["x"], yaw=pose["yaw"], reason=next(x for x in p["candidates"] if x["cand"] == "c0")["outcome"]["reason"], t_lift=(kl + 40 if kl is not None else None), t_drop=(kd + 40 if kd is not None else None), rise_max=float(rise.max()))
    if kd is not None:
        bx = Rb["box"][kd]; rec.update(box_xy_to_tray_cm_at_drop=float(np.linalg.norm(bx[:2] - tray) * 100), box_tcp_sep_cm_at_drop=float(np.linalg.norm(Rb["box"][kd] - Rb["tcp"][kd]) * 100), box_tcp_sep_cm_one_step_before=float(np.linalg.norm(Rb["box"][kd - 1] - Rb["tcp"][kd - 1]) * 100), box_z_trace_after_lift_cm=[round(float(v * 100), 1) for v in rise[kl:kl + 40:3]])
    rec["end_box_xy_to_tray_cm"] = float(np.linalg.norm(Rb["box"][-1][:2] - tray) * 100); rec["end_box_tcp_sep_cm"] = float(np.linalg.norm(Rb["box"][-1] - Rb["tcp"][-1]) * 100); out.append(rec)
res = dict(label="POST-HOC DIAGNOSTIC of the frozen G2a 'dropped' label (nominal ACT only); not a protocol change", n=len(out), cpu_s=P.cpu() - c0, records=out)
P.atomic_json(G.OUT / "results/drop_diagnostic.json", res); G.ledger_add("post_hoc_drop_diagnostic", res["cpu_s"]); print(json.dumps(res, default=float)[:3500])
