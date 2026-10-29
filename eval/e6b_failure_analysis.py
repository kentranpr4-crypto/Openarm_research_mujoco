#!/usr/bin/env python3
"""Why do S1/S2 fail the nominal task? Timeline diagnostics on the dev-seed rollouts (ground truth used for DIAGNOSTICS only).
Per rollout: close time, box lift, arrival at the tray (TCP xy within 2.5 cm of the tray centre and low), re-open time, box-TCP separation during transport,
final box-to-tray distance. Output: outputs/e6b/failure_analysis.json + P6_failure_timing.png.  ACT venv: python eval/e6b_failure_analysis.py"""
import sys, json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from eval.audit_v1 import events, classify, HZ, TRAY
E6 = ROOT / "outputs/eval/e6b"; OUT = ROOT / "outputs/e6b"; ARMS = ["v1", "c0", "s1", "s2"]; PLACE_Z = 0.36 + 0.004 + 0.02 + 0.025 + 0.003   # TCP height when the box is lowered into the tray

def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def load(arm, tag, cond):
    d = E6 / arm / tag / "act" / cond
    if not (d / "episodes.jsonl").exists(): return []
    out = []
    for e in jl(d / "episodes.jsonl"):
        f = d / "rollouts" / f"ep{e['episode_index']:04d}.npz"
        if f.exists():
            with np.load(f) as z: e = dict(e, **{k: z[k] for k in z.files})
        out.append(dict(e, arm=arm, tag=tag))
    return out

def timeline(e):
    a, B, T = e["action"], e["box_pos"], e["tcp_pos"]; n = len(a); g = a[:, 7]; ev = events(e)
    tc = ev["t_close"]; reopen = None
    if tc is not None:
        r = np.nonzero(g[tc:] > 0.035)[0]; reopen = tc + int(r[0]) if len(r) else None
    d_tray = np.linalg.norm(T[:n, :2] - TRAY, axis=1); arrive = np.nonzero((d_tray < 0.025) & (T[:n, 2] < PLACE_Z + 0.015))[0]; arrive = int(arrive[0]) if len(arrive) else None
    rise = B[:n, 2] - B[0, 2]; sep = np.linalg.norm(B[:n] - T[:n], axis=1); held = (sep < 0.06) & (rise > 0.03)
    t_lift = int(np.argmax(rise > 0.03)) if (rise > 0.03).any() else None
    lost = None
    if t_lift is not None:
        after = np.nonzero(~held[t_lift:])[0]; lost = t_lift + int(after[0]) if len(after) else None        # first step after lift where the box is no longer carried
    return dict(t_close=tc, t_lift=t_lift, t_arrive=arrive, t_reopen=reopen, t_box_lost=lost, box_tray_dist_at_reopen=float(np.linalg.norm(B[reopen, :2] - TRAY)) if reopen is not None else None,
                tcp_tray_dist_at_reopen=float(d_tray[reopen]) if reopen is not None else None, box_z_rise_at_reopen=float(rise[reopen]) if reopen is not None else None, final_box_tray=float(np.linalg.norm(B[-1, :2] - TRAY)),
                carried_frac_lift_to_reopen=float(held[t_lift:reopen].mean()) if (t_lift is not None and reopen is not None and reopen > t_lift) else None)

def main():
    res = {}; rows = []
    for arm in ARMS:
        eps = load(arm, "nom21", "nominal") + load(arm, "bound21", "unseen_pose") + load(arm, "bound22", "unseen_pose")
        for e in eps:
            if "action" not in e: continue
            ev = events(e); tl = timeline(e); cls = classify(e, ev); rows.append(dict(arm=arm, tag=e["tag"], idx=e["episode_index"], seed=e["seed"], success=bool(e["task_success"]), cls=cls, **tl))
    res["rows"] = rows; summ = {}
    for arm in ARMS:
        for tag in ("nom21", "bound21", "bound22"):
            R = [r for r in rows if r["arm"] == arm and r["tag"] == tag]
            if not R: continue
            F = [r for r in R if not r["success"]]; late = [r for r in F if r["t_reopen"] is not None and r["t_arrive"] is not None and r["t_reopen"] < r["t_arrive"]]
            summ[f"{arm}/{tag}"] = dict(n=len(R), failures=len(F), classes={c: sum(r["cls"] == c for r in F) for c in sorted({r["cls"] for r in F})},
                                        reopen_before_arrival=len(late), never_arrived=sum(r["t_arrive"] is None for r in F), box_lost_in_transit=sum(r["t_box_lost"] is not None and (r["t_arrive"] is None or r["t_box_lost"] < r["t_arrive"]) for r in F),
                                        success_reopen_minus_arrival_steps=[r["t_reopen"] - r["t_arrive"] for r in R if r["success"] and r["t_reopen"] is not None and r["t_arrive"] is not None][:40])
    res["summary"] = summ; OUT.mkdir(exist_ok=True, parents=True); (OUT / "failure_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, arm in enumerate(ARMS):
        R = [r for r in rows if r["arm"] == arm and r["tag"] == "nom21"]
        for r in R:
            if r["t_reopen"] is None or r["t_arrive"] is None: continue
            ax[0].scatter(i + np.random.default_rng(r["idx"]).uniform(-0.2, 0.2), (r["t_reopen"] - r["t_arrive"]) / HZ, c="C2" if r["success"] else "C3", s=24, edgecolors="k", linewidths=.3)
        F = [r for r in R if not r["success"] and r["box_tray_dist_at_reopen"] is not None]
        for r in F: ax[1].scatter(i + np.random.default_rng(r["idx"] + 99).uniform(-0.2, 0.2), r["box_tray_dist_at_reopen"] * 100, c="C3", s=26, edgecolors="k", linewidths=.3)
    for a_ in ax: a_.set_xticks(range(len(ARMS))); a_.set_xticklabels([x.upper() for x in ARMS])
    ax[0].axhline(0, c="k", lw=.5); ax[0].set_ylabel("re-open time - arrival time at tray [s]  (green success, red fail)"); ax[0].set_title("NOM: gripper opens before/after the TCP reaches the tray")
    ax[1].set_ylabel("box-to-tray distance when the gripper re-opens [cm], failures only"); ax[1].set_title("NOM failures: where was the box when released?"); fig.tight_layout(); fig.savefig(OUT / "P6_failure_timing.png", dpi=110); plt.close(fig)
    print(json.dumps(summ, indent=1, default=float)[:4000])

if __name__ == "__main__":
    main()
