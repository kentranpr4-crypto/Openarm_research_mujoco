#!/usr/bin/env python3
"""DESIGN-PHASE, READ-ONLY feasibility check of the draft Outcome Contract v2 (event-based) on SAVED data only (no simulator, no policy inference, no new rollout).  NOT FROZEN, NOT a result:
  (1) schedule-dependence audit of the frozen G2a 'dropped' label from the saved G2a summaries;
  (2) draft v2 labeller (pure function of logged signals) + synthetic scenario tests;
  (3) application to the saved FULL-LENGTH (200-step) episodes: xsweep replays (6 ACT checkpoints x 60 poses + expert x 60) and Stage-1 trajectories, compared with the legacy task labels.
HONEST PROVENANCE: the first draft thresholds were physically motivated, but THREE development iterations were made after seeing agreement/disagreement with the saved development episodes (iter1 -> iter2: opening ONSET (+4 mm above the running minimum) replaces an absolute 0.030 m crossing, and loss of grasp between acquisition and confirmed lift is its own event; iter2 -> iter3: Z_REL 0.03 -> 0.05 m (one box height) after a trace showed ACT releasing at ~0.031 m above the tray base, and the borderline band widened to [0.03, 0.05)). The saved development episodes are therefore NOT valid for validating the final contract; validation must use fresh, blinded data. The run is not blinded.
Run: python3 eval/outcome_v2_feasibility.py -> outputs/g2b_design/outcome_v2_feasibility.json"""
import json, glob, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; DT = 0.05
P = dict(G_CLOSE=0.040, G_ACQ=0.030, G_OPEN=0.030, NF_ACQ=2, F_ACQ=10.0, K_ACQ=5, R_LIFT=0.03, K_LIFT=3, LIFT_BUDGET=40, NF_HELD=2, F_HELD=5.0, K_LOSS=3, RET_WIN=60, EARLY_SLIP=15, ACQ_BUDGET=25,
         TRAY=(0.44, 0.30), REST_Z=0.384, R_REL=0.05, Z_REL=0.05, PLACE_XY=0.03, PLACE_Z=0.006, REST_V=0.02, K_SETTLE=10, SETTLE_BUDGET=40, BORDER_LO=0.03, BORDER_HI=0.05, OPEN_RISE=0.004, K_OPEN=2, REL_PRELIFT_MIN_RISE=0.01)
def first_run(mask, start, k):
    n = len(mask)
    for t in range(start, n - k + 1):
        if mask[t:t + k].all(): return t
    return None
def label_v2(box, g, nf, ff, hf=None, p=P):
    """box (T,3) [m, world]; g (T,) executed left-gripper target [m]; nf (T,) finger-box contact points; ff (T,) finger normal force [N]; hf (T,) palm force [N].  Row t = observation before action t.  Returns events (step indices), terminal label, grasp_retained, flags."""
    T = len(g); z0 = box[0, 2]; rise = box[:, 2] - z0; tray = np.array(p["TRAY"]); dxy = np.linalg.norm(box[:, :2] - tray, axis=1); v = np.zeros(T)
    if T > 2: v[1:-1] = np.linalg.norm(box[2:] - box[:-2], axis=1) / (2 * DT)
    ev = {}; flags = []; out = dict(events=ev, flags=flags, terminal=None, grasp_retained=None, max_rise_m=float(rise.max()))
    ci = np.where(g < p["G_CLOSE"])[0]
    if len(ci) == 0: out["terminal"] = "not_closed"; return out
    tc = int(ci[0]); ev["close_command"] = tc; ta = first_run((nf >= p["NF_ACQ"]) & (ff >= p["F_ACQ"]) & (g < p["G_ACQ"]), tc, p["K_ACQ"])
    if ta is None: out["terminal"] = "timeout" if (T - tc) < p["ACQ_BUDGET"] else "not_closed"; flags.append("no_acquisition"); return out
    ev["grasp_acquired"] = ta; tl = first_run(rise >= p["R_LIFT"], ta, p["K_LIFT"]); held0 = (nf >= p["NF_HELD"]) & (ff >= p["F_HELD"])
    pre = [t for t in range(ta, min(T - p["K_LOSS"] + 1, tl if tl is not None else T)) if (~held0[t:t + p["K_LOSS"]]).all() and (g[t:t + p["K_LOSS"]] < p["G_OPEN"]).all()]
    if pre and rise[ta:pre[0] + 1].max() >= p["REL_PRELIFT_MIN_RISE"]: ev["premature_drop"] = pre[0]; out["terminal"] = "premature_drop"; out["subtype"] = "acquisition_lost"; out["grasp_retained"] = False; return out
    if tl is None or tl > ta + p["LIFT_BUDGET"]:
        out["terminal"] = "timeout" if (T - ta) < p["LIFT_BUDGET"] + p["K_LIFT"] else "not_lifted"; out["grasp_retained"] = False if out["terminal"] == "not_lifted" else None; return out
    ev["lift"] = tl
    if p["BORDER_LO"] <= rise[tl:].max() < p["BORDER_HI"]: flags.append("borderline_lift")
    held = (nf >= p["NF_HELD"]) & (ff >= p["F_HELD"]); closed_cmd = g < p["G_OPEN"]; runmin = np.minimum.accumulate(g[ta:]); onset = first_run((g[ta:] - runmin) >= p["OPEN_RISE"], 0, p["K_OPEN"]); tr = int(ta + onset) if (onset is not None and ta + onset >= tl) else None; lost = None
    for t in range(tl, T - p["K_LOSS"] + 1):
        if (~held[t:t + p["K_LOSS"]]).all() and closed_cmd[t:t + p["K_LOSS"]].all() and (tr is None or t < tr): lost = t; break
    win_end = tl + p["RET_WIN"]
    if lost is not None and (tr is None or lost < tr):
        ev["premature_drop"] = lost; out["terminal"] = "premature_drop"; out["subtype"] = "early_slip" if lost - tl <= p["EARLY_SLIP"] else "carry_drop"; out["over_tray_region"] = bool(dxy[lost] < p["R_REL"]); out["grasp_retained"] = bool(lost > min(win_end, tr if tr is not None else win_end)); return out
    if tr is not None:
        ev["release_command"] = tr; zrel = box[tr, 2] - p["REST_Z"]
        if dxy[tr] >= p["R_REL"]: out["terminal"] = "premature_drop"; out["subtype"] = "release_outside_target"; ev["premature_drop"] = tr; out["grasp_retained"] = bool(tr > win_end); return out
        if zrel > p["Z_REL"]: out["terminal"] = "premature_drop"; out["subtype"] = "release_in_air"; ev["premature_drop"] = tr; out["grasp_retained"] = bool(tr > win_end); return out
        ev["intended_release"] = tr; out["grasp_retained"] = True if (T >= min(win_end, tr)) else None; stop = min(tr + p["SETTLE_BUDGET"], T)
        ok = (dxy < p["PLACE_XY"]) & (np.abs(box[:, 2] - p["REST_Z"]) < p["PLACE_Z"]) & (v < p["REST_V"]); ts = first_run(ok[:stop], tr, p["K_SETTLE"])
        if ts is not None: ev["placement_success"] = ts; out["terminal"] = "placement_success"
        else: out["terminal"] = "placement_failure" if tr + p["SETTLE_BUDGET"] <= T else "timeout"
        return out
    out["terminal"] = "timeout"; out["grasp_retained"] = True if T >= win_end else None; return out
# ---------------------------------------------------------------------------------------------------------------------------------- synthetic scenarios
def synth(kind, T=200, tl=70, rng=None):
    rng = rng or np.random.default_rng(0); box = np.zeros((T, 3)); box[:, 0] = 0.40; box[:, 1] = 0.10; box[:, 2] = 0.38; g = np.full(T, 0.044); nf = np.zeros(T); ff = np.zeros(T); tc = 42; tray = np.array(P["TRAY"]); rest = 0.384
    for t in range(tc, T): g[t] = max(0.012, 0.044 - 0.0032 * (t - tc))
    held_from = tc + 10; nf[held_from:] = 8; ff[held_from:] = 48.0
    def ramp(a, b, t0, t1): return lambda t: a + (b - a) * np.clip((t - t0) / max(t1 - t0, 1), 0, 1)
    z = ramp(0.38, 0.38 + 0.10, tl, tl + 20); xy = ramp(0, 1, tl + 20, tl + 55)
    for t in range(T):
        s = float(xy(t)); box[t, :2] = np.array([0.40, 0.10]) * (1 - s) + tray * s; box[t, 2] = float(z(t))
    low = ramp(1, 0, tl + 55, tl + 65)
    for t in range(T): box[t, 2] = rest + (box[t, 2] - rest) * float(low(t)) if t >= tl + 55 else box[t, 2]
    t_rel = tl + 75
    if kind in ("success", "tray_arrival_mistaken_for_drop"):
        for t in range(t_rel, T): g[t] = min(0.044, 0.012 + 0.0032 * (t - t_rel + 1)); nf[t] = 0 if g[t] > 0.030 else 8; ff[t] = 0 if g[t] > 0.030 else 48
        nf[t_rel + 3:] = 0; ff[t_rel + 3:] = 0
    elif kind == "carry_drop":
        td = tl + 45; nf[td:] = 0; ff[td:] = 0; box[td:, 2] = np.maximum(0.38, box[td, 2] - 0.02 * np.arange(1, T - td + 1)); box[td:, :2] = box[td, :2]
    elif kind == "early_slip":
        td = tl + 8; nf[td:] = 0; ff[td:] = 0; box[td:, 2] = np.maximum(0.38, box[td, 2] - 0.02 * np.arange(1, T - td + 1)); box[td:, :2] = box[td, :2]
    elif kind == "no_lift":
        box[:, 2] = 0.38; box[:, :2] = [0.40, 0.10]
    elif kind == "incomplete":
        pass
    elif kind == "release_outside_target":
        tr_ = tl + 35; box[tr_:, :2] = box[tr_, :2]; box[tr_ + 3:, 2] = np.maximum(0.38, box[tr_, 2] - 0.02 * np.arange(1, T - tr_ - 2)); box[tr_:tr_ + 3, 2] = box[tr_, 2]; g[tr_:] = np.minimum(0.044, 0.012 + 0.0032 * np.arange(1, T - tr_ + 1)); nf[tr_ + 3:] = 0; ff[tr_ + 3:] = 0
    elif kind == "release_in_air":
        tr_ = tl + 58; g[tr_:] = np.minimum(0.044, 0.012 + 0.0032 * np.arange(1, T - tr_ + 1)); nf[tr_ + 3:] = 0; ff[tr_ + 3:] = 0; box[tr_:, :2] = tray; zz = box[tr_, 2]; box[tr_:tr_ + 3, 2] = zz; box[tr_ + 3:, 2] = np.maximum(rest, zz - 0.02 * np.arange(1, T - tr_ - 2))
    elif kind in ("borderline_lift_035", "borderline_lift_025"):
        peak = 0.035 if kind.endswith("035") else 0.025; box[:, 2] = 0.38 + np.minimum(peak, np.where(np.arange(T) >= tl, (np.arange(T) - tl) * 0.005, 0.0)); box[:, :2] = [0.40, 0.10]
    return box, g, nf, ff
EXPECT = {"success": "placement_success", "tray_arrival_mistaken_for_drop": "placement_success", "carry_drop": "premature_drop", "early_slip": "premature_drop", "no_lift": "not_lifted", "incomplete": "timeout", "release_outside_target": "premature_drop", "release_in_air": "premature_drop", "borderline_lift_035": "timeout", "borderline_lift_025": "not_lifted"}
def run_synth():
    rng = np.random.default_rng(1); res = {}
    for k, exp in EXPECT.items():
        ok = 0; n = 0; labs = []
        for tl in (66, 70, 74, 78, 82):
            T = 200 if k != "incomplete" else tl + 25; box, g, nf, ff = synth(k, T=200, tl=tl, rng=rng)
            if k == "incomplete": box, g, nf, ff = box[:T], g[:T], nf[:T], ff[:T]
            o = label_v2(box, g, nf, ff); labs.append(o["terminal"]); ok += o["terminal"] == exp; n += 1
        res[k] = dict(expected=exp, got=sorted(set(labs)), agree=f"{ok}/{n}")
    return res
# ---------------------------------------------------------------------------------------------------------------------------------- audits on saved data
def g2a_schedule_audit():
    R = json.loads((ROOT / "outputs/g2a_act_corrections/results/g2a_results.json").read_text()); rows = [(c["cand"], c["outcome"]) for p in R["poses"] if p["applicable"] for c in p["candidates"] if c["outcome"]]
    tl = np.array([o["t_lift"] if o["t_lift"] is not None else -1 for _, o in rows]); dr = np.array([o["dropped"] for _, o in rows]); out = {}
    for lo, hi in ((60, 68), (68, 70), (70, 72), (72, 80)): m = (tl >= lo) & (tl < hi); out[f"t_lift[{lo},{hi})"] = dict(n=int(m.sum()), dropped_label_rate=float(dr[m].mean()) if m.any() else None)
    out["dropped_label_by_candidate"] = {c: int(sum(o["dropped"] for cc, o in rows if cc == c)) for c in sorted({c for c, _ in rows})}; out["note"] = "later lift (later arrival at the tray) -> fewer 'dropped' labels: the frozen label depends on schedule"; return out
def load_xsweep():
    cfg = json.loads((ROOT / "outputs/xsweep/xsweep_config.json").read_text()); idxs = [p["idx"] for p in cfg["poses"]]; eps = {}
    for ck, d in (("c0_s0", "c0_seed0"), ("c0_s1", "c0_seed1"), ("c0_s2", "c0_seed2"), ("s1_s0", "s1_seed0"), ("s1_s1", "s1_seed1"), ("s1_s2", "s1_seed2")):
        for ln in open(ROOT / f"outputs/xsweep/eval/{d}/act/sweep/episodes.jsonl"): r = json.loads(ln); eps[(ck, r["episode_index"])] = r
    return idxs, eps
def sig_replay(Z):
    n = len(Z["action"]); return dict(box=Z["box_pos"][:n].astype(float), g=Z["action"][:, 7].astype(float), nf=Z["finger_contacts"][:n].astype(float), ff=Z["finger_force"][:n].astype(float), hf=Z["hand_force"][:n].astype(float))
def apply_saved():
    idxs, eps = load_xsweep(); lab = [];
    for agent in ("c0_s0", "c0_s1", "c0_s2", "s1_s0", "s1_s1", "s1_s2", "expert"):
        for i in idxs:
            Z = np.load(ROOT / f"outputs/xsweep/replay/{agent}_SW_{i:03d}.npz"); o = label_v2(**sig_replay(Z)); legacy = eps[(agent, i)] if agent != "expert" else None
            lab.append(dict(agent=agent, pose=i, v2=o["terminal"], subtype=o.get("subtype"), retained=o["grasp_retained"], flags=o["flags"], legacy_success=(bool(legacy["task_success"]) if legacy else None), legacy_outcome=(legacy["outcome"] if legacy else None), t_lift=o["events"].get("lift"), t_rel=o["events"].get("intended_release"), t_drop=o["events"].get("premature_drop"), t_settle=o["events"].get("placement_success"),
                            tray_before_100=bool((np.linalg.norm(Z["box_pos"][:101, :2] - np.array(P["TRAY"]), axis=1) < 0.01).any()), max_rise=o["max_rise_m"]))
    return lab
def apply_stage1():
    R = json.loads((ROOT / "outputs/closed_loop_pilot/results/pilot_results.json").read_text()); lab = []
    for e in [x for x in R["episodes"] if x["status"] == "OK"]:
        Z = np.load(ROOT / f"outputs/closed_loop_pilot/results/traj/{e['arm']}_p{e['pose_idx']:02d}_r{e['repeat']}.npz"); c = Z["c_rows"]
        o = label_v2(box=Z["box_pos"][:200].astype(float), g=Z["action"][:, 7].astype(float), nf=c[:, 1].astype(float), ff=c[:, 0].astype(float) * 50.0, hf=c[:, 4].astype(float)); lab.append(dict(arm=e["arm"], v2=o["terminal"], subtype=o.get("subtype"), legacy_success=bool(e["task_success"]), legacy_outcome=e["outcome"]))
    return lab
def summarize(lab, key_legacy="legacy_success"):
    L = [l for l in lab if l.get(key_legacy) is not None]; tab = {}
    for l in L: tab.setdefault((l[key_legacy], l["v2"]), []).append(l)
    conf = {f"legacy_success={k[0]}|v2={k[1]}": len(v) for k, v in sorted(tab.items(), key=lambda kv: str(kv[0]))}; dis = [l for l in L if (l["v2"] == "placement_success") != l[key_legacy]]
    return dict(n=len(L), confusion=conf, n_disagree_success_vs_placement=len(dis), disagreements=[{k: l.get(k) for k in ("agent", "arm", "pose", "legacy_outcome", "v2", "subtype", "t_lift", "t_rel", "t_drop")} for l in dis[:12]])
if __name__ == "__main__":
    out = dict(label="DESIGN-PHASE READ-ONLY feasibility of draft Outcome Contract v2 on saved development data; NOT FROZEN; NOT blinded; thresholds were iterated three times on these data (see module docstring) -> not an independent validation", iterations="iter1 json: outcome_v2_feasibility_iter1.json; iter2: outcome_v2_feasibility_iter2.json; this file = iter3", params=P)
    out["synthetic_scenarios"] = run_synth(); out["g2a_frozen_label_schedule_dependence"] = g2a_schedule_audit()
    xs = apply_saved(); out["xsweep_saved_full_episodes"] = dict(act=summarize([l for l in xs if l["agent"] != "expert"]), expert_v2_labels={k: sum(1 for l in xs if l["agent"] == "expert" and l["v2"] == k) for k in sorted({l["v2"] for l in xs if l["agent"] == "expert"})})
    act = [l for l in xs if l["agent"] != "expert"]; fails = [l for l in act if not l["legacy_success"]]
    out["xsweep_act_v2_labels"] = {k: sum(1 for l in act if l["v2"] == k) for k in sorted({l["v2"] for l in act})}; out["xsweep_act_legacy_failures_v2_labels"] = {f"{k}/{s}": sum(1 for l in fails if l["v2"] == k and l["subtype"] == s) for k, s in sorted({(l["v2"], l["subtype"]) for l in fails}, key=str)}
    suc = [l for l in act if l["legacy_success"]]; out["tray_arrival_before_step_100_among_legacy_successes"] = dict(n=len(suc), n_tray_xy_within_1cm_by_step100=sum(l["tray_before_100"] for l in suc), frac=float(np.mean([l["tray_before_100"] for l in suc])))
    q = lambda v: {f"p{p}": float(np.percentile(v, p)) for p in (5, 50, 95, 100)} if len(v) else None
    out["timing_saved_episodes_steps"] = dict(t_lift_act=q([l["t_lift"] for l in act if l["t_lift"] is not None]), t_release_act_success=q([l["t_rel"] for l in act if l["t_rel"] is not None and l["v2"] == "placement_success"]), t_settle_act_success=q([l["t_settle"] for l in act if l["t_settle"] is not None]), t_release_expert=q([l["t_rel"] for l in xs if l["agent"] == "expert" and l["t_rel"] is not None]), t_settle_expert=q([l["t_settle"] for l in xs if l["agent"] == "expert" and l["t_settle"] is not None]),
                                              drop_minus_lift_act=q([l["t_drop"] - l["t_lift"] for l in act if l["t_drop"] is not None and l["t_lift"] is not None and l["v2"] == "premature_drop"]), n_unresolved_timeout=sum(l["v2"] == "timeout" for l in act))
    out["borderline_lift_flags_act"] = dict(n_flagged=sum("borderline_lift" in l["flags"] for l in act), max_rise_between_0p03_0p05=sum(1 for l in act if 0.03 <= l["max_rise"] < 0.05), max_rise_between_0p05_0p06=sum(1 for l in act if 0.05 <= l["max_rise"] < 0.06))
    out["stage1_saved_full_episodes"] = summarize(apply_stage1())
    Path(ROOT / "outputs/g2b_design").mkdir(parents=True, exist_ok=True); (ROOT / "outputs/g2b_design/outcome_v2_feasibility.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(out, indent=1, default=float)[:9000])
