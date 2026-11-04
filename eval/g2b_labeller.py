#!/usr/bin/env python3
"""G2b Outcome Contract v2-r1 LABELLER (B2) - a PURE function of a logged branch trace.  DRAFT: not frozen; thresholds are the r1 values (development-derived, see the r1 document); nothing here was tuned on any new evaluation outcome.
Three layers (kept separate): (A) event timeline with timestamps and signed threshold margins, (B) ONE mutually exclusive terminal outcome, (C) binary endpoints E1/E2/E3 with value in {TRUE, FALSE, CENSORED, INVALID} and an INDEPENDENT ambiguity flag.
Ambiguity is determined by perturbation: an endpoint is `ambiguous` iff moving any decisive threshold inside its pre-registered band (BANDS / +-1 on the debounce lengths) changes its value; the label itself is never altered by the flag.
Trace schema (arrays of equal length N = observed rows; row i = observation before the action of step t0+i):
  g_exec [m] executed left-gripper target; nL,nR [count] finger-box contact POINTS (left/right finger mesh); FL,FR [N] summed contact normal force per finger; box_pos [m] world; box_vel [m/s] world (optional, else central difference);
  meta: t0 (int, absolute step of row 0), box_z0 [m] (box height at episode start), valid (bool).  Optional: Fh [N] palm force (reported, not used by the labels).
Units: metres, newtons, seconds (dt = 0.05 s), steps.  World frame z up; tray centre (0.44, 0.30) m; resting box-centre height 0.384 m."""
import json, hashlib, numpy as np
DT = 0.05
PARAMS = dict(G_CLOSE=0.040, G_ACQ=0.030, F_ACQ=5.0, K_ACQ=5, ACQ_BUDGET=25, R_LIFT=0.03, K_LIFT=3, LIFT_BUDGET=40, F_HELD=2.5, K_LOSS=3, OPEN_RISE=0.004, K_OPEN=2, RET_WIN=60, EARLY_SLIP=15, SHORT_WINDOW=20,
              TRAY_X=0.44, TRAY_Y=0.30, REST_Z=0.384, R_REL=0.05, Z_REL=0.05, PLACE_XY=0.03, PLACE_Z=0.006, REST_V=0.02, K_SETTLE=10, SETTLE_BUDGET=40, T_MAX=160)
BANDS = dict(R_LIFT=0.005, F_ACQ=1.0, F_HELD=0.5, OPEN_RISE=0.001, R_REL=0.005, Z_REL=0.005, PLACE_XY=0.003, PLACE_Z=0.0015, REST_V=0.004)       # +- perturbation of a decisive threshold [its unit]
KBANDS = ("K_ACQ", "K_LIFT", "K_LOSS", "K_OPEN", "K_SETTLE")                                                                                         # +-1 step
TERMINALS = ("INVALID", "ACQUISITION_FAILURE", "CARRY_DROP", "RELEASE_OUTSIDE_TARGET", "PLACEMENT_FAILURE", "PLACEMENT_SUCCESS", "TIMEOUT")
def params_hash(p=PARAMS): return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
def first_run(mask, lo, hi, k):
    """first start s in [lo, hi] with mask[s:s+k] all True (fully inside the observed rows), else None"""
    n = len(mask); hi = min(hi, n - k)
    for s in range(max(lo, 0), hi + 1):
        if mask[s:s + k].all(): return s
    return None
def _arrays(tr):
    g = np.asarray(tr["g_exec"], float); N = len(g); a = dict(g=g, nL=np.asarray(tr["nL"], float), nR=np.asarray(tr["nR"], float), FL=np.asarray(tr["FL"], float), FR=np.asarray(tr["FR"], float), box=np.asarray(tr["box_pos"], float))
    if "box_vel" in tr and tr["box_vel"] is not None: a["v"] = np.linalg.norm(np.asarray(tr["box_vel"], float), axis=1)
    else:
        v = np.zeros(N)
        if N > 2: v[1:-1] = np.linalg.norm(a["box"][2:] - a["box"][:-2], axis=1) / (2 * DT)
        a["v"] = v
    return a
def _core(tr, p):
    a = _arrays(tr); g, box = a["g"], a["box"]; N = len(g); t0 = int(tr["meta"].get("t0", 0)); z0 = float(tr["meta"]["box_z0"]); rise = box[:, 2] - z0; tray = np.array([p["TRAY_X"], p["TRAY_Y"]]); dxy = np.linalg.norm(box[:, :2] - tray, axis=1)
    out = dict(events={}, terminal=None, subtype=None, E1="CENSORED", E2="CENSORED", E3="CENSORED", N=N, t0=t0, intended_release=None)
    ev = out["events"]; ok_all = all(np.isfinite(np.asarray(tr[k], float)).all() for k in ("g_exec", "nL", "nR", "FL", "FR", "box_pos"))
    if (not ok_all) or (not tr["meta"].get("valid", True)) or N < 2: out.update(terminal="INVALID", E1="INVALID", E2="INVALID", E3="INVALID"); return out
    horizon_reached = (t0 + N) >= p["T_MAX"]
    def finish(term, sub=None):
        out["terminal"] = term; out["subtype"] = sub; return out
    cc = first_run(g < p["G_CLOSE"], 0, N, 1)
    if cc is None:
        if horizon_reached: out["E1"] = out["E2"] = out["E3"] = "FALSE"; return finish("ACQUISITION_FAILURE", "NOT_CLOSED")
        return finish("TIMEOUT")
    ev["close_command"] = cc + t0
    held_acq = (a["nL"] >= 1) & (a["nR"] >= 1) & (a["FL"] >= p["F_ACQ"]) & (a["FR"] >= p["F_ACQ"]) & (g < p["G_ACQ"]); ga = first_run(held_acq, cc, cc + p["ACQ_BUDGET"], p["K_ACQ"])
    if ga is None:
        if N - 1 >= cc + p["ACQ_BUDGET"] + p["K_ACQ"] - 1: out["E1"] = out["E2"] = out["E3"] = "FALSE"; return finish("ACQUISITION_FAILURE", "NOT_ACQUIRED")
        return finish("TIMEOUT")
    ev["grasp_acquired"] = ga + t0; out["E1"] = "TRUE"; lc = first_run(rise >= p["R_LIFT"], ga, ga + p["LIFT_BUDGET"], p["K_LIFT"])
    held = (a["nL"] >= 1) & (a["nR"] >= 1) & (a["FL"] >= p["F_HELD"]) & (a["FR"] >= p["F_HELD"]); runmin = np.minimum.accumulate(g[ga:]); onset = first_run((g[ga:] - runmin) >= p["OPEN_RISE"], 0, N, p["K_OPEN"]); ro = (ga + onset) if onset is not None else None
    gl = first_run((~held) & (g < p["G_ACQ"]), ga, N, p["K_LOSS"]); gl = gl if (gl is not None and (ro is None or gl < ro)) else None
    if lc is not None: ev["lift_confirmed"] = lc + t0
    if ro is not None: ev["release_onset"] = ro + t0
    if gl is not None: ev["grasp_lost"] = gl + t0
    if lc is None or (gl is not None and gl < lc) or (ro is not None and ro < lc):                                       # decided before / without a confirmed lift
        if gl is not None and (lc is None or gl < lc): out["E2"] = out["E3"] = "FALSE"; return finish("ACQUISITION_FAILURE", "ACQUISITION_LOST")
        if ro is not None and (lc is None or ro < lc): out["E2"] = out["E3"] = "FALSE"; return finish("ACQUISITION_FAILURE", "OPENED_BEFORE_LIFT")
        if N - 1 >= ga + p["LIFT_BUDGET"] + p["K_LIFT"] - 1: out["E2"] = out["E3"] = "FALSE"; return finish("ACQUISITION_FAILURE", "NOT_LIFTED")
        return finish("TIMEOUT")
    # lift confirmed
    rel_ok = None
    if ro is not None and gl is None:
        zrel = box[ro, 2] - p["REST_Z"]; sub = "OUTSIDE" if dxy[ro] >= p["R_REL"] else ("TOO_HIGH" if zrel > p["Z_REL"] else None); rel_ok = sub is None
        if rel_ok: out["intended_release"] = ro; ev["intended_release"] = ro + t0
        else: ev["unintended_release"] = ro + t0; out["unintended_subtype"] = sub
    end_ret = lc + p["RET_WIN"]
    if out["intended_release"] is not None: end_ret = min(end_ret, out["intended_release"])
    lost_in_window = (gl is not None and gl < end_ret) or (rel_ok is False and ro < end_ret)
    if lost_in_window: out["E2"] = "FALSE"
    elif N - 1 >= end_ret: out["E2"] = "TRUE"
    else: out["E2"] = "CENSORED"
    ev["retention_window_end"] = end_ret + t0; out["window_len"] = end_ret - lc
    if gl is not None: out["E3"] = "FALSE"; return finish("CARRY_DROP", "EARLY_SLIP" if gl - lc <= p["EARLY_SLIP"] else "LATE_CARRY_DROP")
    if ro is None: return finish("TIMEOUT")                                                                               # carry unresolved at the end of the observation
    if rel_ok is False: out["E3"] = "FALSE"; return finish("RELEASE_OUTSIDE_TARGET", out["unintended_subtype"])
    ok = (dxy < p["PLACE_XY"]) & (np.abs(box[:, 2] - p["REST_Z"]) < p["PLACE_Z"]) & (a["v"] < p["REST_V"]); ts = first_run(ok, ro, ro + p["SETTLE_BUDGET"] - p["K_SETTLE"] + 1, p["K_SETTLE"])
    if ts is not None: ev["placement_settled"] = ts + t0; out["E3"] = "TRUE"; return finish("PLACEMENT_SUCCESS")
    if N - 1 >= ro + p["SETTLE_BUDGET"]: out["E3"] = "FALSE"; return finish("PLACEMENT_FAILURE")
    return finish("TIMEOUT")
def _margins(tr, p, core):
    """signed margins at the decisive steps (positive = inside the 'satisfied' side of the threshold, in the threshold's unit)"""
    a = _arrays(tr); g, box = a["g"], a["box"]; z0 = float(tr["meta"]["box_z0"]); rise = box[:, 2] - z0; t0 = core["t0"]; ev = core["events"]; m = {}
    if "grasp_acquired" in ev:
        i = ev["grasp_acquired"] - t0; s = slice(i, i + p["K_ACQ"]); m["acq_force_N"] = float(min(a["FL"][s].min(), a["FR"][s].min()) - p["F_ACQ"])
    if "lift_confirmed" in ev:
        i = ev["lift_confirmed"] - t0; m["lift_rise_m"] = float(rise[i:i + p["K_LIFT"]].min() - p["R_LIFT"])
    if "release_onset" in ev:
        i = ev["release_onset"] - t0; ga = ev["grasp_acquired"] - t0; m["release_onset_dg_m"] = float(g[i] - g[ga:i + 1].min() - p["OPEN_RISE"]); dxy = float(np.linalg.norm(box[i, :2] - np.array([p["TRAY_X"], p["TRAY_Y"]]))); m["release_dxy_m"] = float(p["R_REL"] - dxy); m["release_height_m"] = float(p["Z_REL"] - (box[i, 2] - p["REST_Z"]))
    if "grasp_lost" in ev:
        i = ev["grasp_lost"] - t0; s = slice(i, i + p["K_LOSS"]); m["loss_force_N"] = float(p["F_HELD"] - np.maximum(np.minimum(a["FL"][s], a["FR"][s]), 0).max())
    if "placement_settled" in ev:
        i = ev["placement_settled"] - t0; s = slice(i, i + p["K_SETTLE"]); dxy = np.linalg.norm(box[s, :2] - np.array([p["TRAY_X"], p["TRAY_Y"]]), axis=1); m["place_dxy_m"] = float(p["PLACE_XY"] - dxy.max()); m["place_z_m"] = float(p["PLACE_Z"] - np.abs(box[s, 2] - p["REST_Z"]).max()); m["place_v_ms"] = float(p["REST_V"] - a["v"][s].max())
    return m
def label(tr, p=PARAMS):
    """-> dict(terminal, subtype, events, margins, endpoints{E1,E2,E3: {value, ambiguous, adverse, favourable}}, ambiguous_terminal, observed{...}, params_hash)."""
    core = _core(tr, p); out = dict(terminal=core["terminal"], subtype=core["subtype"], events=core["events"], params_hash=params_hash(p), observed=dict(rows=core["N"], t0=core["t0"], abs_end=core["t0"] + core["N"], horizon_reached=bool(core["t0"] + core["N"] >= p["T_MAX"]), window_len=core.get("window_len")))
    out["margins"] = _margins(tr, p, core) if core["terminal"] != "INVALID" else {}
    alt = []
    for k, d in BANDS.items():
        for s in (-1, 1): q = dict(p); q[k] = p[k] + s * d; alt.append(_core(tr, q))
    for k in KBANDS:
        for s in (-1, 1): q = dict(p); q[k] = max(1, p[k] + s); alt.append(_core(tr, q))
    order = {"FALSE": 0, "CENSORED": 1, "TRUE": 2}; ep = {}
    for e in ("E1", "E2", "E3"):
        v = core[e]; vals = {x[e] for x in alt} | {v}
        if v == "INVALID": ep[e] = dict(value=v, ambiguous=False, adverse=v, favourable=v); continue
        nz = [x for x in vals if x in order]; ep[e] = dict(value=v, ambiguous=bool(len(vals) > 1), adverse=min(nz, key=lambda x: order[x]) if nz else v, favourable=max(nz, key=lambda x: order[x]) if nz else v)
    out["endpoints"] = ep; out["ambiguous_terminal"] = bool({x["terminal"] for x in alt} - {core["terminal"]})
    q = dict(p); q["R_LIFT"] = 0.05; out["sensitivity_R_LIFT_0p05"] = {e: _core(tr, q)[e] for e in ("E1", "E2", "E3")}
    out["short_window"] = bool(core.get("window_len") is not None and core["window_len"] < p["SHORT_WINDOW"]); return out
def resolved(tr, p=PARAMS):
    """True once the terminal outcome is decided (the runner may stop the branch); TIMEOUT = undecided."""
    return _core(tr, p)["terminal"] != "TIMEOUT"
