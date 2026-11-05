#!/usr/bin/env python3
"""G2b Outcome Contract v2-r1 LABELLER (B2), CONTRACT v2-r2 DRAFT - a PURE function of a logged branch trace.  NOT frozen; thresholds are the r1 values (development-derived, see the r1 document); nothing here was tuned on any new evaluation outcome.
Three layers (kept separate): (A) event timeline with timestamps and signed threshold margins, (B) ONE mutually exclusive terminal outcome, (C) binary endpoints E1/E2/E3 with value in {TRUE, FALSE, CENSORED, INVALID} and an INDEPENDENT ambiguity flag.
Ambiguity is determined by perturbation: an endpoint is `ambiguous` iff moving any decisive threshold inside its pre-registered band (BANDS / +-1 on the debounce lengths) changes its value; the label itself is never altered by the flag.
Trace schema (arrays of equal length N = observed rows; row i = observation before the action of step t0+i):
  g_exec [m] executed left-gripper target; nL,nR [count] finger-box contact POINTS (left/right finger mesh); FL,FR [N] summed contact normal force per finger; box_pos [m] world; box_vel [m/s] world (optional, else central difference);
  meta: t0 (int, absolute step of row 0), box_z0 [m] (box height at episode start), valid (bool).  Optional: Fh [N] palm force (reported, not used by the labels).
r2 TIME INDEXING (X2): row i = observation BEFORE the action of step t0+i.  The command g_exec[i] is issued AFTER row i; its first physical consequence is in row i+1.  Hence: COMMAND events (close_command, release_command) carry the ISSUE index i; OBSERVATION conjunctions (acquisition, held, loss) use the command that produced the row, g_app[i] = g_exec[i-1] (row 0: meta.g_prev0, else g_exec[0] and flagged).  Physical opening (g_meas, optional) is a separate timeline event.
r2 INDEXING RULES (C3/C4, frozen text): first_run(mask, lo, hi, k) = first START s in [lo, hi] with mask[s:s+k] all True and s+k <= N.  A failure/decision that needs a run starting at the last allowed start is declared only if rows up to hi+k-1 are observed (else TIMEOUT/CENSORED).  Retention window = rows [t_lc, t_end) HALF-OPEN, t_end = min(t_lc+RET_WIN, t_release_cmd+1) (row t_release_cmd is the last row not yet affected by the release command); a loss/unintended release is inside iff its onset row s < t_end (unintended release: command row < t_lc+RET_WIN).  E2 TRUE needs rows through t_end+K_LOSS-2 observed.  grasp_lost counts only if its onset row <= t_release_cmd.
r2-X4 (E3 requires a CONFIRMED physical release): events target_arrival (box inside the release region R_REL and <= Z_REL above rest for K_ARR rows, start in [t_lc, t_release_cmd]); release_command (commanded opening); release_physical_onset (measured finger rise, timeline only); release_confirmed = first_run(not clamped AND opened, from t_release_cmd+1, start <= t_release_cmd+CONFIRM_BUDGET, K_REL) where clamped = held (both fingers in contact with >= F_HELD) and opened = g_meas - running minimum of g_meas since t_ga >= OPEN_CONFIRM (8 mm); placement_settled = first_run(box in the 3 cm region at rest height, speed < REST_V, AND not clamped, from t_release_confirmed (= VERIFICATION row), K_SETTLE) - so the 10-row window can only START at/after the verified release.
X4 TIMESTAMPS (three distinct kinds): ONSET - release_command (issue row), release_physical_onset, release_confirmed_onset (first row of the confirmed state, rc); VERIFICATION - release_confirmed = rc + K_REL - 1 (the first row at which all K_REL required observations exist; the release is never marked verified earlier); DECISION - e3_decision_row (first observed row from which the E3 value is fixed: SUCCESS ts+K_SETTLE-1; NOT_SETTLED rv+SETTLE_BUDGET; RELEASE_NOT_CONFIRMED t_cmd+CONFIRM_BUDGET+K_REL-1; NO_TARGET_ARRIVAL t_cmd+K_ARR-1; None when undecided/other terminals).
X4 policy: no confirmed release within CONFIRM_BUDGET+K_REL-1 observed rows -> PLACEMENT_FAILURE/RELEASE_NOT_CONFIRMED (E3 FALSE); trace (or horizon) ends earlier -> TIMEOUT (E3 CENSORED, never TRUE); g_meas is REQUIRED (missing/non-finite -> INVALID).
Units: metres, newtons, seconds (dt = 0.05 s), steps.  World frame z up; tray centre (0.44, 0.30) m; resting box-centre height 0.384 m."""
import json, hashlib, numpy as np
DT = 0.05
PARAMS = dict(G_CLOSE=0.040, G_ACQ=0.030, F_ACQ=5.0, K_ACQ=5, ACQ_BUDGET=25, R_LIFT=0.03, K_LIFT=3, LIFT_BUDGET=40, F_HELD=2.5, K_LOSS=3, OPEN_RISE=0.004, K_OPEN=2, RET_WIN=60, EARLY_SLIP=15, SHORT_WINDOW=20, OPEN_CONFIRM=0.008, K_REL=3, K_ARR=3, CONFIRM_BUDGET=40, CONTRACT="v2-r2-x4-draft",
              TRAY_X=0.44, TRAY_Y=0.30, REST_Z=0.384, R_REL=0.05, Z_REL=0.05, PLACE_XY=0.03, PLACE_Z=0.006, REST_V=0.02, K_SETTLE=10, SETTLE_BUDGET=40, T_MAX=160)
BANDS = dict(R_LIFT=0.005, F_ACQ=1.0, F_HELD=0.5, OPEN_RISE=0.001, R_REL=0.005, Z_REL=0.005, PLACE_XY=0.003, PLACE_Z=0.0015, REST_V=0.004, OPEN_CONFIRM=0.002)       # +- perturbation of a decisive threshold [its unit]
KBANDS = ("K_ACQ", "K_LIFT", "K_LOSS", "K_OPEN", "K_SETTLE", "RET_WIN", "K_REL", "K_ARR")                                                                                         # +-1 step
TERMINALS = ("INVALID", "ACQUISITION_FAILURE", "CARRY_DROP", "RELEASE_OUTSIDE_TARGET", "PLACEMENT_FAILURE", "PLACEMENT_SUCCESS", "TIMEOUT")
def params_hash(p=PARAMS): return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
AMBIGUITY_SPEC = dict(bands=BANDS, k_bands=list(KBANDS), k_delta=1, extra=dict(R_LIFT_sensitivity=0.05))
def spec_hash(): return hashlib.sha256(json.dumps(dict(params=PARAMS, spec=AMBIGUITY_SPEC), sort_keys=True).encode()).hexdigest()
def first_run(mask, lo, hi, k):
    """first start s in [lo, hi] with mask[s:s+k] all True (fully inside the observed rows), else None"""
    n = len(mask); hi = min(hi, n - k)
    for s in range(max(lo, 0), hi + 1):
        if mask[s:s + k].all(): return s
    return None
def _arrays(tr):
    g = np.asarray(tr["g_exec"], float); N = len(g); g0 = tr["meta"].get("g_prev0"); a = dict(g=g, gap=np.concatenate([[g[0] if g0 is None else float(g0)], g[:-1]]), meas=(None if tr.get("g_meas") is None else np.asarray(tr["g_meas"], float)), nL=np.asarray(tr["nL"], float), nR=np.asarray(tr["nR"], float), FL=np.asarray(tr["FL"], float), FR=np.asarray(tr["FR"], float), box=np.asarray(tr["box_pos"], float))
    if "box_vel" in tr and tr["box_vel"] is not None: a["v"] = np.linalg.norm(np.asarray(tr["box_vel"], float), axis=1)
    else:                                                                                                   # X3: fallback finite differences; one-sided (first order) at the two endpoints, NEVER a zero placeholder
        v = np.full(N, np.nan)
        if N >= 2: v[0] = np.linalg.norm(a["box"][1] - a["box"][0]) / DT; v[-1] = np.linalg.norm(a["box"][-1] - a["box"][-2]) / DT
        if N > 2: v[1:-1] = np.linalg.norm(a["box"][2:] - a["box"][:-2], axis=1) / (2 * DT)
        a["v"] = v
    return a
def _core(tr, p):
    a = _arrays(tr); g, gap, box = a["g"], a["gap"], a["box"]; N = len(g); t0 = int(tr["meta"].get("t0", 0)); z0 = float(tr["meta"]["box_z0"]); rise = box[:, 2] - z0; tray = np.array([p["TRAY_X"], p["TRAY_Y"]]); dxy = np.linalg.norm(box[:, :2] - tray, axis=1)
    out = dict(events={}, terminal=None, subtype=None, E1="CENSORED", E2="CENSORED", E3="CENSORED", N=N, t0=t0, intended_release=None, vel_unavailable=False)
    ev = out["events"]; ok_all = all(np.isfinite(np.asarray(tr[k], float)).all() for k in ("g_exec", "nL", "nR", "FL", "FR", "box_pos")) and a["meas"] is not None and bool(np.isfinite(a["meas"]).all())
    if (not ok_all) or (not tr["meta"].get("valid", True)) or N < 2: out.update(terminal="INVALID", E1="INVALID", E2="INVALID", E3="INVALID"); return out
    horizon_reached = (t0 + N) >= p["T_MAX"]
    def finish(term, sub=None):
        out["terminal"] = term; out["subtype"] = sub; return out
    cc = first_run(g < p["G_CLOSE"], 0, N, 1)
    if cc is None:
        if horizon_reached: out["E1"] = out["E2"] = out["E3"] = "FALSE"; return finish("ACQUISITION_FAILURE", "NOT_CLOSED")
        return finish("TIMEOUT")
    ev["close_command"] = cc + t0
    held_acq = (a["nL"] >= 1) & (a["nR"] >= 1) & (a["FL"] >= p["F_ACQ"]) & (a["FR"] >= p["F_ACQ"]) & (gap < p["G_ACQ"]); ga = first_run(held_acq, cc + 1, cc + p["ACQ_BUDGET"], p["K_ACQ"])
    if ga is None:
        if N - 1 >= cc + p["ACQ_BUDGET"] + p["K_ACQ"] - 1: out["E1"] = out["E2"] = out["E3"] = "FALSE"; return finish("ACQUISITION_FAILURE", "NOT_ACQUIRED")
        return finish("TIMEOUT")
    ev["grasp_acquired"] = ga + t0; out["E1"] = "TRUE"; lc = first_run(rise >= p["R_LIFT"], ga, ga + p["LIFT_BUDGET"], p["K_LIFT"])
    held = (a["nL"] >= 1) & (a["nR"] >= 1) & (a["FL"] >= p["F_HELD"]) & (a["FR"] >= p["F_HELD"]); runmin = np.minimum.accumulate(g[ga:]); onset = first_run((g[ga:] - runmin) >= p["OPEN_RISE"], 0, N, p["K_OPEN"]); ro = (ga + onset) if onset is not None else None
    gl = first_run((~held) & (gap < p["G_ACQ"]), ga, N, p["K_LOSS"]); gl = gl if (gl is not None and (ro is None or gl <= ro)) else None
    if a["meas"] is not None:                                                                                           # physical opening (timeline only): measured finger position rises over its running minimum
        pm = np.minimum.accumulate(a["meas"][ga:]); po = first_run((a["meas"][ga:] - pm) >= p["OPEN_RISE"], 0, N, p["K_OPEN"])
        if po is not None: ev["release_physical_onset"] = ga + po + t0
    if lc is not None: ev["lift_confirmed"] = lc + t0
    if ro is not None: ev["release_command"] = ro + t0
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
    if out["intended_release"] is not None: end_ret = min(end_ret, out["intended_release"] + 1)
    lost_in_window = (gl is not None and gl < end_ret) or (rel_ok is False and ro < lc + p["RET_WIN"])
    if lost_in_window: out["E2"] = "FALSE"
    elif N - 1 >= end_ret + p["K_LOSS"] - 2: out["E2"] = "TRUE"
    else: out["E2"] = "CENSORED"
    ev["retention_window_end"] = end_ret + t0; out["window_len"] = end_ret - lc
    if gl is not None: out["E3"] = "FALSE"; return finish("CARRY_DROP", "EARLY_SLIP" if gl - lc <= p["EARLY_SLIP"] else "LATE_CARRY_DROP")
    if ro is None: return finish("TIMEOUT")                                                                               # carry unresolved at the end of the observation
    if rel_ok is False: out["E3"] = "FALSE"; return finish("RELEASE_OUTSIDE_TARGET", out["unintended_subtype"])
    inreg = (dxy < p["R_REL"]) & ((box[:, 2] - p["REST_Z"]) <= p["Z_REL"]); arr = first_run(inreg, lc, ro, p["K_ARR"])
    if arr is None:
        if N - 1 >= ro + p["K_ARR"] - 1: out["E3"] = "FALSE"; out["e3_decision_row"] = ro + p["K_ARR"] - 1 + t0; return finish("PLACEMENT_FAILURE", "NO_TARGET_ARRIVAL")
        return finish("TIMEOUT")
    ev["target_arrival"] = arr + t0
    pm = np.minimum.accumulate(a["meas"][ga:]); opened = np.zeros(N, bool); opened[ga:] = (a["meas"][ga:] - pm) >= p["OPEN_CONFIRM"]; rc = first_run((~held) & opened, ro + 1, ro + p["CONFIRM_BUDGET"], p["K_REL"])
    if rc is None:
        if N - 1 >= ro + p["CONFIRM_BUDGET"] + p["K_REL"] - 1: out["E3"] = "FALSE"; out["e3_decision_row"] = ro + p["CONFIRM_BUDGET"] + p["K_REL"] - 1 + t0; return finish("PLACEMENT_FAILURE", "RELEASE_NOT_CONFIRMED")
        return finish("TIMEOUT")
    rv = rc + p["K_REL"] - 1; ev["release_confirmed_onset"] = rc + t0; ev["release_confirmed"] = rv + t0
    ok = (dxy < p["PLACE_XY"]) & (np.abs(box[:, 2] - p["REST_Z"]) < p["PLACE_Z"]) & np.isfinite(a["v"]) & (a["v"] < p["REST_V"]) & (~held); ts = first_run(ok, rv, rv + p["SETTLE_BUDGET"] - p["K_SETTLE"] + 1, p["K_SETTLE"]); out["vel_unavailable"] = bool((~np.isfinite(a["v"][rv:rv + p["SETTLE_BUDGET"]])).any())
    if ts is not None: ev["placement_settled"] = ts + t0; out["E3"] = "TRUE"; out["e3_decision_row"] = ts + p["K_SETTLE"] - 1 + t0; return finish("PLACEMENT_SUCCESS")
    if N - 1 >= rv + p["SETTLE_BUDGET"]: out["E3"] = "FALSE"; out["e3_decision_row"] = rv + p["SETTLE_BUDGET"] + t0; return finish("PLACEMENT_FAILURE", "NOT_SETTLED")
    return finish("TIMEOUT")
def _margins(tr, p, core):
    """signed margins at the decisive steps (positive = inside the 'satisfied' side of the threshold, in the threshold's unit)"""
    a = _arrays(tr); g, box = a["g"], a["box"]; z0 = float(tr["meta"]["box_z0"]); rise = box[:, 2] - z0; t0 = core["t0"]; ev = core["events"]; m = {}
    if "grasp_acquired" in ev:
        i = ev["grasp_acquired"] - t0; s = slice(i, i + p["K_ACQ"]); m["acq_force_N"] = float(min(a["FL"][s].min(), a["FR"][s].min()) - p["F_ACQ"])
    if "lift_confirmed" in ev:
        i = ev["lift_confirmed"] - t0; m["lift_rise_m"] = float(rise[i:i + p["K_LIFT"]].min() - p["R_LIFT"])
    if "release_command" in ev:
        i = ev["release_command"] - t0; ga = ev["grasp_acquired"] - t0; m["release_onset_dg_m"] = float(g[i] - g[ga:i + 1].min() - p["OPEN_RISE"]); dxy = float(np.linalg.norm(box[i, :2] - np.array([p["TRAY_X"], p["TRAY_Y"]]))); m["release_dxy_m"] = float(p["R_REL"] - dxy); m["release_height_m"] = float(p["Z_REL"] - (box[i, 2] - p["REST_Z"]))
    if "grasp_lost" in ev:
        i = ev["grasp_lost"] - t0; s = slice(i, i + p["K_LOSS"]); m["loss_force_N"] = float(p["F_HELD"] - np.maximum(np.minimum(a["FL"][s], a["FR"][s]), 0).max())
    if "release_confirmed_onset" in ev and a["meas"] is not None:
        i = ev["release_confirmed_onset"] - t0; ga_ = ev["grasp_acquired"] - t0; pm_ = np.minimum.accumulate(a["meas"][ga_:]); m["release_open_m"] = float(((a["meas"][ga_:] - pm_)[i - ga_:i - ga_ + p["K_REL"]]).min() - p["OPEN_CONFIRM"])
    if "placement_settled" in ev:
        i = ev["placement_settled"] - t0; s = slice(i, i + p["K_SETTLE"]); dxy = np.linalg.norm(box[s, :2] - np.array([p["TRAY_X"], p["TRAY_Y"]]), axis=1); m["place_dxy_m"] = float(p["PLACE_XY"] - dxy.max()); m["place_z_m"] = float(p["PLACE_Z"] - np.abs(box[s, 2] - p["REST_Z"]).max()); m["place_v_ms"] = float(p["REST_V"] - a["v"][s].max())
    return m
def label(tr, p=PARAMS):
    """-> dict(terminal, subtype, events, margins, endpoints{E1,E2,E3: {value, ambiguous, adverse, favourable}}, ambiguous_terminal, observed{...}, params_hash)."""
    core = _core(tr, p); out = dict(terminal=core["terminal"], subtype=core["subtype"], events=core["events"], params_hash=params_hash(p), observed=dict(rows=core["N"], t0=core["t0"], abs_end=core["t0"] + core["N"], horizon_reached=bool(core["t0"] + core["N"] >= p["T_MAX"]), window_len=core.get("window_len")))
    out["contract"] = p["CONTRACT"]; out["spec_hash"] = spec_hash() if p is PARAMS else None; out["ambiguity_spec"] = AMBIGUITY_SPEC; out["velocity_unavailable"] = bool(core.get("vel_unavailable")); out["e3_decision_row"] = core.get("e3_decision_row"); out["g_prev0_assumed"] = tr["meta"].get("g_prev0") is None
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
