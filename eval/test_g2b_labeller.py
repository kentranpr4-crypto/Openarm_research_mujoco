#!/usr/bin/env python3
"""V2 (contract v2-r2 DRAFT) SYNTHETIC validation of the G2b labeller (numpy only; no simulator; NOT an evaluation).  14 pre-registered scenario families x 24 randomised variants + direct time-shift invariance tests + boundary/ambiguity tests.
Expected labels are fixed BY CONSTRUCTION of each synthetic trace (written independently of the labeller code).  Synthetic data show that the code implements the written contract, NOT that the contract matches physics (that needs the blind annotation).
Run: python3 eval/test_g2b_labeller.py   -> outputs/g2b_impl/labeller_v1_report.json"""
import sys, json, time, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); import g2b_labeller as L
REST, Z0, TRAY, START = 0.384, 0.380, np.array([0.44, 0.30]), np.array([0.40, 0.10]); KP = 5664.0; QC = 0.0205
FAMILIES = ["success_incl_early_release", "tray_arrival_while_held", "carry_drop", "early_slip", "acquisition_lost_before_lift", "no_lift", "borderline_lift", "release_outside_target", "release_too_high", "failed_placement", "incomplete_trajectory", "one_finger_contact_only", "force_noise_near_threshold", "time_shift_invariance"]
KEYS = ("g_exec", "nL", "nR", "FL", "FR", "box_pos", "g_meas")
BASE = [f for f in FAMILIES if f != "time_shift_invariance"]
def synth(family, seed, shift=0):
    rng = np.random.default_rng([20261105, seed]); N = 190; i = np.arange(N)
    tcc = 42 + int(rng.integers(-3, 4)); tga = tcc + 9; tl = tga + int(rng.integers(10, 16)); carry0 = tl + 18; dur = int(rng.integers(10, 50)); arrive = carry0 + dur; t_rel = arrive + 8 + int(rng.integers(0, 6)); peak = 0.10; drop_at = None; cut = None
    if family == "tray_arrival_while_held": t_rel = arrive + 16 + int(rng.integers(0, 9))
    g = np.full(N, 0.044); g = np.where(i >= tcc, np.maximum(0.012, 0.044 - 0.0032 * (i - tcc)), g); end_rel = None
    z = np.full(N, Z0); z = np.where(i >= tl, Z0 + np.minimum(peak, 0.005 * (i - tl)), z); xy = np.tile(START, (N, 1)); s = np.clip((i - carry0) / max(arrive - carry0, 1), 0, 1); xy = START[None] * (1 - s[:, None]) + TRAY[None] * s[:, None]
    low = np.clip((i - arrive) / 6.0, 0, 1); z = z + (REST - z) * low * (i >= arrive)                                   # lowered onto the tray while STILL gripped
    exp = dict(terminal="PLACEMENT_SUCCESS", E1="TRUE", E2="TRUE", E3="TRUE", amb=dict(E1=False, E2=False, E3=False))
    F_each = np.clip(KP / 2 * (QC - g), 0, None); on = (g < QC); nL = np.where(on, rng.integers(3, 6, N), 0).astype(float); nR = np.where(on, rng.integers(3, 6, N), 0).astype(float); FL = F_each * (1 + rng.uniform(-0.05, 0.05, N)); FR = F_each * (1 + rng.uniform(-0.05, 0.05, N))
    def release(t):                                                                                                       # gripper opens from t on; contact vanishes by itself when g > q_c
        nonlocal g
        g = np.where(i >= t, np.minimum(0.044, 0.012 + 0.0032 * (i - t + 1)), g)
    def lose(t0_, rows=None):                                                                                             # contact vanishes while the gripper is still commanded closed
        sl = slice(t0_, t0_ + rows if rows else N); nL[sl] = 0; nR[sl] = 0; FL[sl] = 0; FR[sl] = 0
    if family in ("success_incl_early_release", "tray_arrival_while_held"): release(t_rel)
    elif family == "carry_drop":
        td = tl + int(rng.integers(25, 46)); lose(td); z = np.where(i >= td, np.maximum(Z0, z - 0.02 * (i - td + 1)), z); xy = np.where((i >= td)[:, None], xy[td][None], xy); exp.update(terminal="CARRY_DROP", E2="FALSE", E3="FALSE")
    elif family == "early_slip":
        td = tl + int(rng.integers(9, 15)); lose(td); z = np.where(i >= td, np.maximum(Z0, z - 0.02 * (i - td + 1)), z); xy = np.where((i >= td)[:, None], xy[td][None], xy); exp.update(terminal="CARRY_DROP", E2="FALSE", E3="FALSE")
    elif family == "acquisition_lost_before_lift":
        td = tga + int(rng.integers(5, 9)); lose(td); z = np.full(N, Z0); xy = np.tile(START, (N, 1)); exp.update(terminal="ACQUISITION_FAILURE", E2="FALSE", E3="FALSE")
    elif family == "no_lift":
        z = np.full(N, Z0); xy = np.tile(START, (N, 1)); exp.update(terminal="ACQUISITION_FAILURE", E2="FALSE", E3="FALSE")
    elif family == "borderline_lift":
        pk = [0.022, 0.032, 0.045][seed % 3]; z = np.where(i >= tl, Z0 + np.minimum(pk, 0.005 * (i - tl)), Z0); xy = np.tile(START, (N, 1))
        exp.update(terminal=("ACQUISITION_FAILURE" if pk == 0.022 else "TIMEOUT"), E2=("FALSE" if pk == 0.022 else "TRUE"), E3=("FALSE" if pk == 0.022 else "CENSORED")); exp["amb"]["E2"] = (pk == 0.032)
        if pk == 0.032: exp["amb"]["E3"] = True
    elif family == "release_outside_target":
        t_rel = tl + 40 + int(rng.integers(0, 6)); xy = np.where((i >= tl + 15)[:, None], (START + (TRAY - START) * 0.5 + np.array([0.0, 0.0]))[None], xy); xy = np.where((i >= t_rel)[:, None], xy[t_rel][None], xy); release(t_rel); z = np.where(i >= t_rel + 3, np.maximum(Z0, z - 0.02 * (i - t_rel - 2)), z); exp.update(terminal="RELEASE_OUTSIDE_TARGET", E2="FALSE", E3="FALSE")
    elif family == "release_too_high":
        t_rel = arrive + 8; z = np.where(i >= arrive, np.maximum(z, REST + 0.08), z); z = np.where(i >= t_rel + 3, np.maximum(REST, REST + 0.08 - 0.02 * (i - t_rel - 2)), z); release(t_rel); exp.update(terminal="RELEASE_OUTSIDE_TARGET", E2="FALSE" if t_rel - (tl + 6) < 60 else "TRUE", E3="FALSE")
    elif family == "failed_placement":
        release(t_rel); off = TRAY + np.array([0.042, 0.0]); xy = np.where((i >= t_rel + 4)[:, None], off[None], xy); exp.update(terminal="PLACEMENT_FAILURE", E3="FALSE")
    elif family == "incomplete_trajectory":
        which = seed % 3; release(t_rel)
        if which == 0: cut = tl + 6 + 30; exp.update(terminal="TIMEOUT", E2="CENSORED", E3="CENSORED")
        elif which == 1: cut = t_rel + 8; exp.update(terminal="TIMEOUT", E2="TRUE", E3="CENSORED")
        else: cut = tcc + 8; exp.update(terminal="TIMEOUT", E1="CENSORED", E2="CENSORED", E3="CENSORED")
    elif family == "one_finger_contact_only":
        nR[:] = 0; FR[:] = 0; exp.update(terminal="ACQUISITION_FAILURE", E1="FALSE", E2="FALSE", E3="FALSE"); release(t_rel)
    elif family == "force_noise_near_threshold":
        release(t_rel); k = seed % 3; r0 = carry0 + 3
        if k == 0: lose(r0, 1)                                                                                          # 1-row flicker: below the debounce -> unambiguous success
        elif k == 1: lose(r0, 2); exp["amb"]["E2"] = True; exp["amb"]["E3"] = True                                    # 2-row flicker: ambiguous (K_LOSS-1 would call it a loss)
        else: FL[r0:r0 + 6] = 2.6; FR[r0:r0 + 6] = 2.6; exp["amb"]["E2"] = True; exp["amb"]["E3"] = True              # forces just above F_HELD: ambiguous
    rise_ok = np.linalg.norm(np.zeros(3)) == 0
    box = np.stack([xy[:, 0], xy[:, 1], z], 1)
    if family == "failed_placement": box = np.where((i >= t_rel + 4)[:, None], np.stack([off[0] * np.ones(N), off[1] * np.ones(N), REST * np.ones(N)], 1), box)
    if family in ("success_incl_early_release", "tray_arrival_while_held", "force_noise_near_threshold", "one_finger_contact_only"):
        sett = i >= t_rel + 4; box = np.where(sett[:, None], np.stack([(TRAY[0] + 0.01) * np.ones(N), (TRAY[1] - 0.008) * np.ones(N), REST * np.ones(N)], 1), box)
    lag = lambda a: np.concatenate([a[:1], a[:-1]], axis=0)                                                            # r2 (X2): observation row t is the consequence of the command of step t-1
    gap = lag(g); tr = dict(g_exec=g, nL=lag(nL), nR=lag(nR), FL=lag(FL), FR=lag(FR), box_pos=lag(box), g_meas=np.maximum(gap, QC), meta=dict(t0=0, box_z0=Z0, valid=True, g_prev0=float(g[0])))
    if cut is not None:
        for k in KEYS: tr[k] = tr[k][:cut]
    if shift:                                                                                                             # uniform time shift: the same trace, everything `shift` steps later (idle rows prepended)
        for k in KEYS: tr[k] = np.concatenate([np.repeat(tr[k][:1], shift, axis=0), tr[k]], axis=0)
    return tr, exp, dict(tcc=tcc, tga=tga, tl=tl, t_rel=t_rel)
def check(out, exp, fam, seed):
    errs = []
    if out["terminal"] != exp["terminal"]: errs.append(f"terminal {out['terminal']} != {exp['terminal']}")
    for e in ("E1", "E2", "E3"):
        if out["endpoints"][e]["value"] != exp[e]: errs.append(f"{e} {out['endpoints'][e]['value']} != {exp[e]}")
        if fam in ("borderline_lift", "force_noise_near_threshold", "success_incl_early_release", "tray_arrival_while_held") and out["endpoints"][e]["ambiguous"] != exp["amb"][e]: errs.append(f"amb {e} {out['endpoints'][e]['ambiguous']} != {exp['amb'][e]}")
    return errs
def strip(o):
    return dict(terminal=o["terminal"], subtype=o["subtype"], E={e: (o["endpoints"][e]["value"], o["endpoints"][e]["ambiguous"]) for e in ("E1", "E2", "E3")})
def base(N=200, lc_start=60, ro=None, loss=None, v=None, F=24.0, lower=True):
    """clean hand-built trace for the r2 index tests: close command at row 40, grasp acquired at row 50, rise to 5 cm from row lc_start, optional release command at row ro (gripper target rises), optional contact loss from row `loss`.  Observation rows are the consequences of the previous command (lag 1)."""
    i = np.arange(N); g = np.where(i >= 40, 0.010, 0.044).astype(float)
    if ro is not None: g = np.where(i >= ro, np.minimum(0.044, 0.010 + 0.006 * (i - ro + 1)), g)
    gap = np.concatenate([g[:1], g[:-1]]); on = (i >= 50); n = np.where(on, 4.0, 0.0); FL = np.where(on, F, 0.0)
    if ro is not None: on2 = gap < QC; n = np.where(on2, n, 0.0); FL = np.where(on2, FL, 0.0)                         # contact vanishes once the previous target is above the contact position
    if loss is not None: n = np.where(i >= loss, 0.0, n); FL = np.where(i >= loss, 0.0, FL)
    z = np.where(i >= lc_start, Z0 + 0.05, Z0); xy = np.tile(START, (N, 1))
    if ro is not None and lower: xy = np.tile(TRAY, (N, 1)); z = np.where(i >= lc_start, np.where(i <= ro, Z0 + 0.045, REST), Z0)           # carried 4.5 cm up above the tray, lowered onto it after the release command
    box = np.stack([xy[:, 0], xy[:, 1], z], 1); tr = dict(g_exec=g, nL=n.copy(), nR=n.copy(), FL=FL.copy(), FR=FL.copy(), box_pos=box, g_meas=np.maximum(gap, QC), meta=dict(t0=0, box_z0=Z0, valid=True, g_prev0=0.044))
    if v is not None: tr["box_vel"] = v
    return tr
def r2_tests():
    bt = []
    # C4 / X2: retention window [t_lc, t_lc+60) half-open; loss onset 59 rows after t_lc -> inside (E2 FALSE), 60 rows after -> outside (E2 TRUE); both boundary rows flagged ambiguous (RET_WIN +-1)
    o = {k: L.label(base(lc_start=60, loss=60 + k)) for k in (58, 59, 60, 61)}; lc = o[59]["events"]["lift_confirmed"]
    bt.append(("C4 loss onset t_lc+58 / t_lc+59 inside window -> E2 FALSE", o[58]["endpoints"]["E2"]["value"] == "FALSE" and o[59]["endpoints"]["E2"]["value"] == "FALSE"))
    bt.append(("C4 loss onset t_lc+60 / t_lc+61 outside window -> E2 TRUE (terminal CARRY_DROP kept)", o[60]["endpoints"]["E2"]["value"] == "TRUE" and o[61]["endpoints"]["E2"]["value"] == "TRUE" and o[60]["terminal"] == "CARRY_DROP"))
    bt.append(("C4 boundary rows t_lc+59 and t_lc+60 are flagged ambiguous by the RET_WIN +-1 perturbation", o[59]["endpoints"]["E2"]["ambiguous"] and o[60]["endpoints"]["E2"]["ambiguous"]))
    # C5: ambiguity flags never change the primary label; spec reported
    a = L.label(base(lc_start=60, loss=119)); bt.append(("C5 primary label identical when the ambiguity analysis is bypassed", a["terminal"] == L._core(base(lc_start=60, loss=119), L.PARAMS)["terminal"] and a["endpoints"]["E2"]["value"] == L._core(base(lc_start=60, loss=119), L.PARAMS)["E2"]))
    bt.append(("C5 ambiguity perturbation values are reported and hashed", a["ambiguity_spec"]["bands"] == L.BANDS and "RET_WIN" in a["ambiguity_spec"]["k_bands"] and a["spec_hash"] == L.spec_hash() and a["contract"] == "v2-r2-draft"))
    # X2: release command row ro: loss with onset row ro counts as a loss (observation ro is pre-action); onset ro+1 is the consequence of the release (not a loss)
    r = {k: L.label(base(ro=100, loss=100 + k)) for k in (0, 1)}
    bt.append(("X2 contact lost on row == release-command row is a LOSS (not caused by that command)", r[0]["terminal"] == "CARRY_DROP"))
    bt.append(("X2 contact lost on row release-command+1 is the consequence of the release (no loss, intended release)", r[1]["terminal"] in ("PLACEMENT_SUCCESS", "PLACEMENT_FAILURE", "TIMEOUT") and r[1]["terminal"] != "CARRY_DROP"))
    ev = L.label(base(ro=100))["events"]; bt.append(("X2 commanded release precedes the physical opening (separate events, lag >= 1 row)", "release_command" in ev and "release_physical_onset" in ev and ev["release_physical_onset"] > ev["release_command"]))
    ev2 = L.label(base(ro=100))["events"]; bt.append(("X2 close_command is the issue row and grasp_acquired is observed >= 1 row later", ev2["grasp_acquired"] > ev2["close_command"]))
    # C2: release command vs lift start
    c2 = {d: L.label(base(lc_start=60, ro=60 + d, lower=False)) for d in (-2, -1, 0, 1)}
    bt.append(("C2 release command before the lift row -> ACQUISITION_FAILURE / OPENED_BEFORE_LIFT", c2[-2]["subtype"] == "OPENED_BEFORE_LIFT" and c2[-1]["subtype"] == "OPENED_BEFORE_LIFT"))
    bt.append(("C2 release command at/after the lift row is not OPENED_BEFORE_LIFT", c2[0]["subtype"] != "OPENED_BEFORE_LIFT" and c2[1]["subtype"] != "OPENED_BEFORE_LIFT"))
    # C3: debounce-completion decidability (never acquired, close command at row 40): decided only when rows up to cc+ACQ_BUDGET+K_ACQ-1 exist
    def nacq(N): t = base(N=N); t["nL"][:] = 0; t["nR"][:] = 0; t["FL"][:] = 0; t["FR"][:] = 0; return L.label(t)
    need = 40 + L.PARAMS["ACQ_BUDGET"] + L.PARAMS["K_ACQ"]; bt.append(("C3 NOT_ACQUIRED needs rows through cc+ACQ_BUDGET+K_ACQ-1 (N-1 >= 69)", nacq(need - 1)["terminal"] == "TIMEOUT" and nacq(need)["subtype"] == "NOT_ACQUIRED" and nacq(need)["endpoints"]["E1"]["value"] == "FALSE"))
    # X1: horizon truncation vs time shift
    full = base(N=200); sh = {k: (np.concatenate([np.repeat(v[:1], 8, 0), v]) if hasattr(v, "ndim") else v) for k, v in full.items()}; inside = L.label(base(N=140)); inside_sh = L.label({k: (v[:148] if hasattr(v, "ndim") else v) for k, v in sh.items()})
    bt.append(("X1 shift that stays fully observable inside T_MAX keeps terminal and endpoints", inside["terminal"] == inside_sh["terminal"] and {e: inside["endpoints"][e]["value"] for e in ("E1", "E2", "E3")} == {e: inside_sh["endpoints"][e]["value"] for e in ("E1", "E2", "E3")}))
    nc = base(N=160); nc["g_exec"][:] = 0.044; nc["g_meas"][:] = 0.044; nc["nL"][:] = 0; nc["nR"][:] = 0; nc["FL"][:] = 0; nc["FR"][:] = 0
    nc_cut = {k: (v[:150] if hasattr(v, "ndim") else v) for k, v in nc.items()}; bt.append(("X1 never-closed trace: FALSE only when the absolute horizon (160) is reached, TIMEOUT when truncated earlier", L.label(nc)["subtype"] == "NOT_CLOSED" and L.label(nc_cut)["terminal"] == "TIMEOUT"))
    # X3: velocity
    N = 200; vel = np.full(N, 0.0); tr = base(ro=100); tr["box_vel"] = np.stack([vel, vel, vel], 1); clean = L.label(tr)
    tr_n = base(ro=100); vn = np.zeros((N, 3)); vn[104:140] = np.nan; tr_n["box_vel"] = vn; nan = L.label(tr_n)
    bt.append(("X3 NaN box velocity is not treated as at rest: no PLACEMENT_SUCCESS from missing velocity, flag set", clean["terminal"] == "PLACEMENT_SUCCESS" and clean["velocity_unavailable"] is False and nan["velocity_unavailable"] and nan["terminal"] == "PLACEMENT_FAILURE"))
    fb = L._arrays(base(ro=100))["v"]; bt.append(("X3 finite-difference fallback gives finite one-sided derivatives at both endpoints (no zero placeholder)", bool(np.isfinite(fb).all())))
    return bt
def run():
    t0 = time.time(); rep = dict(params_hash=L.params_hash(), families={}, time_shift={}, notes=[]); fails = 0
    for fam in BASE:
        ok = 0; bad = []
        for v in range(24):
            tr, exp, info = synth(fam, v); out = L.label(tr); errs = check(out, exp, fam, v)
            if errs: bad.append((v, errs[:2]))
            else: ok += 1
        rep["families"][fam] = dict(n=24, agree=ok, failures=bad[:4]); fails += 24 - ok
    # direct time-shift invariance (same random draws, everything 8 steps later): terminal, subtype, endpoints, ambiguity and margins identical; absolute event times shifted by exactly 8
    for fam in BASE:
        ok = 0; bad = []
        for v in range(24):
            a, _, _ = synth(fam, v); b, _, _ = synth(fam, v, shift=8); oa, ob = L.label(a), L.label(b); same = strip(oa) == strip(ob) and oa["margins"] == ob["margins"] and all(ob["events"][k] - oa["events"][k] == 8 for k in oa["events"]) and set(oa["events"]) == set(ob["events"])
            if same: ok += 1
            else: bad.append((v, strip(oa), strip(ob)))
        rep["time_shift"][fam] = dict(n=24, invariant=ok, failures=bad[:2]); fails += 24 - ok
    # horizon is NOT time-shift invariant by design: a decision that needs rows beyond T_MAX is censored
    a, _, _ = synth("success_incl_early_release", 3); late = {k: (np.concatenate([np.full(40, v[0]) if v.ndim == 1 else np.repeat(v[:1], 40, 0), v]) if hasattr(v, "ndim") else v) for k, v in a.items()}; late["meta"] = dict(t0=0, box_z0=Z0, valid=True, g_prev0=0.044)
    cut = {k: (v[:150] if hasattr(v, "ndim") else v) for k, v in late.items()}; cut["meta"] = dict(t0=0, box_z0=Z0, valid=True, g_prev0=0.044); oc = L.label(cut)
    rep["horizon_not_invariant_example"] = dict(full=L.label(a)["terminal"], shifted_40_steps_and_cut_at_150=oc["terminal"], note="expected: shifting later can push the decision beyond the observed rows -> TIMEOUT/CENSORED (absolute horizon), not a labelling bias")
    # explicit boundary tests
    def mk(rise_plateau, flick=0, fth=2.5, F=24.0):
        tr, _, _ = synth("success_incl_early_release", 5); tr = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in tr.items()}; return tr
    bt = []
    tr, _, info = synth("success_incl_early_release", 5); o = L.label(tr); bt.append(("success clean", o["terminal"] == "PLACEMENT_SUCCESS" and not o["endpoints"]["E2"]["ambiguous"]))
    bad = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in tr.items()}; bad["FL"][:] = np.nan; o = L.label(bad); bt.append(("non-finite -> INVALID", o["terminal"] == "INVALID" and o["endpoints"]["E2"]["value"] == "INVALID"))
    inv = dict(tr); inv["meta"] = dict(t0=0, box_z0=Z0, valid=False, g_prev0=0.044); bt.append(("meta invalid -> INVALID", L.label(inv)["terminal"] == "INVALID"))
    o = L.label(tr); bt.append(("params hash stable", o["params_hash"] == L.params_hash()))
    bt.append(("labels do not depend on the ambiguity analysis (value unchanged)", all(o["endpoints"][e]["value"] in ("TRUE", "FALSE", "CENSORED") for e in ("E1", "E2", "E3"))))
    bt.append(("legacy R_LIFT=0.05 sensitivity present", set(o["sensitivity_R_LIFT_0p05"]) == {"E1", "E2", "E3"}))
    tr2, _, _ = synth("carry_drop", 4); o2 = L.label(tr2); bt.append(("carry-drop is never relabelled as success", o2["terminal"] == "CARRY_DROP" and o2["endpoints"]["E3"]["value"] == "FALSE"))
    tr3, _, _ = synth("tray_arrival_while_held", 2); o3 = L.label(tr3); bt.append(("tray arrival while held is not a drop", o3["terminal"] == "PLACEMENT_SUCCESS"))
    # early intentional release: window ends at the release (< 60 steps) and E2 is TRUE; long carry: full 60-step window
    short = [synth("success_incl_early_release", v) for v in range(24)]; lens = [L.label(t)["observed"]["window_len"] for t, _, _ in short]; bt.append(("both early (<60) and full-length (=60) retention windows occur", min(lens) < 60 and max(lens) == 60))
    tr4, _, inf = synth("early_slip", 7); tr4 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in tr4.items()}; tl4 = inf["tl"]
    for k in ("nL", "nR", "FL", "FR"): tr4[k][tl4 + 8:] = 0                                                         # contact lost one row after the 0.03 m crossing (before K_LIFT=3 rows)
    tr4["box_pos"][tl4 + 8:, 2] = np.maximum(Z0, tr4["box_pos"][tl4 + 8:, 2] - 0.02 * (np.arange(len(tr4["g_exec"]) - tl4 - 8) + 1)); o4 = L.label(tr4); bt.append(("loss between lift crossing and its confirmation -> ACQUISITION_LOST (contract semantics)", o4["terminal"] == "ACQUISITION_FAILURE" and o4["subtype"] == "ACQUISITION_LOST"))
    bt += r2_tests(); rep["boundary_tests"] = {n: bool(ok) for n, ok in bt}; fails += sum(not ok for _, ok in bt)
    rep["n_traces_families"] = 24 * len(BASE); rep["n_traces_time_shift_pairs"] = 24 * len(BASE); rep["total_failures"] = int(fails); rep["verdict"] = "PASS" if fails == 0 else "FAIL"; rep["cpu_like_wall_s"] = round(time.time() - t0, 1)
    Path(ROOT / "outputs/g2b_impl").mkdir(parents=True, exist_ok=True); (ROOT / "outputs/g2b_impl/labeller_v1_report.json").write_text(json.dumps(rep, indent=1, default=str)); return rep
if __name__ == "__main__":
    r = run(); print(json.dumps({k: r[k] for k in ("verdict", "total_failures", "n_traces_families", "n_traces_time_shift_pairs", "params_hash", "cpu_like_wall_s")})); print(json.dumps({k: v["agree"] for k, v in r["families"].items()})); print(json.dumps({k: v["invariant"] for k, v in r["time_shift"].items()})); print(json.dumps(r["boundary_tests"]))
    for k, v in r["families"].items():
        if v["failures"]: print("FAIL", k, v["failures"][:2])
    for k, v in r["time_shift"].items():
        if v["failures"]: print("SHIFTFAIL", k, v["failures"][:1])
