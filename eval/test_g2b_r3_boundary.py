#!/usr/bin/env python3
"""r3 boundary consistency: the release-confirmation budget boundary (confirmed-state onset at t_cmd+30 ... +50, every truncation length) must never produce contradictory labels.  Synthetic only.
Run: python3 eval/test_g2b_r3_boundary.py -> outputs/g2b_impl/r3_boundary_report.json"""
import sys, json, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import g2b_labeller as L, test_g2b_labeller as T
P = L.PARAMS; B, K = P["CONFIRM_BUDGET"], P["K_REL"]; FAIL_TERMS = {"PLACEMENT_FAILURE", "CARRY_DROP", "RELEASE_OUTSIDE_TARGET", "ACQUISITION_FAILURE"}
def late(d, n):
    t = T.mod(T.base(ro=100), jam=True); R0 = 100 + d; t["nL"][R0:] = 0; t["nR"][R0:] = 0; t["FL"][R0:] = 0; t["FR"][R0:] = 0; t["g_meas"][R0:] = 0.0295
    return {k: (v[:n] if hasattr(v, "ndim") else v) for k, v in t.items()}
viol = []; table = {}; amb = {}
for d in range(30, 51):
    full = L.label(late(d, 220)); table[d] = (full["terminal"], full["subtype"], full["endpoints"]["E3"]["value"], full["e3_decision_row"])
    amb[d] = full["endpoints"]["E3"]["ambiguous"]
    want_ok = d <= B                                                                                                       # start <= t_cmd + 40 is verified
    if want_ok != (full["terminal"] == "PLACEMENT_SUCCESS"): viol.append(("budget rule", d, full["terminal"]))
    if not want_ok and (full["subtype"] != "RELEASE_NOT_CONFIRMED" or full["e3_decision_row"] != 100 + B + K - 1): viol.append(("not-confirmed rule", d, full["subtype"], full["e3_decision_row"]))
    for n in range(100, 221, 1):
        o = L.label(late(d, n)); e3 = o["endpoints"]["E3"]["value"]; ev = o["events"]; term = o["terminal"]; N = n
        if (e3 == "TRUE") != (term == "PLACEMENT_SUCCESS"): viol.append(("E3 TRUE iff SUCCESS", d, n))
        if e3 == "FALSE" and term not in FAIL_TERMS: viol.append(("E3 FALSE needs a failure terminal", d, n, term))
        if e3 == "CENSORED" and term != "TIMEOUT": viol.append(("E3 CENSORED iff TIMEOUT", d, n, term))
        if "release_confirmed" in ev and o["subtype"] == "RELEASE_NOT_CONFIRMED": viol.append(("verified yet not confirmed", d, n))
        if "release_confirmed" in ev and ev["release_confirmed"] > N - 1: viol.append(("verified beyond the observed rows", d, n))
        if "release_confirmed" in ev and ev["release_confirmed"] != ev["release_confirmed_onset"] + K - 1: viol.append(("verification != onset + K - 1", d, n))
        if o["e3_decision_row"] is not None and o["e3_decision_row"] > N - 1: viol.append(("decision uses unobserved rows", d, n))
        if term == "PLACEMENT_SUCCESS" and not (ev["placement_settled"] >= ev["release_confirmed"] > ev["release_command"]): viol.append(("settle before verification", d, n))
        if e3 in ("TRUE", "FALSE") and e3 != full["endpoints"]["E3"]["value"]: viol.append(("decided label differs from the full-trace label", d, n, e3))
        if o["endpoints"]["E1"]["value"] != full["endpoints"]["E1"]["value"] and o["endpoints"]["E1"]["value"] != "CENSORED": viol.append(("E1 changed by truncation", d, n))
        if o["endpoints"]["E2"]["value"] not in (full["endpoints"]["E2"]["value"], "CENSORED"): viol.append(("E2 changed by truncation", d, n))
    # the exact decision rows: d = 40 needs rows through t_cmd+42 to verify; d = 41 is rejected at t_cmd+42
rep = dict(verdict="PASS" if not viol else "FAIL", n_violations=len(viol), violations=viol[:10], table={str(k): v for k, v in table.items()}, e3_ambiguous_flag_by_onset=amb, note="CONFIRM_BUDGET itself is not in the ambiguity bands: onsets t_cmd+39 and +40 are flagged ambiguous by OTHER perturbations (K_REL, OPEN_CONFIRM, ...), but onsets +41 and +42 (one and two rows beyond the budget) are NOT flagged (r3 proposal: add CONFIRM_BUDGET +-1)")
out = ROOT / "outputs/g2b_impl"; (out / "r3_boundary_report.json").write_text(json.dumps(rep, indent=1, default=str))
if viol: (out / f"r3_boundary_report_FAIL_{time.strftime('%H%M%S')}.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(dict(verdict=rep["verdict"], n_violations=len(viol), examples=viol[:5]))); print({d: table[d][:3] for d in (39, 40, 41, 42)}); print({d: amb[d] for d in (39, 40, 41, 42)})
