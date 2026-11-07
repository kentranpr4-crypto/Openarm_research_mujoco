#!/usr/bin/env python3
"""CONFIRM_BUDGET +-1 ambiguity sensitivity (SYNTHETIC / BOUNDARY tests only; pure python).  The change is limited to the ambiguity/sensitivity analysis: the PRIMARY labels (terminal, subtype, events, E1/E2/E3 values, decision rows) must be identical to the committed labeller (read from git ac0eabe) on every case.
Boundary: the confirmed state starts t_cmd+39 / +40 / +41 / +42 rows after the release command.  Run: python3 eval/test_g2b_confirm_budget.py -> outputs/g2b_impl/confirm_budget_report.json (a failing run is also kept as ..._FAIL_<time>.json)"""
import sys, json, time, subprocess, types, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import g2b_labeller as L, test_g2b_labeller as T
src = subprocess.run(["git", "show", "ac0eabe:eval/g2b_labeller.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout; OLD = types.ModuleType("old_labeller"); exec(compile(src, "old_labeller", "exec"), OLD.__dict__)
P = L.PARAMS; B, K = P["CONFIRM_BUDGET"], P["K_REL"]; RES = []
def t(name, ok, ev=None): RES.append(dict(test=name, result="PASS" if ok else "FAIL", evidence=ev)); print(("PASS " if ok else "FAIL ") + name, "" if ok else json.dumps(ev, default=str)[:400], flush=True)
def late(d, n=220):
    tr = T.mod(T.base(ro=100), jam=True); R0 = 100 + d; tr["nL"][R0:] = 0; tr["nR"][R0:] = 0; tr["FL"][R0:] = 0; tr["FR"][R0:] = 0; tr["g_meas"][R0:] = 0.0295
    return {k: (v[:n] if hasattr(v, "ndim") else v) for k, v in tr.items()}
def prim(o): return dict(terminal=o["terminal"], subtype=o["subtype"], events=o["events"], E={e: o["endpoints"][e]["value"] for e in ("E1", "E2", "E3")}, margins=o["margins"], e3_decision_row=o["e3_decision_row"], observed=o["observed"])
t("CONFIRM_BUDGET stays 40 (primary) and is now one of the +-1 perturbed parameters; params hash unchanged, spec hash changed", B == 40 and "CONFIRM_BUDGET" in L.KBANDS and L.params_hash() == OLD.params_hash() and L.spec_hash() != OLD.spec_hash(), dict(params_hash=L.params_hash()[:12], spec_old=OLD.spec_hash()[:12], spec_new=L.spec_hash()[:12]))
# boundary table
tab = {}; reg = []
for d in (39, 40, 41, 42):
    tr = late(d); new, old = L.label(tr), OLD.label(tr); tab[d] = dict(terminal=new["terminal"], subtype=new["subtype"], E3=new["endpoints"]["E3"]["value"], E3_ambiguous_new=new["endpoints"]["E3"]["ambiguous"], E3_ambiguous_old=old["endpoints"]["E3"]["ambiguous"], verified=new["events"].get("release_confirmed"), onset=new["events"].get("release_confirmed_onset"), decision_row=new["e3_decision_row"], same_primary=prim(new) == prim(old))
    alt = {}
    for s_ in (-1, 1): q = dict(P); q["CONFIRM_BUDGET"] = B + s_; alt[s_] = L._core(tr, q)["E3"]
    tab[d]["E3_with_budget_minus1_plus1"] = [alt[-1], alt[1]]; tab[d]["flag_caused_by_CONFIRM_BUDGET"] = bool(len({new["endpoints"]["E3"]["value"], alt[-1], alt[1]}) > 1)
t("PRIMARY labels identical to the committed labeller at onsets t_cmd+39 / +40 / +41 / +42 (terminal, subtype, events, E1/E2/E3, margins, decision rows)", all(v["same_primary"] for v in tab.values()), tab)
t("primary outcome at the boundary is unchanged: +39 and +40 SUCCESS (verified at +41 and +42 rows), +41 and +42 RELEASE_NOT_CONFIRMED with decision row t_cmd+42", tab[39]["E3"] == tab[40]["E3"] == "TRUE" and tab[41]["E3"] == tab[42]["E3"] == "FALSE" and tab[41]["decision_row"] == tab[42]["decision_row"] == 100 + B + K - 1 and tab[39]["verified"] == 100 + 39 + K - 1 and tab[40]["verified"] == 100 + 40 + K - 1, tab)
t("new sensitivity: onset +40 (budget 39 would reject it) and onset +41 (budget 41 would accept it) are now flagged ambiguous BY CONFIRM_BUDGET; onsets +39 and +42 are not flipped by a +-1 change", tab[40]["flag_caused_by_CONFIRM_BUDGET"] and tab[41]["flag_caused_by_CONFIRM_BUDGET"] and not tab[39]["flag_caused_by_CONFIRM_BUDGET"] and not tab[42]["flag_caused_by_CONFIRM_BUDGET"] and tab[41]["E3_ambiguous_new"] and not tab[41]["E3_ambiguous_old"], {d: (v["E3_ambiguous_old"], v["E3_ambiguous_new"], v["E3_with_budget_minus1_plus1"]) for d, v in tab.items()})
# debounce: nothing is verified before all K_REL observations exist, for every onset and every truncation, in the primary label AND in the perturbed (budget +-1) evaluations
viol = []
for d in (38, 39, 40, 41, 42, 43):
    for n in range(100, 150):
        tr = late(d, n)
        for s_ in (-1, 0, 1):
            q = dict(P); q["CONFIRM_BUDGET"] = B + s_; o = L._core(tr, q); ev = o["events"]
            if "release_confirmed" in ev and ev["release_confirmed"] > n - 1: viol.append(("verified beyond observed rows", d, n, s_))
            if "release_confirmed" in ev and ev["release_confirmed"] != ev["release_confirmed_onset"] + K - 1: viol.append(("verification != onset+K-1", d, n, s_))
            if o["E3"] == "TRUE" and o["terminal"] != "PLACEMENT_SUCCESS": viol.append(("E3 vs terminal", d, n, s_))
            if o["E3"] == "FALSE" and o["subtype"] == "RELEASE_NOT_CONFIRMED" and n - 1 < 100 + B + s_ + K - 1: viol.append(("decided before the budget could be observed", d, n, s_))
t("no event is verified before all K_REL debounce observations exist, for onsets +38..+43, every truncation length and budget 39 / 40 / 41", not viol, viol[:5])
# primary regression over the whole synthetic corpus
cases = [(f"{fam}:{v}", T.synth(fam, v)[0]) for fam in T.BASE for v in range(6)] + [("base", T.base(ro=100)), ("jam", T.mod(T.base(ro=100), jam=True)), ("lag2", T.mod(T.base(ro=100), lag=2)), ("slide", T.mod(T.base(ro=100), slide=108))] + [(f"late{d}", late(d)) for d in range(30, 50)]
bad = [n for n, tr in cases if prim(L.label(tr)) != prim(OLD.label(tr))]; t(f"PRIMARY-LABEL REGRESSION over {len(cases)} synthetic cases: none (any difference would be a BLOCKER)", not bad, bad[:5])
# ambiguity flags only ever ADD (budget perturbation cannot remove a flag the old labeller raised)
lost = [n for n, tr in cases if any(OLD.label(tr)["endpoints"][e]["ambiguous"] and not L.label(tr)["endpoints"][e]["ambiguous"] for e in ("E1", "E2", "E3"))]; t("the new perturbation never removes an ambiguity flag raised before", not lost, lost[:5])
# the saved real I1 trace: primary labels identical (offline, no simulation)
try:
    import g2b_trace as TR
    A, m = TR.load_trace(ROOT / "outputs/g2b_integration/traces/I1v2_expert_p3000"); lt = TR.to_labeller_trace(A, dict(t0=0, box_z0=m["meta"]["box_z0"], g_prev0=m["meta"]["g_prev0"])); a_, b_ = L.label(lt), OLD.label(lt)
    t("saved REAL I1 trace: primary labels identical to the committed labeller; E3 not ambiguous by CONFIRM_BUDGET (verification far from the budget edge)", prim(a_) == prim(b_) and a_["endpoints"]["E3"]["ambiguous"] == b_["endpoints"]["E3"]["ambiguous"], dict(terminal=a_["terminal"], onset=a_["events"]["release_confirmed_onset"], cmd=a_["events"]["release_command"], margin_rows=40 - (a_["events"]["release_confirmed_onset"] - a_["events"]["release_command"])))
except Exception as e: t("saved real I1 trace could be re-labelled", False, dict(exception=f"{type(e).__name__}: {e}"))
rep = dict(verdict="PASS" if all(r["result"] == "PASS" for r in RES) else "FAIL", n=len(RES), results=RES, boundary_table=tab); out = ROOT / "outputs/g2b_impl"; (out / "confirm_budget_report.json").write_text(json.dumps(rep, indent=1, default=str))
if rep["verdict"] != "PASS": (out / f"confirm_budget_report_FAIL_{time.strftime('%H%M%S')}.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(dict(verdict=rep["verdict"], n=rep["n"])))
