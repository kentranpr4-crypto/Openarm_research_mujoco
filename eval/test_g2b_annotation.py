#!/usr/bin/env python3
"""Unit tests of the blind-annotation tooling on SYNTHETIC annotations (numpy only).  Not a validation of any label."""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); import g2b_annotation as A
RES = []
def test(name):
    def deco(fn):
        try: ok, ev = fn()
        except Exception as e: ok, ev = False, repr(e)
        RES.append(dict(test=name, result="PASS" if ok else "FAIL", evidence=ev)); print(("PASS " if ok else "FAIL ") + name, "" if ok else ev, flush=True); return fn
    return deco
rng = np.random.default_rng(1); ARMS = ["c0", "O1", "O3", "D1", "D2", "H1", "S0"]; TERM = ["PLACEMENT_SUCCESS"] * 60 + ["CARRY_DROP"] * 25 + ["ACQUISITION_FAILURE"] * 18 + ["RELEASE_OUTSIDE_TARGET"] * 8 + ["PLACEMENT_FAILURE"] * 12 + ["TIMEOUT"] * 3
REC = [dict(branch_id=f"p{i // 7:02d}_{ARMS[i % 7]}", pose_id=i // 7, arm=ARMS[i % 7], terminal=TERM[i % len(TERM)], ambiguous=bool(i % 11 == 0)) for i in range(280)]
@test("sampling is seeded/reproducible, honours quotas, never reveals arm/pose/machine label in the sheet, ids are anonymous hashes")
def _():
    s1, k1 = A.sample_branches(REC); s2, k2 = A.sample_branches(REC); txt = json.dumps(s1); cnt = {}
    for k in k1.values(): cnt["AMBIGUOUS_FLAGGED" if k["ambiguous"] else k["terminal"]] = cnt.get("AMBIGUOUS_FLAGGED" if k["ambiguous"] else k["terminal"], 0) + 1
    return s1 == s2 and k1 == k2 and not any(w in txt for w in ARMS + TERM[:1] + ["pose"]) and len(s1) == len(set(x["anon_id"] for x in s1)) == len(k1) and cnt.get("AMBIGUOUS_FLAGGED") == 4, dict(n=len(s1), strata=cnt)
@test("kappa and confusion matrix are correct on constructed cases (perfect = 1; known disagreement)")
def _():
    a = ["PLACEMENT_SUCCESS"] * 6 + ["CARRY_DROP"] * 4; b = list(a); k1 = A.cohen_kappa(a, b); b2 = list(a); b2[0] = "CARRY_DROP"; b2[9] = "PLACEMENT_SUCCESS"; k2 = A.cohen_kappa(a, b2); M = A.confusion(a, b2)
    po = 8 / 10; pe = (6 * 6 + 4 * 4) / 100; exp = (po - pe) / (1 - pe); return abs(k1 - 1) < 1e-12 and abs(k2 - exp) < 1e-9 and M["matrix"][A.CLASSES.index("PLACEMENT_SUCCESS")][A.CLASSES.index("CARRY_DROP")] == 1, dict(kappa_perfect=k1, kappa=k2, expected=exp)
def rows_from(key, flip=None, cannot=None, e2flip=None):
    rows = []
    for j, k in enumerate(key.values()):
        h = k["terminal"]; e2m = "TRUE" if k["terminal"] in ("PLACEMENT_SUCCESS", "PLACEMENT_FAILURE", "RELEASE_OUTSIDE_TARGET") and False else ("TRUE" if k["terminal"] in ("PLACEMENT_SUCCESS", "PLACEMENT_FAILURE") else "FALSE"); e2h = e2m
        if flip and j in flip: h = flip[j]
        if cannot and j in cannot: h = "CANNOT_DECIDE"
        if e2flip and j in e2flip: e2h = "FALSE" if e2m == "TRUE" else "TRUE"
        rows.append(dict(pose_id=k["pose_id"], machine=k["terminal"], human=h, ambiguous=k["ambiguous"], machine_E2=e2m, human_E2=e2h))
    return rows
S, K = A.sample_branches(REC); GOLD = [("PLACEMENT_SUCCESS", "PLACEMENT_SUCCESS")] * 10
@test("PASS only with kappa >= 0.80, lower bound >= 0.65, E2 agreement >= 0.90, gold 100 % and ZERO critical disagreements on unambiguous branches")
def _():
    r = A.evaluate(rows_from(K), GOLD, boot=500); return r["verdict"] == "PASS" and r["kappa"] == 1.0, dict(verdict=r["verdict"], kappa=r["kappa"], ci=r["kappa_ci95"])
@test("a critical disagreement on an UNAMBIGUOUS branch (machine drop vs human placement success) fails the validation; the same disagreement on an ambiguity-flagged branch does not count as critical")
def _():
    idx_unamb = next(j for j, k in enumerate(K.values()) if k["terminal"] == "CARRY_DROP" and not k["ambiguous"]); idx_amb = next(j for j, k in enumerate(K.values()) if k["terminal"] == "CARRY_DROP" and k["ambiguous"]) if any(k["terminal"] == "CARRY_DROP" and k["ambiguous"] for k in K.values()) else None
    r1 = A.evaluate(rows_from(K, flip={idx_unamb: "PLACEMENT_SUCCESS"}), GOLD, boot=300); out = dict(unamb=r1["verdict"], n_crit=r1["n_critical"])
    ok = r1["verdict"] == "FAIL" and r1["n_critical"] >= 1
    if idx_amb is not None: r2 = A.evaluate(rows_from(K, flip={idx_amb: "PLACEMENT_SUCCESS"}), GOLD, boot=300); out["amb_n_crit"] = r2["n_critical"]; ok &= r2["n_critical"] == 0
    return ok, out
@test("low kappa fails; CANNOT_DECIDE on unambiguous branches fails; E2 disagreement counts as critical; gold errors fail")
def _():
    many = {j: "PLACEMENT_FAILURE" for j in range(0, 40, 2)}; r = A.evaluate(rows_from(K, flip=many), GOLD, boot=300); c = A.evaluate(rows_from(K, cannot={1}), GOLD, boot=300); e = A.evaluate(rows_from(K, e2flip={3}), GOLD, boot=300); g = A.evaluate(rows_from(K), GOLD[:-1] + [("PLACEMENT_SUCCESS", "CARRY_DROP")], boot=300)
    return r["verdict"] == "FAIL" and c["n_cannot_decide"] >= 1 and e["n_critical"] >= 1 and g["verdict"] == "FAIL", dict(kappa_low=r["kappa"], cannot=c["n_cannot_decide"], e2_crit=e["n_critical"])
@test("annotator record must be complete and independent; treatment-effect analysis is refused without a PASS report")
def _():
    try: A.require_annotator(dict(id="x")); a = False
    except ValueError: a = True
    try: A.require_annotator(dict(id="x", role="independent reviewer", independence_declaration="did not write or see the labeller", is_labeller_author=True)); b = False
    except ValueError: b = True
    ok = A.require_annotator(dict(id="x", role="independent reviewer", independence_declaration="did not write or see the labeller"))
    try: A.require_validation_pass(dict(verdict="FAIL")); c = False
    except PermissionError: c = True
    return a and b and ok and c and A.require_validation_pass(dict(verdict="PASS")), dict(rubric_sha256=A.RUBRIC_SHA256)
rep = dict(verdict="PASS" if all(r["result"] == "PASS" for r in RES) else "FAIL", n_tests=len(RES), results=RES, rubric_sha256=A.RUBRIC_SHA256); Path(ROOT / "outputs/g2b_impl").mkdir(parents=True, exist_ok=True); (ROOT / "outputs/g2b_impl/annotation_report.json").write_text(json.dumps(rep, indent=1, default=float)); print(rep["verdict"], rep["n_tests"])
