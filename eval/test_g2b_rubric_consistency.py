#!/usr/bin/env python3
"""SYNTHETIC rubric-consistency tests (NOT independent annotation; no protected pose is inspected).  (i) the rubric text contains the five X4 distinctions; (ii) its numeric constants equal the labeller parameters; (iii) a literal 'plot reader' written from the rubric words (independent of the labeller code, same author) reaches the same PLACEMENT outcome as the labeller on synthetic X4 cases.
Run: python3 eval/test_g2b_rubric_consistency.py -> outputs/g2b_impl/rubric_consistency_report.json"""
import sys, re, json, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import g2b_annotation as A, g2b_labeller as L, test_g2b_labeller as T
R = A.RUBRIC; P = L.PARAMS; RES = []
def t(name, ok, ev=None): RES.append(dict(test=name, result="PASS" if ok else "FAIL", evidence=ev)); print(("PASS " if ok else "FAIL ") + name, "" if ok else json.dumps(ev, default=str)[:300])
low = R.lower()
need = {"target arrival": ["reached the target region"], "commanded opening": ["opening command", "executed target rising"], "physical opening": ["physically opened", "measured finger position"], "confirmed release": ["confirmed release"], "placement stability": ["rests inside the target", "10 consecutive steps"]}
miss = {k: [s for s in v if s.lower() not in low] for k, v in need.items()}; t("rubric text contains the five X4 distinctions (target arrival, commanded opening, physical opening, confirmed release, placement stability)", all(not v for v in miss.values()), miss)
t("rubric states that a box resting on the tray while still gripped is not placed, and that history (not final location) decides", "still gripped is not yet placed" in low and "judge the history" in low and "never only the final box position" in low)
t("rubric says the PLACEMENT_SUCCESS window starts AFTER the confirmed release", "after (c) the box rests" in low)
consts = [("8 mm", P["OPEN_CONFIRM"] * 1000), ("4 mm", P["OPEN_RISE"] * 1000), ("3 cm", P["PLACE_XY"] * 100), ("6 mm", P["PLACE_Z"] * 1000), ("0.02 m/s", P["REST_V"]), ("5 cm", P["R_REL"] * 100), ("60 steps", P["RET_WIN"])]
bad = [(s, v) for s, v in consts if s not in R or abs(float(s.split()[0]) - v) > 1e-9]; t("rubric numeric constants equal the labeller parameters (8 mm, 4 mm, 3 cm, 6 mm, 0.02 m/s, 5 cm, 60 steps)", not bad, bad)
t("rubric debounce constants equal K_SETTLE = 10 and K_REL = 3", "10 consecutive steps" in R and P["K_SETTLE"] == 10 and ">= 3 consecutive steps" in R and P["K_REL"] == 3)
t("rubric hash is deterministic and recorded", A.RUBRIC_SHA256 == __import__("hashlib").sha256(R.encode()).hexdigest())
def reader(tr):
    """literal reading of rubric outcomes 4/5/6 for traces whose carry is already confirmed; uses only the plotted quantities"""
    g, gm, box = np.asarray(tr["g_exec"]), np.asarray(tr["g_meas"]), np.asarray(tr["box_pos"]); N = len(g); nL, nR, FL, FR = (np.asarray(tr[k]) for k in ("nL", "nR", "FL", "FR")); clamped = (nL >= 1) & (nR >= 1) & (FL >= 2.5) & (FR >= 2.5)
    dxy = np.linalg.norm(box[:, :2] - np.array([P["TRAY_X"], P["TRAY_Y"]]), axis=1); zr = box[:, 2] - P["REST_Z"]; v = np.zeros(N); v[1:-1] = np.linalg.norm(box[2:] - box[:-2], axis=1) / (2 * 0.05); v[0] = np.linalg.norm(box[1] - box[0]) / 0.05; v[-1] = np.linalg.norm(box[-1] - box[-2]) / 0.05
    rise = g[40:] - np.minimum.accumulate(g[40:]); idx = np.where(rise >= 0.004)[0]
    if len(idx) == 0: return "TIMEOUT"                                                                       # no opening command: still carrying / still gripped
    cmd = 40 + int(idx[0]); arrived = bool(((dxy <= P["R_REL"]) & (zr <= P["Z_REL"]))[:cmd + 1].any())
    if not arrived: return "OTHER"
    opened = (gm - np.minimum.accumulate(gm[40:].tolist() and np.r_[np.full(40, gm[40]), gm[40:]])) >= 0.008; ok3 = (~clamped) & opened; conf = None
    for s in range(cmd + 1, N - 2):
        if ok3[s:s + 3].all(): conf = s + 2; break
    if conf is None: return "PLACEMENT_FAILURE" if N - 1 >= cmd + 42 else "TIMEOUT"                                 # 2 s = 40 steps after the command
    stable = (dxy < 0.03) & (np.abs(zr) < 0.006) & (v < 0.02) & (~clamped)
    for s in range(conf, N - 9):
        if stable[s:s + 10].all(): return "PLACEMENT_SUCCESS"
    return "PLACEMENT_FAILURE" if N - 1 >= conf + 40 else "TIMEOUT"
b = T.base(ro=100); cases = {"normal release": b, "jam (command only)": T.mod(b, jam=True), "lag 2": T.mod(b, lag=2), "lag 4": T.mod(b, lag=4), "slides out": T.mod(b, slide=108),
         "window cut early": {k: (v[:112] if hasattr(v, "ndim") else v) for k, v in b.items()}, "jam cut early": {k: (v[:120] if hasattr(v, "ndim") else v) for k, v in T.mod(b, jam=True).items()}}
g = T.base(ro=None, lc_start=60); g["box_pos"] = g["box_pos"].copy(); g["box_pos"][:, 0] = T.TRAY[0]; g["box_pos"][:, 1] = T.TRAY[1]; g["box_pos"][70:, 2] = T.REST; cases["on tray, still gripped, no command"] = g
disc = []
for n, tr in cases.items():
    lab = L.label(tr)["terminal"]; rd = reader(tr); disc.append((n, rd, lab)) if rd != lab else None
t("plot reader (literal rubric) and labeller agree on the PLACEMENT outcome for 8 synthetic X4 cases (self-consistency only, not independent annotation)", not disc, disc)
rep = dict(verdict="PASS" if all(r["result"] == "PASS" for r in RES) else "FAIL", n=len(RES), results=RES); out = ROOT / "outputs/g2b_impl"; (out / "rubric_consistency_report.json").write_text(json.dumps(rep, indent=1, default=str))
if rep["verdict"] != "PASS": (out / f"rubric_consistency_report_FAIL_{time.strftime('%H%M%S')}.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(dict(verdict=rep["verdict"], n=rep["n"])))
