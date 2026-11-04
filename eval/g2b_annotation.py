#!/usr/bin/env python3
"""G2b blind-annotation protocol tooling (frozen rubric, seeded stratified sampling, anonymisation, agreement statistics, acceptance rule).  DRAFT; nothing here has been run on G2b data.
The annotator must be independent of the labeller author and must never see candidate IDs, arm names or machine labels.  The machine labels are hashed BEFORE annotation; the treatment-effect analysis refuses to run without a PASS validation report."""
import json, hashlib, numpy as np
CLASSES = ["ACQUISITION_FAILURE", "CARRY_DROP", "RELEASE_OUTSIDE_TARGET", "PLACEMENT_FAILURE", "PLACEMENT_SUCCESS", "TIMEOUT", "INVALID"]
RUBRIC = """G2b BLIND ANNOTATION RUBRIC v1 (frozen text; its SHA256 is recorded in the protocol freeze)
You see anonymised plots of ONE branch: box height rise [m], box distance to the tray centre [m], executed gripper target [m], left- and right-finger contact force [N] and contact points, box-TCP distance [m], palm force [N]. You do not see the candidate, the arm, or any automatic label.
Choose exactly one TERMINAL OUTCOME:
 1 ACQUISITION_FAILURE - the gripper never closed, or closed without firm two-sided contact, or lost the box before it was lifted >= 3 cm, or never lifted >= 3 cm within 2 s of grasping, or opened before the lift.
 2 CARRY_DROP - after a confirmed lift the contact on one or both fingers disappears for >= 3 consecutive steps while the gripper target is still closed (the box slips or falls), before any opening of the gripper.
 3 RELEASE_OUTSIDE_TARGET - the gripper target starts to open (rises >= 4 mm above its minimum) while the box is farther than 5 cm from the tray centre, or more than 5 cm above the resting height over the tray.
 4 PLACEMENT_FAILURE - the gripper opens over the target region (within 5 cm, <= 5 cm above rest) but the box does not come to rest within 3 cm of the tray centre (and within 6 mm of the resting height, speed < 0.02 m/s for 10 steps) within 2 s.
 5 PLACEMENT_SUCCESS - as 4, and the box does come to rest inside the target.
 6 TIMEOUT - the plotted trajectory ends before any of 1-5 can be decided.
 7 INVALID - the plots contain impossible values (NaN, negative forces, teleporting box).
Rules: a box that sinks onto the tray while it is STILL gripped is NOT a drop. Contact that vanishes because the gripper is opening is NOT a drop. If unsure choose CANNOT_DECIDE (counted as a disagreement unless the branch is flagged ambiguous by the machine).
Also answer: E1 grasp acquired (TRUE/FALSE/CANNOT_DECIDE); E2 grasp retained from the lift until the gripper starts to open or 60 steps after the lift, whichever is first (TRUE/FALSE/CENSORED/CANNOT_DECIDE); E3 box placed (TRUE/FALSE/CENSORED)."""
RUBRIC_SHA256 = hashlib.sha256(RUBRIC.encode()).hexdigest()
DEFAULT_QUOTAS = {"PLACEMENT_SUCCESS": 12, "CARRY_DROP": 8, "ACQUISITION_FAILURE": 6, "RELEASE_OUTSIDE_TARGET": 4, "PLACEMENT_FAILURE": 6, "TIMEOUT": 2, "INVALID": 2, "AMBIGUOUS_FLAGGED": 4}
def anon_id(seed, branch_id): return hashlib.sha256(f"{seed}|{branch_id}".encode()).hexdigest()[:10]
def sample_branches(records, quotas=DEFAULT_QUOTAS, seed=20261106):
    """records: list of dict(branch_id, pose_id, arm, terminal, ambiguous(bool)).  Seeded stratified draw by the MACHINE terminal label (and ambiguity); fills from PLACEMENT_SUCCESS when a stratum is short.
    Returns (annotation_sheet WITHOUT arm/pose/machine label, key with the hidden mapping)."""
    rng = np.random.default_rng(seed); chosen = []; used = set(); by = {}
    for r in sorted(records, key=lambda r: r["branch_id"]): by.setdefault("AMBIGUOUS_FLAGGED" if r.get("ambiguous") else r["terminal"], []).append(r)
    short = 0
    for k, q in quotas.items():
        pool = [r for r in by.get(k, []) if r["branch_id"] not in used]; take = min(q, len(pool)); idx = rng.permutation(len(pool))[:take]
        for i in idx: chosen.append(pool[i]); used.add(pool[i]["branch_id"])
        short += q - take
    if short:
        pool = [r for r in by.get("PLACEMENT_SUCCESS", []) if r["branch_id"] not in used]; idx = rng.permutation(len(pool))[:short]
        for i in idx: chosen.append(pool[i]); used.add(pool[i]["branch_id"])
    order = rng.permutation(len(chosen)); chosen = [chosen[i] for i in order]
    sheet = [dict(anon_id=anon_id(seed, r["branch_id"])) for r in chosen]; key = {anon_id(seed, r["branch_id"]): dict(branch_id=r["branch_id"], pose_id=r["pose_id"], arm=r["arm"], terminal=r["terminal"], ambiguous=bool(r.get("ambiguous"))) for r in chosen}
    return sheet, key
def cohen_kappa(a, b, classes=CLASSES + ["CANNOT_DECIDE"]):
    n = len(a)
    if n == 0: return float("nan")
    idx = {c: i for i, c in enumerate(classes)}; M = np.zeros((len(classes), len(classes)))
    for x, y in zip(a, b): M[idx[x], idx[y]] += 1
    po = np.trace(M) / n; pe = float((M.sum(1) * M.sum(0)).sum() / n ** 2)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
def confusion(machine, human, classes=CLASSES + ["CANNOT_DECIDE"]):
    idx = {c: i for i, c in enumerate(classes)}; M = np.zeros((len(classes), len(classes)), int)
    for x, y in zip(machine, human): M[idx[x], idx[y]] += 1
    prec = {c: (float(M[idx[c], idx[c]] / M[:, idx[c]].sum()) if M[:, idx[c]].sum() else None) for c in classes}; rec = {c: (float(M[idx[c], idx[c]] / M[idx[c]].sum()) if M[idx[c]].sum() else None) for c in classes}
    return dict(classes=classes, matrix=M.tolist(), precision_human_as_reference=prec, recall=rec)
def evaluate(rows, gold, boot=2000, seed=0, extension_used=False):
    """rows: list of dict(pose_id, machine(terminal), human(terminal or CANNOT_DECIDE), ambiguous, machine_E2, human_E2); gold: list of (expected, human) for synthetic gold traces.
    Returns the reliability report with the r1 acceptance decision: PASS | FAIL | EXTEND (one pre-registered extension of 20 branches when kappa >= 0.80 but its lower bound < 0.65)."""
    m = [r["machine"] for r in rows]; h = [r["human"] for r in rows]; kappa = cohen_kappa(m, h); poses = sorted({r["pose_id"] for r in rows}); by = {p: [r for r in rows if r["pose_id"] == p] for p in poses}; rng = np.random.default_rng(seed); ks = []
    for _ in range(boot):
        smp = sum([by[poses[j]] for j in rng.integers(0, len(poses), len(poses))], []); ks.append(cohen_kappa([r["machine"] for r in smp], [r["human"] for r in smp]))
    ks = [k for k in ks if np.isfinite(k)]; lo, hi = (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))) if ks else (float("nan"), float("nan"))
    crit = [r for r in rows if not r["ambiguous"] and (((r["machine"] == "PLACEMENT_SUCCESS") != (r["human"] == "PLACEMENT_SUCCESS") and r["human"] in CLASSES and not (r["machine"] in ("TIMEOUT", "INVALID") or r["human"] in ("TIMEOUT", "INVALID"))) or (r["machine_E2"] in ("TRUE", "FALSE") and r["human_E2"] in ("TRUE", "FALSE") and r["machine_E2"] != r["human_E2"]))]
    cannot = [r for r in rows if r["human"] == "CANNOT_DECIDE" and not r["ambiguous"]]; e2 = [r for r in rows if r["machine_E2"] in ("TRUE", "FALSE") and r["human_E2"] in ("TRUE", "FALSE")]; e2_agree = float(np.mean([r["machine_E2"] == r["human_E2"] for r in e2])) if e2 else float("nan")
    gold_acc = float(np.mean([e == h_ for e, h_ in gold])) if gold else float("nan"); reasons = []
    if not (kappa >= 0.80): reasons.append(f"kappa {kappa:.3f} < 0.80")
    if not (e2_agree >= 0.90): reasons.append(f"E2 agreement {e2_agree:.3f} < 0.90")
    if not (gold_acc == 1.0): reasons.append("gold traces not 100 %")
    if crit: reasons.append(f"{len(crit)} critical disagreement(s) on unambiguous branches")
    if cannot: reasons.append(f"{len(cannot)} CANNOT_DECIDE on unambiguous branches")
    lower_ok = lo >= 0.65; verdict = "PASS" if (not reasons and lower_ok) else ("EXTEND" if (not reasons and not lower_ok and not extension_used) else "FAIL")
    if not reasons and not lower_ok: reasons.append(f"kappa lower bound {lo:.3f} < 0.65" + (" (one extension of 20 branches allowed)" if not extension_used else " (extension already used)"))
    return dict(verdict=verdict, kappa=kappa, kappa_ci95=[lo, hi], e2_agreement=e2_agree, gold_accuracy=gold_acc, n=len(rows), n_critical=len(crit), n_cannot_decide=len(cannot), critical_examples=[(r["pose_id"], r["machine"], r["human"]) for r in crit[:5]], confusion=confusion(m, h), reasons=reasons, rubric_sha256=RUBRIC_SHA256)
def require_annotator(annotator):
    need = ("id", "role", "independence_declaration")
    if not all(annotator.get(k) for k in need): raise ValueError(f"annotator record incomplete: need {need}")
    if annotator.get("is_labeller_author"): raise ValueError("the annotator must not be the labeller author")
    return True
def require_validation_pass(report):
    if report.get("verdict") != "PASS": raise PermissionError("treatment-effect analysis refused: blind validation report is not PASS")
    return True
