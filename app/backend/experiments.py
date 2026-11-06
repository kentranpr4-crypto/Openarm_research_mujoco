"""Experiment cards.  Every number is read from a saved artifact when the card is requested (nothing is hard-coded as a result); a value that cannot be read is reported as UNAVAILABLE with the reason.
Status labels (IMPLEMENTED / VALIDATED / EXPERIMENTAL / DESIGN ONLY / NOT TRAINED / NOT EVALUATED) are derived from the artifacts' own labels and from file existence, not asserted."""
import math
from . import security as S
LABELS = ("IMPLEMENTED", "VALIDATED", "EXPERIMENTAL", "DESIGN ONLY", "NOT TRAINED", "NOT EVALUATED")
def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return [max(0.0, c - h), min(1.0, c + h)]
def J(rel, *path):
    o = S.read_json(rel)
    for k in path:
        try: o = o[k]
        except (KeyError, IndexError, TypeError): raise S.ArtifactUnavailable(f"key {'/'.join(map(str, path))} not found in {rel}")
    return o
def M(id, label, rel=None, path=(), fn=None, unit=None, kind="artifact", ci=None, n=None, note=None, tag=None, docs_check=None):
    """one metric: value computed by fn(J-value) from a saved artifact; verified=True only if it was read from the artifact just now"""
    out = dict(id=id, label=label, unit=unit, source=rel, path="/".join(map(str, path)), kind=kind, note=note, tag=tag)
    try:
        v = J(rel, *path) if rel else None; out["value"] = fn(v) if fn else v; out["verified"] = True; out["status"] = "READ_FROM_ARTIFACT"
        if ci: out["ci95"] = ci(v)
        if n: out["n"] = n(v)
    except (S.ArtifactUnavailable, Exception) as e:
        out.update(value=None, verified=False, status="UNAVAILABLE", reason=str(e))
    if docs_check:
        d, needle = docs_check; out["doc_check"] = dict(doc=d, contains=needle, found=S.exists(d) and S.doc_contains(d, needle))
    return out
def KN(k, n): return dict(k=k, n=n, rate=(k / n if n else None), wilson95=wilson(k, n))
def art(rel, role="artifact"):
    d = dict(path=rel, role=role, exists=S.exists(rel))
    try: d["sha256"] = S.sha256(rel) if d["exists"] else None
    except S.ArtifactUnavailable: d["sha256"] = None
    return d
def derive_labels(impl_files=(), artifact_label=None, independent=False, all_gates_pass=None, design_only=False, results_exist=True, trained=True):
    L = []
    if impl_files and all(S.exists(f) for f in impl_files): L.append("IMPLEMENTED")
    if design_only: L.append("DESIGN ONLY")
    lab = (artifact_label or "").upper()
    if independent and all_gates_pass: L.append("VALIDATED")
    elif lab and any(w in lab for w in ("DEVELOPMENT", "EXPLORATORY", "PILOT")) or (independent and not all_gates_pass): L.append("EXPERIMENTAL")
    if not trained: L.append("NOT TRAINED")
    if not results_exist: L.append("NOT EVALUATED")
    return [x for x in LABELS if x in L]
def _any_checkpoint(prefix):
    base = S.resolve("outputs/checkpoints")
    return base.is_dir() and any(p.name.startswith(prefix) for p in base.iterdir())
# ------------------------------------------------------------------------------------------------------------------------------------------ cards
def act_v1():
    A = "outputs/audit_v1/audit_results.json"; E = "outputs/e6b/stage1_results.json"; tab = lambda cond, key: (lambda v: v[cond][key])
    kn = lambda cond: (lambda v: KN(v[cond]["task"], v[cond]["n"]))
    mets = [M("nominal", "Nominal-pose task success (ACT V1, 3 seeds pooled)", A, ("A1_metrics", "recomputed_table"), fn=kn("act/nominal"), tag="HISTORICAL_AUDIT", docs_check=("docs/ai-native/tasks/2026-10-27-lab-scene-act-pipeline.md", "30/30")),
            M("unseen", "Unseen-pose task success (ACT V1, 3 seeds pooled)", A, ("A1_metrics", "recomputed_table"), fn=kn("act/unseen_pose"), tag="HISTORICAL_AUDIT", docs_check=("docs/ai-native/tasks/2026-10-27-lab-scene-act-pipeline.md", "18/30")),
            M("bound", "BOUND task success (V1, seeds 21+22)", E, ("arms", "v1", "bound"), fn=lambda v: KN(v[0], v[1]), tag="STAGE01_BOUNDARY"),
            M("dz", "TCP height at gripper closing (V1) [cm]", E, ("arms", "v1", "dz_summary"), fn=lambda v: dict(mean=v["mean"], std=v["std"], frac_gt_3p4=v["frac_gt_3p4"], n=v["n"]), unit="cm", note="grasp-height error: the closing happens higher than the expert's"),
            M("jerk_v1", "Action jerk V1 (all / interior / boundary) [rad/s^3]", E, ("arms", "v1", "jerk"), unit="rad/s^3"),
            M("jerk_c0", "Action jerk C0 (re-trained, train-only statistics)", E, ("arms", "c0", "jerk"), unit="rad/s^3")]
    fm = []
    try: fm = [dict(mode=k, count=v) for k, v in J(A, "A2_taxonomy").items()]
    except S.ArtifactUnavailable: pass
    table = []
    try:
        for arm, v in J(E, "arms").items(): table.append(dict(arm=arm, nom=v.get("nom"), bound=v.get("bound"), dz_mean_cm=(v.get("dz_summary") or {}).get("mean"), jerk_all=(v.get("jerk") or {}).get("all")))
    except S.ArtifactUnavailable: pass
    return dict(arms_table=table, id="act_v1", title="ACT V1 (frozen historical checkpoint)", purpose="Characterise the imitation policy trained on scripted-expert demonstrations in the OpenArm v1 lab scene.", dataset="Scripted-expert demonstrations (outputs/datasets); evaluation poses: nominal and unseen-x poses, 3 seeds x 10.", candidates=["ACT V1 (single checkpoint)"],
                method="Closed-loop MuJoCo rollouts, 200 steps at 20 Hz; pick/place/task defined by recorded box rise and final box pose.", metrics=mets, failure_modes=fm,
                outcome_validity=dict(status="DEVELOPMENT", text="n = 10 per seed; configurations share poses, so the 60 % vs 100 % contrast is not an independent test."), implementation=["eval/evaluate.py", "eval/closed_loop_pilot.py"], artifacts=[art(A), art(E), art("outputs/checkpoints/act_full_v1/train_info.json", "checkpoint metadata")],
                labels=derive_labels(["eval/evaluate.py"], "DEVELOPMENT audit"), caveats=["Simulation only; privileged evaluation observer; not Sim2Real."])
def wm3b():
    I = "outputs/indep/evaluation_independent.json"; Wv = "outputs/wm3/evaluation_v2.json"
    P1 = ("paired", "primary", "P1_box_K20_wm3b_vs_c0", "95")
    mets = [M("wm3b_box_k20", "WM3-B box position error at K=20 (independent held-out, 50 groups) [mm]", I, P1 + ("a_mean",), unit="mm", tag="INDEPENDENT_SIM", docs_check=("docs/ai-native/tasks/2026-11-01-indep-eval-report.md", "3.20 mm")),
            M("wm2c0_box_k20", "WM2-C0 baseline box position error at K=20 [mm]", I, P1 + ("b_mean",), unit="mm", tag="INDEPENDENT_SIM", docs_check=("docs/ai-native/tasks/2026-11-01-indep-eval-report.md", "3.83 mm")),
            M("paired_p1", "Paired difference WM3-B minus WM2-C0 (95 % interval) [mm]", I, P1, fn=lambda v: dict(diff=v["diff_mean"], lo=v["lo"], hi=v["hi"], verdict=v["verdict"], n_groups=v["n_groups"]), unit="mm", tag="INDEPENDENT_SIM"),
            M("gates", "Frozen gate verdicts, WM3-B (independent evaluation)", I, ("gate_verdicts", "wm3b"), tag="INDEPENDENT_SIM", note="a FAIL here means a pre-registered gate was not met; see the calibration caveat")]
    curve = {}
    for mname in ("wm3b", "wm2_c0", "persistence", "constvel"):
        try: curve[mname] = {k: J(I, "errors", mname, "ALL_50_groups", k, "box_mm", "mean") for k in ("K=1", "K=5", "K=10", "K=20")}
        except S.ArtifactUnavailable: pass
    cal = None
    try: c = J(I, "classification_calibration"); cal = {k: dict(auroc=v.get("auroc"), ece=v.get("ECE"), brier=v.get("Brier"), evaluable=v.get("evaluable")) for k, v in c.items() if isinstance(v, dict)}
    except S.ArtifactUnavailable: pass
    try: allpass = all(v == "PASS" for v in J(I, "gate_verdicts", "wm3b").values()); lab = J(I, "status")
    except S.ArtifactUnavailable: allpass, lab = None, None
    return dict(id="wm3b", title="WM3-B world model (ridge + GRU)", purpose="Predict short-horizon box and contact evolution (K = 20 steps) for candidate action sequences from privileged simulator state.", dataset="Simulator logs of scripted-expert and ACT rollouts; independent held-out evaluation on 50 new initial-state groups.", candidates=["WM3-B", "WM2-C0", "persistence", "constant velocity", "ridge", "MLP"],
                method="Ridge forecasts future left-arm joint deltas Y_s (20x8); a 2-layer GRU (256) consumes history, candidate action sequence and the normalised ridge forecast, and predicts box (7) and contact (7) per step plus 4 event logits.", metrics=mets, curve_box_mm=curve, calibration=cal,
                failure_modes=["Gate G3 (calibration) FAILs in the independent evaluation" if (allpass is False) else "see gate verdicts"], outcome_validity=dict(status="SIMULATION-ONLY", text="Inputs are privileged simulator quantities (box pose, contact forces); no real-robot sensing path exists. Error numbers are offline prediction errors, not task success."),
                implementation=["eval/wm3b_model.py", "eval/wm2_models.py"], artifacts=[art(I), art(Wv), art("outputs/wm3/models/wm3b.pt", "weights"), art("outputs/wm3/model_config_wm3b.json", "frozen spec")], labels=derive_labels(["eval/wm3b_model.py"], lab if False else "INDEPENDENT", independent=bool(lab and "INDEPENDENT" in lab.upper()), all_gates_pass=allpass),
                caveats=["Privileged simulator-state dependency (not Sim2Real).", "Calibration caveat: see gate G3 and the classification calibration table.", "Historical development splits give smaller errors (2.53 mm vs 2.77 mm); the independent numbers are shown here."])
def stage1():
    A = "outputs/closed_loop_pilot/results/pilot_analysis.json"; F = "outputs/closed_loop_pilot/protocol_freeze.json"
    mets = [M("a0", "A0 ACT default (c0_s0), task success", A, ("success_counts", "A0"), fn=lambda v: KN(*v), tag="PILOT"), M("a2", "A2 WM3-B ranking, task success", A, ("success_counts", "A2"), fn=lambda v: KN(*v), tag="PILOT"),
            M("a1", "A1 random checkpoint selection, task success (3 repeats x 24 poses)", A, ("success_counts",), fn=lambda v: KN(round(v["A1_mean_successes_per_pose_sum"] * 3), v["A1_n_poses"] * 3), tag="PILOT"),
            M("a2_a1", "A2 minus A1 (paired by pose, bootstrap over poses)", A, ("primary_A2_minus_A1",), fn=lambda v: dict(mean_diff=v["mean_diff"], ci95=v["ci95"], verdict=v["verdict"], poses_better=v["poses_better"], poses_worse=v["poses_worse"], poses_equal=v["poses_equal"]), tag="PILOT"),
            M("a2_a0", "A2 minus A0", A, ("secondary_A2_minus_A0",), fn=lambda v: dict(mean_diff=v["mean_diff"], ci95=v["ci95"], verdict=v["verdict"]), tag="PILOT")]
    try: lab = J(A, "label"); cands = J(F, "arms")
    except S.ArtifactUnavailable: lab, cands = None, {}
    return dict(id="stage1", title="Stage 1 - ACT checkpoint ranking with WM3-B", purpose="Test whether WM3-B can rank three ACT checkpoint chunks online and improve task success.", dataset="24 development poses; A0 x1, A1 x3 repeats, A2 x1 = 120 episodes; horizon 200 steps.", candidates=[f"{k}: {v}" for k, v in cands.items()],
                method="At each 20-step decision the candidate chunks come from ACT checkpoints; A2 picks the chunk with the lowest frozen score J of WM3-B predictions.", metrics=mets, failure_modes=[dict(mode=k, count=v) for k, v in (J(A, "failure_classification") if S.exists(A) and isinstance(J(A, "failure_classification"), dict) else {}).items()] if S.exists(A) else [],
                outcome_validity=dict(status="DEVELOPMENT", text="All three pairwise intervals contain 0 (verdict INCONCLUSIVE): the WM ranking did NOT show an overall task-success improvement. Pilot evidence with the privileged observer."), implementation=["eval/closed_loop_pilot.py", "eval/decision_utility_lib.py"], artifacts=[art(A), art(F, "protocol freeze"), art("outputs/closed_loop_pilot/results/pilot_results.json")],
                freeze=dict(file=F, sha256=(S.sha256(F) if S.exists(F) else None)), labels=derive_labels(["eval/closed_loop_pilot.py"], lab), caveats=["Not a validated production pipeline.", "Different experiment from the G2b grasp-correction study (checkpoint ranking vs geometric grasp correction)."])
def branching():
    A = "outputs/branching_replay/results/branching_analysis.json"; F = "outputs/branching_replay/protocol_freeze.json"; J20 = ("J20_overall",)
    mets = [M("states", "Decision states", A, ("counts", "n_states"), tag="OFFLINE_REPLAY"), M("branches", "Evaluated branches", A, ("counts", "n_branches"), tag="OFFLINE_REPLAY"), M("poses", "Poses", A, ("counts", "n_poses")),
            M("j20", "J20 pairwise ranking accuracy (pooled ratio, 95 % CI)", A, J20 + ("pairwise_accuracy_pooled_ratio_ci",), fn=lambda v: dict(value=v[0], lo=v[1], hi=v[2]), tag="OFFLINE_REPLAY", note="local pairwise ranking correctness among meaningful pairs - NOT full-task success"),
            M("pairs", "Meaningful pairs", A, J20 + ("n_meaningful_pairs",)), M("headroom_frac", "States with oracle headroom >= 0.01", A, J20 + ("headroom_frac_ge_0p01",), tag="OFFLINE_REPLAY"),
            M("stop", "Scientific stop rule", A, ("oracle_stop_rule",), tag="OFFLINE_REPLAY", note="rule: oracle headroom < 0.01 in > 80 % of decision states => stop ranking research under this protocol")]
    try: lab = J(A, "label"); cands = J(F, "candidates")
    except S.ArtifactUnavailable: lab, cands = None, {}
    return dict(id="branching", title="Branching replay (offline ranking correctness)", purpose="Check offline whether WM3-B ranks the true executed outcome of alternative ACT chunks correctly, and how much headroom an oracle would have.", dataset="24 poses x 9 decision states = 216 states; 3 ACT checkpoint candidates per state = 648 executed branches (short horizon).", candidates=[f"{k}: {v}" for k, v in cands.items()],
                method="Snapshot at each decision state, execute every candidate chunk from the bit-exact snapshot, compare WM3-B ranking with the executed outcome.", metrics=mets, failure_modes=["Oracle headroom is absent in most states (stop rule triggered)."], outcome_validity=dict(status="DEVELOPMENT", text="Local ranking correctness only; short-horizon labels; privileged observer; the stop rule was triggered."),
                implementation=["eval/branching_replay.py", "eval/decision_utility_lib.py"], artifacts=[art(A), art(F, "protocol freeze")], freeze=dict(file=F, sha256=(S.sha256(F) if S.exists(F) else None)), labels=derive_labels(["eval/branching_replay.py"], lab), caveats=["Local ranking correctness does not imply task-level improvement."])
def g2a():
    A = "outputs/g2a_act_corrections/results/g2a_analysis.json"; F = "outputs/g2a_act_corrections/protocol_freeze.json"
    mets = [M("poses", "Poses", A, ("counts", "n_poses")), M("cands", "Candidates per pose", A, ("counts", "n_candidates"), fn=lambda v: v // 48 if v else None, note="432 candidates / 48 poses"), M("branches", "Executed branches", A, ("counts", "n_executed_branches"), tag="DEVELOPMENT"),
            M("valid", "Physical validity (legal clip->slew, finite, no IK failure, penetration <= 2 mm)", A, ("validity", "valid_fraction"), tag="DEVELOPMENT"), M("ik_pos", "Max IK position error [m]", A, ("validity", "ik_max_pos_err_m"), unit="m")]
    try: lab = J(A, "label"); cd = J(F, "candidates")
    except S.ArtifactUnavailable: lab, cd = None, []
    cands = [f"{c['id']}: {c['kind']}" + (f" dims={c.get('dims')}" if c.get("dims") else "") + (f" value={c['value']}" if "value" in c else "") + (f" ({c.get('timing')})" if c.get("timing") else "") for c in cd]
    return dict(id="g2a", title="G2a - oracle-assisted ACT grasp corrections", purpose="Test whether geometric corrections of the ACT chunk at one decision state change short-horizon lift-and-hold behaviour.", dataset="48 development poses (32 hard + 16 mid), 9 candidates each = 432 branches, snapshot at t = 40, short horizon (t = 40..100).", candidates=cands,
                method="Remove gravity-comp offset, FK, geometric correction in the box frame (oracle reference), IK, absolute targets, clip->slew; nominal c0 is untouched ACT.", metrics=mets, failure_modes=["The frozen G2a 'dropped' rule counted arrival at the tray as a drop (tray arrivals were labelled drops)."],
                outcome_validity=dict(status="INVALID_FOR_GRASP_QUALITY", text="Only the physical-validity results stand. The G2a grasp-quality outcome claims are INVALIDATED and are intentionally NOT displayed as a success metric here.", doc_check=S.doc_contains("docs/ai-native/tasks/2026-11-04-g2b-outcome-contract-v2-r1-and-preregistration.md", "tray arrival") if S.exists("docs/ai-native/tasks/2026-11-04-g2b-outcome-contract-v2-r1-and-preregistration.md") else None),
                implementation=["eval/g2a_act_corrections.py"], artifacts=[art(A), art(F, "protocol freeze")], freeze=dict(file=F, sha256=(S.sha256(F) if S.exists(F) else None)), labels=derive_labels(["eval/g2a_act_corrections.py"], lab), caveats=["Oracle-assisted (uses the true box pose); not a deployable policy.", "No G2a success rate is shown on purpose."])
G2B_CANDIDATES = [dict(id="c0", name="Nominal ACT", kind="reference", oracle=False, text="Unmodified ACT chunk."), dict(id="O1", name="Oracle depth cancel", kind="oracle-assisted", oracle=True, g2a_id="c1", text="Cancels the depth error using the true box pose (early ramp)."), dict(id="O3", name="Oracle depth + lateral + yaw cancel", kind="oracle-assisted", oracle=True, g2a_id="c3", text="Cancels depth, lateral and yaw errors using the true box pose."),
                  dict(id="D1", name="2 cm shallower", kind="deployable (no oracle input)", oracle=False, g2a_id="c5", text="Fixed 2 cm shallower grasp depth."), dict(id="D2", name="Close delayed 8 steps", kind="deployable (no oracle input)", oracle=False, g2a_id="c4", text="Gripper closing command delayed by 8 control steps."), dict(id="H1", name="2 cm deeper (harm control)", kind="harm control", oracle=False, g2a_id="c6", text="Fixed 2 cm deeper grasp; expected to increase palm contact."),
                  dict(id="S0", name="Time-shift null", kind="validity control", oracle=False, text="Holds the last target for 8 steps then resumes ACT: a pure schedule shift used as a timing control.")]
G2B_EVENTS = [dict(id="close_command", kind="command", text="first executed gripper target < 0.040 m (issue row)"), dict(id="grasp_acquired", kind="observation", text="both fingers in contact with >= 5 N for 5 rows while the commanded gripper is closed"), dict(id="lift_confirmed", kind="observation", text="box rise >= 0.03 m for 3 rows"), dict(id="release_command", kind="command", text="commanded gripper rises >= 4 mm over its minimum for 2 rows"),
              dict(id="release_physical_onset", kind="physical", text="measured finger rises >= 4 mm over its minimum for 2 rows"), dict(id="release_confirmed", kind="verification", text="not clamped AND fingers opened >= 8 mm for 3 rows (verified at onset + 2)"), dict(id="placement_settled", kind="observation", text="10 quiet rows in the target region starting at or after the verified release"), dict(id="decision_row", kind="decision", text="row from which the E3 label is fixed")]
def g2b():
    R = "outputs/g2b_integration"; docs = ["docs/ai-native/tasks/2026-11-05-g2b-outcome-contract-r3-freeze-candidate.md", "docs/ai-native/tasks/2026-11-05-g2b-blind-annotation-plan.md", "docs/ai-native/tasks/2026-11-05-g2b-compute-budget-and-pose-plan.md"]
    gates = []
    for g, rel, key in (("I1", R + "/integration_v3_run_BLOCKED_after_I1.json", "I1"), ("I2", R + "/integration_v4_I2_I4.json", "I2"), ("I3", R + "/integration_v4_I2_I4.json", "I3"), ("I4", R + "/integration_v5_I4.json", "I4")):
        try:
            rep = J(rel); gl = (rep.get("child_report") or rep)["gates"]; row = [x for x in gl if x["gate"] == key][0]; gates.append(dict(gate=g, result=row["result"], cpu_s=row.get("cpu_s"), source=rel, note="integration self-test on known development poses; not an evaluation"))
        except Exception as e: gates.append(dict(gate=g, result="UNAVAILABLE", reason=str(e)))
    frozen = None
    r3 = docs[0]
    try: frozen = not S.doc_contains(r3, "NOT FROZEN")
    except S.ArtifactUnavailable: frozen = None
    run_exists = S.exists("outputs/g2b_run")
    mets = [M("integration", "Integration gates I1-I4 (self-tests)", None), M("effects", "Treatment-effect results", None)]
    mets[0].update(value=gates, verified=bool(gates), status="READ_FROM_ARTIFACT"); mets[1].update(value=None, verified=False, status="NOT_EVALUATED", reason="the G2b evaluation has not been executed; no success rate exists")
    return dict(id="g2b", title="G2b - grasp-correction outcome study (DESIGN; evaluation not executed)", purpose="Pre-registered paired study of seven grasp-correction candidates against nominal ACT with an event-based outcome contract.", dataset="Planned: 40 development poses (not generated), 7 candidates, <= 280 branches, horizon 160 steps; sealed reserve poses not generated.", candidates=[f"{c['id']}: {c['name']} ({c['kind']})" for c in G2B_CANDIDATES], candidate_defs=G2B_CANDIDATES, endpoint_events=G2B_EVENTS,
                endpoints=[dict(id="E1", name="GRASP_ACQUIRED"), dict(id="E2", name="GRASP_RETAINED (primary)"), dict(id="E3", name="PLACEMENT_SUCCESS (requires a verified physical release)")], method="Paired risk differences per pose with pose-cluster bootstrap; two primary comparisons O1-c0 and D1-c0 (97.5 % intervals); S0 equivalence margin +-0.10 as a timing diagnostic; blind annotation validation gate.", metrics=mets, failure_modes=["Not applicable: no evaluation outcome exists."],
                outcome_validity=dict(status="NOT_EVALUATED", text="Outcome Contract r3 is a freeze CANDIDATE (not frozen). Labeller validation is synthetic plus integration self-tests; independent blind annotation has not been run (annotator unassigned)."), implementation=["eval/g2b_labeller.py", "eval/g2b_trace.py", "eval/g2b_s0.py", "eval/g2b_stats.py", "eval/g2b_integration_v2.py"], integration_gates=gates, artifacts=[art(d, "design document") for d in docs] + [art(R + "/integration_v5_I4.json")],
                labels=derive_labels(["eval/g2b_labeller.py", "eval/g2b_integration_v2.py"], None, design_only=True, results_exist=run_exists), contract=dict(r3_frozen=frozen, note="r3 freeze candidate document states NOT FROZEN" if frozen is False else "freeze state could not be determined"), caveats=["Do not read any success rate into this card: none exists.", "Integration gates were run on known development poses only (self-tests)."])
BUILDERS = dict(act_v1=act_v1, wm3b=wm3b, stage1=stage1, branching=branching, g2a=g2a, g2b=g2b)
ORDER = ["act_v1", "wm3b", "stage1", "branching", "g2a", "g2b"]
def summary():
    out = []
    for i in ORDER:
        try: c = BUILDERS[i](); out.append(dict(id=c["id"], title=c["title"], labels=c["labels"], validity=c["outcome_validity"]["status"], purpose=c["purpose"], n_metrics=len(c["metrics"]), unavailable=sum(1 for m in c["metrics"] if m["status"] == "UNAVAILABLE")))
        except Exception as e: out.append(dict(id=i, title=i, labels=[], validity="UNAVAILABLE", purpose="", error=str(e)))
    return out
def detail(eid):
    S.safe_id(eid)
    if eid not in BUILDERS: raise S.ArtifactUnavailable(f"unknown experiment: {eid}")
    return BUILDERS[eid]()
