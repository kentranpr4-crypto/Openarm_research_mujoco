"""Applies the PRE-REGISTERED P1 pilot gates (gates/p1_pilot_gates.yaml, protocol v1.0) to measured JSON files and prints one row per gate plus an OVERALL decision.  It never edits a threshold, never touches a run directory, and writes nothing unless --out is given (a NEW file).
Decision rules (checker-level clarifications of protocol v1.0; they change NO threshold):
  * a non-finite (NaN / inf) metric  -> STOP at once (the model or the optimisation diverged); no 'consecutive' grace
  * a missing metric or too few samples -> NOT_AVAILABLE; at a scheduled check that makes OVERALL = INCOMPLETE (exit 2): do NOT continue until it is measured, unless the gate is named in --allow-missing (e.g. G4 before the half-way point)
  * 'consecutive' gates STOP only if the last k evaluations are all beyond the STOP threshold; otherwise WARN
  * every file must come from the SAME checkpoint (sha256) and the SAME validation dataset; a mismatch -> INCONSISTENT -> OVERALL = INCOMPLETE
  * the gates file hash must equal the registered hash, else exit 2 (a new protocol version has to be documented and acknowledged with --protocol-ack <sha256>)
Sample floors (documented here, not in the yaml): G1 >= 100 free windows, G2 >= 30 windows per cell, G3 >= min_windows, G4 >= min_episodes, G5 >= min_states.
Checkpoint policy: gates are evaluated on a COPY of the newest latest.pt at iteration 400, 800, ... and at the end; best.pt (selected on the return of the 9,000,000+ goals) is reported descriptively only.
Inputs: --wm-eval (p1_wm_eval.py outputs, oldest first; run with --checkpoint, --actor-episodes >= 32 and --grad-states >= 24), --actor-eval (p1_actor_eval.py outputs, newest last), --benchmark (p1_benchmark.py output), --iteration N, --budget-iterations M.
Usage (repo root):  python -m research.supermimic_openarm.p1_gate_check --wm-eval a.json b.json --actor-eval c.json --benchmark d.json --iteration 4000 --budget-iterations 8000
Exit code: 0 CONTINUE, 1 STOP, 2 INCOMPLETE / protocol error."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import yaml
GATES = "research/supermimic_openarm/gates/p1_pilot_gates.yaml"; REGISTERED_SHA256 = "a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca"; INTERVAL = 400
IDS = ("G1_free_space", "G2_contact", "G3_return_gap", "G4_heldout_success", "G5_gradient_quality", "G6_action_safety", "G7_gpu_resources")
MIN_FREE_WINDOWS = 100; MIN_CELL_WINDOWS = 30
def fin(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def j(p):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError) as e: LOAD_ERRORS.append(f"{p}: {type(e).__name__}"); return None                     # an unreadable input is INCOMPLETE (exit 2), never an uncaught crash (a Python traceback exits with 1 == STOP)
LOAD_ERRORS = []
def level(v, warn, stop, hiw=True):
    if not fin(v): return "STOP" if v is not None else "NOT_AVAILABLE"                                            # None = missing; NaN/inf = diverged
    bad = (lambda x, t: x > t) if hiw else (lambda x, t: x < t); return "STOP" if bad(v, stop) else ("WARN" if warn is not None and bad(v, warn) else "PASS")
def consec(vals, warn, stop, k, hiw=True):
    """vals oldest -> newest (None = missing).  Non-finite newest value -> STOP immediately.  Otherwise STOP only if the last k values are all finite and beyond STOP."""
    if not vals or vals[-1] is None: return "NOT_AVAILABLE", None
    if not fin(vals[-1]): return "STOP", vals[-1]
    st = level(vals[-1], warn, stop, hiw)
    if st == "STOP":
        lk = vals[-k:]
        if len(lk) < k or not all(fin(x) and level(x, warn, stop, hiw) == "STOP" for x in lk): st = "WARN"
    return st, vals[-1]
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--gates", default=GATES); ap.add_argument("--wm-eval", nargs="*", default=[]); ap.add_argument("--actor-eval", nargs="*", default=[]); ap.add_argument("--benchmark"); ap.add_argument("--iteration", type=int, default=0); ap.add_argument("--budget-iterations", type=int, default=0)
    ap.add_argument("--allow-missing", default="", help="comma-separated gate ids that may be NOT_AVAILABLE (e.g. G4_heldout_success before the half-way point)"); ap.add_argument("--protocol-ack", default="", help="sha256 of a NEW protocol version you have documented"); ap.add_argument("--out"); a = ap.parse_args(argv)
    raw = Path(a.gates).read_bytes(); sha = hashlib.sha256(raw).hexdigest(); print("gates file sha256:", sha)
    if sha != REGISTERED_SHA256 and a.protocol_ack != sha: print(f"PROTOCOL MISMATCH: registered {REGISTERED_SHA256}; refusing to apply a changed gates file without --protocol-ack {sha} (document the new protocol version first)", file=sys.stderr); return 2
    G = yaml.safe_load(raw)["gates"]
    if tuple(G) != IDS: print(f"MALFORMED gates file: expected gates {IDS}, found {tuple(G)}", file=sys.stderr); return 2
    allow = {x.strip() for x in a.allow_missing.split(",") if x.strip()}; W = [j(p) for p in a.wm_eval]; A = [j(p) for p in a.actor_eval]; rows = []; bad_inputs = [e for e in LOAD_ERRORS]
    if any(w is None for w in W) or any(x is None for x in A): print("INCOMPLETE: an evaluation input file is missing or unreadable:", bad_inputs, file=sys.stderr); return 2
    LOAD_ERRORS.clear()
    def add(gid, st, val, note=""): rows.append((gid, st, val, G[gid].get("warn"), G[gid].get("stop"), note))
    # ---- consistency (same checkpoint, same dataset, scheduled iteration)
    issues = []
    if W and A:
        wc, ac = (W[-1].get("checkpoint") or {}).get("sha256"), A[-1].get("checkpoint_sha256")
        if wc is None or ac is None: issues.append("checkpoint sha256 missing in a wm-eval/actor-eval file")
        elif wc != ac: issues.append("newest wm-eval and actor-eval come from DIFFERENT checkpoints")
    its = {x for x in [(W[-1].get("checkpoint") or {}).get("iteration") if W else None, A[-1].get("checkpoint_iteration") if A else None] if x is not None}
    if len(its) > 1: issues.append(f"checkpoint iterations differ: {sorted(its)}")
    if a.iteration and its and a.iteration not in its: issues.append(f"--iteration {a.iteration} does not match the evaluated checkpoint iteration {sorted(its)}")
    dsets = {w.get("dataset_sha256") for w in W}
    if len(dsets) > 1 or (W and None in dsets): issues.append("wm-eval files used different (or unrecorded) validation datasets")
    if a.iteration and a.iteration % INTERVAL and not (a.budget_iterations and a.iteration == a.budget_iterations): print(f"NOTE: iteration {a.iteration} is off the registered schedule (every {INTERVAL} iterations and at the end)")
    # ---- G1
    g = G["G1_free_space"]; v = []
    for w in W:
        c = w.get("box_prediction", {}).get("table", {}).get("h16_free"); v.append(c["wm_mm"]["mean"] if c and c.get("n", 0) >= MIN_FREE_WINDOWS else None)
    st, val = consec(v, g["warn"], g["stop"], g["consecutive"]); add("G1_free_space", st, val, "mean WM box error mm, h=16, free-space windows (n >= %d)" % MIN_FREE_WINDOWS)
    # ---- G2
    g = G["G2_contact"]; v = []
    for w in W:
        t = w.get("box_prediction", {}).get("table", {}); rr = [t[k]["wm_mm"]["mean"] / max(t[k]["stationary_mm"]["mean"], 1e-9) for k in ("h8_first_contact", "h8_sustained") if k in t and t[k].get("n", 0) >= MIN_CELL_WINDOWS]; v.append(max(rr) if rr else None)
    st, val = consec(v, g["warn"], g["stop"], g["consecutive"]); add("G2_contact", st, val, "max ratio WM/stationary error, h=8, first_contact & sustained (cells with n >= %d)" % MIN_CELL_WINDOWS)
    # ---- G3
    g = G["G3_return_gap"]; v = [(w["matched_returns_actor"]["mean_gap"] if "matched_returns_actor" in w and w["matched_returns_actor"].get("windows", 0) >= g["min_windows"] else None) for w in W]; st, val = consec(v, g["warn"], g["stop"], g["consecutive"]); add("G3_return_gap", st, val, "mean imagined-minus-real return (16 steps) on the actor's own held-out trajectories")
    # ---- G4
    g = G["G4_heldout_success"]
    if A and A[-1].get("episodes", 0) >= g["min_episodes"] and fin(A[-1].get("success_rate")):
        ae = A[-1]; frac = (a.iteration / a.budget_iterations) if a.budget_iterations else None; p = ae["success_rate"]; lo = ae["wilson95"][0]; n = ae["episodes"]
        if frac is not None and frac >= g["mid_fraction"] and p < g["mid_stop_below"]: st = "STOP"
        elif frac is not None and frac >= 0.999: st = "PASS" if (p >= g["final_pass_at_least"] and lo >= g["final_wilson_lower_at_least"]) else "WARN"
        else: st = "PASS"
        add("G4_heldout_success", st, p, f"{ae['success']}/{n}, Wilson95 {lo:.3f}-{ae['wilson95'][1]:.3f}, iteration {a.iteration}/{a.budget_iterations}")
    elif A and A[-1].get("episodes", 0) < g["min_episodes"]: add("G4_heldout_success", "NOT_AVAILABLE", None, f"insufficient samples: {A[-1].get('episodes')} < {g['min_episodes']}")
    elif A and not fin(A[-1].get("success_rate")): add("G4_heldout_success", "STOP", A[-1].get("success_rate"), "non-finite success rate")
    else: add("G4_heldout_success", "NOT_AVAILABLE", None, "no actor-eval file")
    # ---- G5
    g = G["G5_gradient_quality"]; v = [(w["return_gradient_check"]["fraction_real_return_higher_along_plus_gradient_than_minus"] if "return_gradient_check" in w and w["return_gradient_check"].get("n", 0) >= g["min_states"] else None) for w in W]
    st, val = consec(v, g["warn"], g["stop"], g["consecutive"], hiw=False); add("G5_gradient_quality", st, val, "fraction of states where the real return is higher along +WM gradient than along -WM gradient (0.5 = chance)")
    # ---- G6
    g = G["G6_action_safety"]
    if A and "safety" in A[-1] and A[-1].get("episodes", 0) >= G["G4_heldout_success"]["min_episodes"]:
        sf = A[-1]["safety"]; vals = [sf.get(k) for k in ("executed_outside_ctrl_range", "executed_slew_over_limit", "non_finite_states", "right_arm_command_changed_steps")]; pen = sf.get("min_box_robot_contact_distance_m"); fl = sf.get("fraction_box_left_table")
        if not all(fin(x) for x in vals + [pen, fl]): st, hard = "STOP", None; note = "non-finite or missing safety statistic"
        else:
            hard = int(sum(vals)); st = "STOP" if (hard > 0 or fl > g["stop_box_left_table"] or pen < g["stop_penetration_m"]) else ("WARN" if pen < g["warn_penetration_m"] else "PASS"); note = f"hard violations={hard}, episodes ended terminated {fl:.3f}, min penetration {pen * 1000:.2f} mm"
        add("G6_action_safety", st, hard, note)
    else: add("G6_action_safety", "NOT_AVAILABLE", None, "no (or too small) actor-eval file")
    # ---- G7
    g = G["G7_gpu_resources"]; b = j(a.benchmark) if a.benchmark else None
    if b is None: add("G7_gpu_resources", "NOT_AVAILABLE", None, "no benchmark file given" if not a.benchmark else f"benchmark file missing or unreadable ({a.benchmark})")
    elif str(b.get("verdict", "")).startswith("NOT_GPU"): add("G7_gpu_resources", "NOT_AVAILABLE", None, "benchmark file is a CPU self-test, not GPU evidence")
    else:
        peak = (b.get("benchmark") or {}).get("peak_vram_gb"); it = (b.get("training_step_latency") or {}).get("iteration_estimate_s")
        if str(b.get("verdict", "")).startswith(("FAILED", "BUDGET")) or b.get("timed_out_budget") or (peak is not None and not fin(peak)) or (peak is not None and peak > g["stop_peak_vram_gb"]): add("G7_gpu_resources", "STOP", peak, f"verdict {b.get('verdict')}")
        elif peak is None or not fin(it): add("G7_gpu_resources", "NOT_AVAILABLE", peak, "peak VRAM or iteration time missing")
        else: iters = g["wall_clock_cap_s"] / it; add("G7_gpu_resources", "WARN" if iters < g["min_iterations_in_cap"] else "PASS", peak, f"peak VRAM GiB; projected iterations in cap {round(iters)} (plan must be reduced BEFORE launching if < {g['min_iterations_in_cap']})")
    print(f"{'gate':22s} {'status':14s} {'value':>10s}  warn / stop   note")
    for gid, st, val, w, s, note in rows: print(f"{gid:22s} {st:14s} {('%.4g' % val) if fin(val) else str(val):>10s}  {w} / {s}   {note}")
    for i in issues: print("INCONSISTENT:", i)
    missing = [r[0] for r in rows if r[1] == "NOT_AVAILABLE" and r[0] not in allow]
    if any(r[1] == "STOP" for r in rows): overall, code = "STOP", 1
    elif issues or missing: overall, code = "INCOMPLETE (do not continue until resolved: " + ("; ".join(issues) + ("; " if issues and missing else "") + (("missing " + ",".join(missing)) if missing else "")) + ")", 2
    else: overall, code = ("CONTINUE (%d WARN)" % sum(r[1] == "WARN" for r in rows)) if any(r[1] == "WARN" for r in rows) else "CONTINUE", 0
    print("OVERALL:", overall)
    if a.out:
        if Path(a.out).exists(): sys.exit(f"refusing to overwrite {a.out}")
        Path(a.out).write_text(json.dumps(dict(gates_sha256=sha, registered_sha256=REGISTERED_SHA256, rows=[dict(gate=r[0], status=r[1], value=(r[2] if fin(r[2]) else None), note=r[5]) for r in rows], inconsistencies=issues, overall=overall), indent=1))
    return code
if __name__ == "__main__": sys.exit(main())
