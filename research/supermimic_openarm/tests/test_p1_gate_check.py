"""Tests of the monitoring contract: 7 registered gates, registered SHA-256, WARN/STOP thresholds, missing / NaN / insufficient-sample handling, checkpoint + dataset consistency, consecutive-evaluation logic."""
import contextlib, copy, hashlib, io, json, math, tempfile
from pathlib import Path
import pytest, yaml
from research.supermimic_openarm import p1_gate_check as GC
GATES = Path(GC.GATES)
def wm(free=4.0, ratio=0.4, gap=0.1, frac=0.85, sha="a" * 64, it=400, ds="d" * 64, n_free=500, n_cell=100, windows=200, states=24):
    return dict(checkpoint=dict(sha256=sha, iteration=it, name="latest.pt"), dataset_sha256=ds, box_prediction=dict(table={"h16_free": dict(n=n_free, wm_mm=dict(mean=free)), "h8_first_contact": dict(n=n_cell, wm_mm=dict(mean=ratio * 10), stationary_mm=dict(mean=10.0)), "h8_sustained": dict(n=n_cell, wm_mm=dict(mean=ratio * 10), stationary_mm=dict(mean=10.0))}),
                matched_returns_actor=dict(windows=windows, mean_gap=gap), return_gradient_check=dict(n=states, fraction_real_return_higher_along_plus_gradient_than_minus=frac))
def actor(rate=0.5, n=64, sha="a" * 64, it=400, hard=0, pen=-0.0005, left=0.0):
    k = round(rate * n) if math.isfinite(rate) else 0; return dict(checkpoint_sha256=sha, checkpoint_iteration=it, episodes=n, success=k, success_rate=rate, wilson95=[max(0, rate - 0.1) if math.isfinite(rate) else 0, min(1, rate + 0.1) if math.isfinite(rate) else 1],
                                                                   safety=dict(executed_outside_ctrl_range=hard, executed_slew_over_limit=0, non_finite_states=0, right_arm_command_changed_steps=0, min_box_robot_contact_distance_m=pen, fraction_box_left_table=left))
def bench(peak=9.0, it=1.0, verdict="BENCHMARK_OK"): return dict(verdict=verdict, benchmark=dict(peak_vram_gb=peak), training_step_latency=dict(iteration_estimate_s=it))
def run(tmp, wms=(), actors=(), bm=None, extra=(), gates=None):
    args = []; files = []
    for i, w in enumerate(wms): p = Path(tmp) / f"w{i}.json"; p.write_text(json.dumps(w)); files.append(str(p))
    if files: args += ["--wm-eval", *files]
    files = []
    for i, x in enumerate(actors): p = Path(tmp) / f"a{i}.json"; p.write_text(json.dumps(x)); files.append(str(p))
    if files: args += ["--actor-eval", *files]
    if bm is not None: p = Path(tmp) / "b.json"; p.write_text(json.dumps(bm)); args += ["--benchmark", str(p)]
    if gates: args += ["--gates", gates]
    buf = io.StringIO(); err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err): rc = GC.main(args + list(extra))
    return rc, buf.getvalue(), err.getvalue()
def status(out, gid): return next(l.split()[1] for l in out.splitlines() if l.startswith(gid))
def test_gates_file_has_seven_gates_and_the_registered_hash():
    raw = GATES.read_bytes(); assert hashlib.sha256(raw).hexdigest() == GC.REGISTERED_SHA256 == "a7dc91ca66f6350d4798602aa86258a4c01777d9d123f2e724519499d24a8dca"; G = yaml.safe_load(raw)["gates"]; assert tuple(G) == GC.IDS and len(G) == 7
    assert (G["G1_free_space"]["warn"], G["G1_free_space"]["stop"]) == (5.0, 8.0) and (G["G2_contact"]["warn"], G["G2_contact"]["stop"]) == (0.6, 0.8) and (G["G3_return_gap"]["warn"], G["G3_return_gap"]["stop"]) == (1.0, 3.0)
    assert (G["G5_gradient_quality"]["warn"], G["G5_gradient_quality"]["stop"]) == (0.65, 0.45) and G["G4_heldout_success"]["mid_stop_below"] == 0.05 and G["G6_action_safety"]["stop_penetration_m"] == -0.004
    for gid, g in G.items(): assert g["metric"] and g["validation_data"] and g["justification"], gid
def test_all_healthy_continues(tmp_path):
    rc, out, _ = run(tmp_path, [wm()], [actor()], bench(), ["--iteration", "400", "--budget-iterations", "8000"]); assert rc == 0 and out.strip().splitlines()[-1].startswith("OVERALL: CONTINUE"), out
def test_thresholds_warn_and_stop(tmp_path):
    rc, out, _ = run(tmp_path, [wm(free=6.0)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G1_free_space") == "WARN" and rc == 0
    rc, out, _ = run(tmp_path, [wm(free=9.0)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G1_free_space") == "WARN" and rc == 0            # a single evaluation beyond STOP is only a WARN ...
    rc, out, _ = run(tmp_path, [wm(free=9.0), wm(free=9.5)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G1_free_space") == "STOP" and rc == 1       # ... two consecutive evaluations STOP
    rc, out, _ = run(tmp_path, [wm(free=9.0), wm(free=4.0)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G1_free_space") == "PASS"
    rc, out, _ = run(tmp_path, [wm(ratio=0.9), wm(ratio=0.9)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G2_contact") == "STOP"
    rc, out, _ = run(tmp_path, [wm(gap=3.5), wm(gap=3.2)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G3_return_gap") == "STOP"
    rc, out, _ = run(tmp_path, [wm(frac=0.40), wm(frac=0.42)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G5_gradient_quality") == "STOP"
    rc, out, _ = run(tmp_path, [wm(frac=0.60)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G5_gradient_quality") == "WARN"
def test_nan_metric_stops_immediately(tmp_path):
    rc, out, _ = run(tmp_path, [wm(free=float("nan"))], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G1_free_space") == "STOP" and rc == 1
    rc, out, _ = run(tmp_path, [wm(gap=float("nan"))], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G3_return_gap") == "STOP" and rc == 1
    rc, out, _ = run(tmp_path, [wm()], [actor(rate=float("nan"))], bench(), ["--iteration", "400"]); assert status(out, "G4_heldout_success") == "STOP" and rc == 1
    rc, out, _ = run(tmp_path, [wm()], [actor()], bench(peak=float("nan")), ["--iteration", "400"]); assert status(out, "G7_gpu_resources") == "STOP" and rc == 1
def test_missing_or_insufficient_data_is_incomplete_not_continue(tmp_path):
    rc, out, _ = run(tmp_path, [wm()], [], bench(), ["--iteration", "400"]); assert rc == 2 and "INCOMPLETE" in out and status(out, "G4_heldout_success") == "NOT_AVAILABLE" and status(out, "G6_action_safety") == "NOT_AVAILABLE"
    rc, out, _ = run(tmp_path, [wm(n_free=10)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G1_free_space") == "NOT_AVAILABLE" and rc == 2                 # too few free-space windows
    rc, out, _ = run(tmp_path, [wm(n_cell=5)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G2_contact") == "NOT_AVAILABLE" and rc == 2
    rc, out, _ = run(tmp_path, [wm(windows=10)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G3_return_gap") == "NOT_AVAILABLE"
    rc, out, _ = run(tmp_path, [wm(states=5)], [actor()], bench(), ["--iteration", "400"]); assert status(out, "G5_gradient_quality") == "NOT_AVAILABLE"
    rc, out, _ = run(tmp_path, [wm()], [actor(n=16)], bench(), ["--iteration", "400"]); assert status(out, "G4_heldout_success") == "NOT_AVAILABLE" and status(out, "G6_action_safety") == "NOT_AVAILABLE"
    rc, out, _ = run(tmp_path, [wm()], [actor()], bench(verdict="NOT_GPU_EVIDENCE_SELFTEST"), ["--iteration", "400"]); assert status(out, "G7_gpu_resources") == "NOT_AVAILABLE" and rc == 2          # a CPU self-test is not GPU evidence
    rc, out, _ = run(tmp_path, [wm()], [actor()], None, ["--iteration", "400", "--allow-missing", "G7_gpu_resources"]); assert rc == 0                                                    # explicit, named exception
def test_checkpoint_and_dataset_consistency(tmp_path):
    rc, out, _ = run(tmp_path, [wm(sha="a" * 64)], [actor(sha="b" * 64)], bench(), ["--iteration", "400"]); assert rc == 2 and "DIFFERENT checkpoints" in out
    rc, out, _ = run(tmp_path, [wm(it=400)], [actor(it=800)], bench(), ["--iteration", "400"]); assert rc == 2 and "iterations differ" in out
    rc, out, _ = run(tmp_path, [wm(it=400)], [actor(it=400)], bench(), ["--iteration", "800"]); assert rc == 2 and "does not match" in out
    rc, out, _ = run(tmp_path, [wm(ds="1" * 64), wm(ds="2" * 64)], [actor()], bench(), ["--iteration", "400"]); assert rc == 2 and "different (or unrecorded) validation datasets" in out
def test_heldout_success_gate_midway_and_final(tmp_path):
    rc, out, _ = run(tmp_path, [wm(it=4000)], [actor(rate=0.03, it=4000)], bench(), ["--iteration", "4000", "--budget-iterations", "8000"]); assert status(out, "G4_heldout_success") == "STOP" and rc == 1
    rc, out, _ = run(tmp_path, [wm(it=2000)], [actor(rate=0.03, it=2000)], bench(), ["--iteration", "2000", "--budget-iterations", "8000"]); assert status(out, "G4_heldout_success") == "PASS"             # before the half-way point a low rate is not a STOP
    rc, out, _ = run(tmp_path, [wm(it=8000)], [actor(rate=0.60, it=8000)], bench(), ["--iteration", "8000", "--budget-iterations", "8000"]); assert status(out, "G4_heldout_success") == "PASS"
    rc, out, _ = run(tmp_path, [wm(it=8000)], [actor(rate=0.30, it=8000)], bench(), ["--iteration", "8000", "--budget-iterations", "8000"]); assert status(out, "G4_heldout_success") == "WARN"
def test_safety_gate(tmp_path):
    rc, out, _ = run(tmp_path, [wm()], [actor(hard=1)], bench(), ["--iteration", "400"]); assert status(out, "G6_action_safety") == "STOP" and rc == 1
    rc, out, _ = run(tmp_path, [wm()], [actor(left=0.2)], bench(), ["--iteration", "400"]); assert status(out, "G6_action_safety") == "STOP"
    rc, out, _ = run(tmp_path, [wm()], [actor(pen=-0.005)], bench(), ["--iteration", "400"]); assert status(out, "G6_action_safety") == "STOP"
    rc, out, _ = run(tmp_path, [wm()], [actor(pen=-0.003)], bench(), ["--iteration", "400"]); assert status(out, "G6_action_safety") == "WARN"
def test_gpu_gate(tmp_path):
    rc, out, _ = run(tmp_path, [wm()], [actor()], bench(peak=17.0), ["--iteration", "400"]); assert status(out, "G7_gpu_resources") == "STOP" and rc == 1
    rc, out, _ = run(tmp_path, [wm()], [actor()], bench(it=3.0), ["--iteration", "400"]); assert status(out, "G7_gpu_resources") == "WARN"                    # 7200 s / 3 s = 2400 iterations < 4000
    rc, out, _ = run(tmp_path, [wm()], [actor()], bench(verdict="BUDGET_EXCEEDED"), ["--iteration", "400"]); assert status(out, "G7_gpu_resources") == "STOP"
def test_changed_protocol_is_refused_unless_acknowledged(tmp_path):
    G = yaml.safe_load(GATES.read_text()); G["gates"]["G1_free_space"]["stop"] = 20.0; p = tmp_path / "gates_v2.yaml"; p.write_text(yaml.safe_dump(G)); sha = hashlib.sha256(p.read_bytes()).hexdigest()
    rc, out, err = run(tmp_path, [wm()], [actor()], bench(), ["--iteration", "400"], gates=str(p)); assert rc == 2 and "PROTOCOL MISMATCH" in err
    rc, out, err = run(tmp_path, [wm()], [actor()], bench(), ["--iteration", "400", "--protocol-ack", sha], gates=str(p)); assert rc == 0
    G2 = copy.deepcopy(G); del G2["gates"]["G7_gpu_resources"]; q = tmp_path / "bad.yaml"; q.write_text(yaml.safe_dump(G2)); rc, out, err = run(tmp_path, [wm()], [actor()], bench(), ["--protocol-ack", hashlib.sha256(q.read_bytes()).hexdigest()], gates=str(q)); assert rc == 2 and "MALFORMED" in err
def test_missing_or_unreadable_inputs_are_incomplete_not_a_crash_that_looks_like_stop(tmp_path):
    good = tmp_path / "w.json"; good.write_text(json.dumps(wm())); act = tmp_path / "a.json"; act.write_text(json.dumps(actor()))
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err): rc = GC.main(["--wm-eval", str(good), "--actor-eval", str(act), "--benchmark", str(tmp_path / "nope.json"), "--iteration", "400"])
    assert rc == 2 and status(buf.getvalue(), "G7_gpu_resources") == "NOT_AVAILABLE" and "missing or unreadable" in buf.getvalue()                       # a missing benchmark file is INCOMPLETE (2), never exit 1 (== STOP)
    bad = tmp_path / "bad.json"; bad.write_text("{not json")
    for wmf, af in ((str(tmp_path / "gone.json"), str(act)), (str(good), str(bad))):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err): rc = GC.main(["--wm-eval", wmf, "--actor-eval", af, "--iteration", "400"])
        assert rc == 2 and "INCOMPLETE" in err.getvalue()
