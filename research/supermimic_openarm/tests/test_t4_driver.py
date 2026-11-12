"""Quick tests of the T4 driver's pure logic: the pilot decision rule and the local verification (registered hashes, bundle content policy).  The end-to-end control flow is exercised by tests/run_t4_driver_selftest.py (slow, uses a fake CLI)."""
import hashlib, importlib.util, io, json, tarfile
from pathlib import Path
import pytest
REPO = Path(__file__).resolve().parents[3]
def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"research/supermimic_openarm/scripts/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
D = load("colab_t4_run")
def smoke(ok=True, peak=9000, t_it=0.8, ck=0.5, ev=30.0, gate=300.0, start=60.0):
    return dict(verdict="SMOKE_OK" if ok else "SMOKE_FAILED", nvidia_smi_peak_used_mib_incl_context=peak, steps={"2_training": dict(seconds_per_iteration_steady=t_it, wall_s_incl_warmup=3 * t_it + start), "3_checkpoint": dict(save_s=ck), "4_rollout_eval": dict(projected_trainer_eval_32_episodes_s=ev), "5_gate_evaluation_cost": dict(total_gate_evaluation_s=gate)})
def bench(ok=True, peak=9000, t_it=0.8, ram=3.0): return dict(verdict="BENCHMARK_OK" if ok else "FAILED_EXIT_1", gpu=dict(mem_used_mib_peak_nvidia_smi=peak), training_step_latency=dict(iteration_estimate_s=t_it), host_memory=dict(available_min_gib=ram))
def test_decision_go_when_every_condition_holds():
    d = D.decide(smoke(), bench(), 7200, 15360); assert d["go"] and d["projected_iterations_in_budget"] >= 4000 and all(d["conditions"].values())
def test_decision_blocks_each_condition_independently():
    assert not D.decide(smoke(ok=False), bench(), 7200, 15360)["go"]
    assert not D.decide(smoke(), bench(ok=False), 7200, 15360)["go"]
    d = D.decide(smoke(peak=12800), bench(peak=9000), 7200, 15360); assert not d["conditions"]["vram_headroom_ok"] and not d["go"]            # 80% of 15360 MiB = 12288 MiB
    d = D.decide(smoke(), bench(ram=0.8), 7200, 15360); assert not d["conditions"]["host_ram_headroom_ok"] and not d["go"]
    d = D.decide(smoke(t_it=1.6), bench(t_it=1.6), 7200, 15360); assert not d["conditions"]["projected_iterations_ge_4000"] and d["projected_iterations_in_budget"] < 4000 and not d["go"]     # too slow: stop at the benchmark
    d = D.decide(smoke(), bench(t_it=0.8), 7200, 15360); d2 = D.decide(smoke(gate=900.0), bench(), 7200, 15360); assert d2["projected_iterations_in_budget"] < d["projected_iterations_in_budget"]      # eval + gate cost is counted
    assert not D.decide({}, {}, 7200, 15360)["go"]                                                                                                 # missing measurements never pass
def test_decision_uses_the_slower_of_benchmark_and_smoke_iteration_time():
    d = D.decide(smoke(t_it=0.5), bench(t_it=1.2), 7200, 15360); assert d["seconds_per_iteration"] == 1.2
def mini_package(tmp, forbidden=None, gates_hash=None, extra_missing_hash=False):
    files = {"research/supermimic_openarm/gates/p1_pilot_gates.yaml": (REPO / "research/supermimic_openarm/gates/p1_pilot_gates.yaml").read_bytes(), "research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml": (REPO / "research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml").read_bytes(),
             "research/supermimic_openarm/scripts/colab_p1_bootstrap.py": b"print('x')\n", "sim/a.py": b"1\n"}
    if forbidden: files[forbidden] = b"x"
    man = dict(git_commit="a" * 40, dirty_paths_in_bundle=[], files={k: hashlib.sha256(v).hexdigest() for k, v in files.items()}); p = Path(tmp) / "p1_bundle.tar.gz"
    with tarfile.open(p, "w:gz") as t:
        for k, v in files.items(): ti = tarfile.TarInfo(k); ti.size = len(v); t.addfile(ti, io.BytesIO(v))
        b = json.dumps(man).encode(); ti = tarfile.TarInfo("BUNDLE_MANIFEST.json"); ti.size = len(b); t.addfile(ti, io.BytesIO(b))
    (Path(tmp) / "wm_eval_dataset_v1.npz").write_bytes(b"dataset"); exp = dict(gates_sha256=gates_hash or hashlib.sha256(files["research/supermimic_openarm/gates/p1_pilot_gates.yaml"]).hexdigest(), protocol_sha256=hashlib.sha256(files["research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml"]).hexdigest(), dataset_sha256=hashlib.sha256(b"dataset").hexdigest())
    if extra_missing_hash: exp["dataset_sha256"] = ""
    (Path(tmp) / "EXPECTED_HASHES.json").write_text(json.dumps(exp)); res = Path(tmp) / "res"; res.mkdir(); return Path(tmp), res
def test_verify_accepts_a_clean_package(tmp_path):
    pkg, res = mini_package(tmp_path); ok, info = D.verify(pkg, res); assert ok and not info["problems"] and (res / "bundle_listing.txt").exists()
def test_verify_stops_on_hash_mismatch_missing_reference_and_forbidden_members(tmp_path):
    for kw, frag in ((dict(gates_hash="0" * 64), "gates_sha256 differs"), (dict(extra_missing_hash=True), "reference hash is missing"), (dict(forbidden="sim/.venv/bin/activate"), "forbidden"), (dict(forbidden="research/supermimic_openarm/x.pt"), "forbidden"), (dict(forbidden="outputs/a.json"), "forbidden"), (dict(forbidden="research/openarm_wam/x.py"), "forbidden"), (dict(forbidden="research/supermimic_openarm/oauth_client.json"), "forbidden")):
        d = tmp_path / frag.split()[0] / str(abs(hash(str(kw)))); d.mkdir(parents=True); pkg, res = mini_package(d, **kw); ok, info = D.verify(pkg, res); assert not ok and any(frag in p for p in info["problems"]), (kw, info["problems"])
def test_the_real_registered_hashes_match_the_working_tree():
    m = load("make_t4_package"); assert hashlib.sha256((REPO / "research/supermimic_openarm/gates/p1_pilot_gates.yaml").read_bytes()).hexdigest() == m.REGISTERED["gates_sha256"]
    assert hashlib.sha256((REPO / "research/supermimic_openarm/experiments/p1_actor_ab_protocol.yaml").read_bytes()).hexdigest() == m.REGISTERED["protocol_sha256"]
def test_decision_blocks_when_the_memory_probe_predicts_the_registered_config_does_not_fit():
    ok = dict(fits_with_15pct_headroom=True, predicted_host_gib_for_registered_config=9.0); bad = dict(fits_with_15pct_headroom=False, predicted_host_gib_for_registered_config=41.0)
    assert D.decide(smoke(), bench(), 7200, 15360, ok)["go"]
    d = D.decide(smoke(), bench(), 7200, 15360, bad); assert not d["conditions"]["host_ram_predicted_ok"] and not d["go"] and d["memprobe"]["predicted_host_gib_for_registered_config"] == 41.0
    assert not D.decide(smoke(), {}, 7200, 15360, bad)["go"]                                                                                          # benchmark skipped for RAM: still NO-GO, never a silent GO
