#!/usr/bin/env python3
"""FREEZE Gate v3 (protocol gate-v3.0) — runs every pre-freeze check; if ANY check fails nothing is written and no training may start.
Checks: (1) all Gate v3 unit tests (incl. geometry/symmetry, aggregation, G3 rare-label calibration on the validation split only, G7 regimes/boundaries/support); (2) contract-v2 tests (leakage, rotation-vector round trip, tilt repair, losses);
(3) dataset integrity (v1.1 and v2); (4) the approved parameter list equals CONFIG; (5) calibration uses the validation split only (config + enforcement test); (6) no model number is read by this script.
Writes outputs/gates_v3/gates_v3_config.json (frozen path) and outputs/gates_v3/freeze.json.   Run: python3 eval/freeze_gate_v3.py"""
import sys, json, hashlib, subprocess, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm4_gates as G
PY = "/home/user/.venvs/act/bin/python"; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd, tail): r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True); return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-tail:]
APPROVED = {"calibration_quantile": 0.96, "calibration_fpr": 0.04, "eval_fpr_max": 0.05, "tpr_min": 0.70, "tumble_step_geodesic_rad": 0.30, "tau_hnorm_singular": 0.20, "rotating_total_geodesic_min_rad": 0.02, "sensitivity": [0.20, 0.50],
            "static_abs_max_rad": 0.01, "static_relative_max": 3.0, "static_denominator_floor_rad": 0.003, "yaw_ratio_max": 1.10, "yaw_ratio_upper_max": 1.25, "hard_subset_min_prevalence": 0.05, "calibration_split": "val"}
def main():
    if G.CFG_PATH.exists(): sys.exit(f"{G.CFG_PATH} exists (frozen)")
    out = {}; fails = []
    rc, t = run(["python3", "eval/test_wm4_gates.py"], 2); ok = rc == 0 and t and t[-1].startswith("ALL PASS"); out["gate_v3_unit_tests"] = dict(passed=bool(ok), last=t[-1] if t else None); (fails.append("gate_v3_unit_tests") if not ok else None)
    ut = json.loads((ROOT / "outputs/gates_v3/unit_tests.json").read_text()); same = ut["config_sha256"] == G.config_sha() and ut["all_pass"]; out["unit_tests_bound_to_current_config"] = bool(same); (fails.append("unit_tests_config_mismatch") if not same else None)
    rc, t = run([PY, "eval/test_contract_v2.py"], 2); ok = rc == 0 and t and t[-1].startswith("ALL PASS"); out["contract_v2_tests"] = dict(passed=bool(ok), last=t[-1] if t else None); (fails.append("contract_v2_tests") if not ok else None)
    for v in ("v1_1", "v2"):
        rc, t = run([PY, "eval/dataset_integrity.py", v], 3); res = json.loads((ROOT / f"outputs/wm2_{'v1_1' if v == 'v1_1' else 'v2'}/integrity.json").read_text()); ok = rc == 0 and res["overall"] == "PASS"; out[f"dataset_integrity_{v}"] = dict(passed=bool(ok), gates={k: g["passed"] for k, g in res["gates"].items()}, integrity_sha256=sha(ROOT / f"outputs/wm2_{'v1_1' if v == 'v1_1' else 'v2'}/integrity.json"), index_sha256=res["dataset_index_sha256"]); (fails.append(f"integrity_{v}") if not ok else None)
    C = G.CONFIG; op = C["G3"]["rare_label"]["operating_point"]; r = C["G7"]["regimes"]; st = C["G7"]["static_input_subset"]
    got = {"calibration_quantile": op["calibration_quantile"], "calibration_fpr": op["calibration_fpr"], "eval_fpr_max": op["eval_fpr_max"], "tpr_min": op["tpr_min"], "tumble_step_geodesic_rad": r["tumble_step_geodesic_rad"], "tau_hnorm_singular": r["tau_hnorm_singular"], "rotating_total_geodesic_min_rad": r["rotating_total_geodesic_min_rad"],
           "sensitivity": C["G7"]["sensitivity_diagnostics"]["tumble_step_geodesic_rad"], "static_abs_max_rad": st["abs_max_rad"], "static_relative_max": st["relative"]["max_ratio"], "static_denominator_floor_rad": st["relative"]["denominator_floor_rad"], "yaw_ratio_max": C["G7"]["ratio_max"], "yaw_ratio_upper_max": C["G7"]["ratio_upper_bound_max"],
           "hard_subset_min_prevalence": C["G3"]["hard_subset"]["min_prevalence"], "calibration_split": op["calibration_split"]}
    out["approved_parameters_equal_config"] = dict(passed=got == APPROVED, config=got); (fails.append("approved_parameters") if got != APPROVED else None)
    out["calibration_uses_validation_split_only"] = dict(passed=op["calibration_split"] == "val" and ut["results"].get("G3_calibration_must_use_the_validation_split_only_other_split_names_NOT_EVALUABLE") is True and ut["results"].get("G3_operating_threshold_tau_is_independent_of_evaluation_labels") is True); (fails.append("calibration_split") if not out["calibration_uses_validation_split_only"]["passed"] else None)
    if fails: print(json.dumps(out, indent=1)); sys.exit(f"PRE-FREEZE CHECKS FAILED: {fails} — Gate v3 NOT frozen, no training allowed")
    G.CFG_PATH.write_bytes(G.config_bytes()); code = {f: sha(ROOT / f) for f in ("eval/wm4_gates.py", "eval/geometry_v2.py", "eval/wm3_gates.py", "eval/wm2_gates.py", "data/contract_v2.py", "eval/orientation_loss.py", "eval/test_wm4_gates.py", "eval/test_contract_v2.py", "eval/dataset_integrity.py", "eval/freeze_gate_v3.py")}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(); dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    F = dict(protocol_version="gate-v3.0", frozen_at=datetime.datetime.utcnow().isoformat() + "Z", config_sha256=sha(G.CFG_PATH), status="Gate v3 is PROSPECTIVE: it applies only to evaluations designed and frozen after this freeze. It does not change any historical verdict (WM-2 v1 gates, WM-3B development v2 verdicts, I-1 independent evaluation: G3 FAIL, yaw 0.0515 vs 0.0146 rad).",
             inherits_gate_v2_config_sha256=C["inherits"]["gates_v2_config_sha256"], approved_parameters=APPROVED, pre_freeze_checks=out, code_sha256=code, unit_tests_sha256=sha(ROOT / "outputs/gates_v3/unit_tests.json"), unit_tests=dict(n=ut["n"], all_pass=ut["all_pass"]), code_commit=head, uncommitted_files_at_freeze=dirty,
             attestation="No threshold was chosen or changed using any model prediction or result; the numeric values are the user-approved list; calibration of the rare-event operating point uses the validation split only; Euler yaw is never ground truth at singular orientations; full and D2 symmetry-aware geodesic distances are reported separately; FAIL outranks NOT_EVALUABLE; NOT_APPLICABLE is never a pass.")
    (ROOT / "outputs/gates_v3/freeze.json").write_text(json.dumps(F, indent=1)); print(json.dumps({k: F[k] for k in ("protocol_version", "config_sha256", "unit_tests", "code_commit")}, indent=1)); print("FROZEN")
if __name__ == "__main__": main()
