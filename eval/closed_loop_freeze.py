#!/usr/bin/env python3
"""Writes outputs/closed_loop_pilot/protocol_freeze.json BEFORE any pilot episode exists (refuses to overwrite).  Run: python3 eval/closed_loop_freeze.py"""
import json, sys, time, hashlib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); D = ROOT / "outputs/closed_loop_pilot"; F = D / "protocol_freeze.json"
if F.exists(): sys.exit("freeze exists")
if (D / "results").exists(): sys.exit("results exist before freeze")
sh = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
import closed_loop_pilot as P, decision_utility_lib as L
rep = json.loads((D / "integration_report.json").read_text()); assert rep["verdict"] == "PASS" and rep["n_gates"] == 11
assert rep["code_sha256"]["eval/closed_loop_pilot.py"] == sh("eval/closed_loop_pilot.py") and rep["code_sha256"]["eval/test_closed_loop_pilot.py"] == sh("eval/test_closed_loop_pilot.py")
assert sh("outputs/xsweep/xsweep_config.json") == "6c59292d6d951673ab813fc50eba38612c01777534ccdad2a6f3c5797b1f596c"
poses = P.pilot_poses(); hist = P.historical_a0(); ho = {str(p["idx"]): hist[p["idx"]]["outcome"] for p in poses}
files = ["eval/closed_loop_pilot.py", "eval/test_closed_loop_pilot.py", "eval/closed_loop_pilot_analysis.py", "eval/closed_loop_freeze.py", "eval/decision_utility_lib.py", "eval/wm2_run.py", "eval/wm2_models.py", "eval/wm3b_model.py", "eval/metrics.py", "eval/evaluate.py",
         "data/wm1_sim.py", "control/safety_limits.py", "control/action_adapter.py", "sim/lab_env.py", "sim/lab_scene.py", "sim/cameras.py", "policies/baseline.py", "policies/act_infer.py", "outputs/xsweep/xsweep_config.json", "outputs/closed_loop_pilot/integration_report.json",
         "outputs/wm3/models/wm3b.pt", "outputs/wm3/gates_v2_config.json", "outputs/gates_v3/gates_v3_config.json", "outputs/decision_utility/protocol_freeze.json", "outputs/xsweep/eval/c0_seed0/act/sweep/episodes.jsonl"]
for k, d in P.CK.items(): files += [f"outputs/checkpoints/{d}/last/model.safetensors", f"outputs/checkpoints/{d}/last/config.json", f"outputs/checkpoints/{d}/last/norm_contract.json", f"outputs/checkpoints/{d}/last/policy_preprocessor_step_3_normalizer_processor.safetensors", f"outputs/checkpoints/{d}/last/policy_postprocessor_step_0_unnormalizer_processor.safetensors"]
fz = dict(name="Stage 1 closed-loop ACT + WM3-B pilot - protocol freeze", frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pilot_episodes_exist_at_freeze=False, label="EXPLORATORY SIMULATION PILOT; DEVELOPMENT evidence; privileged oracle observer; not Sim2Real",
          poses=poses, poses_sha256=hashlib.sha256(json.dumps(poses, sort_keys=True).encode()).hexdigest(), pose_source="outputs/xsweep/xsweep_config.json (sha 6c59292d...), x in {0.34..0.39} x 4 blocks = 24; no seeds 31-33",
          arms=dict(A0="ACT c0_seed0 (e6b_c0_seed0), 1 episode/pose", A1="chunk1 c0_seed0; chunks 2..10 uniform random pick among the 3 C0 checkpoints; 3 repeats/pose; rng=default_rng([20261101,8001,pose_idx,repeat]); one integer(3) draw per decision from chunk 2 on", A2="chunk1 c0_seed0; chunks 2..10 argmin frozen J of WM3-B on executed sequences (row t+20); 1 episode/pose; tie tol 1e-9, tie-break sha1(f'{pose_idx}|{k}|tie') over candidate index k in [c0_s0,c0_s1,c0_s2]"),
          candidate_recipe="C0 x 3 frozen training seeds: " + json.dumps(P.CK), horizon_steps=P.HORIZON, chunk=P.CHUNK, control_hz=20, executed_semantics="16-D absolute actuator targets -> clip(ctrlrange) then slew(2.0 rad/s arm, 0.2 m/s gripper) from last executed command, sequential per step; WM A_fut = executed[:, :8]",
          J=dict(formula="dxy + |z-0.384| + 0.05*palm_any(>0.01N) + 0.05*(max rise<0.05m) at row t+20", TRAY=L.TRAY.tolist(), Z_TRAY=L.Z_TRAY, PALM_THR=L.PALM_THR, LIFT_M=L.LIFT_M, W_PALM=L.W_PALM, W_LIFT=L.W_LIFT), wm="WM3-B outputs/wm3/models/wm3b.pt (ridge refit deterministically from frozen train windows, event heads stubbed - not used by J)",
          outcomes="eval/evaluate.py classify + eval/metrics.task_labels (success / no_pick / dropped_or_misplaced) at horizon 200; flags palm_contact (hand force > 0.01 N any step), cluster_flag (xsweep definition), action_invalid_chunks, non-finite; no episode is discarded",
          phases="approach: no finger contact yet & rise<0.01; grasp: rise<0.05; carry: rise>=0.05 & dxy_tray>=0.03; place: rise>=0.05 & dxy_tray<0.03 (privileged state so far)",
          primary="pose-level success: A2 - mean(A1 repeats), paired by pose; percentile bootstrap over the 24 poses (A1 repeats kept together), 10000 resamples, default_rng(0), 95% CI; verdict SUPPORTED_BETTER if CI lower>0, SUPPORTED_WORSE if upper<0 else INCONCLUSIVE; effect size, counts, per-pose differences, discordant poses reported",
          secondary="A2 vs A0, A1 vs A0 (same bootstrap); failure classification; tie fraction and J spread by phase; continuous (dxy+dz) vs binary (palm, lift) contribution; candidate-level regret/pairwise ranking = NOT_EVALUABLE (no counterfactual outcomes)",
          historical_A0_outcomes_c0_seed0_xsweep=ho,
          two_phase_execution="phase 1 = A0 on all 24 poses; STOP (report, no phase 2) if A0 outcome differs from historical xsweep on >2/24 poses (3-way outcome class), or A0 success >= 0.90, or incomplete; phase 2 = A2 + A1x3 per pose",
          stop_conditions=["CPU cap", "any action_invalid chunk or non-finite state/action -> STOP and report", "worker ERROR", "A0 deviation >2/24", "A0 success >= 90%", "frozen hash change", "pilot results already exist"],
          compute=dict(cap_total_stage1_cpu_s=P.CAP_TOTAL_S, note="entire Stage 1 incl. integration tests (ledger outputs/closed_loop_pilot/cpu_ledger.json); workers=4; per-worker cap=(cap-ledger_used-60)/4; PARTIAL when reached, never raised", ledger_used_at_freeze_s=P.ledger_used()),
          files_sha256={f: sh(f) for f in files})
P.atomic_json(F, fz); print(hashlib.sha256(F.read_bytes()).hexdigest(), fz["frozen_at"], len(files), "files; ledger", fz["compute"]["ledger_used_at_freeze_s"])
