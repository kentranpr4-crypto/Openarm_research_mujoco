# Validation report (self-tested, 2026-11-06) - status: PARTIAL (GPU not available)

**TIP.** Goal: portable SuperMimic-style training package for an RTX 4090. Output: source, 3 tasks, 4 configs, setup/smoke scripts, CLI, tests, audit. Limits: no commit/push, no full training, no ACT/G2a/G2b data, no seeds 31-33, upstream XML untouched, everything under `research/supermimic_openarm/`. Assumption: CPU MuJoCo for simulation, CUDA only for networks (MuJoCo-Warp unverified).
These are **self-checks**, not an independent audit. Nothing here is a research result.

## Test environments
| Env | Python | torch | mujoco | numpy | Result |
|---|---|---|---|---|---|
| A (cloud `act` venv) | 3.13 | 2.11.0+cu130 (CPU use) | 3.17.0 | 2.2.6 | 44 tests OK, 52.9 s |
| B (**pinned**, clean venv from the lock) | 3.10.20 | 2.9.1+cu128 (no GPU) | 3.17.0 | 2.2.6 | 44 tests OK, 67.1 s; `smoke_test.sh` exit 0 |
The same CPU mini-training printed identical numbers in A and B (e.g. iteration 3 `wm_loss=1.542`): deterministic across the two stacks.

## Acceptance checks (Phase 7)
| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Syntax and imports | PASS | `py_compile`, all modules imported by tests; package import loads no torch (`test_no_torch_import...`) |
| 2 | OpenArm model loading | PASS | `TestEnvAndTasks` builds all 3 tasks from `sim/lab_scene.py` |
| 3 | Observation/action dimensions | PASS | `test_state_and_action_dimensions` (state 34, action 8, actor obs per task spec) |
| 4 | Action clip / slew | PASS | `test_clip_and_slew_applied_to_executed_action`; torch mirror parity vs original `ActionAdapter` (`test_torch_adapter_parity...`) |
| 5 | Deterministic reset | PASS | `test_deterministic_reset`, `test_deterministic_rollout`, sync == subproc env |
| 6 | World-model one-step prediction | PASS | `test_one_step_prediction` (shape, finite, `prev_ctrl == executed action`) |
| 7 | Multi-step imagined rollout shapes | PASS | H = 8, 16, 32 |
| 8 | Actor gradient through the world model | PASS | non-zero finite actor grad; WM params get no grad; cutting the dynamics changes the gradient; gradient from the LAST reward reaches the FIRST action (BPTT over 8 steps) |
| 9 | Critic gradients, terminal masking | PASS | `test_lambda_returns_terminal_masking`, `test_critic_gradients_and_update` |
| 10 | Replay sampling | PASS | windows never cross episodes, ring wrap, state round-trip |
| 11 | Checkpoint save / load / resume | PASS | resume restores weights + replay; metrics continue 1..4; COMPLETED run refused |
| 12 | Finite losses and gradients | PASS | `bad_grads == 0`, `actor_grad_norm > 0` in every mini-training row (3 tasks) |
| 13 | CPU mini-training | PASS | reaching 4 it, pushing 2 it, pick_place 2 it; PPO and BC mini-runs |
| 14 | Evaluation correctness | PASS (unit level) | success criteria unit tests incl. "held box / never lifted / fingers touching is NOT success"; evaluate CLI on held-out seeds >= 9,000,000 |
| 15 | No protected data access | PASS | static scan (no datasets, ACT, G2a/G2b, pose files) + seed guard rejects 31-33 |
| 16 | No change to ACT / G2b / upstream | PASS | `git status` of `policies control sim app` empty; upstream clone clean; `eval/` shows only the 2 pre-existing pending G2b edits |
| 17 | No privileged leakage in deployment obs | PASS | deployment observation has no box terms; reaching actor input == deployment observation; push/pick-place flagged `deployable_without_perception=False` |
Extra: FK matches `mj_forward` to 1e-6 (25 random poses); corruption of a checkpoint is detected; CLI fails with exit 2 and a clear message for missing config, bad override, CUDA requested without CUDA; Studio adapter works with torch and mujoco blocked.

## Measured on CPU (not GPU)
Reaching config shapes on 1 CPU process (4 sync envs, hidden 512, window 8, batch 2048, imagination batch 1024, H = 16): **332 env transitions/s (3.0 ms/step incl. policy)**, world-model update (10 updates) 2.34 s, actor+critic update 0.87 s. `debug_cpu`: 292 transitions/s. `benchmark` verdict on CPU: `NOT_TESTED_NO_GPU`.

## NOT TESTED (no GPU, no workstation)
CUDA device use, GPU determinism, TF32, bf16 autocast, the hard VRAM cap (`set_per_process_memory_fraction`), real VRAM peak, GPU throughput, the 64-process subproc backend at scale, multi-hour stability, driver compatibility of the cu128 wheel, and a full `setup_ubuntu.sh` run (container disk too small for a second install; its components were tested separately: `check_gpu.sh`, `smoke_test.sh`, pinned install via `uv`; syntax checked with `bash -n`). Studio hook not applied or tested inside the live app (adapter tested stand-alone).

## ESTIMATES (labelled, unmeasured on GPU)
* VRAM: 2-4 GB (saved activations 0.41 GB per 1024-sample chunk at H = 16, 0.82 GB at H = 32, window batch 0.21 GB, weights + Adam < 0.02 GB, plus CUDA context / allocator overhead). Cap in configs: 20 GB.
* Time: environment cost dominates: about `332 x min(num_envs, free cores)` transitions/s as an optimistic bound; reaching pilot (3000 iterations x 256 transitions) plausibly tens of minutes to a few hours. **Convergence is unknown**: no claim.

## Remaining blockers / risks
1. No GPU run yet (all GPU items above). 2. Pushing and pick-and-place are not proven trainable; reward shaping and horizons are untuned. 3. Pick-and-place relies on contact flags only in evaluation (the imagined reward uses predicted finger opening and box height). 4. Full paper text not read (appendix hyperparameters unverified). 5. MuJoCo-Warp backend not implemented. 6. `app/backend/server.py` hook is only a proposed patch. 7. BC baseline exists for reaching only. 8. Environment episode stream restarts with new seeds after a resume.

## GO / NO-GO for the RTX 4090
**GO to the next step only**: clone, run `setup_ubuntu.sh`, `check_gpu.sh`, `smoke_test.sh`, then `benchmark --steps 1000`. **NO-GO for long training** until the benchmark gate (VRAM within cap, `bad_grads == 0`, checkpoint resume) passes on the workstation and a short reaching pilot is evaluated on held-out goals.
