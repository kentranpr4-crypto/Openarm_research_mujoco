# Stage 1 closed-loop ACT + WM3-B pilot — REPORT (2026-11-01) — **EXPLORATORY SIMULATION PILOT**
Status: **DONE** (protocol executed as frozen, 120/120 episodes, all integration gates PASS). Development evidence, privileged oracle observer (exact MuJoCo box pose/contacts), 1 simulator, not Sim2Real, not independent. Tự kiểm bởi cùng operator.
Raw: `outputs/closed_loop_pilot/results/{pilot_results,pilot_analysis}.json`, freeze `outputs/closed_loop_pilot/protocol_freeze.json` (sha256 `8fd93a31ccfedcc5b9cc5184171f69b1e124fd1989dc58cf997c83b879220b2c`, 2026-11-01T07:53:54Z, written before any pilot episode).

## Integration gates (11/11 PASS in final run; CPU 103.6 s)
Run 1 FAIL (G1 compared two different env instances -> 2e-5 state / 1.0 contact-count divergence from solver warm-start history, not an observer bug; G7-G10 truncated by my own 0.06 CPU-h test cap); G1 redefined to the user's wording "same simulation state" (hooked into `wm1_sim.simulate`: diff exactly 0.0); both earlier reports preserved (`integration_report_run1_FAIL.json`, `..._run2_before_orchestrator_PASS.json`). Evidence: observer==wm1_sim 0.0; no leakage (window identical when later rows are removed/poisoned); A0 runner == ACTRunner.select_action and == stored xsweep rollout, max |diff| 0.0 (60 steps); time indexing; executed = clip then slew == ActionAdapter; no cross-checkpoint/episode leakage; batch == single prediction (3.8e-7); malformed shapes rejected 5/5; RNG reproducible; CPU guard + atomic writer; orchestrator OK/CAP/ERROR.

## Result (24 poses; A0 x1, A1 x3, A2 x1; 120 episodes, 0 errors, 0 invalid chunks, 0 non-finite)
| Arm | success | no_pick | dropped/misplaced | palm contact | cluster flag |
|---|---|---|---|---|---|
| A0 | 18/24 (75 %) | 0 | 6 | 4 | 6 |
| A1 | 49/72 (68 %), 16.33/24 per-pose mean | 1 | 22 | 18 | 14 |
| A2 | 17/24 (71 %) | 0 | 7 | 7 | 6 |
- **Primary A2 − mean(A1): +0.028 (+2.8 pp), 95 % CI [−0.194, +0.236], INCONCLUSIVE.** 7 poses better, 6 worse, 11 equal; 8 discordant poses (A2 better: 17, 18, 45, 46; worse: 0, 30, 32, 47).
- A2 − A0: −0.042 [−0.25, +0.167] INCONCLUSIVE; A1 − A0: −0.069 [−0.278, +0.125] INCONCLUSIVE.
- A0 reproduced xsweep c0_seed0 outcomes on 24/24 poses (final xy diff 0.0 m): deterministic simulator/policy.
- Stratum (descriptive, post hoc): low x 0.34–0.37: A0 10/16, A1 26/48, A2 9/16; edge 0.38–0.39: A0 8/8, A1 23/24, A2 8/8 (no headroom at the edge).
- Selection: A1 picks c0_s0/s1/s2 = 223/210/215; A2 = 92/49/75 (A2 keeps the default 43 % of decisions).

## A2 ranking diagnostics by phase (216 decisions, chunk 2..10)
| Phase | n | J spread (median) | margin best-2nd (median) | tie frac margin ≤1e-3 / ≤5e-3 | picked ≠ default | argmin changes if only continuous dxy+dz used |
|---|---|---|---|---|---|---|
| approach | 48 | 0.0087 | 0.0014 | 0.375 / 0.625 | 0.40 | 0.15 |
| grasp | 24 | 0.0363 | 0.0147 | 0.0 / 0.25 | 0.96 | 0.17 |
| carry | 62 | 0.0168 | 0.0030 | 0.37 / 0.63 | 0.61 | 0.10 |
| place | 82 | 0.0006 | 0.0004 | 0.85 / 0.95 | 0.54 | 0.02 |
Tie under the registered tolerance 1e-9: 0 % in all phases (the SHA-1 tie-break never fires); selections at margins ≤1e-3 are decided by numerical noise. Binary palm/lift penalties have median spread 0 in every phase: ranking is driven by continuous predicted distance. Candidate-level regret and ranking accuracy vs ground truth: **NOT_EVALUABLE** (only the selected chunk is executed).

## Compute / safety
Stage 1 total 1 443 s = **0.401 CPU-h** of 0.90 (tests 431 s incl. the failed first run; pilot 1 013 s, wall 259 s with 4 workers); median episode wall A0 6.1 s, A1 6.1 s, A2 7.7 s (ACT 0.8 s/episode for A0/A1, 2.25 s for A2; WM3-B 64 ms/episode). 0 safety violations; ctrlrange clips (4 442 / 12 713 / 4 105 values) are the gripper-open saturation found in Stage 0.1.

## Deviations / limitations
1. Estimated 21 s/episode; actual ~6 s because cameras are rendered only on chunk boundaries (`lean_step`, physics identical, verified by G3 and by A0 == stored xsweep rollouts); added before the freeze.
2. First integration run failed/truncated (see above), its CPU is in the ledger.
3. 24 poses, 1 A2 run per pose: the CI half-width is ~±21 pp; effects below ~20 pp are undetectable. Single policy-seed set (C0 x3), 1 WM3-B checkpoint, 1 simulator, oracle observer.
4. J is applied myopically at t+20 inside a closed loop (extrapolation from the offline decision-utility study); place-phase decisions carry almost no signal.
5. Phase labels and the continuous/binary decomposition are post-registered diagnostics defined in the freeze (not outcome-dependent).
6. Not independent evidence; Sim2Real would need a perception-based observer.

## Recommendation
No evidence in this pilot that WM3-B ranking helps ACT in closed loop (+2.8 pp, CI spans ±20 pp); equally no evidence of harm. Do **not** scale up with the same J/candidate recipe. A larger run is justified only if a design change addresses the diagnostics (candidates with more behavioural spread at grasp/carry; fewer decisions where margin ≤1e-3), which needs a new pre-registration; do not retune J on these outcomes.
