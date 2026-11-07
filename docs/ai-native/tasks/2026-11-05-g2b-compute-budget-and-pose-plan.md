# G2b compute reconciliation, 7-candidate budget and pose plan — PLAN ONLY (nothing executed; cap NOT AUTHORIZED) — 2026-11-05
Sources: saved logs only (`integration_v5_I4.json`, `integration_v4_I2_I4.json`, `intervention_benchmark_v2.json`, the CPU ledgers) plus two cheap offline measurements (bootstrap time on synthetic data, `save_trace` time). No simulator was run. Report: `outputs/g2b_impl/cpu_reconciliation_report.json`.

## 1. Reconciliation of the saved I4 CPU accounting (no unresolved discrepancy above 0.01 s per pose; one process-level residual of 1.59 s explained by inference only)
**Timer scopes** (all use the same process CPU clock `getrusage(SELF)+getrusage(CHILDREN)`):
| Timer | Scope |
|---|---|
| setup | env.set_perturbation + env.reset + creation of the recorder (box_z0 read) |
| prefix (0-40) | the first `nominal_i4` call: 2 ACT queries (camera rendering + network + slew_chain), 40 `lean_step`s, 40 `record`s, progress writes |
| branch (40-160) | the second `nominal_i4` call: 6 ACT queries, 120 steps, 120 records, progress writes, plus `record_final` |
| ACT | observation rendering (`obs_full`, 3 cameras) + `policy.chunk` + `slew_chain`, summed over ALL 8 queries of the episode (prefix and branch) |
| physics | `P.lean_step` (50 `mj_step` substeps per control step), all 160 steps |
| recorder | `rec.record` (scratch copy + `mj_forward` + contact scan), all 160 rows |
| labeller | `rec.arrays` + `to_labeller_trace` + `label` (about 40 perturbed re-evaluations) |
| per-pose total | from the creation of the timer dict to the end of labelling |
**Nesting.** ACT, physics and recorder are components that run INSIDE prefix + branch and span both; they are not nested in each other. The labeller runs after the branch and is outside prefix / branch. Therefore ACT can exceed the branch-only CPU: the branch holds only 6 of the 8 queries (e.g. pose 3000: ACT 2.945 s > branch 2.571 s).
| Pose | setup | prefix | branch | labeller | stage sum | per-pose total | residual | ACT+physics+recorder | prefix+branch | residual |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | 0.313 | 0.880 | 2.571 | 0.029 | 3.793 | 3.795 | 0.002 | 3.442 | 3.451 | 0.009 |
| 3001 | 0.321 | 0.828 | 2.440 | 0.015 | 3.604 | 3.605 | 0.001 | 3.258 | 3.268 | 0.010 |
| 3002 | 0.263 | 0.774 | 2.464 | 0.016 | 3.517 | 3.519 | 0.002 | 3.228 | 3.238 | 0.010 |
The 0.001-0.002 s stage residual is guard checks and slicing; the 0.009-0.010 s component residual is the atomic JSON progress writes (8 per pose) and bookkeeping. Sum of the three pose totals 10.919 s vs gate CPU 10.922 s (0.003 s).
**Process level.** In-process timeline: startup 25.028 (numpy 0.47, torch 3.88, project modules 0.67, checkpoint 6.12, MuJoCo env + camera/renderer 13.84) + reuse verification 0.43 + I4 10.922 + exit 0.032 = 36.413 (last milestone). `getrusage(RUSAGE_CHILDREN)` of the launcher = 38.003; sampled live-tree peak 37.91. **Residual 1.59 s** = interpreter start before the first milestone plus interpreter / library teardown after the last milestone (the final milestone is written before teardown). This is INFERRED, not measured; the sampled tree peak (37.91) and rusage (38.003) agree within 0.09 s. Launcher own CPU 1.432 s (polling every 0.25 s over 149 s wall). Aggregate of the run = max(38.003, 37.91) + 1.432 = 39.435 s (matches the report).
**Ledger.** Preparation / post-checks 39.91 s (unmetered preparation estimate 2.0 + manifest 0.07 + first complete dry-run 17.76 + final complete dry-run 19.50 + post-checks 0.58) + run 39.435 = **79.35 s of the 100 s attempt cap**. The first complete dry-run contained two test-design failures and was repeated in full; both are charged. Wall 149 s vs CPU 39.4 s: the machine was contended (torch import 54 s wall, first pose 7.8 s wall for 3.8 s CPU).
**Unresolved discrepancies:** none above 0.01 s per pose; the 1.59 s process residual is explained but unmeasured.

## 2. Measured / assumed cost table (CPU-s)
| Item | Value | Basis |
|---|---|---|
| Start-up (imports, ACT checkpoint, env + camera) | 22.8, 23.8, 25.0 (uncontended); 40.1, 40.9 (contended) | measured 5 times |
| Per pose: prefix 0-40 | 0.77-0.88 | measured (3 poses) |
| Per pose: decision-state ACT query | 0.36 | measured (benchmark) |
| Per branch c0 / O3 / D1 / D2 / S0 (5-6 ACT queries, recorder, labeller) | 2.246 / 2.167 / 2.243 / 2.136 / 2.485 | measured (benchmark, pose 3000) |
| Per branch O1 / H1 | 2.2 / 2.24 | **ASSUMED** equal to O3 / D1 |
| IK / correction | 0.035-0.046 per branch (2 % of a branch) | measured |
| ACT query incl. rendering | 0.34-0.37 | measured; about 80 % of a branch |
| `save_trace` per branch (npz + json) | 0.0125-0.015 | measured offline |
| Bootstrap + report (`full_report`, B = 10 000, all modes, 40 poses x 7 arms) | 0.30 | measured offline on synthetic data |
| Integrity verification (hashes of about 20 files + checkpoint) | about 0.4 | measured (reuse verification 0.32-0.43) |
| Annotation packaging (54 plots) | about 3 | **ASSUMED** |
The earlier assumption of 20-60 s for bootstrap was wrong by two orders of magnitude (0.3 s measured). **Already-completed development tests are NOT G2b-run costs** (labeller suites, integration gates I1-I4, dry-runs, benchmarks were paid under their own authorizations and are not repeated inside the cap). Unbenchmarked O1 / H1: ACT dominates and does not depend on the correction; IK is 2 % of a branch; uncertainty +-0.1 s per branch, +-8 s over 40 poses.

## 3. Mandatory G2b-run costs and scenarios (cap 1 080 CPU-s, NOT AUTHORIZED, not spent)
Per pose = prefix 0.83 + decision query 0.36 + 7 branches (2.246 + 2.2 + 2.167 + 2.243 + 2.136 + 2.243 + 2.485 = 15.72) = **16.9 s**. Branches run the FULL 160-step horizon (no outcome-dependent early stop), so the cost of a pose does not depend on its outcome.
| Scenario | Per pose | 40 poses | Fixed (start-up + save + integrity + stats + packaging) | Total | % of cap | Margin |
|---|---|---|---|---|---|---|
| Expected (uncontended) | 16.9 | 676 | 25 + 4.2 + 0.4 + 0.3 + 3 = 33 | **709 s (0.197 CPU-h)** | 66 % | +371 s |
| Conservative (x1.25 per pose, contended start-up 41, assumed costs doubled, O1/H1 +8) | 21.1 | 845 | 41 + 8.4 + 0.8 + 0.6 + 6 + 8 = 65 | **910 s (0.253 CPU-h)** | 84 % | +170 s |
| Cap-overrun (x1.75 contention per pose) | 29.6 | 1 183 | 65 | **1 248 s** | 116 % | -168 s (the stop rule ends the run at about pose 33) |
Conclusion: 40 poses fit the proposed cap in the expected and conservative scenarios; they do not under sustained x1.75 contention (observed so far only for start-up and the I2-I4 gates, not for I4 itself, which ran uncontended per operation). The cap stays NOT AUTHORIZED; N is NOT reduced or expanded by this plan.

## 4. Atomic per-pose reporting
After each pose (fixed pre-registered order = ascending pose index) write `outputs/g2b_run/poses/pose_<idx>.json` atomically with, for every arm: status, validity checks, CPU components, trace sha256, label (terminal, events, endpoints, ambiguity), code / protocol hashes; and append the pose total to the CPU ledger. A pose is COMPLETE only if all 7 arms finished and validated; an interrupted pose keeps its in-progress record but is excluded from every comparison (arm-independent). No automatic retry; a pose is never re-run to improve a result.

## 5. Objective cap stop rule (independent of outcomes and of candidate success)
Before starting pose k, stop (status PARTIAL) iff `CPU_used + 1.25 x max(per-pose CPU of completed poses) + R_reserve > cap`, where R_reserve = 20 s (statistics 0.3 s, packaging 3 s, margin). The only inputs are CPU ledger values; the order of poses is fixed in advance; the number of ACT queries per branch is fixed; therefore the stopping time does not depend on any label. The launcher's cooperative guard (92 %) and hard kill (cap - 0.5 s) remain as backstops.

## 6. Effect of a partial evaluation on paired inference
The completed poses are a deterministic, outcome-independent prefix of the pre-registered order, so the paired risk differences on them are unbiased for the population of the first n poses; the cost is precision (the CI half-width grows roughly as 1/sqrt(n)) and power. Rules (set now, not after results): N planned = 40 and is never changed after seeing outcomes; a CONFIRMATORY claim on O1-c0 / D1-c0 or an S0 equivalence statement requires all 40 poses (minus pre-specified NOT_APPLICABLE ones) complete; with fewer, the analysis is reported as PARTIAL / exploratory with the number of completed poses and the achieved CI width, and equivalence is reported as "insufficient precision" rather than as a failure. Applicability is decided from c0 and applies to all arms; in G2a 48 of 48 poses were applicable, so the expected effective N is 40.

## 7. Pose-generation and sealed-reserve protocol (DESIGN ONLY: no pose is generated, no file is created)
**Development poses (N = 40):** separate RNG namespace `[20261106, 9301]`; strata as in G2a (24 poses x 0.34-0.38 m, 16 poses x 0.38-0.45 m; y in [0.07, 0.13]; yaw in [-0.20, 0.20]); rejection if within Chebyshev distance 0.003 of ANY previously used initial state (all `episodes.jsonl` poses, WM datasets, the independent manifest, the G1 poses, the 48 G2a poses) or of another development / reserve pose; no outcome or model is consulted; the list is written once, its sha256 recorded, and then fixed.
**Sealed reserve (40 poses):** generated in a separate step that needs its own authorization, in a separate namespace, with the same exclusion rules checked INSIDE the generator (it outputs only a boolean and the minimum distance, never the poses); file mode 0400 under `outputs/g2b_reserve/`, never opened by analysis code; only `sha256(list || salt)` is committed in the freeze record; the reserve is used only after the development protocol has been frozen, run and reported, for an independent confirmation under a separately approved protocol; any access is logged; it is never used for tuning. Protected ACT seeds 31-33 stay unused.
