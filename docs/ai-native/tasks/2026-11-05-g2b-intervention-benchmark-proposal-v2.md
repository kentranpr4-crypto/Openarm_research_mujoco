# G2b intervention CPU benchmark — REVISED PROPOSAL v2 (one attempt AUTHORIZED 2026-11-05 with a fresh hard cap of 90 CPU-s; the first 108 s attempt stays charged as consumed) — 2026-11-05
Why a v2: attempt 1 was killed by an operator-added wall-clock `timeout 110` during start-up; no measurement exists, the CPU actually used is unknown, and the 108 CPU-s authorization is treated as fully consumed (not a claim that 108 s were used). Attempt 1 report: `outputs/g2b_integration/intervention_benchmark_run1_ABORTED.json` (kept).

## 1. Harness (implemented, mock-tested only: `eval/g2b_intervention_benchmark.py`, tests `eval/test_g2b_benchmark_accounting.py`)
Launcher (stdlib only) -> child. The launcher polls `/proc/<pid>/stat` (all threads + reaped children) every 0.2 s and SIGKILLs the child at the cap; the authoritative figure is `getrusage(RUSAGE_CHILDREN)` + the launcher's own CPU. The child has a cooperative guard at 92 % of the cap (checked at milestones and every 20 steps). CPU and wall are reported separately; no wall-clock timeout exists. Atomic incremental reports (launcher report + `.child.json` with milestones: process start, numpy/torch/project imports, G2a freeze check, ACT checkpoint, MuJoCo env + camera/renderer (not separable without touching frozen code), IK, prefix, decision query, every branch start/end). Statuses: COMPLETE / PARTIAL (guard or kill) / STOPPED_ON_FAIL / BLOCKED (also when /proc is missing). No retries. Mock results: see the session report (11/11 PASS). Limitation: the hard kill overshoots by at most poll_s x busy threads; un-reaped grandchildren are only seen after they are reaped.

## 2. Scope (unchanged from the 2026-11-05 approval)
One previously used development pose (G2a idx 3000); snapshot at t = 40; horizon 160; branches c0, O3 (G2a c3), D1 (c5), D2 (c4), S0 (k = 8); frozen ACT checkpoint `c0_s0` and frozen G2a correction code (sha256 checked against the G2a freeze, read-only); trace recorder on; no protected seeds 31-33, no prospective 40 poses, no reserve poses. Reports only CPU components (setup/IK, ACT, mj_step, recorder, labelling), validity (legal clip->slew, finite, penetration <= 2 mm, IK failures = 0, label not INVALID) and action semantics (O3/D1 distinct from c0 by the G2a thresholds; D2 changes only the gripper channel; S0 holds 8 identical targets and resets the policy once; queries at 48, 68, 88, 108, 128, 148). No outcome comparison, no treatment effect, no scoring.

## 3. Expected cost and recommended fresh cap
| Component | Basis | Expected CPU-s |
|---|---|---|
| process + numpy/torch/project imports, checkpoint load, env + cameras, IK init | measured 16.7 s for torch + checkpoint + camera env in the integration run; G2a imports and Kin init not measured | 18-22 |
| prefix 0-40 with recorder | measured 1.0-1.3 s per episode | 1.2 |
| decision-state query | one ACT query, about 0.4 s | 0.4 |
| 5 branches x (restore, ACT, mj_step, recorder, label) | nominal branch measured 2.82 s (6 queries); S0 has 6 queries too | 14 |
| IK / correction for O3 and D1 | unmeasured (G2a: about 0.09 s per branch overhead; O3 does 20 IK solves per chunk) | 1-4 |
| Total expected | | about 36-42 |
**Recommended hard cap: 90 CPU-s (0.025 CPU-h)** = about 2.2x the expected total, because start-up behaved unexpectedly in attempt 1 (more than 110 s of wall time without reaching the first milestone) and the true start-up CPU is the main unknown. Cooperative guard at 92 % = 82.8 s. **Start-up sub-limit:** if the milestone `ik_initialized` is not reached within 45 CPU-s, stop (PARTIAL) rather than spend the remainder.

## 4. Stop conditions (all end the run immediately, no retry, no cap increase)
cap or guard reached (PARTIAL); any branch invalid (illegal clip->slew, non-finite, penetration > 2 mm, IK failure, INVALID label) or action-semantics check failed (STOPPED_ON_FAIL); frozen G2a file hash mismatch; pose NOT_APPLICABLE; /proc missing (BLOCKED); any exception (BLOCKED); start-up sub-limit exceeded.

## 5. Permissions needed for the retry
Simulator + ACT rollouts under the new cap (CPU only, no GPU, no training); read-only import of the frozen G2a correction code; writing `outputs/g2b_integration/intervention_benchmark_v2.json` (+ `.child.json`); run via the launcher only: `/home/user/.venvs/act/bin/python eval/g2b_intervention_benchmark.py --cap-s 90 --report outputs/g2b_integration/intervention_benchmark_v2.json`.

## 6. What the result would change
Only the cost model of the 280-branch budget: measured per-branch cost of IK-heavy (O3, D1) and non-IK (D2, S0) branches and the real start-up cost. It cannot support any statement about candidate quality.

## 7. Harness v3 additions required by the 2026-11-05 authorization (mock tested, 15/15 PASS)
Workload in its own session / process group; the launcher samples the WHOLE LIVE TREE (members found by parent links and group membership; each member contributes utime+stime of all threads plus the cutime+cstime of its already-reaped children, so nothing is counted twice); the final aggregate is max(getrusage(RUSAGE_CHILDREN) delta, running maximum of the sampled tree) + launcher CPU; the group is killed (killpg SIGKILL) at cap - 0.5 s (kill margin) and the launcher verifies that no member of the group remains; overshoot, kill-triggering sample and kill-to-exit time are reported; start-up sub-limit: 45 CPU-s before the milestone `ik_initialized` => PARTIAL; cooperative guard at 92 %; BLOCKED when the live tree cannot be read; wall time reported separately; no wall-clock timeout; no retry. Per branch: setup/IK, ACT, mj_step, recording + labelling, aggregate CPU, wall time, validity and execution status.
