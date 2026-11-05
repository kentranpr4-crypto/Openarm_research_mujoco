# G2b Outcome Contract r3 — CONSOLIDATED DRAFT (design only; not frozen; not approved; not executed) — 2026-11-05
Consolidates, without overwriting, the earlier documents: baseline design `9d6bcce`, r1 `a77f0fb`, r2 draft + I1-I4 `b838870`, X4 `413891a`. Code state described here = commit `413891a` (`eval/g2b_labeller.py`, contract string `v2-r2-x4-draft`, params hash `f112ab37…ea5`, spec hash `fe1a540e…627`) plus the uncommitted rubric draft (`eval/g2b_annotation.py`, RUBRIC sha256 `23a31723…6ba`). Where r3 PROPOSES something that the code does not do yet it is marked **[PROPOSED]**. No G2a/G1/Stage-1 frozen result is touched. Tự kiểm; nothing here is independent validation. Simulation only; privileged state; not Sim2Real.

## 0. Scientific status (explicit)
| Item | Status |
|---|---|
| G2b labeller | SYNTHETIC VALIDATION ONLY (plus I1-I4 self-tests on 3 known development poses; no independent validation) |
| Annotation rubric | SELF-CONSISTENCY TESTED (same author); no annotation run |
| Benchmark harness | MOCK TESTED |
| Intervention branch costs | NOT YET MEASURED |
| Independent blind annotation | NOT_CHECKED |
| Outcome Contract r3 | DRAFT, NOT FROZEN |
| G2b execution | NOT AUTHORIZED |

## 1. Scope and fixed design decisions
7 candidates (c0, O1, O3, D1, D2, H1, S0); 40 prospective development poses; 40 sealed reserve poses (not generated, not authorized); maximum 280 branches; horizon T_MAX = 160 steps (absolute); primary endpoint E2 GRASP_RETAINED; S0 equivalence margin +-0.10; two Bonferroni primary comparisons (O1-c0, D1-c0); proposed G2b CPU cap 0.30 CPU-h NOT AUTHORIZED. The G2a frozen outcome is INVALID for grasp-quality conclusions (it counted tray arrival as a drop); only its palm-contact finding is definition-independent.

## 2. Time indexing (X2) — five separate things
Row i = observation BEFORE the action of step t0+i (dt = 0.05 s). A command issued at row i first has a physical consequence in row i+1. Observation conjunctions use the command that produced the row, g_app[i] = g_exec[i-1] (row 0: meta.g_prev0, else flagged `g_prev0_assumed`).
| # | Kind | Event | Row index |
|---|---|---|---|
| 1 | command | release_command | issue row t_cmd (running-minimum rise of the commanded target >= 4 mm for 2 command rows) |
| 2 | physical | release_physical_onset | measured finger position `g_meas` >= 4 mm above its running minimum since t_ga for 2 rows (timeline only, never changes a label) |
| 3 | verified | release_confirmed_onset / release_confirmed | onset rc = first row of the confirmed state; verification rv = rc + K_REL - 1 (first row with all K_REL observations) |
| 4 | decision | e3_decision_row | first observed row from which E3 is fixed (section 5) |
| 5 | placement | placement_settled | first row of the first 10-row quiet window that starts at or after rv |
Ordering required for E3 TRUE: target_arrival <= release_command < release_physical_onset <= release_confirmed_onset < release_confirmed <= placement_settled < e3_decision_row (checked by invariant tests; the physical onset can be later than the confirmed onset only if the measured opening is jerky — not asserted).

## 3. Signals and per-finger contact definitions (B1)
n_L, n_R = number of MuJoCo contact POINTS between the box collision geom and the left / right finger collision mesh; F_L, F_R = sum over those points of |f_n| (normal component of `mj_contactForce`, contact frame, newtons; f_n >= 0; world force on geom2 = frame.T @ f[:3]); F_h = palm (hand mesh); box pose and velocity (free-joint qvel, instantaneous), TCP, `g_exec` (command), `g_meas` (measured finger joint). Required signals: g_exec, g_meas, n_L, n_R, F_L, F_R, box_pos (non-finite or missing g_meas => INVALID). Trace sampling: after the 50 substeps of a control step the recorder refreshes a scratch copy of the data with mj_forward so contacts, xpos and qpos/qvel belong to the same instant (end of step); the live data are never modified (verified bit-exact, I1/I2). `held` = n_L, n_R >= 1 and F_L, F_R >= F_HELD (2.5 N) ("clamped").

## 4. Event definitions (exact, with the frozen index rules)
`first_run(mask, lo, hi, k)` = first START s in [lo, hi] with mask[s:s+k] all true and s+k <= N. A decision that needs a run starting at the last allowed start is made only when rows through hi+k-1 are observed; otherwise TIMEOUT/CENSORED.
| Event | Definition |
|---|---|
| close_command | first row with g_exec < 0.040 (issue row) |
| grasp_acquired | first_run(n_L,n_R >= 1 and F_L,F_R >= 5 N and g_app < 0.030, from t_cc+1, start <= t_cc+25, K_ACQ = 5) |
| lift_confirmed | first_run(rise >= R_LIFT = 0.03 m, from t_ga, start <= t_ga+40, K_LIFT = 3); sensitivity R_LIFT = 0.05 |
| release_command | as section 2, computed on g_exec from t_ga |
| grasp_lost | first_run(not held and g_app < 0.030, from t_ga, K_LOSS = 3) counted only if its onset row <= t_release_command (a loss on the command row is NOT caused by that command; a loss one row later is its consequence) |
| intended release | release_command with d_xy < R_REL = 0.05 m and z - z_rest <= Z_REL = 0.05 m at the command row; else RELEASE_OUTSIDE_TARGET (OUTSIDE / TOO_HIGH) |
| OPENED_BEFORE_LIFT | release_command row < lift start row (ACQUISITION_FAILURE); a command on the lift row is not OPENED_BEFORE_LIFT |
| target_arrival | first_run(d_xy < R_REL and z - z_rest <= Z_REL, from t_lc, start <= t_release_command, K_ARR = 3) |
| release_confirmed_onset | first_run(not held and opened, from t_cmd+1, start <= t_cmd+CONFIRM_BUDGET = +40, K_REL = 3), opened = g_meas - running min since t_ga >= OPEN_CONFIRM = 8 mm |
| placement_settled | first_run(d_xy < 0.03 m and abs(z - z_rest) < 6 mm and speed < 0.02 m/s and not held, from rv, start <= rv+31, K_SETTLE = 10); speed = free-joint qvel (instantaneous); missing/NaN speed never counts and sets `velocity_unavailable`; the finite-difference fallback is one-sided at the ends |
Thresholds R_LIFT .. K_SETTLE are the r1 development-derived values (3 iterations on development data, so NOT independently validated); OPEN_CONFIRM, K_REL, K_ARR, CONFIRM_BUDGET come from geometry/convention and were fixed before the X4 tests.

## 5. Terminal outcome, endpoints, decision rows
Terminal priority: INVALID > ACQUISITION_FAILURE (NOT_CLOSED, NOT_ACQUIRED, ACQUISITION_LOST, OPENED_BEFORE_LIFT, NOT_LIFTED) > CARRY_DROP (EARLY_SLIP <= 15 rows after lift, LATE) > RELEASE_OUTSIDE_TARGET > PLACEMENT_FAILURE (NO_TARGET_ARRIVAL, RELEASE_NOT_CONFIRMED, NOT_SETTLED) > PLACEMENT_SUCCESS > TIMEOUT (undecided).
- **E1** TRUE iff grasp_acquired; FALSE if NOT_CLOSED/NOT_ACQUIRED decided; else CENSORED.
- **E2 (primary)** window = rows [t_lc, t_end) HALF-OPEN, t_end = min(t_lc + 60, t_release_command + 1); a loss onset or an unintended release inside it => FALSE; TRUE needs rows through t_end + K_LOSS - 2; otherwise CENSORED; any acquisition failure => FALSE. E2 judges retention up to the intended end of the carry (the COMMAND); a physical release failure afterwards does not change E2.
- **E3** TRUE iff PLACEMENT_SUCCESS; FALSE for decided failures; CENSORED iff TIMEOUT. **E2 TRUE with E3 FALSE is legitimate** (grasp retained, release physically failed: RELEASE_NOT_CONFIRMED).
- **e3_decision_row**: SUCCESS ts+9; NOT_SETTLED rv+40; RELEASE_NOT_CONFIRMED t_cmd+42; NO_TARGET_ARRIVAL t_cmd+2; None while undecided.
- **X1 horizon**: T_MAX = 160 is absolute. Uniform time-shift invariance is required only while the shifted trace stays fully observable inside the horizon; horizon-truncated traces are labelled TIMEOUT/CENSORED (NOT_CLOSED is FALSE only when the horizon was reached).
- **INVALID**: non-finite signal, missing g_meas, meta.valid false, IK/legality/penetration (> 2 mm) gate failed, snapshot test failed.

## 6. Release-confirmation boundary t_cmd+40 .. t_cmd+42 (verified, synthetic)
Rule: the confirmed state may START at most CONFIRM_BUDGET = 40 rows after the command (start-based, inclusive) and needs K_REL = 3 rows of observation. Exhaustive synthetic check (`eval/test_g2b_r3_boundary.py`: onset d = 30..50 x every truncation length 100..220; 0 violations):
| onset (rows after t_cmd) | full-trace label | decision row |
|---|---|---|
| +39 | PLACEMENT_SUCCESS (verified at +41) | settle + 9 |
| +40 | PLACEMENT_SUCCESS (verified at +42, needs rows through +42) | settle + 9 |
| +41 | PLACEMENT_FAILURE / RELEASE_NOT_CONFIRMED, E3 FALSE | t_cmd + 42 |
| +42 | PLACEMENT_FAILURE / RELEASE_NOT_CONFIRMED, E3 FALSE | t_cmd + 42 |
Invariants verified for every (onset, truncation): E3 TRUE iff SUCCESS; E3 FALSE only with a failure terminal; E3 CENSORED iff TIMEOUT; no release_confirmed with RELEASE_NOT_CONFIRMED; verification never beyond the observed rows and always onset + K_REL - 1; decision row never beyond the observed rows; placement_settled >= release_confirmed > release_command; decided E3 values never differ from the full-trace value; E1/E2 only move to CENSORED under truncation. A branch whose onset is +40 but whose trace ends before +42 is TIMEOUT (E3 CENSORED), not FALSE.
**Reviewer decision 2026-11-05:** CONFIRM_BUDGET +-1 is APPROVED as a PROPOSED ambiguity rule of r3 only; it is NOT implemented in the labeller, NOT frozen, and no data are re-labelled (primary CONFIRM_BUDGET stays 40; release-confirmation onset/verification semantics and E1/E2/E3 behaviour unchanged).
**Finding [PROPOSED fix]:** CONFIRM_BUDGET is not in the ambiguity bands, so an onset one or two rows beyond the budget (+41, +42) is a decided FALSE that is NOT flagged ambiguous, whereas +39/+40 are flagged by other perturbations. r3 proposes adding CONFIRM_BUDGET +-1 (and the other budgets) to the K-type perturbations; not implemented.

## 7. Ambiguity, censoring, invalid
Ambiguity = the endpoint (terminal) changes under FIXED perturbations: R_LIFT +-5 mm, F_ACQ +-1 N, F_HELD +-0.5 N, OPEN_RISE +-1 mm, R_REL +-5 mm, Z_REL +-5 mm, PLACE_XY +-3 mm, PLACE_Z +-1.5 mm, REST_V +-0.004 m/s, OPEN_CONFIRM +-2 mm and K_ACQ, K_LIFT, K_LOSS, K_OPEN, K_SETTLE, RET_WIN, K_REL, K_ARR +-1; values and `spec_hash` are written into every label; flags never change the primary label. Censored never becomes success or failure; invalid is counted and excluded per the policy of section 8.

## 8. Statistics (unchanged from r1/r2)
Unit = pose; paired risk differences on E2 with pose-cluster percentile bootstrap (10 000, seed 0) keeping all arms together; one invalid/censoring policy for all arms (drop a pose from all comparisons if c0 is invalid/censored/not applicable; complete-case primary; worst-case/best-case/all-arms-complete/ambiguity/R_LIFT sensitivity); oracle (O1, O3) and deployable (D1, D2, H1, S0) reported separately; primary O1-c0 and D1-c0 at 97.5 %; others 95 %; arms with validity < 80 % excluded. A claim of improvement needs SUPPORTED_BETTER and worst-case point estimate >= 0 and S0 equivalence. Power of the +-0.10 TOST at N = 40: 56 / 24 / 9 % at 5 / 10 / 15 % discordance — inconclusive equivalence is likely (decision open: keep, widen to +-0.15, or enlarge N).

## 9. S0 specification
At the decision state (t = 40) hold the last executed 16-D target for K_S0 = 8 control steps (each held command still passes clip -> slew), reset any stateful policy cache, then query ACT from the observation AFTER the pause (queries at 48, 68, 88, ...); later chunk boundaries shift by 8; the horizon stays absolute (160). With k = 0 the runner reproduces the nominal schedule bit-exactly (I3: executed commands and qpos identical). S0 may genuinely change the physics, so a failed equivalence does not by itself prove a labelling-timing bias; the labeller's own shift invariance is tested separately.

## 10. Blind annotation (design only; no annotation run; annotator UNASSIGNED)
Frozen rubric text = `eval/g2b_annotation.py` RUBRIC v2-X4 DRAFT (sha256 above; it distinguishes target arrival, commanded opening, physical opening, confirmed release and placement stability, and states that history, not the final box location, decides; a box resting on the tray while still clamped is not placed). Independence: the annotator did not write the labeller and never sees candidate IDs or machine labels; stratified seeded sample of 44 branches + 10 gold traces; kappa >= 0.80 with 95 % lower bound >= 0.65 (one 20-branch extension), E2 agreement >= 0.90, gold 100 %, zero critical disagreements on unambiguous branches; failure => labels void, revise/re-freeze/re-run on new development poses; the analysis refuses to run without a PASS report. Rubric-vs-labeller consistency was only checked synthetically (`eval/test_g2b_rubric_consistency.py`, same author).

## 11. Compute
Nominal branch at horizon 160 with trace: 2.82 CPU-s (3 known poses; 1.17 s prefix per episode). Intervention branches NOT measured: the first benchmark attempt was aborted by an operator wall-clock timeout and its 108 CPU-s authorization is treated as consumed. 40 x 7 projection ~0.26 CPU-h if intervention branches cost like nominal, at the 0.30 cap if they cost 20 % more. Hardened benchmark harness and proposal: `2026-11-05-g2b-intervention-benchmark-proposal-v2.md`.

## 12. Open decisions for the reviewer
1. Add CONFIRM_BUDGET (and the other budgets) to the ambiguity perturbations? 2. TOST margin / N. 3. Is a held-at-rest success before the physical opening really excluded (yes, by X4) — confirm no other reading. 4. Annotator identity. 5. Reserve-pose generation authorization. 6. CPU caps (benchmark and G2b).
