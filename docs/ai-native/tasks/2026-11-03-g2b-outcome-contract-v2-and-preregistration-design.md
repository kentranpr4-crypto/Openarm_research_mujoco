# G2a outcome-validity audit, Outcome Contract v2 (DRAFT) and G2b pre-registration proposal — 2026-11-03
Status: **DESIGN ONLY.** Nothing here was executed in a simulator; no ACT/WM inference; no G2a label, freeze or result was modified. Read-only analysis of saved data: `eval/outcome_v2_feasibility.py` → `outputs/g2b_design/outcome_v2_feasibility{,_iter1,_iter2}.json`. Tự kiểm.

## 0. Record and verdicts (G2a, preserved)
- G2a commit `c44fb3db81e2d52ce539685a3c0af665907419df`; freeze sha256 `8b6fdf893d7743506310a3044f01754ef5cea0d90a10c4cf9ae171f18011a558`; 48 poses, 432 branches, 12/12 gates PASS, 779.5 CPU-s; all failed test attempts and pre-freeze changes (horizon 120→100, penetration-based collision validity, tolerance and synthetic-test fixes) remain documented in the G2a report.
- **TECHNICAL VALIDITY: PASS** (bit-exact snapshots, bit-exact nominal reproduction, IK ≤0.5 mm/0.005 rad, legal chains).
- **FROZEN STOP-RULE VERDICT: PASS under the original, flawed outcome** (headroom 25 %, c1 − nominal +4.2 pp [−6.2, +16.7], INCONCLUSIVE).
- **GRASP-QUALITY CONCLUSION: INVALID / NOT ESTABLISHED.** Only the directly measured palm-contact effect (deeper +2 cm → 35 % vs 6 %; depth-only oracle correction → 0 %) is outcome-definition independent.

## 1. Outcome validity audit (saved data only)
| Evidence | Result |
|---|---|
| Schedule dependence of the frozen 'dropped' label (432 G2a branches) | dropped-label rate by lift time: t_lift 60–67: **80.6 %** (n=31); 68–69: 34.8 % (92); 70–71: 11.1 % (180); 72–79: 8.1 % (111). Per candidate dropped labels: c0 11, c1 13, c2 12, c3 8, c4 8, c5 3, c6 10, c7 15, c8 8. A delayed trajectory (c4) is rewarded for arriving after t=100. |
| Nominal 'dropped' labels re-executed (post-hoc diagnostic) | 11/11 were successful carries reaching the tray at t=94–100 (box 0.4–0.9 cm from the tray centre, still held one step earlier). |
| Prevalence of the artifact | 248/276 (89.9 %) of successful ACT episodes in the saved xsweep set have the box within 1 cm of the tray centre by step 100. |
| Marginal lift | the 4 nominal 'not_lifted' G2a states peaked at 5.37–5.67 cm vs the 5 cm threshold; across xsweep ACT episodes 19 peak in 5–6 cm, 1 in 3–5 cm. |
| Re-labelling the 432 G2a branches under v2 | **NOT_EVALUABLE**: G2a stored per-branch summaries only (no box/contact/gripper traces). |
Conclusion: the frozen outcome conflated 'reached the tray before the horizon' with 'dropped', and used a horizon (t=100) before the median ACT carry drop; its success/headroom numbers cannot be used for grasp-quality claims.

## 2. Outcome Contract v2 (DRAFT; to be hashed and frozen only after approval)
**Principle:** event-based labels measured relative to event times (never absolute time), from signals that every branch logs. Dropped events are **not** relabelled as success; arrival at the tray while the grasp is held is simply not a drop.
**Signals (per control step t, row = observation before action t; Δt = 0.05 s):** box centre b(t) [m, world, z up]; executed left-gripper target g(t) [m] (16-D absolute target, ctrlrange −0.005…0.044); finger–box contact points n_f(t) [count]; finger normal force F_f(t) [N]; palm force F_h(t) [N]; TCP p(t) [m]. Derived: rise(t) = b_z(t) − b_z(0) [m]; d_xy(t) = ‖b_xy − tray centre‖ [m]; box speed v(t) [m/s] (central difference).
**Geometry (lab_scene constants):** tray centre (0.44, 0.30) m; tray top z = 0.364 m; resting box-centre height z_rest = 0.384 m; box 0.05×0.04×0.04 m.
**Parameters (iteration 3 of the draft; physical meaning in brackets):**
| Name | Value | Meaning |
|---|---|---|
| G_CLOSE / G_ACQ | 0.040 / 0.030 m | close command / commanded closure for acquisition |
| NF_ACQ, F_ACQ, K_ACQ | ≥2 contacts, ≥10 N, 5 steps | grasp_acquired = 5 consecutive steps of firm two-sided contact with the gripper commanded closed |
| R_LIFT, K_LIFT, LIFT_BUDGET | 0.03 m, 3 steps, 40 steps | lifted: box ≥3 cm above its initial height (clears the 1.2 cm tray rim with margin) for 3 steps within 40 steps of acquisition; legacy 0.05 m kept as a sensitivity analysis |
| NF_HELD, F_HELD, K_LOSS | ≥2, ≥5 N, 3 steps | held(t); loss of grasp = not held for 3 consecutive steps while the gripper is still commanded closed |
| OPEN_RISE, K_OPEN | 4 mm, 2 steps | release command onset: executed target rises ≥4 mm above its running minimum for 2 steps |
| R_REL, Z_REL | 0.05 m, 0.05 m | intended release: onset while d_xy < 5 cm and box ≤5 cm (one box height) above z_rest |
| PLACE_XY, PLACE_Z, REST_V, K_SETTLE, SETTLE_BUDGET | 0.03 m, 0.006 m, 0.02 m/s, 10 steps, 40 steps | placement_success: within 40 steps of release, 10 consecutive steps in the target region at rest |
| RET_WIN | 60 steps | retention window after lift |
| borderline band | max rise in [0.03, 0.05) m | flag `borderline_lift` (sensitivity at R_LIFT = 0.05) |
**State machine:** APPROACH → CLOSING (g < G_CLOSE) → ACQUIRED → LIFTED → CARRY → (OVER_TRAY → LOWER) → RELEASE (onset) → SETTLED. Transitions out of the nominal path are the labels below.
**Events and terminal labels (mutually exclusive):**
- `not_closed` (no close command, or closed but never acquired within 25 steps);
- `grasp_acquired` (event);
- `not_lifted` (acquired, no lift within the budget);
- `premature_drop` with subtype: `acquisition_lost` (lost after acquisition, before confirmed lift, box was raised ≥1 cm), `early_slip` (lost ≤15 steps after lift), `carry_drop` (later loss, gripper still commanded closed), `release_outside_target` (release onset with d_xy ≥ 5 cm), `release_in_air` (onset >5 cm above z_rest). Flag `over_tray_region` when the loss happens within 5 cm of the tray centre;
- `intended_release` (event: onset over the target and ≤5 cm above z_rest) followed by `placement_success` or `placement_failure` (released over the target but not settled in the target region);
- `timeout` (unresolved when the trajectory ends: carry unresolved, release unresolved, or acquisition/lift budget not elapsed);
- `invalid` (non-finite data, failed snapshot/legality/penetration gates).
**Primary grasp endpoint:** `GRASP_RETAINED` = grasp_acquired ∧ lifted ∧ no loss of grasp before min(release onset, t_lift + 60). Unresolved (trajectory ends before that) → None (excluded, sensitivity reported). **Secondary:** `placement_success`, time-to-lift, time-to-release, palm contact (F_h > 0.01 N), subtype distribution.
**Near-threshold handling:** hysteresis via K_LIFT/K_LOSS/K_OPEN; every label stores the margin to its decisive threshold; events within ±5 mm / ±1 step of a threshold carry an `ambiguous_*` flag; the analysis reports the primary result with and without flagged branches and under R_LIFT ∈ {0.03, 0.05}; ambiguous branches are never hand-relabelled.
**Horizon and timing:** saved ACT/expert episodes (360+60): lift t = 65–79 (median 67), release median 107 (p95 112, max 122), settle p95 118 (max 126); detected loss occurs 12–39 steps after lift (median 25.5). → retention window 60 steps (>max 39); trajectory horizon T_MAX = 160 control steps (27 % beyond the latest observed settle); branches end early once the terminal label is resolved. Event-relative windows make the labels invariant to a uniform time shift of the whole trajectory; arrival times are reported, not penalised.
**Validity rules:** candidate invalid if IK residual > 2 mm / 0.03 rad at any corrected step, executed chain illegal/non-finite, robot–obstacle penetration > 2 mm, or the snapshot/restore test fails; invalid branches are excluded from that candidate's endpoints (pose-wise pairing drops the pose for that comparison) and counted.
**Immutability:** contract text + labeller source + parameter table → SHA256 recorded in the G2b freeze before any branch is run; labels are a pure function of stored traces so they can be recomputed and audited.
**Provenance (honest):** thresholds were physically motivated, then modified in **three development iterations after viewing the saved development episodes** (iter1→2: release onset defined by a 4 mm rise instead of an absolute 0.030 m crossing, and a separate acquisition-lost event; iter2→3: Z_REL 0.03→0.05 m after a trace showed ACT releasing at 0.031 m above the tray base, borderline band widened to [0.03, 0.05)). The saved episodes are therefore **not** a valid validation set for the final contract.

## 3. Label-validation test plan
Draft feasibility (read-only, saved development data, not blinded, thresholds iterated): synthetic scenarios 10/10 groups (50 traces: success, tray arrival while held, carry drop, early slip, no lift, incomplete trajectory, release outside target, release in air, borderline lift 3.5 cm → timeout/unresolved, 2.5 cm → not_lifted). Saved full episodes, iteration 3: xsweep ACT (360): legacy success 276 → v2 placement_success 267, premature_drop 9; legacy failure 84 → not_closed 13, not_lifted 3, placement_failure 53, premature_drop 14, placement_success 1 (a 3–5 cm lift counted 'no_pick' by the legacy 5 cm rule); expert 60/60 placement_success; Stage 1 (120): legacy success 84 → 82 placement_success, 2 premature_drop. Trace inspection of one disagreement (c0_s2 pose 3): the box fell ≈4 cm onto the tray in 2 steps with zero contact while the gripper was still commanded closed — a real loss of grasp that the legacy final-state label counts as success.
**Planned validation (to be run before/with G2b):**
1. **V1 synthetic, automated:** 10 scenario types × 20 randomised variants (timings, noise, threshold-adjacent perturbations); acceptance 100 % on unambiguous cases; boundary variants must raise the corresponding ambiguity flag.
2. **V2 blinded human/independent adjudication on fresh data:** after the G2b run but **before** any treatment comparison is computed, 40 random G2b branch traces (stratified by v2 label, hashed IDs, plots of b_z, d_xy, g, n_f, F_f, box–TCP distance, labels hidden) are labelled by a reviewer who has not seen the labeller output (the user or a separate agent instance given only the written rubric). Acceptance: Cohen's κ ≥ 0.80 and zero disagreements between `premature_drop` and `placement_success`/`intended_release` on clear cases; failure ⇒ the G2b labels are void (no claim) and the contract must be revised and revalidated on a new run.
3. **V3 historical data:** saved xsweep (all signals present) and Stage-1 (contact block present) episodes may be used for descriptive consistency only; **G2a branches: NOT_EVALUABLE** (no traces). Signal availability: xsweep replay ✓ all; Stage-1 trajectories ✓ (no gripper_q); G2a ✗.
4. **Schedule-invariance test inside G2b:** the pure time-shift control (S0) must be equivalent to nominal on `GRASP_RETAINED` (§4).

## 4. G2b pre-registration proposal (not executed)
**Question:** does a physically grounded change of the ACT's grasp geometry improve **grasp acquisition and retention** (primary) and placement (secondary), relative to the unchanged ACT from the same simulator state?
**Design:** one decision state per pose (start of the 20-step ACT chunk at t=40, applicable iff the gripper is still open and the chunk contains the first close command); every candidate branches from the same bit-exact snapshot (validated machinery); ACT is re-queried normally afterwards; each branch runs until its v2 terminal label is resolved or T_MAX = 160; full traces (b, p, g, n_f, F_f, F_h, F_hand) stored per branch (needed for labelling and audit). Corrections persist as the G2a joint-space offset.
**Poses:** 40 new development poses (24 'hard' x 0.34–0.38, 16 'mid' x 0.38–0.45), new RNG namespace, Chebyshev ≥0.003 from every earlier pose including the 48 G2a poses; pose list SHA256 in the freeze. **Sealed reserve:** a further 40 poses are generated in a separate namespace, hashed, **never executed or inspected**; any confirmatory run on them needs separate approval after G2b is analysed and its protocol frozen; no design decision may use them. Reserved ACT test seeds 31–33 untouched. After G2b the 40 poses are development-only.
**Candidates (7, incl. nominal) and controls:**
| ID | Definition | Oracle? | Role |
|---|---|---|---|
| c0 | unmodified ACT | — | nominal control |
| O1 | depth-only correction cancelling the ACT-commanded TCP depth error vs the expert grasp point (clipped to p5/p95 of the pooled audit error), early ramp | **oracle-assisted** | primary oracle comparison |
| O3 | depth + lateral + yaw cancel, early ramp | **oracle-assisted** | upper-bound geometry |
| D1 | fixed 2 cm shallower TCP at the grasp (no oracle input) | deployable | primary deployable comparison |
| D2 | gripper close delayed by 8 steps (arm unchanged) | deployable | timing-only effect |
| H1 | fixed 2 cm deeper TCP | deployable | harm control (palm mechanism) |
| S0 | **pure time-shift null**: hold the last executed command for 8 steps at the decision state, then proceed with nominal ACT (all later chunk boundaries shift by 8) | deployable | **schedule-invariance control** |
Oracle-assisted and deployable results are reported in separate tables; only D1/D2 can support a deployability statement, and only for simulation.
**Endpoints and statistics:** unit = pose; paired differences vs c0 (risk difference), improved/harmed/unchanged counts, 10 000 pose-cluster bootstrap resamples (seed 0), 95 % percentile CI; two primary comparisons (O1−c0 oracle, D1−c0 deployable) with a Bonferroni 97.5 % interval; secondary: O3, D2, H1, placement_success, palm contact, time-to-events; S0 equivalence margin ±0.10 on GRASP_RETAINED (the 95 % CI of S0−c0 must lie inside ±0.10). Expected power: with 40 poses and 20–30 % nominal failure, effects below ≈ +0.15 will usually remain INCONCLUSIVE — reported, not hidden.
**Invalid / unresolved handling (pre-specified):** invalid branches excluded for that candidate (pose-wise pairing), counted; candidate-level validity <80 % → that candidate is excluded from claims; overall validity <80 % → STOP; nominal invalid/unresolved → pose excluded from all comparisons; unresolved candidate branches excluded with a worst-case (count as failure) sensitivity; flagged ambiguous branches analysed with/without.
**Conditions for a VALID grasp-quality claim (all required):** (C1) V1 and V2 pass; (C2) integration gates pass (bit-exact snapshots/nominal, IK, legality, no leakage, trace storage, deterministic labelling); (C3) S0 equivalence holds, i.e. the outcome is schedule-invariant; (C4) validity ≥80 %; (C5) the primary comparison's CI lower bound > 0 (SUPPORTED_BETTER), with the sensitivity (R_LIFT 0.05, with/without ambiguous) not reversing the sign; (C6) mechanism consistency (palm contact decreases for O1/D1, increases for H1). Any claim is limited to simulation, development poses and the oracle/deployable status of the candidate; a confirmatory run on the sealed reserve is required before a stronger statement.
**Stop rules:** validity <80 %; critical gate failure; V2 failure; S0 equivalence failure (claims void, results reported descriptively); no nominal failures at all (NO_HEADROOM, no proxy added afterwards); CPU cap (PARTIAL at a pose boundary, no extension).
**No tuning on future outcomes:** contract, candidates, ranges, poses, statistics and thresholds are frozen with a SHA256 before the first branch; the only data-dependent step is the pre-registered blind label audit, which can void but not modify labels.

## 5. Compute budget (estimate; cap 0.25 process CPU-h = 900 s for everything)
| Item | CPU-s (est.) |
|---|---|
| integration tests (G2a gates + labeller/trace-storage gates + schedule-shift gate), up to 2 runs | 150 |
| freeze + label-validation tooling (plots for V2, scenario generator) | 20 |
| run: setup 25 + 40 poses × (prefix 1 + 7 branches × 1.6) | 513 (630 if branches cost 2.1 s) |
| analysis + V2 packaging | 15 |
| **total** | **≈ 700 (≈ 815 worst case)**; reserve ≈ 85–200 s |
The 1.6 s per branch is an extrapolation of the measured G2a cost (1.03 s for 60 steps/2 queries) to ≈ 90 steps/4 queries with early termination; if the measured cost in the tests exceeds 2.3 s per branch the pose count is reduced **before** the freeze (pre-registered rule), never after.

## 6. What must be approved before any execution
1. Outcome Contract v2 text, parameter table and labeller code (hash to be recorded); acknowledgement of the iteration provenance in §2.
2. The V2 blind-review arrangement (who labels the 40 traces; rubric) and its κ ≥ 0.80 acceptance rule.
3. The 7 candidate definitions, oracle/deployable split, S0 equivalence margin ±0.10, primary comparisons and decision rules.
4. 40 development poses (hash) and the sealed 40-pose reserve (hash, no execution).
5. CPU cap 0.25 CPU-h including tests/analysis, the early-termination/horizon (T_MAX = 160) and the pose-reduction rule.
6. Commit/push of the design, contract, labeller and feasibility artifacts (currently uncommitted).
7. Confirmation that the G2a frozen outcomes remain labelled “invalid for grasp-quality claims” and that no GF-WM dataset/training follows from G2a.
