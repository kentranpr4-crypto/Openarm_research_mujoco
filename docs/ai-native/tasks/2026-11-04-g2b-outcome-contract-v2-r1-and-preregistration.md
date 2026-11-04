# G2b — Outcome Contract v2 (r1) and pre-registration, REVISION 1 — 2026-11-04
Status: **DESIGN ONLY — NOT FROZEN — NOT EXECUTED.** Supersedes the committed baseline `2026-11-03-g2b-outcome-contract-v2-and-preregistration-design.md` (commit `9d6bcce84d220aa7554765a92f03994de555b80d`). No simulator rollout, training or dataset generation was done; the only new computations are read-only on saved data (force-law check on G1 results, G2a cost accounting, equivalence-test power simulation on synthetic data). G2a freeze `8b6fdf89…a558` and all G2a/G1/Stage-1 results are untouched. Tự kiểm.

## 1. Outcome Contract v2 — revised

### 1.1 Three layers (kept separate)
**Layer A — event timeline** (timestamps in control steps since episode start, Δt = 0.05 s; row t = observation before action t; an action executed at step t first affects row t+1). Each event has a timestamp `t_*`, the decisive margin and ambiguity flags.
**Layer B — one terminal outcome per branch** (mutually exclusive).
**Layer C — binary evaluation endpoints** E1, E2, E3 (each TRUE / FALSE / CENSORED / INVALID, plus an independent `ambiguous` flag).

### 1.2 Signals, units, frames, geometry
| Signal | Definition | Unit | Logged today? |
|---|---|---|---|
| b(t) | box-centre position, MuJoCo world frame (z up) | m | yes |
| g(t) | executed left-gripper target (absolute, after clip→slew; v1 finger slide joint, ctrlrange −0.005…0.044) | m | yes |
| n_L(t), n_R(t) | number of MuJoCo contact **points** between the box and the left / right finger collision mesh | count | **only the sum n_f** |
| F_L(t), F_R(t) | Σ over those contact points of the normal-force magnitude |f_n| from `mj_contactForce` (normal component in the contact frame) | N | **only the sum F_f** |
| F_h(t) | Σ |f_n| over hand-mesh–box contact points (palm) | N | yes |
| p(t) | TCP position (world) | m | yes |
| v(t) | box speed, central difference of b | m/s | derived |
World/target frame: tray centre (0.44, 0.30) m; tray top z = 0.364 m; resting box-centre height z_rest = 0.384 m (box 0.05×0.04×0.04 m, mass 0.05 kg); rise(t) = b_z(t) − b_z(0); d_xy(t) = ‖b_xy(t) − tray centre‖; release region = horizontal disc R_REL = 0.05 m around the tray centre (inside the 0.14×0.14 m tray interior); placement region = disc PLACE_XY = 0.03 m.
**Force/contact semantics verified (G1 data, read-only):** the summed finger force is the servo force of the equality-coupled gripper: median F_f = 70.0 / 47.9 / 25.9 N at closing targets 0.008 / 0.012 / 0.016 m, slope −5507 N/m (actuator kp = 5664 N/m, 2.8 % lower) with contact position q_c ≈ 0.0207 m, i.e. **F_f(total) ≈ kp·(q_c − g)** and each finger carries ≈ F_f/2 (finger joint 2 is tied to joint 1). n_f counts contact **points**: 8 in 631/648 G1 holds, 7 in 17 — it is not a finger count, and a total n_f ≥ 2 can be met by one finger. Weight of the box 0.49 N, friction ≈ 1.0 ⇒ ≥ 0.25 N normal force per finger suffices to hold, so the thresholds below are ≥ 10× the physical minimum. **Consequence: G2b must log n_L, n_R, F_L, F_R** (new logging code = blocker B1); the definitions below use per-finger quantities; the saved-data feasibility run could only use the sums.

### 1.3 Event definitions (exact)
Notation: first_run(cond, from, K) = first t ≥ from with cond true on K consecutive rows.
- `close_command` t_cc = first t with g(t) < 0.040.
- `grasp_acquired` t_ga = first_run(n_L ≥ 1 ∧ n_R ≥ 1 ∧ F_L ≥ 5 N ∧ F_R ≥ 5 N ∧ g < 0.030, from t_cc, K = 5). Not reached within 25 steps of t_cc ⇒ NOT_ACQUIRED.
- `lift_confirmed` t_lc = first_run(rise ≥ R_LIFT = 0.03 m, from t_ga, K = 3) with t_lc ≤ t_ga + 40, else NOT_LIFTED (once the 40 steps have elapsed observed). Sensitivity: R_LIFT = 0.05 m (legacy).
- `held(t)` = n_L ≥ 1 ∧ n_R ≥ 1 ∧ F_L ≥ 2.5 N ∧ F_R ≥ 2.5 N.
- `release_onset` t_ro = first t ≥ t_ga with g(t) − min_{s ∈ [t_ga, t]} g(s) ≥ 0.004 m for 2 consecutive rows (opening ONSET, not an absolute crossing).
- `grasp_lost` t_gl = first_run(¬held ∧ g < 0.030, from t_ga, K = 3) **with t_gl < t_ro** (loss while still commanded closed; an opening gripper loses contact legitimately). Types by timing: before t_lc ⇒ ACQUISITION_LOST; t_gl − t_lc ≤ 15 ⇒ early slip; else carry loss. Flag `over_tray` if d_xy(t_gl) < R_REL.
- `intended_release` iff t_ro exists, no earlier grasp_lost, d_xy(t_ro) < R_REL and b_z(t_ro) − z_rest ≤ Z_REL = 0.05 m (one box height). Release with d_xy ≥ R_REL ⇒ `release_outside_target` (subtype outside); with b_z − z_rest > Z_REL over the target ⇒ release_outside_target (subtype too_high).
- `placement_settled` t_ps = first_run(d_xy < 0.03 ∧ |b_z − z_rest| < 0.006 ∧ v < 0.02, from t_ro, K = 10) with t_ps + 9 ≤ t_ro + 40; else PLACEMENT_FAILURE once 40 steps after t_ro were observed.
- `PREMATURE_DROP` (umbrella event) = grasp_lost (after acquisition) ∪ release_outside_target; it is never relabelled as success, and arrival at the tray while `held` is not a drop.

### 1.4 Layer B — terminal outcome (priority order)
1. `INVALID` (non-finite signal, IK/legality/penetration gate failed, snapshot test failed) — excluded, counted.
2. `ACQUISITION_FAILURE` with subtype NOT_CLOSED (no close command through the observed horizon), NOT_ACQUIRED, ACQUISITION_LOST, NOT_LIFTED.
3. `CARRY_DROP` (grasp_lost after t_lc, early or late).
4. `RELEASE_OUTSIDE_TARGET` (outside / too high).
5. `PLACEMENT_FAILURE` (intended release, not settled).
6. `PLACEMENT_SUCCESS`.
7. `TIMEOUT` = trajectory ended (T_MAX = 160 steps or early termination) before any of 2–6 was decided.

### 1.5 Layer C — endpoints and retention window
- **E1 GRASP_ACQUIRED**: TRUE if t_ga exists; FALSE if NOT_CLOSED/NOT_ACQUIRED decided; CENSORED otherwise.
- **E2 GRASP_RETAINED (primary):** window W = [t_lc, t_end_ret] with **t_end_ret = min(t_ro_intended, t_lc + 60)** (steps). TRUE iff E1 ∧ lift_confirmed ∧ no grasp_lost and no release_outside_target inside W. **If the intended release occurs before 60 steps, the window ends at the release** (retention is judged up to the intended end of the carry; a release before t_lc + 60 that is NOT intended ⇒ FALSE). FALSE also for any acquisition failure (composite acquisition-and-retention). CENSORED iff the observation ends before t_end_ret with no loss. A window shorter than 20 steps is flagged `short_window`.
- **E3 PLACEMENT_SUCCESS (secondary):** TRUE iff terminal PLACEMENT_SUCCESS; FALSE for decided failures; CENSORED for TIMEOUT.

### 1.6 Horizon, censoring, expiry
T_MAX = 160 control steps (the latest settle in 420 saved ACT/expert episodes was step 126; lift 65–79; detected losses 12–39 steps after lift; retention window 60 > max 39). Branches end early when the terminal outcome is decided (placement_settled + K_SETTLE, loss + K_LOSS, budget expiry). Horizon expiry with an undecided state ⇒ terminal TIMEOUT and CENSORED endpoints; CENSORED is never converted to success or failure in the primary analysis. Event times are relative; a uniform time shift changes no label.

### 1.7 Borderline / ambiguous handling
Every decisive threshold stores a signed margin; a branch carries `ambiguous` flags when any decisive margin is inside the band: lift |rise − R_LIFT| < 5 mm; acquisition min(F_L,F_R) within 20 % of 5 N; loss force within 20 % of 2.5 N or a 1–2-row `held` flicker next to the decision; release onset |Δg − 4 mm| < 1 mm; release |d_xy − R_REL| or |z − Z_REL| < 5 mm; placement |d_xy − 3 cm| < 3 mm, |z − z_rest| within 1.5 mm of 6 mm, v within 20 % of 0.02. **The label is kept as computed**; the flag is stored. Analyses: primary = all branches with their computed labels; sensitivity (a) exclude flagged branches, (b) adverse and (c) favourable assignment of flagged branches (§3.3). No silent conversion.

### 1.8 Provenance of the thresholds (development-derived decisions)
| Iteration | Change | Data viewed | Disagreement with legacy success label (ACT 360 / Stage-1 120) |
|---|---|---|---|
| draft 1 | physically motivated thresholds; release = absolute g ≥ 0.030 m; Z_REL 0.03 m | none before the run | 36 / 13 |
| draft 1 → 2 | release **onset** (+4 mm over the running minimum); separate acquisition-lost event | aggregate confusion tables | 15 / 5 |
| draft 2 → 3 | Z_REL 0.03 → 0.05 m (one box height) after a trace showed an ACT release at 0.031 m above the tray base; borderline band [0.03, 0.05) | aggregate tables + traces of 2 ACT episodes (c0_s2 poses 3 and 6) | 10 / 2 |
All three changes were made **after seeing the 480 saved development episodes**. Legacy labels are not ground truth (e.g. a 4 cm drop onto the tray is a legacy 'success'). Therefore **no independent-validation claim is made on those traces**; the saved-data run only demonstrates implementability. This r1 additionally changes the contact definition to per-finger quantities (not yet testable on saved data) — a further design change, not validated.

## 2. G2b candidates and exact branch count
Candidates **7 including the nominal** (no other candidate exists): c0 (unmodified ACT); **oracle-assisted:** O1 (depth-only cancel), O3 (depth+lateral+yaw cancel); **deployable (no oracle input):** D1 (2 cm shallower), D2 (close delayed 8 steps), H1 (2 cm deeper, harm control), S0 (pure 8-step time-shift null). 40 development poses × 7 = **280 branches maximum** (fewer if poses are NOT_APPLICABLE: the gripper already closing at t=40 or no close inside the chunk); applicability is decided from c0 and applies to all arms.

## 3. Statistical protocol (pairing preserved)
### 3.1 Estimand and pairing
Unit = pose. Every comparison is a within-pose paired risk difference RD(X−c0) = mean over poses of (E2_X − E2_c0) on identical simulator state; resampling is over poses with all 7 arms kept together (10 000 resamples, seed 0, percentile intervals). Counts of improved / harmed / unchanged poses are reported.
### 3.2 One invalid/censoring policy for all arms
1. A pose is dropped from **all** comparisons if c0 is INVALID, CENSORED or NOT_APPLICABLE (arm-independent rule).
2. For comparison X−c0 the **primary population = complete-case**: poses where c0 and X are both valid and non-censored (attrition per arm reported).
3. Sensitivity (pre-specified, all reported): **worst-case for X** (X invalid/censored counted FALSE, c0 unchanged), **best-case for X** (counted TRUE); all-arms-complete subpopulation (poses valid in all 7 arms) to check selection bias; excluding `ambiguous` branches; adverse/favourable assignment of ambiguous branches; R_LIFT = 0.05 m.
4. Any arm with validity < 80 % is excluded from claims; overall validity < 80 % stops the study.
### 3.3 Claims, multiplicity, CI levels (fixed before freeze)
| Family | Comparison | Role | Interval | Decision |
|---|---|---|---|---|
| **Oracle** (mechanism, not deployable) | O1 − c0 (primary), O3 − c0 (secondary) | E2 | 97.5 % CI (Bonferroni, 2 primary tests overall) for O1; 95 % for O3 | SUPPORTED_BETTER iff lower > 0 |
| **Deployable** | D1 − c0 (primary), D2 − c0 (secondary), H1 − c0 (harm control) | E2 | 97.5 % CI for D1; 95 % for the others | same |
| Secondary endpoints | E3, palm contact, time-to-events | — | 95 % CI, exploratory, no multiplicity correction | descriptive |
Oracle and deployable results are reported in separate tables; only D-family results can support a deployability statement (simulation only). A *claim of improvement* additionally requires the **worst-case-for-X point estimate ≥ 0** and the sign unchanged under R_LIFT = 0.05 m.
### 3.4 S0 equivalence (schedule-invariance validity gate)
Test (TOST, α = 0.05 per side): declare equivalence iff the **90 % pose-cluster percentile CI of RD(S0−c0) on E2 lies entirely inside (−0.10, +0.10)**; the 95 % interval is reported alongside. If equivalence is not shown, the outcome is **not** certified schedule-invariant: all grasp-quality claims are void and the results are reported descriptively. **Power warning (simulation of this exact test, true difference 0):** N = 40 poses: 56 % if 5 % of poses are discordant, 24 % at 10 %, 9 % at 15 %; N = 60: 82 / 45 / 23 %; margin ±0.15 with N = 40: 93 / 69 / 50 %. With the proposed margin and N = 40 the gate passes reliably only if S0 and c0 differ in ≲ 5 % of poses, so **inconclusive equivalence is a likely outcome** and would void claims. Decision needed: keep ±0.10 (accept the risk), widen to ±0.15 (needs justification: ±6 poses of 40), or increase N (budget).

## 4. CPU budget — recomputed
Measured G2a (432 branches of 60 steps with 2 ACT queries, IK on 8/9, bit-exact snapshots): total 517 s = setup 19.2 s + 10.37 s per pose (prefix ≈ 1 s + 9 branches ≈ 1.04 s). Cost model per branch ≈ 0.09 s (IK/restore) + 0.0158 s × steps, × 1.15 for per-finger/trace logging. Mean horizon ≈ 82 steps (≈ 70 % of branches run to placement ≈ 90 steps from t=40, the rest stop at a loss/budget ≈ 58 steps) ⇒ **1.6 s/branch expected, 2.3 s if every branch ran to T_MAX (120 steps)**; S0 adds 8 steps.
| Item | Expected (s) | Pessimistic (s) |
|---|---|---|
| integration tests (≈ 90 s per run × 2) | 180 | 180 |
| setup | 25 | 25 |
| prefixes 40 × 1.0 | 40 | 40 |
| **280 branches** × 1.6 / 2.3 | 448 | 638 |
| analysis, trace packaging, V2 plots | 25 | 25 |
| **Total** | **≈ 720 s (0.20 CPU-h)** | **≈ 910 s (0.25 CPU-h)** |
**Proposed hard cap: 0.30 CPU-h (1 080 s)** for everything (expected 67 %, pessimistic 84 % of the cap). Alternative within 0.25 CPU-h: 34 poses (238 branches; expected ≈ 645 s, pessimistic ≈ 810 s). The cap is enforced per pose (PARTIAL at a pose boundary; no extension); the pose count is fixed **before** the freeze from the measured test cost (> 2.3 s/branch ⇒ fewer poses).

## 5. Validation design (frozen before the run; labels never edited afterwards)
### 5.1 Ordering (enforced by the analysis code)
(1) V1 synthetic suite (below) must pass before the run. (2) Run G2b; per-branch traces and **machine labels are written and hashed**. (3) Blind annotation of a seeded sample (5.3). (4) Reliability report (κ, confusions, critical disagreements) written and hashed. (5) Only if the report is PASS does the treatment-effect analysis run (the script refuses without the PASS hash). The sealed reserve poses are never used.
### 5.2 Synthetic coverage (V1, automated, ≥ 20 randomised variants per cell)
success; tray arrival while held (the G2a artifact); carry drop; early slip; acquisition lost before lift; no lift (never acquires); borderline lift (rise 2.5 / 3.5 / 5.5 cm); release outside target; release too high; failed placement (released over the target, ends outside / moving); incomplete trajectory (cut mid-carry, mid-placement); one-finger contact only (per-finger rule); force-sensor flicker next to the loss threshold; uniform 8-step time shift of every case (labels must be identical). Acceptance: 100 % agreement on unambiguous cells; boundary cells must raise the matching ambiguity flag. Synthetic data prove the code implements the rubric, **not** that the rubric matches physics.
### 5.3 Blind annotation protocol
- **Annotator:** a person (or independent agent instance) who did not write the labeller and has not seen machine labels; written rubric version-hashed; annotator sees only plots of b_z, d_xy, g, F_L/F_R, n_L/n_R, box–TCP distance, F_h with anonymised IDs and randomised order, and chooses one terminal outcome + E1/E2/E3 + "cannot decide" (counted as a disagreement unless the branch is ambiguity-flagged).
- **Rubric (frozen text):** (i) acquired = both fingers in firm contact with the gripper closed; (ii) retained = the box stays gripped (contact on both fingers) from lift until the gripper starts to open or 60 steps after lift, whichever first; (iii) a box that sinks onto the tray while still gripped is NOT a drop; (iv) contact vanishing while the gripper target is still closed = grasp lost; (v) the gripper target rising over the tray, within one box height, = intended release; (vi) placement = the box rests inside the 3 cm region.
- **Sampling:** seeded (fixed before the run) stratified random draw from all G2b branches (all arms) using the machine label only for stratification: PLACEMENT_SUCCESS 12; CARRY_DROP 8; ACQUISITION_FAILURE 6; RELEASE_OUTSIDE_TARGET 4; PLACEMENT_FAILURE 6; TIMEOUT/INVALID 4; plus 4 ambiguity-flagged branches (take all available if fewer, then fill from PLACEMENT_SUCCESS); plus 10 interleaved synthetic gold traces (attention check, not in κ).
### 5.4 Metrics and acceptance
Cohen's κ on the terminal label (point estimate and 95 % bootstrap interval), the full confusion matrix, per-class precision/recall, binary agreement for E1/E2/E3. **Critical disagreements:** machine PLACEMENT_SUCCESS vs annotator drop/failure; machine CARRY_DROP/RELEASE_OUTSIDE_TARGET vs annotator success; E1/E2 disagreements. **PASS** iff κ ≥ 0.80, lower bound ≥ 0.65, E2 agreement ≥ 0.90, gold traces 100 %, and **0** critical disagreements among non-flagged branches (flagged ones reported separately). If the lower bound is < 0.65 with κ ≥ 0.80, one pre-registered extension of 20 additional branches is allowed; no other adaptation.
### 5.5 If reliability fails
Labels are void; no treatment-effect analysis is run or reported; the failed audit is reported; the contract must be revised, re-frozen and re-run on new development poses (the sealed reserve stays sealed); no claim from this run. No part of the saved development traces may be presented as independent validation.

## 6. Pose sets (unchanged from the baseline design)
40 development poses (24 x 0.34–0.38, 16 x 0.38–0.45; Chebyshev ≥ 0.003 from every earlier pose incl. G2a) and a **sealed reserve of 40 poses** (separate namespace, hashed, never executed or inspected, outside the exclusion check for development use until separately approved); ACT test seeds 31–33 unused.

## 7. Remaining blockers before protocol freeze
- B1 per-finger contact/force logging (n_L, n_R, F_L, F_R) and full per-branch trace storage: new code + tests.
- B2 final labeller code (per-finger definitions, margins/flags, endpoints, censoring) + V1 synthetic suite incl. time-shift cell; hash.
- B3 S0 implementation (8-step pause at the decision state) and its integration gate; re-run of the 12 G2a gates plus new gates.
- B4 decision on the S0 equivalence margin / N (power in §3.4) and on the cap (0.30 vs 0.25 CPU-h with 34 poses).
- B5 named independent annotator and acceptance of the rubric; sealed-reserve pose generation and hashing (file creation approval).
- B6 measured per-branch cost from the tests to fix the pose count.
- B7 written approval of the full contract (thresholds, provenance) and of this statistical plan.

## 8. Recommendation
**NO-GO for executing G2b now.** **Conditional GO** for the next design/implementation step (B1–B3 code and tests, still without outcome collection) once B4, B5 and B7 are decided. Nothing in G2a changes: grasp-quality conclusion NOT ESTABLISHED.
