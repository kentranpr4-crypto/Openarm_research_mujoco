# G2b Outcome Contract r3 — FREEZE CANDIDATE (NOT FROZEN; not approved; design + synthetic tests only) — 2026-11-05
Consolidates, without overwriting, the earlier documents: baseline design `9d6bcce`, r1 `a77f0fb`, r2 draft + I1-I4 `b838870`, X4 `413891a`, r3 draft + harness `98a7251`, integration harness `ac0eabe`. Historical frozen reports (G2a `8b6fdf89…`, G1 `309d0edf…`, branching `f2249bd3…`, Stage-1 `8fd93a31…`) are immutable and untouched. Tự kiểm; simulation only; privileged state (not Sim2Real); no outcome of any G2b treatment exists.

## 0. Version hashes (as of this document)
| Item | Value |
|---|---|
| Contract string | `v2-r2-x4-draft` |
| Labeller PARAMS hash | `f112ab37b2ce73afd3051242d8319093a5451136d7fcb014686ee0691ad49ea5` |
| Ambiguity spec hash (includes CONFIRM_BUDGET +-1, uncommitted) | `5040776731dcecad75a4b0f808697b8ff658cfd41d32ca77cc731471c0cfe2e5` |
| Annotation rubric sha256 (v2-X4 draft) | `23a31723f7a10e0ecd7eb5929f0597246622315d0e82d8739e8b327c1d28e6ba` |
| Git HEAD of the committed state | `ac0eabea237b09ea23892b03d5cb47dad8aeccac` |
Working-tree file hashes (the labeller differs from the committed one by the CONFIRM_BUDGET line only):
| File | sha256 |
|---|---|
| `eval/g2b_labeller.py` | `d30e99d2007d24881f10501bce8c74caea40f0e2c23da8fa9b309fc0e650d975` |
| `eval/g2b_trace.py` | `16426d995fbaf74d24444b53aa4873fac387fed8fb011c5f225d62e14b5d7df8` |
| `eval/g2b_s0.py` | `8c823f7a69abd82a8e40d07581b870e58fa44d0f2b8dcf07b892bf817173e76c` |
| `eval/g2b_stats.py` | `9165c72655ca9c7d243b31110d73966bac7ce1609bc62a3a12afcc1b186c145e` |
| `eval/g2b_annotation.py` | `59b67e2dc3d0b3dbb3316555c1d2f2bf0eb2a326deeb9de86125faa2108f6187` |
| `eval/g2b_integration_v2.py` | `7cdf51dd24a679ecd80682b116ab76ce2268e93ca976c86422ed22ee33392a8b` |
| `eval/g2b_intervention_benchmark.py` | `2e0e14a7997983ce3fb86109c6343ae7497278b7dc2f81e0714f2b3bde87dbdc` |
At freeze time these hashes, the checkpoint hashes (`outputs/checkpoints/e6b_c0_seed0/last/*`), the G2a freeze hash and the pose-list hash must be recorded in a protocol-freeze file.

## 1. Fixed design decisions
7 candidates (c0, O1, O3, D1, D2, H1, S0); 40 planned development poses (no reserve pose generated); max 280 branches; absolute horizon T_MAX = 160 control steps; primary endpoint E2 GRASP_RETAINED; secondary E1, E3; two primary comparisons O1-c0 and D1-c0; S0 equivalence margin +-0.10; proposed G2b CPU cap 1 080 CPU-s NOT AUTHORIZED. G2a's frozen outcome is INVALID for grasp-quality claims (it counted tray arrival as a drop); only its palm-contact finding is definition-independent.

## 2. Time-indexing convention (X2)
Row i = observation BEFORE the action of step t0+i (dt = 0.05 s). A command issued at row i has its first physical consequence in row i+1. Command events carry the ISSUE row; observation conjunctions use g_app[i] = g_exec[i-1] (row 0: meta.g_prev0). A rollout of T actions has T+1 observation rows in EVERY trace field (the final observation is added by `record_final`, which repeats the last executed command and fabricates no measurement). Verified offline on the saved real I1 trace (24/24 alignment checks) and by re-simulation from scratch copies (29 checks; a[t-1] instead of a[t] changed s[t+1] in 23/23 command-change checks).
| Kind | Event | Row index |
|---|---|---|
| command | close_command | issue row of the first g_exec < 0.040 (threshold crossing of the COMMAND; the first commanded departure from open is earlier) |
| command | release_command | issue row (running-minimum rise of the commanded gripper >= 4 mm for 2 command rows) |
| physical | release_physical_onset | measured finger >= 4 mm above its running minimum for 2 rows (timeline only) |
| onset | release_confirmed_onset | first row of the confirmed state (not clamped AND opened >= 8 mm for K_REL = 3 rows) |
| verification | release_confirmed | onset + K_REL - 1 (first row with all required observations; never earlier) |
| window onset | placement_settled | FIRST row of the first 10-row quiet window that starts at or after the verification row |
| decision | e3_decision_row | settled + 9 (SUCCESS); verification + 40 (NOT_SETTLED); t_cmd + 42 (RELEASE_NOT_CONFIRMED); t_cmd + 2 (NO_TARGET_ARRIVAL) |

## 3. Signals, contact and force semantics (B1)
Per-finger contact points n_L, n_R between the box collision geom and each finger collision mesh; F_L, F_R = sum of |f_n| (normal component of `mj_contactForce`, contact frame, newtons, f_n >= 0; world force on geom2 = frame.T @ f[:3]; verified by box-weight balance 0.488-0.490 N vs 0.4905 N and finger action-reaction within 0.4 %); `held` ("clamped") = n_L, n_R >= 1 and F_L, F_R >= 2.5 N. Contacts are counted as POINTS (about 4 per finger), never as fingers. Trace sampling: after the 50 substeps of a control step `xpos` and contacts belong to the start of the last substep and qpos/qvel to its end (1 ms); the recorder refreshes a scratch copy with `mj_forward` so every logged quantity is end-of-step, and the live data (including solver warm-start) is never touched (verified: raw xpos minus qpos equals v x 1 ms; refreshed minus qpos = 0.0). Box speed = free-joint qvel (instantaneous; missing/NaN never counts as at-rest; finite-difference fallback is one-sided at the ends).

## 4. Event definitions, indexing rules and parameters
`first_run(mask, lo, hi, k)` = first START s in [lo, hi] with mask[s:s+k] all true and s+k <= N; a decision that needs a run starting at the last allowed start is made only when rows through hi+k-1 are observed, otherwise TIMEOUT / CENSORED.
| Event | Definition |
|---|---|
| grasp_acquired | first_run(n_L,n_R >= 1 and F_L,F_R >= 5 N and g_app < 0.030, from t_cc+1, start <= t_cc+25, K_ACQ = 5) |
| lift_confirmed | first_run(rise >= 0.03 m, from t_ga, start <= t_ga+40, K_LIFT = 3); sensitivity R_LIFT = 0.05 |
| grasp_lost | first_run(not held and g_app < 0.030, from t_ga, K_LOSS = 3) counted only if its onset row <= t_release_command |
| intended release | release_command with d_xy < 0.05 m and z - z_rest <= 0.05 m; else RELEASE_OUTSIDE_TARGET (OUTSIDE / TOO_HIGH) |
| OPENED_BEFORE_LIFT | release_command row < lift start row (ACQUISITION_FAILURE) |
| target_arrival | first_run(d_xy < 0.05 and z - z_rest <= 0.05, from t_lc, start <= t_release_command, K_ARR = 3) |
| release_confirmed_onset | first_run(not held and opened, from t_cmd+1, start <= t_cmd + CONFIRM_BUDGET (= 40), K_REL = 3); opened = g_meas - running min since t_ga >= 8 mm |
| placement_settled | first_run(d_xy < 0.03 and abs(z - z_rest) < 6 mm and speed < 0.02 m/s and not held, from verification, start <= verification+31, K_SETTLE = 10) |

## 5. Terminal outcomes, endpoints, ambiguity, censoring, invalid
Terminal priority: INVALID > ACQUISITION_FAILURE (NOT_CLOSED, NOT_ACQUIRED, ACQUISITION_LOST, OPENED_BEFORE_LIFT, NOT_LIFTED) > CARRY_DROP (EARLY_SLIP <= 15 rows after lift, LATE) > RELEASE_OUTSIDE_TARGET > PLACEMENT_FAILURE (NO_TARGET_ARRIVAL, RELEASE_NOT_CONFIRMED, NOT_SETTLED) > PLACEMENT_SUCCESS > TIMEOUT (undecided).
- **E1** TRUE iff grasp_acquired; FALSE if NOT_CLOSED / NOT_ACQUIRED decided; else CENSORED. **E2 (primary)**: retention window = rows [t_lc, t_end) half-open, t_end = min(t_lc+60, t_release_command+1); a loss onset or an unintended release inside => FALSE; TRUE needs rows through t_end + K_LOSS - 2; otherwise CENSORED; any acquisition failure => FALSE. **E3** TRUE iff PLACEMENT_SUCCESS (needs a verified physical release); FALSE for decided failures; CENSORED iff TIMEOUT. E2 TRUE with E3 FALSE (jammed gripper) is legitimate. E3 is NOT monotone relative to the pre-X4 labels.
- **INVALID**: non-finite signal, missing g_meas, meta.valid false, IK / legality / penetration (> 2 mm) gate, snapshot test failed. **CENSORED/TIMEOUT are never converted to success or failure.** A uniform time shift keeps every label only while the shifted trace stays fully observable inside T_MAX; horizon-truncated cases are labelled TIMEOUT / CENSORED.
- **Ambiguity** = an endpoint (or the terminal) changes under FIXED perturbations: R_LIFT +-5 mm, F_ACQ +-1 N, F_HELD +-0.5 N, OPEN_RISE +-1 mm, R_REL +-5 mm, Z_REL +-5 mm, PLACE_XY +-3 mm, PLACE_Z +-1.5 mm, REST_V +-0.004 m/s, OPEN_CONFIRM +-2 mm and +-1 step on K_ACQ, K_LIFT, K_LOSS, K_OPEN, K_SETTLE, RET_WIN, K_REL, K_ARR and **CONFIRM_BUDGET (implemented in the working tree, NOT committed)**. Flags never alter the primary label (tested: 0 primary-label differences on 102 synthetic cases and on the saved real I1 trace). Boundary result: confirmed-state onset at t_cmd+39 / +40 / +41 / +42: SUCCESS, SUCCESS, FAILURE (RELEASE_NOT_CONFIRMED), FAILURE; E3 flagged ambiguous at +39, +40, +41 and not at +42 (the budget perturbation flags +40 and +41; +39 is flagged by other perturbations). No contradictory label exists for any onset 30-50 and any truncation length (0 violations).

## 6. Threshold provenance and development-data exposure
| Parameter group | Source | Exposure |
|---|---|---|
| G_CLOSE, G_ACQ, F_ACQ, K_ACQ, ACQ_BUDGET, R_LIFT, K_LIFT, LIFT_BUDGET, F_HELD, K_LOSS, OPEN_RISE, K_OPEN, RET_WIN, EARLY_SLIP, R_REL, Z_REL, PLACE_XY, PLACE_Z, REST_V, K_SETTLE, SETTLE_BUDGET | r1: physically motivated, then changed in 3 iterations after seeing 480 saved development episodes (release onset, Z_REL 0.03 -> 0.05, acquisition-lost event) | exposed to development outcomes; NOT independently validated |
| OPEN_CONFIRM 8 mm, K_REL, K_ARR, CONFIRM_BUDGET | X4: geometry and the existing K / budget conventions, fixed before the X4 tests | the author had seen the 3-5 row command-to-physical lag of poses 3000-3002 before writing X4; no value was tuned or changed afterwards |
| Real-data margins (saved I1 trace, pose 3000; signed margins, positive = inside the satisfied side) | acquisition force +1.49 N over 5 N; lift rise +12.0 mm over 30 mm; release onset +2.9 mm over 4 mm; release dxy +49.2 mm and height +49.9 mm (box released essentially at the tray centre and rest height); release_open +4.2 mm over 8 mm; place_dxy +29.3 mm, place_z +5.94 mm, place_v +0.01999 m/s (box at rest within 0.06 mm of the rest height and about 1e-5 m/s) | all margins are comfortably inside their ambiguity bands on this trajectory (an earlier statement in this session that place_z / place_v were "near the threshold" misread these margins and is retracted); this says nothing about harder trajectories, which the blind validation must cover |
Poses seen so far: G2a development poses 3000-3047 (G2a definition), I1-I4 and benchmark on poses 3000-3002. The 40 prospective development poses do not exist yet; no sealed reserve pose exists; ACT test seeds 31-33 are unused. No threshold may be changed after the prospective run.

## 7. Statistical protocol (clarified; estimands unchanged, one interpretation change PROPOSED)
Unit = pose; paired risk differences on E2, RD(X-c0) = mean over poses of (E2_X - E2_c0); pose-cluster percentile bootstrap (10 000 resamples, seed 0, all arms kept together). One invalid / censoring policy: a pose is dropped from ALL comparisons if c0 is invalid / censored / not applicable (arm-independent); primary population = complete case; sensitivity: worst-case / best-case for X, all-arms-complete, exclude ambiguous, adverse / favourable assignment of ambiguous, R_LIFT = 0.05. Oracle (O1, O3) and deployable (D1, D2, H1, S0) families are reported separately. Primary: O1-c0 and D1-c0 at 97.5 % (Bonferroni, family alpha 0.05); secondary O3, D2, H1 at 95 %. A comparison is SUPPORTED_BETTER iff the lower bound > 0.
**Proposed clearer interpretation (needs approval; the code `eval/g2b_stats.py` still implements the older conjunction):**
1. Primary paired treatment-effect evidence (O1-c0, D1-c0) is separate from S0 equivalence. A treatment claim requires SUPPORTED_BETTER and a worst-case point estimate >= 0 (and an unchanged sign under R_LIFT = 0.05).
2. S0 is a timing-control diagnostic of the schedule change, not a gate on every comparison. Report its 90 % CI against +-0.10 as exactly one of: **equivalence established** (CI inside), **timing effect detected** (CI excludes 0 and reaches outside the margin), **inconclusive** (CI contains 0 and reaches outside the margin).
3. Failure to establish equivalence does not by itself invalidate an otherwise valid treatment comparison; it requires an explicit disclosure of possible timing confounding, and the schedule-changing candidate D2 (close delayed 8 steps) must be interpreted with that caveat.
4. Statistical non-significance (a CI containing 0), equivalence (a CI inside +-0.10) and insufficient power / precision (CI half-width larger than the margin, or fewer than the planned complete poses) are three different findings and must be labelled separately; at N = 40 the TOST has 56 / 24 / 9 % power at 5 / 10 / 15 % discordance, so "inconclusive" is expected unless S0 and c0 differ in <= 5 % of poses.
5. Outcome-label validity is a separate gate (section 9): if the blind validation fails, the labels are void and NO treatment-effect analysis is run.
No statistical criterion may be revised after seeing G2b treatment outcomes.

## 8. S0 specification
At the decision state (t = 40) hold the last executed 16-D target for 8 control steps (every held command passes clip -> slew), reset any stateful policy cache, query ACT from the observation AFTER the pause (queries 48, 68, 88, ...), later chunk boundaries shift by 8, the horizon stays absolute. With k = 0 the runner reproduces the nominal schedule bit-exactly (I3: executed commands and qpos error 0.0). S0 may genuinely change the physics.

## 9. Blind annotation validation (details: `2026-11-05-g2b-blind-annotation-plan.md`)
Independent annotator (UNASSIGNED), frozen rubric (sha above), anonymised randomised plots, seeded stratified sample of 44 branches + 10 gold traces, kappa >= 0.80 with 95 % lower bound >= 0.65 (one 20-branch extension), E2 agreement >= 0.90, gold 100 %, 0 critical disagreements on unambiguous branches. Failure => labels void; revise, re-freeze and re-run on new development poses; sealed reserve stays sealed. The existing development traces are NOT valid independent validation.

## 10. CPU stop rules (details: `2026-11-05-g2b-compute-budget-and-pose-plan.md`)
Aggregate process CPU (launcher + workload + descendants) with a cooperative guard at 92 % and a hard kill at cap - 0.5 s; pose-boundary stop rule independent of outcomes; atomic per-pose records; PARTIAL on any cap stop; no retry; no cap increase.

## 11. Freeze checklist (all must hold; none is satisfied yet)
(1) Reviewer approval of r3 and of the proposed section-7 interpretation (and the matching `g2b_stats.py` change). (2) Commit of the CONFIRM_BUDGET change (requires separate approval) and re-run of I1 labelling offline (done: identical primary labels). (3) Independent annotator and adjudicator named. (4) 40 development poses generated under the written protocol and hashed. (5) Sealed reserve protocol approved (generation separately authorized). (6) G2b CPU cap authorized. (7) Protocol-freeze file with all hashes. (8) Decision on N and power (keep 40, no reduction after results). 
