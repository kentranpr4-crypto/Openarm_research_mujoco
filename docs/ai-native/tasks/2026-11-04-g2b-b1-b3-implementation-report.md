# G2b B1–B3 implementation report (non-evaluation prototype)

Date: 2026-11-04 · Base commit: `a77f0fb403e5ce7701f537de487fa13c380ce1bd` (r1 design doc, unchanged)
Status: **PARTIAL** (implementation prototype; no G2b execution, no freeze, no new evaluation rollout). Test levels: B1 UNIT-LEVEL PASS · B2 SYNTHETIC PASS · B3 MOCK PASS · annotation tooling PASS · real MuJoCo/ACT integration NOT_CHECKED · independent blind label validation NOT_CHECKED · G2b evaluation readiness BLOCKED.

## Files (all new)
| File | Role |
|---|---|
| eval/g2b_trace.py | B1 per-finger trace recorder, ContactMap, atomic save/load |
| eval/test_g2b_trace_contacts.py | B1 contact/force/unit verification (9 gates, needs ACT venv) |
| eval/g2b_labeller.py | B2 Outcome Contract v2-r1 labeller (pure function) |
| eval/test_g2b_labeller.py | B2 V1 synthetic suite |
| eval/g2b_s0.py | B3 S0 hold-8-steps branch runner |
| eval/g2b_stats.py | paired statistics, TOST, sensitivity modes |
| eval/test_g2b_s0_stats.py | S0 mock + stats tests (17) |
| eval/g2b_annotation.py | blind annotation tooling (rubric, sampling, kappa) |
| eval/test_g2b_annotation.py | annotation tests (6) |

## Test results (self-run; not independent)
| Suite | Result | Evidence |
|---|---|---|
| py_compile, 9 files | PASS | run 2026-11-04 |
| Labeller V1 | PASS: 312 traces + 312 time-shift pairs + 10 boundary tests, 0 failures | outputs/g2b_impl/labeller_v1_report.json |
| Trace/contacts | PASS 9/9 (after 2 corrected test designs, T07) | trace_contacts_report.json; failed runs preserved `*_run{1,2}_FAIL_T07.json` |
| S0 + stats | PASS 17/17 (first run failed on test-setup errors, preserved) | s0_stats_report.json |
| Annotation | PASS 6/6 | annotation_report.json |

## Contact/force verification
mj_contactForce returns contact-frame SI force (N); f6[0] normal >= 0; world force on geom2 = frame.T @ f6[:3]; verified by box-weight balance (+0.4905 N). n* are contact POINTS (~4/finger). Per-finger force measured directly (finger joint 2 is equality-coupled, so servo law cannot split fingers). Use qpos, not xpos, right after mj_step.

## Limitations / open items
- S0 tested with mock env only; may genuinely change physics, so TOST failure != label-timing bias.
- TOST +-0.10 at N=40 has power ~56/24/9 % for 5/10/15 % discordance.
- Labeller clarifications C1–C5 (section below) are implementation choices that differ from, or complete, the r1 text; they are NOT approved contract changes; the r1 document is unchanged.
- Development traces are not valid for independent validation.

## Freeze blockers
Real-env runner integration (trace + S0); re-run G2a gates + new gates; per-branch cost measurement; annotator identity + rubric approval; sealed-reserve pose generation/hash approval; written approval of contract/statistics; CPU cap (0.30 h) authorization.

## C1–C5 — implementation choices versus the r1 contract (review required; none is an approved contract change)
Provenance note: the original C1–C5 list was not written to any file and could not be recovered; this list was rebuilt on 2026-11-04 by reading eval/g2b_labeller.py against r1 §1.3–1.7. It names the five most material differences; further smaller observations are in X1–X3.
| # | r1 text | Current implementation | Reason | Affects timing / terminal / E2 | Boundary example (C3, C4: labeller run on hand-made traces; C1, C2, C5: reasoned from the code, NOT run) | New revision needed? |
|---|---|---|---|---|---|---|
| C1 | v = central difference of b | box speed = free-joint qvel (instantaneous, from the trace); central difference only as fallback when box_vel is absent | qvel is the simulator's own velocity; B1 test T07 showed control-step differences are 20-90 % off under acceleration | placement_settled time and E3 only; no effect on E1/E2 | REST_V = 0.02 m/s: a box decelerating through 0.02 can pass on qvel and fail on the 0.1 s central difference (and vice versa) | Yes, one-line change of the signal definition (or switch to fallback) |
| C2 | subtypes NOT_CLOSED, NOT_ACQUIRED, ACQUISITION_LOST, NOT_LIFTED only; release before lift not addressed | extra subtype OPENED_BEFORE_LIFT (release onset before lift confirmation, no earlier loss) under ACQUISITION_FAILURE | a release before any lift is not a placement; keeps RELEASE_OUTSIDE_TARGET for released-after-lift | terminal label only (E2 and E3 are FALSE under either reading) | gripper target opens at row 30 while rise reaches 3 cm only at row 31 | Yes (new subtype) |
| C3 | failure declared "once the 40 steps have elapsed observed" (lift); "not reached within 25 steps of t_cc" (acquisition) | a failure is declared only after the last possible debounce run could have completed (acquisition: row t_cc+29; lift: t_ga+42; settle: t_ro+40); earlier ends are TIMEOUT / CENSORED | a run that starts at the budget edge needs K rows to be judged | decidability only; delays when a branch may stop; E1 CENSORED vs FALSE on short traces | never-acquired trace, t_cc=0: N=28 -> TIMEOUT; N=29 -> TIMEOUT (flag ambiguous); N=30 -> ACQUISITION_FAILURE / NOT_ACQUIRED, E1 FALSE | Yes (state the decidability rule) |
| C4 | E2 window [t_lc, min(t_intended_release, t_lc+60)]; loss "inside W" | strict: loss counts iff its onset row t_gl < t_end_ret (onset of the not-held run, not its detection); E2 TRUE needs row t_end_ret observed, else CENSORED | window end is exclusive of the final row; onset time is the event timestamp | changes E2 at the boundary only | lift at 10 (window end 70): loss onset 69 -> E2 FALSE; 70 -> E2 TRUE; 71 -> E2 TRUE. **The boundary flip at 70 is NOT flagged ambiguous**, because RET_WIN is not in BANDS | Yes: define the boundary and add RET_WIN (+-1) to the ambiguity analysis |
| C5 | ambiguous when a decisive margin lies inside the §1.7 band | ambiguous iff perturbing a band parameter (and K +-1) changes the endpoint value (or the terminal for ambiguous_terminal); margins stored | flags then mean "label sensitive" instead of "margin small"; avoids flagging irrelevant margins | flags only; no label, no event time | a lift margin of 2 mm is not flagged when another condition already decides the endpoint | Yes (flag semantics) |
Further observations (not claimed to be part of the original list): **X1** the horizon test uses absolute t0+N >= T_MAX (only NOT_CLOSED depends on it), so a uniform shift is label-invariant except when it moves the end of the trace across T_MAX; r1 §1.6 says shifts change no label. **X2** g_exec[t] is the command applied at step t (after observation row t), so contacts/box in row t+1 are its first effect; conjunctions such as g < 0.030 and F >= 5 N at the same row mix a not-yet-effective command with the preceding state (events are stamped up to one step early; K+-1 partly covers it). **X3** the central-difference fallback sets the first and last rows to 0 speed.
Do not freeze until C1-C5 and X1-X3 are reviewed.

## Trace-sampling and contact-semantics audit (static reading; no simulator run beyond the unit tests)
- Control step = env.substeps (= 50 at 20 Hz, dt 0.001 s) calls of mj_step with a constant ctrl; there is no mj_forward after the loop (sim/lab_env.py:62, eval/closed_loop_pilot.py:109).
- After mj_step, d.xpos/xmat/cvel and d.contact/efc forces belong to the state at the START of the last substep (pre-integration), whereas qpos/qvel are post-integration: a 1 ms (0.001 s) offset. The recorder reads box_pos/tcp from xpos and contacts from d.contact (mutually consistent), qvel and finger qpos (post-integration, 1 ms newer). The offset is below every threshold scale (10 cm/s x 1 ms = 0.1 mm) but is a documented inconsistency; T07 uses qpos for that reason.
- Row t is recorded before the action of step t (rec.record before lean_step); g_exec[t] is that action's gripper target; the action's first effect is visible in row t+1 (X2).
- Contact frame: f6 in the contact frame, normal = row 0 of contact.frame, from geom1 to geom2, f6[0] >= 0; world force on geom2 = frame.T @ f6[:3]; verified by box-weight balance (Fz = 0.4878-0.4904 N against 0.4905 N) and by action-reaction between fingers (FL 23.72 N, FR 23.78 N, imbalance < 0.4 %).
- Per-finger mapping by geom name (blue_box_collision with openarm_left_{left,right}_finger_collision; palm = openarm_left_hand_collision); forces summed as |f_n| over contact points; per-finger sums reproduce the historical observer on every row (differences <= 2e-6 N, counts 0).
- Contacts and object state in one row come from the same mj_step call (same logical timestep); finger qpos and qvel are 1 substep newer.
- Event coverage by trace fields: close_command <- g_exec; grasp_acquired <- g_exec, nL, nR, FL, FR; lift_confirmed <- box_pos, box_z0; release_onset <- g_exec; grasp_lost <- nL, nR, FL, FR, g_exec; release location <- box_pos; placement_settled <- box_pos, box_vel; ambiguity <- all of them. Not covered by the trace: nothing required by r1; g_meas is logged but not used by any event.

## Integration plan (PREPARED, NOT EXECUTED; each step needs separate approval)
| Gate | Content | Real simulator | Estimated CPU |
|---|---|---|---|
| I1 | scripted expert (no policy), 1 development pose, full 160-step recording, label complete trace, truncation at 3 cut points -> CENSORED/TIMEOUT behaviour, NaN injection -> INVALID | yes | setup 19 s + 160 x 0.0158 s = about 22 s |
| I2 | snapshot (mj_copyData) bit-exactness with trace recorder attached; c0 branch replay equals the original rollout | yes | about 10 s |
| I3 | ACT runner compatibility: one c0 branch with trace, then S0 (k=8) from the same snapshot: held target for 8 steps through clip->slew, action history/observation after the pause, policy-state reset, equals nominal when k=0 | yes (ACT checkpoint) | about 25 s |
| I4 | CPU benchmark: 3 development poses, c0 only, horizon 160, with trace logging | yes | about 45 s |
| Total | | | about 100 s = 0.03 CPU-h (estimate from the measured G2a cost model, not measured for this scope) |
Permissions needed: running simulator rollouts with the ACT checkpoint; a CPU cap for the integration gate (proposed 0.05 CPU-h, separate from and not part of the 0.30 CPU-h G2b cap, NOT AUTHORIZED); 1-3 development poses outside the sealed reserve; ACT test seeds 31-33 stay unused; outputs/ stays untracked. Invalid or truncated traces: INVALID (non-finite, meta.valid False) never enters the labeller as a label; truncated traces give TIMEOUT / CENSORED, never FALSE.
