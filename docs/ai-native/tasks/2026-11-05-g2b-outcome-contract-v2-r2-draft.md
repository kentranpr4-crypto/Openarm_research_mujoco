# G2b Outcome Contract v2-r2 — DRAFT (not frozen, not approved, not executed) — 2026-11-05
Status: **DRAFT.** Supersedes nothing: r1 (`a77f0fb`) and the G2a/G1/Stage-1 frozen artifacts are untouched. The C1–C5 list is the *reconstructed* list (the original list was not recovered) and is treated as NEW PROPOSED contract changes. Implementation: `eval/g2b_labeller.py` (contract string `v2-r2-draft`), tests `eval/test_g2b_labeller.py` (synthetic). Tự kiểm; nothing here is independent validation.

## 1. Time indexing (X2 — mandatory fix)
- Row `i` = observation BEFORE the action of step `t0+i`. Command `g_exec[i]` is issued after row `i`; its first physical consequence is in row `i+1`.
- COMMAND events carry the ISSUE row: `close_command` (first `g_exec < 0.040`), `release_command` (running-minimum rise >= 4 mm for 2 consecutive command rows).
- OBSERVATION conjunctions use the command that produced the row, `g_app[i] = g_exec[i-1]` (row 0: `meta.g_prev0`, else `g_exec[0]` and the flag `g_prev0_assumed`).
- PHYSICAL opening is a separate timeline event `release_physical_onset` (measured `g_meas` rise >= 4 mm over its running minimum for 2 rows); it never changes a label.
- Historical G2a results are not re-labelled.

## 2. Changes (exact)
| # | r1 | r2 draft | Effect |
|---|---|---|---|
| C1 | v = central difference | v = free-joint qvel (instantaneous). State-time consistency: with trace refresh (below) positions, contacts and qvel belong to the same instant (end of the control step) | placement_settled / E3 only |
| C2 | release before lift not defined | `OPENED_BEFORE_LIFT` (ACQUISITION_FAILURE) iff the release-command row < the lift-confirmation start row; command row == lift row is not OPENED_BEFORE_LIFT (then outside/too-high rules apply) | terminal subtype only |
| C3 | budgets ambiguous | `first_run(mask, lo, hi, k)` = first START s in [lo, hi] with `mask[s:s+k]` all True and `s+k <= N`; acquisition search starts at row `t_cc+1`, start <= `t_cc+25`; lift start <= `t_ga+40`; settle start <= `t_ro+31`. A failure is declared only when rows up to `hi+k-1` are observed, else TIMEOUT/CENSORED | decidability |
| C4 | window `[t_lc, min(t_ro, t_lc+60)]` | HALF-OPEN rows `[t_lc, t_end)`, `t_end = min(t_lc+60, t_release_cmd+1)`; a loss (onset row) is inside iff `onset < t_end`; an unintended release is inside iff its command row `< t_lc+60`; E2 TRUE needs rows through `t_end+K_LOSS-2`; `grasp_lost` counts only if its onset row `<= t_release_cmd`. `RET_WIN` is perturbed by +-1 in the ambiguity analysis | E2 at the boundary |
| C5 | margin-in-band flags | ambiguity = the endpoint (or terminal) changes under the FIXED perturbations: `BANDS` (R_LIFT 5 mm, F_ACQ 1 N, F_HELD 0.5 N, OPEN_RISE 1 mm, R_REL 5 mm, Z_REL 5 mm, PLACE_XY 3 mm, PLACE_Z 1.5 mm, REST_V 0.004 m/s) and K +-1 for K_ACQ, K_LIFT, K_LOSS, K_OPEN, K_SETTLE, RET_WIN; the perturbation values, `spec_hash` and `contract` are written into every label. Flags never alter the primary label | flags only |
| X1 | shift invariance | absolute `T_MAX = 160` stays; shift invariance is required only when the shifted trace is still fully observable inside the horizon; horizon-truncated cases are tested separately (never-closed trace is NOT_CLOSED only at the horizon, TIMEOUT when truncated earlier) | tests |
| X3 | endpoint speed 0 | no zero placeholder: one-sided first-order derivative at endpoints when `box_vel` is absent; missing/NaN `box_vel` rows never satisfy `v < REST_V` and set `velocity_unavailable` | placement |

## 3. Endpoints, censoring, invalid (unchanged semantics unless listed)
E1 GRASP_ACQUIRED, E2 GRASP_RETAINED (primary), E3 PLACEMENT_SUCCESS (secondary): TRUE / FALSE / CENSORED / INVALID + independent `ambiguous`. INVALID = non-finite signal or `meta.valid == False`. TIMEOUT / CENSORED are never converted to FALSE. Per-finger definitions as r1 §1.2–1.3 (n_L, n_R, F_L, F_R from `mj_contactForce` normal components in newtons; total-only counts are not used).

## 4. Trace collection (C-section of the review)
- After 50 `mj_step` substeps `d.xpos/d.contact` belong to the START of the last substep; `qpos/qvel` to its end (1 ms apart). The recorder (`TraceRecorder(refresh=True)`) copies `MjData` to a scratch object and calls `mj_forward` there, so every logged quantity is end-of-step and the live data (including solver warm-start) is never modified. No physics is altered.

## 5. Statistics and protocol (unchanged from r1 unless stated)
S0 equivalence margin +-0.10 (TOST, 90 % CI inside); N = 40 prospective development poses; two Bonferroni-adjusted primary comparisons (O1-c0, D1-c0 at 97.5 %); paired pose-cluster bootstrap; worst/best-case, all-arms-complete, ambiguity sensitivity; blind validation: kappa >= 0.80 (lower bound >= 0.65), E2 agreement >= 0.90, zero critical disagreements on unambiguous branches, failure => labels void. The 0.30 CPU-h G2b cap is **NOT authorized**. Power limitation of the +-0.10 TOST at N = 40 (56 / 24 / 9 % at 5 / 10 / 15 % discordance) stands.

## 6. Open questions (need a decision before any freeze)
1. Is `OPEN_RISE` onset (command +4 mm, 2 rows) the right release definition for ACT, whose opening ramp may differ from the expert's? Analytically a monotone opening command always satisfies onset before the held force falls below 2.5 N (needs about +7.6 mm), but this was not tested on ACT data.
2. Should RET_WIN and the three budgets be formally listed as ambiguity bands (only RET_WIN is added here)?
3. Keep +-0.10 / N = 40, or widen / enlarge (power).

## 7. Integration record (2026-11-05; tự kiểm; not an evaluation)
- **I1 run 1 FAIL (kept: `outputs/g2b_integration/integration_report_run1_I1_FAIL.json`, trace `traces/I1_expert_p3000_run1.*`).** The harness assertion "close command precedes physical closing" compared the 0.040 m THRESHOLD CROSSING (`close_command`, row 47) with the physical closing onset (row 46), although the command had started to move at row 45. This was a harness interpretation error, not a product failure. `close_command` and all outcome events were NOT redefined; labeller and trace code hashes were identical before/after (verified by sha256); the corrected assertion compares the first commanded departure (> 1 mm, row 45) with the first physical departure (row 46).
- **Rerun (one manual rerun, approved):** I1 PASS, I2 PASS, I3 PASS, I4 PASS. Aggregate ledger 133.8 CPU-s of 180 (includes 45 s unmetered estimate + 22.5 s conservative adjustment). Numbers: see the final report of the session and `outputs/g2b_integration/integration_report.json`.
- **Open semantic observation (X4):** `placement_settled` can be stamped at the release-COMMAND row (e.g. 106 vs physical opening 110): the box already rests on the tray while still gripped, and the 10-row settle run starts at the command row. Consistent with r1 ("tray arrival while held is not a drop") but E3 success is therefore not conditional on the physical opening; needs a reviewer decision.
