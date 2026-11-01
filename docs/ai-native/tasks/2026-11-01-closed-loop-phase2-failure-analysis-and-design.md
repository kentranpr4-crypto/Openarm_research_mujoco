# Stage 1 failure analysis (read-only, POST-HOC) + Phase 2 design — 2026-11-01
Status: **DONE for analysis; design only (no execution).** Inputs: frozen checkpoints' saved results `outputs/closed_loop_pilot/results/{pilot_results.json,traj/*.npz}`; script `eval/closed_loop_failure_analysis.py` (numpy, ~1 s CPU, no simulator/model); raw `outputs/closed_loop_pilot/failure_analysis.json`. Post-hoc diagnostic, NOT independent evidence. Stage 1 protocol (sha 8fd93a31…2c), outcomes and J untouched. Tự kiểm.

## A. Findings (S = supported by the saved data, U = unsupported/not concluded, N = NOT_EVALUABLE)
1. **Pose reconciliation (S).** better/equal/worse = sign of diff = A2 success − mean(A1 successes): 7/11/6. Discordant = A2 differs from the A1 *majority*: 8. Cross-tab: better+discordant {17,18,45,46}; better+concordant {15,16,19} (A2 succeeded, A1 majority also succeeded but 1 of 3 failed); worse+discordant {0,30,32,47}; worse+concordant {1,2}; equal {11 poses: 10 always-success, 1 always-fail (pose 31)}. Both counts are driven by single episodes; the 8 discordant poses split 4/4.
2. **Switching (S).** A1 and A2 switch checkpoint at 67.7 % / 67.6 % of chunk boundaries (random expectation ≈ 2/3); 6.10 vs 6.08 switches/episode. A2 is behaviourally indistinguishable from random in switching pattern. Switch count does not differ between successes and failures (A1 6.06 vs 6.17; A2 6.12 vs 6.00) → no support that switching frequency explains failures (U).
3. **Boundary discontinuity (S, executed targets only).** Mean executed arm step at chunk boundaries / within chunk: A0 1.57×, A1 1.95×, A2 1.80×; when the checkpoint switched 0.0148 rad vs 0.0123 rad when not (A1), 0.0132 vs 0.0127 (A2); gripper boundary step 0.00050–0.00059 m vs 0.00028 m (A0). Boundary jumps are modest (≤0.1 rad slew limit, at limit in <0.5 % of boundaries). Raw ACT target jump before slew: N (not saved).
4. **Failure timing (S).** All arms fail the same way: A0 6/6 and A2 6/7 drops are “dropped far from tray (carry)”, A1 14/22 carry + 8 at tray + 1 never lifted; median drop step 104–108 (≈ chunk 6) in every arm. The failure is an ACT transport/carry failure at low x, not a decision-specific effect of A2.
5. **J prediction vs realised (S, selected chunks only, last chunk excluded).** Within phase Spearman(J_pred, J_real) = 0.65–0.79 and dxy Spearman 0.88–0.98, dxy MAE 1.6–4.5 mm (the WM tracks box progress). Pooled 0.94 is inflated by between-phase variance. **Palm contact is over-predicted**: predicted 50 % vs realised 11 % of chunks (carry 67 % vs 11 %, place 38 % vs 0 %).
6. **Near ties (S).** ΔJ(best, second) ≤1e-3 in 111/216 (51 %) A2 decisions (place 70/82, carry 23/62, approach 18/48, grasp 0/24), ≤5e-3 in 153/216 (71 %); non-default picked in 55 and 83 of them. Every episode contains near-tie decisions. The official tie rule (1e-9) was not changed; it never fired. Margin ≥0.01 occurs in only 47/216 (22 %) decisions (≥0.02: 15 %).
7. **A2 non-default picks (descriptive, confounded).** 124 non-default / 92 default picks; realised J 0.137 vs 0.138 overall; by phase no consistent difference (grasp: 0.240 non-default vs 0.263 default, n_default = 1). No inference.
8. **Real same-state pairs (S, the only counterfactual in the data).** A0 and A2 are bit-identical until A2’s first non-default pick (asserted from saved actions), so at that state we have two *executed* continuations of one chunk (default in A0, WM-selected in A2). 21 usable pairs: WM-predicted ΔJ (always favouring the selection by construction) matches the sign of realised ΔJ in 11/21 (0.52). With |realised ΔJ| ≥ 0.01 (the frozen meaningful-pair threshold): **2/6 agree**; in 4 of the 6 (poses 1, 16, 30, 46) the selected chunk realised a palm-sized J increase (+0.04–0.05, a palm contact the WM did not predict as a penalty for the default). Interpretation limited by n: ranking skill at first divergence is not demonstrated (U), not refuted.
9. **A2 failure poses vs A0/A1 (S).** A2 failed 7: 0, 1, 30 (A0 also failed); 2, 31, 32, 47 (A0 succeeded; first divergence at t=20–40 near ties with margins 0.0011, 0.0030, 0.0014, 0.0011 — three of four ≤ 1e-3, one 3e-3). A2 succeeded where A0 failed: 15, 16, 46 (margins 0.0016, 0.0066, 0.0069). A2 − A0 = +3 / −4 poses, consistent with noise.
10. **Not evaluable (N):** counterfactual outcomes of unselected candidates; WM ranking accuracy on A1 picks (A1 never ran WM; b rows with yaw/tilt/speed not saved); raw pre-clip ACT jumps; causal attribution of any single failure to a single decision.

**Supported conclusions:** A2 ≈ random at the decision level (switching rate identical, >50 % near-ties); all arms fail by the same carry-phase drop; WM tracks box distance well but over-predicts palm contact; at the first divergence the WM’s predicted preference was not borne out by executed chunks in this small sample.
**Not supported:** that switching or discontinuities cause the failures; that WM ranking harms or cannot help; any ranking-skill estimate with a CI (n = 21 and 6).

## B. Phase 2 proposals (design only; nothing run, nothing tuned on Stage 1 outcomes)
**Evidence rule for all three:** parameters come from the offline decision-utility protocol / physical reasoning, not from Stage 1 outcomes; Stage 1 statistics are only used to size the experiment.

### P1 — Consistent candidate generation
- Hypothesis: candidates that stay on one policy’s trajectory family (no checkpoint hand-off) give a ranking problem the WM can resolve and avoid hand-off artefacts.
- Options keeping action semantics (absolute 16-D actuator targets, clip→slew): (a) same checkpoint queried at lagged times (execute the tail of an earlier chunk / re-query at t−k); (b) candidate filter: keep only checkpoints whose first executed target is within δ of the last command; (c) per-checkpoint continuation (stay with the chosen checkpoint for ≥m chunks, “hysteresis”) — a selection rule, not a post-hoc blend.
- Instrumentation: save raw chunks, boundary jumps, per-candidate first-step gap.
- Control: Stage 1 A0/A1/A2 numbers. Metrics: boundary jump, realised J/pairwise accuracy by option, success.
- Compute: offline replay 0.05 CPU-h; closed loop ≈0.3 CPU-h per 24-pose arm set (≈6 s/episode).
- Stop: option yields <2 distinct candidates in >50 % of states, or any action-validity violation.
- Leakage risks: choosing δ/m from Stage 1 outcomes (forbidden; set from physical limits, e.g. slew step).
- Falsified if: realised J spread is not related to boundary-gap size, or hysteresis leaves success unchanged within the CI of A1.
- Prior from Stage 1: weak (switch effect on boundary step +20 %, no association of switch count with failure).

### P2 — Horizon-aligned decision scoring
- Hypothesis: the frozen one-chunk J mostly measures approach progress; carry drops at step ≈105 need a longer horizon and carry-specific risk terms (e.g. hold stability: box-in-TCP offset, finger force, box speed).
- Design: new scoring J2 (not a change to J): two-chunk (40-step) autoregressive WM rollout of candidate + default continuation (the rollout code exists in `decision_utility_lib`), plus terms registered from physics/expert statistics (weights fixed before any evaluation; no Stage-1-outcome fitting; or a parameter-free lexicographic rule).
- Instrumentation: store b/c rows and WM predictions; run J and J2 in parallel on identical states.
- Baseline: frozen J20. Metrics: pairwise accuracy, regret and sign agreement against realised outcomes of *executed branches* (see E).
- Compute: WM cost ×2 (≈20 ms/decision), negligible.
- Stop: J2 pairwise accuracy ≤ J20 on development states.
- Leakage: realised outcomes must not select weights; use a held-out set of states for the comparison.
- Falsified if J2 does not beat J20 pairwise accuracy at meaningful (≥0.01) gaps on held-out branch states.

### P3 — Uncertainty-aware ranking with ACT fallback
- Hypothesis: ranking helps only when the WM margin exceeds its noise floor; below it, keep the ACT default (c0_s0).
- Rule: use WM choice iff margin ≥ τ and (optional) WM self-consistency check (batch perturbation of A_hist); else c0_s0. τ = 0.01, the frozen meaningful-pair threshold of the decision-utility protocol (not tuned on Stage 1).
- Coverage from Stage 1: margin ≥0.01 in 22 % of decisions (≥0.02: 15 %) → the gated policy changes behaviour in ≈ one decision in five, so the expected effect is small and detection needs many poses.
- Instrumentation: gate indicator, fallback rate. Control: A0 and always-rank A2. Metrics: success, fallback rate, gated-decision realised pairwise accuracy.
- Compute: same as A2 (≈8 s/episode); no extra WM training. Real WM uncertainty (ensemble) would need new WM training — not requested.
- Stop: fallback rate >90 % (inert) or <10 % (gate irrelevant).
- Leakage: τ from outcomes; margin computed only from the WM at decision time.
- Falsified if: realised-vs-predicted agreement at margin ≥τ is not above chance in the branch data.

### Comparison
| | P1 | P2 | P3 |
|---|---|---|---|
| Addresses Stage-1 finding | hand-off artefacts (weak evidence) | horizon mismatch / carry drop | near-tie randomness (strong evidence) |
| Needs training | no | no | no |
| Cost | low | low | low |
| Risk of being inert | medium | medium | high (22 % coverage) |
| Testable offline first | yes | yes | yes |

## C. Recommended next experiment (not run): offline branching replay — information gain per CPU is highest
Goal: obtain what Stage 1 lacks — *executed* outcomes of every candidate from the *same* state — and use them to test P1/P2/P3 offline before any new closed-loop run.
- Method: for each of the 24 poses, run A0 once (deterministic, verified bit-identical to xsweep), and at each decision state t = 20…180 snapshot `mjData`; branch the 3 C0 candidates for one chunk (20 steps), record realised J (and, for 2-chunk branches, J over 40 steps with the default continuation), then evaluate in parallel WM J20, J2 (P2), margin gating (P3), boundary gap (P1). No closed-loop policy is evaluated; the branch outcomes are labels for the WM ranking, nothing is selected or executed on their basis.
- Size: 24 poses × 9 states × 3 branches = 648 branch chunks; optional second chunk of default continuation = +648.
- Primary metrics: pairwise ranking accuracy at |ΔJ_real| ≥ 0.01 with pose-level bootstrap (10 000, seed 0), regret of J20/J2/gated rules against realised outcomes; candidate-level headroom (oracle best-of-3 realised J vs default) — a ceiling on what ranking can ever give.
- Projected CPU: A0 passes 24×6 s ≈ 145 s; branches 648×(ACT 0.08 s + 20 sim steps ≈0.6 s) ≈ 450 s (+450 s for 2-chunk); WM + load ≈ 60 s ⇒ **≈ 0.18 CPU-h (1-chunk), ≈ 0.30 CPU-h (2-chunk)**; cap proposal 0.40 CPU-h. Cumulative ledger after: ≈ 0.4 (Stage 1) + 0.4 ≈ 0.8 CPU-h of the new accounting, far below 22.
- Stop rules: oracle best-of-3 realised-J headroom vs default < 0.01 in >80 % of states (nothing to gain → stop the line), any action-validity violation, cap reached (PARTIAL).
- Decision it enables: if oracle headroom and WM pairwise accuracy at margin ≥ τ are both substantial, a pre-registered closed-loop P3 (±P2) pilot with ≥48 poses is justified; otherwise stop the WM-ranking line for this ACT.
- Needs separate approval: it runs the simulator (new execution) although it evaluates no policy end to end.
- Limits: 24 poses (same as Stage 1 → states are developed on the same poses; use *new* poses for the confirmatory step), privileged oracle observer, simulation only.
