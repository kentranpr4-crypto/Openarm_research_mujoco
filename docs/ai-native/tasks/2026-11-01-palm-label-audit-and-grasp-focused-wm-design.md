# Palm-contact calibration / label-horizon audit (read-only) + Grasp-Focused World Model design — 2026-11-01
Status: **DONE (audit) / design only (no simulator, no training, no model inference).** Inputs: stored branching-replay results/arrays (freeze `f2249bd3…cfe24` untouched), stored Stage-1 results/trajectories. Scripts: `eval/palm_label_audit.py` (numpy); raw: `outputs/branching_replay/palm_label_audit.json`, `palm_term_removed_diagnostic.json`. POST-HOC diagnostics on development data, not independent evidence. Tự kiểm.

## A. Audit results
| # | Finding | Status |
|---|---|---|
| A1 | Branch-level palm flag (648 branches, threshold 0.01 N): TP 42, **FP 305**, FN 0, TN 301. Predicted rate 53.5 % vs realised 6.5 %; precision **0.12**, recall 1.00, FPR **0.50**. By phase FPR: approach 0.51, grasp 0.70, carry 0.77, place 0.30 (place: 81 FP, 0 realised). By candidate: FPR 0.46 / 0.48 / 0.57. | Supported |
| A2 | **Realised palm contact is binary and large**: true positives have peak hand force median 1.9 N, p90 5.0 N, max 13.5 N; **all 305 false positives have realised hand force exactly 0.0 N** (none in (0, 0.01] N). Changing the realised-label threshold from 1e-4 to 1 N leaves precision at 0.12→0.09. → the problem is not label-threshold sensitivity. | Supported |
| A3 | **Label horizon is not the main cause**: extending the label for the default branch (real A0 continuation) from 20 to 40/60/rest-of-episode raises precision only 0.10→0.14/0.15/0.17 (FP 94→90/89/87 of 105 predicted). J2 (40-step, hold continuation, all candidates): precision 0.13, FPR 0.63. | Supported (default branch only; other candidates have no continuation beyond t+20) |
| A4 | Palm term in the ranking: predicted flag differs across candidates in 55/216 states, **realised flag differs in only 10/216**; removing the 0.05·palm term would change the argmin in 28/216 states. | Supported |
| A5 | **Diagnostic (post hoc, one variant tried, frozen J untouched):** ranking with the palm term removed → pairwise accuracy 0.874 [0.802, 0.953] vs 0.793 [0.716, 0.868] (frozen J20); grasp phase 56/56 pairs correct (1.00 vs 0.875); regret − default −0.00033 [−0.0011, +0.0005] vs −0.00003. Evidence that the over-predicted palm flag adds noise; not a validated improvement. | Exploratory |
| A6 | Headroom decomposition (20 states with headroom ≥0.01): continuous dxy+dz carries it (grasp: 0.0150 of 0.0150; palm 0, lift 0); palm term is the majority in 2/20 states only. The grasp-phase WM skill found in the replay is **box displacement/height ranking, not palm avoidance**. | Supported |
| A7 | **Stage 1 closed loop: palm contact is a real failure risk**: P(fail | palm) vs P(fail | no palm): A0 0.75 vs 0.15 (3/4 vs 3/20), A1 0.56 vs 0.24, A2 0.43 vs 0.24. 29/120 episodes had palm contact; first contact median step 44, **before the first finger contact in 79 %**, median 23 steps of contact, peak force median 3.1 N (max 13.5 N); 13 of those 29 still succeeded. | Supported (descriptive; n small, confounded by pose) |
| N | **NOT_EVALUABLE** from stored data: continuous calibration of the predicted palm force (reliability curve, ECE, threshold sweep of the *prediction*) — only the thresholded boolean was stored; timing of the predicted first contact; candidate-specific long-horizon labels for non-default candidates. | Instrumentation gap |

Conclusion: the frozen palm head is a **high-recall / low-precision alarm** (FPR 50 %), mis-specified rather than mis-labelled; palm contact matters physically (A7) but is rare (6.5 %) and strongly bimodal (A2), so it needs a calibrated, class-balanced classifier with timing, not a 0.01 N threshold on a regressed absolute force.

## B. Grasp-Focused World Model (GF-WM) — design (nothing run)
**Goal:** rank *physically meaningful grasp alternatives* at pre-grasp / approach / close states by predicted grasp quality and palm-contact risk, with calibrated uncertainty and ACT fallback.
**Hypotheses.** H-G1: a classifier-based, calibrated palm/contact head reduces FPR at recall ≥0.9 from 0.50 to ≤0.15. H-G2: grasp-quality heads (lift-and-hold success, box-in-TCP offset, slip) rank candidate grasps better than the frozen J20 on candidate-level executed labels. H-G3: a grasp candidate family with controlled geometric variation has oracle headroom in *outcome* terms (palm-free lift-and-hold), unlike the near-identical C0 chunks (headroom ≥0.01 in 9 % of states).

### B1. Physically meaningful candidate generation (no change of action semantics)
- Family = parametric variations of the scripted grasp primitive (`ScriptedExpert.segments`): TCP depth along the finger axis at close (a_offset; the observed ACT failure at low x is “too deep / palm contact”), lateral offset, yaw offset, pre-grasp standoff/height, closing dwell/speed, commanded finger width. Ranges anchored on geometry (box 4 cm cube, finger ctrlrange −0.005…0.044 m, IK reach) and on the ACT-vs-expert a_close differences already in xsweep — **not** on Stage-1/replay outcomes.
- Every candidate: IK → 16-D absolute joint targets in actuator order → the same clip→slew chain from the last executed target; shared executed prefix up to the decision state (consistent family, no checkpoint hand-off, addresses P1).
- Validity gate per candidate (pre-registered): IK success, ctrlrange/slew legality, no self/table collision before contact, executed-vs-commanded tracking error under a bound. N = 7–9 candidates per state by a fixed factorial/Latin design.
- Modes: (a) **oracle-pose family** (privileged box pose in IK; upper bound, clearly labelled); (b) **ACT-anchored**: ACT chunk + the family’s deltas applied to the grasp-phase portion — only if (a) shows headroom.
- Honest limitation: (a) uses the privileged pose; it measures whether grasp-geometry choice matters, not a deployable system.

### B2. GF-WM architecture
- Inputs (as WM3-B): H=4 histories of joint state, box pose, contact block, executed action history, candidate executed chunk; plus grasp-geometry features (box-in-TCP frame, finger gap, finger/palm contact flags).
- Horizons: K=10 and K=20 (finer timing); optional 40 with hold.
- Outputs: (1) palm contact **hazard/probability by step** + peak force class (focal/class-balanced BCE, positive rate 6.5 %); (2) finger contact count/force; (3) box-in-TCP offset (a_offset) at close; (4) lift-and-hold success within 20 steps (rise ≥5 cm and TCP-box distance < hold bound); (5) slip (box–TCP separation); (6) box position delta. Huber on continuous heads.
- Calibration: temperature/isotonic on a *separate* calibration pose set; report reliability and ECE; **≥3 training seeds** → ensemble disagreement as uncertainty (WM3-B is a single seed); no use of test poses for any threshold.
- Training data design (requires approval, **not run**): candidate-level executed labels from branching (counterfactual labels exist, unlike WM-2 observational data), new pose namespace (not Stage-1/indep poses), grasp-phase states only.

### B3. Scoring (pre-registered, no tuning on outcomes)
Lexicographic risk-aware rule instead of a hand-weighted sum: (1) drop candidates with calibrated P_palm > τ_p (τ_p fixed on the calibration set for recall ≥0.9); (2) among the rest maximise calibrated P(lift-and-hold); (3) tie → smaller predicted |a_offset − nominal|; (4) if the best-vs-second margin < τ_m (set from the calibration set noise floor, initial reference 0.01 from the decision-utility protocol) or ensemble disagreement > τ_u → fall back to the default (ACT chunk / nominal grasp). The frozen J is not modified.

### B4. Controls / baselines
Default (ACT or nominal grasp); uniform random; **oracle best-of-N** (upper bound); frozen WM3-B J20; frozen J20 with palm term removed (diagnostic from A5); a **physics heuristic** (shallowest-depth rule) to test whether the WM adds anything beyond a trivial rule; GF-WM ablations (no palm head, single seed vs ensemble, K=10 vs 20).

### B5. Metrics
Pairwise accuracy at outcome gaps (|Δ| ≥ pre-registered); top-1 and normalised regret; **oracle headroom distribution in outcome terms** (palm-free lift-and-hold; proportion of states with headroom); palm AUROC / PR-AUC / FPR at recall 0.9 / ECE / Brier; contact-time error; gated coverage vs regret; all CIs by bootstrap over **poses**; phase-wise (pre-grasp, approach, close).

### B6. Compute (estimates from measured costs: branch ≈ 0.85 s CPU incl. ACT, ≈0.3–0.6 s expected without ACT; WM3-B training took 2 137 s wall at 3 threads)
| Step | Content | Estimated CPU-h | Proposed cap |
|---|---|---|---|
| G1 | candidate generator validity + outcome-based oracle headroom: 24 new poses × 3 grasp states × 9 candidates = 648 branches (+ IK plans) | 0.10–0.20 | **0.25** |
| G1b | proxy check: expert continuation to episode end for 96 branches (does palm-free lift-and-hold predict task success?) | ≈0.1 | 0.15 |
| G2 | dataset: ~300 poses × 3 states × 9 candidates ≈ 8 100 branches | 0.9–1.9 | 2.0 |
| G3 | train GF-WM, 3 seeds + calibration | 0.9–2.4 | 2.5 (separate approval) |
| G4 | held-out evaluation, 48 new poses | 0.2–0.4 | 0.5 |
Total ≈ 2.2–5.0 CPU-h. Ledger (approx.): earlier use 5.9–10.3 + Stage 0/0.1 0.04 + Stage 1 0.40 + replay 0.21 ⇒ ≈ 6.6–11.0 of the 22 CPU-h ceiling; remaining ≈ 11–15 (estimate, same uncertainty as before).

### B7. Stop criteria
G1: candidate validity <80 % → fix generator, no further steps; outcome-based oracle headroom <0.01-equivalent (pre-registered in outcome units) in >80 % of grasp states → **stop the grasp-focused line**; any legality/semantics violation or snapshot nondeterminism → BLOCKED. G2/G3: caps; calibration ECE > 0.05 or FPR at recall 0.9 > 0.30 on the calibration set → do not proceed to G4. G4: frozen protocol, no changes after seeing results.

### B8. Leakage / validity risks
Oracle pose in candidate IK (privileged; labelled); designing ranges or thresholds from Stage-1/replay outcomes (forbidden — use geometry and calibration poses); re-using the 24 Stage-1 poses as test (they are development data now: new namespace needed); labels computed with the future of the same branch (allowed as labels only, never as inputs); expert-continuation labels bias toward expert-friendly grasps (use only for proxy validation); A5 was a single post-hoc variant on development data and must not be used to set the GF-WM design.

### B9. What would falsify the hypotheses
H-G1: calibrated classifier FPR at recall 0.9 on held-out poses >0.30 (no better than the frozen head’s 0.50→ not meaningfully). H-G2: GF-WM pairwise accuracy/regret not better than the shallow-depth heuristic and J20-no-palm with non-overlapping pose-bootstrap CIs. H-G3: oracle outcome headroom absent in >80 % of grasp states for the generated family.

## C. Recommended next experiment (not run; needs approval): **G1**
Validate the candidate generator and measure outcome-based oracle headroom before any WM training: 24 new poses (new RNG namespace, strata incl. low x), 3 grasp states, 9 parametric candidates, snapshot/branch execution (same bit-exact machinery), labels = palm-free lift-and-hold within 20 steps + final box offset, plus the cheap baselines (default, random, shallow-depth heuristic, frozen J20 / J20-no-palm *predictions only if windows are saved*). Cap 0.25 CPU-h, integration tests + protocol freeze first. Decision it enables: continue to G2/G3 only if headroom and validity pass; otherwise stop the grasp-focused WM line.
