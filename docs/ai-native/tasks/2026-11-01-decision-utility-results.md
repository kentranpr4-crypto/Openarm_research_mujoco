# Offline decision-utility results (DEVELOPMENT; protocol frozen before any ranking result)

Date: 2026-11-01. Protocol: `2026-11-01-decision-utility-protocol.md`; freeze `outputs/decision_utility/protocol_freeze.json` (sha256 `c1082af6ee6574d1eeb4fa71fbd9b1f64ae4f8d0235b77e270d360aa26eea968`, written after the library tests 24/24 and the sanity check passed). Results: `outputs/decision_utility/results.json` (not in git). **Everything here is DEVELOPMENT evidence** (wm2 test poses and I-1 were analysed before; privileged-state WM inputs; scripted-expert candidates; frozen models; one rollout design). Commit hash of the last pushed code: `bb15867` — these files are uncommitted.

## 1. Sanity check (before the freeze, no ranking result)
Constructed block-0 windows equal the stored dataset windows bit for bit (759 + 1 237 candidates, max |Δ| = 0.0); WM3-B predictions on them equal the stored predictions to 7e-7. Library tests: 24/24 (scoring, tie-break, regret, pairwise accuracy, AUROC, ECE, bootstrap, rollout with true deltas reproduces the true rows, **leakage by perturbation: replacing every ground-truth row after t0 leaves the predictor inputs bit-identical**).

## 2. Candidate coverage and support
| | wm2 test poses | I-1 poses |
|---|---|---|
| poses / candidates | 31 / 759 | 50 / 1 237 |
| true success rate of candidates | 0.900 | 0.891 |
| poses where some candidates succeed and some fail | 30 / 31 | 50 / 50 |
| median spread of the true cost J within a pose | 0.244 | 0.256 |
| true horizon end already settled | 99.5 % | 99.6 % |
| rolled blocks (K = 20) | 5–7 | 5–7 |
Ceiling effect: ~90 % of candidates succeed, so random choice already succeeds 0.89–0.90; top-1 success cannot separate models (all CIs overlap); **oracle regret and pairwise ranking accuracy are the informative metrics**. ACT-reachability diagnostic (coarse action statistics, 501 ACT roll-outs): only 12–29 % of the scripted candidates — including the control expert (12–16 %) — fall inside the 1–99 % envelope of ACT roll-outs; the candidate sets are scripted-expert perturbations and say nothing certain about ACT-produced chunks. Ties at the top: 0 % for the action-conditioned models, 20–29 % for the action-blind models, 100 % for the "box stays" null (tie-break by SHA-1 rule).

## 3. Ranking results (macro over poses; 95 % bootstrap CI over poses)
| model | wm2 test: top-1 | regret | pairwise acc. | I-1: top-1 | regret | pairwise acc. |
|---|---|---|---|---|---|---|
| **WM3-B** | 0.968 [0.90, 1.0] | **0.002** [0.001, 0.004] | **0.905** [0.870, 0.935] | 1.00 | **0.001** [0.001, 0.002] | **0.917** [0.893, 0.938] |
| WM2-C0 | 1.00 | 0.006 [0.003, 0.011] | 0.827 [0.771, 0.877] | 0.98 | 0.009 [0.002, 0.021] | 0.847 [0.809, 0.882] |
| post-hoc hybrid | 1.00 | 0.005 [0.002, 0.008] | 0.833 [0.770, 0.889] | 0.98 | 0.009 [0.002, 0.022] | 0.841 [0.800, 0.878] |
| MLP | 0.935 | 0.008 [0.005, 0.011] | 0.767 [0.720, 0.811] | 0.98 | 0.007 [0.005, 0.009] | 0.821 [0.786, 0.854] |
| ridge | 0.871 | 0.039 [0.024, 0.060] | 0.477 [0.430, 0.531] | 0.86 | 0.034 [0.023, 0.049] | 0.469 [0.437, 0.504] |
| MLP action-blind | 0.903 | 0.025 [0.003, 0.054] | 0.586 [0.539, 0.630] | 0.96 | 0.016 [0.004, 0.031] | 0.618 [0.584, 0.649] |
| ridge action-blind | 0.935 | 0.013 | 0.479 | 0.92 | 0.020 | 0.489 |
| "box stays" null | 0.903 | 0.027 | 0.500 | 0.90 | 0.027 | 0.500 |
| random / oracle | 0.839 / 1.0 | 0.027 / 0 | 0.517 / 1.0 | 0.92 / 1.0 | 0.026 / 0 | 0.497 / 1.0 |
Paired differences over poses: WM3-B − WM2-C0 pairwise **+0.077 [0.022, 0.136]** (wm2 test) and **+0.069 [0.036, 0.104]** (I-1); regret −0.004 [−0.010, 0.001] and **−0.007 [−0.021, −0.0003]**; top-1 −0.032 [−0.097, 0.0] and +0.02 [0, 0.06] (one wm2-test pose where WM3-B chose a failing depth candidate). **Action conditioning matters**: WM3-B − action-blind MLP pairwise +0.32 [0.27, 0.37] / +0.30 [0.26, 0.34], regret −0.023 [−0.051, −0.001] / −0.014 [−0.030, −0.003]. The linear ridge has no ranking skill (pairwise 0.47–0.48, consistent with its failed box gate). The hybrid equals C0 (identical box/contact heads).
Per-action-family ranking (control + one family; pairwise accuracy, wm2 test / I-1): WM3-B depth 0.90 / 0.87, timing 0.91 / 0.88, noise 0.91 / 0.92, release_height 0.94 / 0.96; C0 depth 0.72 / 0.73, **timing 0.65 / 0.48**, noise 0.90 / 0.90; action-blind MLP depth 0.44, timing 0.50, release_height 0.50. The speed family has no pair with a true cost difference ≥ 0.01 (not evaluable: the family does not change the placement outcome). Failure mode: top-1 errors concentrate in the depth family (WM3-B 0.90 / 0.96 within-depth top-1; C0 / hybrid errors in the noise family).
Risk classification at the fixed operating point (predicted "placed" at the horizon end vs true), wm2 test / I-1: WM3-B precision 0.36 / 0.41, recall 0.83 / 0.85, accuracy 0.84 / 0.85, AUROC of J for failure 0.915 [0.87, 0.95] / 0.922 [0.90, 0.95]; C0 recall 0.96 / 0.88 but precision 0.18 / 0.20 (it predicts failure far too often after 5–7 autoregressive blocks; ranking AUROC 0.90 / 0.89 is nevertheless good); MLP AUROC 0.90 / 0.92; action-blind AUROC 0.56 / 0.62. Calibration of p(placed) at the last block: ECE WM3-B 0.068 / 0.070 (Brier 0.064 / 0.069), MLP 0.055 / 0.062, C0 0.082 / 0.107. Pairwise accuracy by predicted score difference (WM3-B): < 0.01: 0.73 / 0.77; 0.01–0.03: 0.93 / 0.92; 0.03–0.1: 0.95 / 0.95; > 0.1: 0.98 / 0.99 — larger predicted differences are reliably correct.

## 4. Decision changes caused by yaw handling (inference only; Huber and rotation-vector heads NOT RUN — weights were not persisted and retraining is not allowed → PARTIAL for that item)
Feeding yaw = persistence into the rollout instead of the model's yaw: WM3-B changes the selected candidate in **77 % / 74 %** of the poses (C0: 55 % / 42 %), Kendall τ between score vectors 0.72 / 0.74 (C0 0.85 / 0.86), max |ΔJ| up to 0.32 / 0.51; but top-1 stays 1.00 / 0.98, regret difference −0.001 [−0.003, 0.001] / +0.001 [0.000, 0.003], pairwise accuracy 0.922 vs 0.905 (wm2 test) and 0.925 vs 0.917 (I-1). Reading: most candidates are outcome-equivalent so selection among them is unstable, and the change of selection hardly changes the true outcome; WM3-B's long rollouts depend on its own (hallucinated) yaw feedback, and removing it helps slightly.

## 5. Latency (one CPU thread; 25 candidates; median)
One block, batch 1: WM3-B 4.3 ms, C0 4.3 ms, hybrid 4.3 ms, MLP 0.5 ms, ridge 0.12 ms. A full 25-candidate ranking over ~6 blocks (batched): WM3-B 98 ms, C0 101 ms, hybrid 96 ms, MLP 13 ms, ridge 8 ms (the ≈ 0.1 s ranking fits a 5–10 Hz replanning loop).

## 6. Compute (cap 0.1 process CPU-h for the new offline evaluation, counting everything)
Sanity check 135.6 s + freeze ≈ 5 s + ranking run 85.0 s + task-relevance analysis 3.5 s + library tests ≈ 2 s ≈ **232 s = 0.064 CPU-h** (complete, no PARTIAL except the not-run Huber / rotation-vector items). Project ledger: measured since VM boot 2.76 CPU-h; estimated before boot 3.1–7.5 → used **5.9 – 10.3**, remaining **11.7 – 16.1** of 22.

## 7. Limits
Development only; privileged-state inputs and the true state at t0; scripted-expert candidates (12–29 % inside an ACT-like envelope); outcome ceiling (≈ 90 % success) hides top-1 differences; the scoring weights (0.05 m) are unit choices, not tuned and not sensitivity-tested; WM rollouts compound error over 5–7 blocks (C0 over-predicts failure); one decision time (t0 = 4); no closed-loop execution.
