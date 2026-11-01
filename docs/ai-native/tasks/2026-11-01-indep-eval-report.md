# Independent held-out evaluation of WM3-B / WM2-C0 / hybrid / ridge / MLP — REPORT (2026-11-01)

**Status: DONE (protocol executed as frozen; one frozen gate = FAIL, reported as FAIL).** Self-check by the same operator; not an external audit.
Privileged-state, simulation only, no vision, not Sim2Real. Single training seed per model.
Raw results: `outputs/indep/evaluation_independent.json` (not in git). Protocol: `2026-11-01-indep-eval-protocol.md`.

## 1. What is INDEPENDENT and what is DEVELOPMENT
| Item | Label |
|---|---|
| 50 new initial-state groups x 25 variants (1 250 episodes, 1 237 usable), `outputs/indep/*` | **INDEPENDENT** |
| `outputs/wm2*`, `outputs/wm3*`, WM-2/2.1/3B reports, Gate v1 and Gate v2 *development* verdicts | **DEVELOPMENT** (unchanged) |
| `outputs/indep_dry/*` (plumbing check on development windows) | DEVELOPMENT, not a result |

Freeze (`outputs/indep/protocol_freeze.json`, sha256 `4d0d43c7…f4af`) was written before any independent data existed (verified `independent_data_exist_at_freeze=false`). At evaluation: 0 frozen files changed, model checkpoint hashes match, closed-form baseline hashes match, window hashes match, 0 overlap with development groups (min Chebyshev distance to any previously used state 0.0035), 0 groups in two strata, forbidden seeds 31–33 unused.

## 2. Data audit
- Safety: executed arm rate ≤ 2.0 rad/s (ratio to limit 1.0000002), gripper ≤ 0.095 m/s (limit 0.2), 0 rate violations, 0 ctrlrange violations. Control variant success 50/50.
- 13 episodes failed planning (IK; all depth variants at 8 poses) and are excluded (1.0 %). Windows: 10 319 / 6 011 / 6 087 / 6 524 (active ≈ 99.8 %).
- Support: groups per stratum 18 / 10 / 10 / 12. Palm-contact positives: 123 windows in 20 groups (prevalence 1.0 %).

## 3. Primary pre-registered comparisons (K=20, 50 groups, paired group bootstrap)
| Comparison | Mean diff | 95 % CI | Bonferroni 98.33 % | Verdict |
|---|---|---|---|---|
| P1 box: WM3-B (3.20 mm) − WM2-C0 (3.83 mm) | −0.62 mm (−16 %) | [−0.91, −0.43] | [−1.00, −0.41] | SUPPORTED_LOWER_ERROR |
| P2 box: WM3-B − hybrid | identical to P1 by construction (hybrid keeps C0's box head) | | | not independent evidence |
| P3 joint: WM3-B (0.0015 rad) − WM2-C0 (0.0093 rad) | −0.0078 rad | [−0.0101, −0.0062] | | SUPPORTED_LOWER_ERROR (WM3-B joint = ridge) |

## 4. Errors (macro mean over groups, 95 % group-bootstrap CI in `evaluation_independent.json`)
ALL 50 groups, box position [mm]:
| Model | K=1 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| persistence | 2.95 | 13.60 | 25.75 | 46.05 |
| ridge | 1.08 | 5.79 | 9.15 | 13.86 |
| MLP (cond) | 0.86 | 2.61 | 3.86 | 6.02 |
| MLP (blind) | 0.87 | 3.65 | 5.30 | 8.24 |
| WM2-C0 | 0.73 | 1.72 | 2.53 | 3.83 |
| hybrid | 0.73 | 1.72 | 2.53 | 3.83 |
| **WM3-B** | **0.50** | **1.31** | **2.05** | **3.20** |

K=20 other metrics (ALL): joint [rad] persistence 0.1705, ridge 0.0015, MLP 0.0163, C0 0.0093, hybrid 0.0015, WM3-B 0.0015; a-offset [cm] ridge 0.750, MLP 0.422, C0 0.262, WM3-B 0.246; contact (std units) ridge 0.365, MLP 0.240, C0 0.130, WM3-B 0.122.
By stratum, K=20 box [mm], WM3-B vs C0 (paired 95 % CI of difference): test_id 2.22 vs 2.64 (−0.42 [−0.60,−0.29], 18 groups); low_heldout 2.93 vs 3.32 (−0.39 [−0.50,−0.28], 10); strict low OOD 4.35 vs 4.91 (−0.56 [−0.77,−0.32], 10); high OOD 3.94 vs 5.12 (−1.18 [−2.26,−0.51], 12). Per-stratum n is small: descriptive.
Per variant family (50 groups each, K=20 box): WM3-B lower than C0 in all six families (control 1.89/2.42, depth 3.86/4.65, timing 2.43/3.01, noise 4.75/5.34, speed 2.19/2.76, release_height 2.26/2.77); per-family paired CIs not pre-registered.

**Negative finding — box yaw (reported metric, no gate/threshold):** K=20 yaw error [rad]: persistence 0.0146, ridge 0.0147, MLP 0.0157, C0 0.0460, WM3-B 0.0515. Both GRU models predict yaw ~3x worse than "no change", and WM3-B is worse than C0 (higher error SUPPORTED in 3 of 4 strata). The same pattern exists in the development data (persistence 0.028, C0 0.068, WM3-B 0.080) so it is a model property, not a distribution-shift effect. Gates v1/v2 do not cover yaw, so all PASS verdicts are blind to it.

## 5. Gate v2 verdicts (frozen config, applied unchanged; separate from development verdicts)
| Model | G0 | G1 | G2 | G3 | G4 | G5 | G6 |
|---|---|---|---|---|---|---|---|
| WM3-B independent | PASS | PASS | PASS | **FAIL** | PASS | PASS | PASS |
| WM2-C0 independent | – | FAIL (joint) | PASS | **FAIL** | PASS | PASS | – |
| hybrid independent | – | PASS | PASS | **FAIL** | not run | PASS | – |
| WM3-B development (WM-3B pilot) | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

- **G3 FAIL cause (frozen rule):** close/test_id AUROC = 1.0000 (CI lower 1.0), but the rule requires margin ≥ 0.05 over the outcome-only baseline, whose AUROC is 0.969, so the margin is at most 0.031. The criterion is unsatisfiable when the reference baseline exceeds AUROC 0.95; the same FAIL applies to C0 and hybrid. Palm (pooled low sets) passes: AUROC 0.9989, CI lower 0.998, margin 0.273. This is reported as FAIL; it is a gate-design limitation, not evidence of a weak classifier, but I do not re-define the gate within this experiment.
- G1: box superiority over best baseline (MLP) 58 % [54, 61 %], a-offset 45 %, contact 59 %, joint = ridge. G2: R = 1.21 / 1.55 / 1.39 (box), 1.02–1.17 (a-offset), all ≤ 2.0, non-inferior on every OOD set. G4: beats best action-blind by 69 % [66, 71 %]; action-shuffle raises box error 2.22 → 57.9 mm (+2505 %). G5: close ECE 0.0068, Brier 0.0019; palm ECE 0.0028, Brier 0.0028. G6: median 3.70 ms (p95 6.3), 879 634 params + ridge block.
- Negative controls: persistence as candidate FAILS G1; label-shuffle AUROC 0.502.

## 6. Action effects (paired variants from the same initial state; 50 poses)
| Family (eligible?) | true |Δ| | WM3-B sign agr. | WM3-B magnitude ratio |pred|/|true| [95 % CI] | C0 magnitude ratio |
|---|---|---|---|---|
| depth (a-offset cm; yes) | 2.09 | 1.00 | 1.10 [1.07, 1.13] | 0.93 [0.88, 0.98] |
| speed (box xy mm; yes) | 38.7 | 1.00 | 1.01 [0.995, 1.03] | 0.98 |
| release_height (yes) | 4.6 | 1.00 | 1.07 [1.04, 1.12] | 1.17 (sign 0.95) |
| noise (box z mm; yes) | 21.1 | 0.994 | 0.97 [0.94, 0.99] | 0.98 |
| timing (NOT eligible: 17 % of effects ≥ τ) | 0.17 | 1.00 | 1.06 | 1.07 |
Direction is essentially perfect for both GRU models; WM3-B overestimates the depth effect by ~10 % and C0 underestimates it by ~7 %.

## 7. Classification / calibration (all strata pooled, 50 groups)
AUROC (prevalence): close 1.000 (41.5 %), palm 0.9989 (1.0 %), lifted 0.9996 (26.8 %), placed 0.9985 (17.2 %); ECE ≤ 0.0055; Brier close 0.0018 (outcome-only 0.0915), palm 0.0025 (prevalence-only 0.0102). C0/hybrid are within 0.0005 AUROC of WM3-B. These tasks are near ceiling; they do not discriminate between the models.

## 8. Ablations (test_id, K=20 box)
clean 2.22 mm; ridge-forecast zeroed 47.4 mm; ridge-forecast shuffled across windows 61.7 mm; action-shuffled 57.9 mm. The model depends strongly on both the ridge forecast and the action sequence. (The ridge-forecast sensitivity is a property of the architecture being fed the forecast, not a robustness result.)

## 9. Deviations and limitations
1. D-I1: the dry-run showed the logistic event heads of ridge/ridge_blind are not bit-reproducible (≤ 9e-4 in probability; LBFGS float32). All other outputs were bit-identical to the stored development predictions. Before the freeze, only the deterministic closed-form part (Wt, mu, sd) is hashed; logistic heads are excluded from the hash. Effect on conclusions: none expected (ridge event probabilities are not used in any primary comparison).
2. D-I2: 13 IK/plan failures (1 %) excluded; all in depth variants.
3. D-I3: frozen G0 item "labelled DEVELOPMENT" was satisfied by the label INDEPENDENT (declared in the protocol before collection).
4. D-I4: noise-variant AR(1) sequences are identical to development by variant definition (new initial states, not new noise draws). P2 duplicates P1 by construction.
5. D-I5 (carried over): WM3-B per-epoch training log lost (WM-3B D-1); ridge forecast is in-sample on the training windows.
6. Single seed per model: bootstrap covers initial-state sampling only, not training-seed variance. The 0.62 mm gain is a statement about this WM3-B checkpoint vs this C0 checkpoint.
7. Same operator wrote, ran and evaluated (processes separated, hashes frozen); no external review.
8. Scope: same task family, same variant families, sim only; no new tasks, no new action types, no vision.

## 10. AC
| AC | Result | Evidence |
|---|---|---|
| Freeze before collection | PASS | protocol_freeze.json, flag false at freeze |
| Manifest, no overlap, namespace 7001, no seeds 31–33 | PASS | manifest checks, dataset_audit |
| Actuator limits/action semantics checked | PASS | audit: 0 violations |
| No retraining / tuning / gate edits | PASS | frozen hashes unchanged at evaluation |
| Metrics requested in decision D | PASS (all computed) | evaluation_independent.json |
| Gate v2 independent verdict separate | PASS (reported) – content: G3 FAIL | §5 |
| Independent claims of WM3-B vs C0 | SUPPORTED for box/joint/contact; NOT supported for yaw | §3–4 |
| Independent audit by someone else | NOT_CHECKED | n/a |

Not committed: all `outputs/indep*`, `data/indep_*.py`, `eval/indep_*.py`, protocol and this report (await separate approval).
