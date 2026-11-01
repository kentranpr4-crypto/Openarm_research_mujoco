# Independent held-out evaluation — PROTOCOL (pre-registered, frozen before data collection)

Date: 2026-11-01. Scope: privileged-state world model (box/contact ground truth from MuJoCo; no vision; NOT Sim2Real).
Status of everything produced under this protocol: **INDEPENDENT** (new simulator initial states). Everything in `outputs/wm2*`, `outputs/wm3*` is **DEVELOPMENT**.

## 1. What is frozen before collection (hashes in `outputs/indep/protocol_freeze.json`)
- Models (no re-training, no tuning): WM2-C0 (`outputs/wm2_1/models/gru_c0.pt`), WM3-B (`outputs/wm3/models/wm3b.pt`), MLP cond / blind (`outputs/wm2_1/models/mlp*.pt`); closed-form ridge, ridge_blind, outcome-only logistic re-fitted deterministically from frozen train windows and verified against stored weight hashes. Post-hoc hybrid = ridge `Y_s` + WM2-C0 `Y_b, Y_c, Y_e` (no new fitting).
- Gate v2 config (`outputs/wm3/gates_v2_config.json`, sha256 `f6c32391…0627`), normaliser statistics (train-only, from frozen wm2 windows), model configs.
- Code: `eval/wm3_gates.py`, `wm3b_model.py`, `wm2_models.py`, `wm2_common.py`, `wm3_data.py`, `indep_predict.py`, `indep_evaluate.py`, `data/indep_*.py`, `data/wm1_intervene.py`, `data/wm1_sim.py`, this protocol.
- Manifest (`outputs/indep/manifest.json`): 50 new initial-state groups x 25 development variants = 1 250 episodes (cap 1 250).

## 2. Initial states and actions
- Separate seed namespace: `eval_seed_label = 7001`, `rng = default_rng([20261101, 7001])`. Reserved seeds 31–33 never used.
- Strata = same x-ranges / y / yaw ranges as the development intervention manifest: `test_id` (x 0.38–0.45, 18 groups), `test_low_heldout` (0.36–0.38, 10), `test_ood_low_strict` (0.34–0.36, 10), `test_ood_high` (0.4501–0.48, 12). Poses are drawn without looking at any model or prediction.
- A candidate pose is rejected if its Chebyshev distance (x[m], y[m], yaw[rad]) to ANY previously used initial state (wm2 dataset groups, WM-1 intervention manifest, every ACT/sweep episode) is < 0.003. Recorded minimum distance is in the manifest.
- 25 variants per group = the development variants, unchanged (control; depth x5; timing x6; noise x6 [AR(1) draws identical to development by definition of the variants]; speed x4; release_height x3). All 25 variants of a group start from the same simulator initial state, which is the paired-intervention set used for causal action-effect measurement.
- Safety: executed actions pass through the same `SafetyLimits.clip_and_slew` (ctrlrange; 2.0 rad/s arm; 0.2 m/s gripper). After collection the audit recomputes max executed rate/ctrlrange violations (must be 0).
- Episodes with IK/plan failure or non-finite states are excluded from windows and counted.

## 3. Windows
Contract v1 unchanged (H=4, K=20, stride 5, t in [H, T-1-K]); `active` subset, at-risk subsets and phase-aligned family windows exactly as in `eval/wm3_gates.py` / `eval/wm3_data.py`.

## 4. Statistical analysis (fixed)
- Sampling unit = **initial-state group**. Window errors -> per-group mean -> macro mean over groups. Overlapping windows are never treated as independent samples.
- CI: percentile bootstrap over groups, 10 000 resamples, seed 0, 95 % (AUROC/effect statistics: 1 000–2 000 resamples over groups/poses). "ALL" = the 50 groups pooled with equal group weight; per-stratum results are reported separately and are low-powered (10–18 groups).
- **Pre-registered primary comparisons** (K=20, ALL groups, paired bootstrap on per-group differences, same resamples): P1 box error WM3-B − WM2-C0; P2 box error WM3-B − hybrid; P3 joint error WM3-B − WM2-C0. Verdict per comparison: SUPPORTED_LOWER_ERROR if the 95 % CI upper bound < 0, SUPPORTED_HIGHER_ERROR if lower bound > 0, else INCONCLUSIVE. A Bonferroni (3 comparisons, 98.33 %) interval is reported next to it. All other comparisons are secondary/descriptive.
- Reported metrics: horizons 1/5/10/20 (and curves 1–20); box position (mm), box yaw (rad), joint (rad), a-offset (cm), contact (std-normalised); close/palm/lifted/placed AUROC, prevalence, Brier, ECE (15 bins); paired action-effect direction (sign agreement), Spearman, magnitude ratio |pred|/|true| and regression slope per family; per-variant-family errors (NOT_EVALUABLE if < 10 groups); latency (same machine, 1 thread, batch 1).
- **Gate v2** (frozen config, unchanged thresholds) is evaluated separately from the development verdicts. Adaptation declared here: the frozen G0 item "evaluation labelled DEVELOPMENT" is satisfied by the correct label INDEPENDENT; the dataset-independence checks (0 overlap with development groups, audit clean, hashes) are reported as extra integrity items.
- Models: WM2-C0, WM3-B, hybrid, ridge, MLP cond/blind (+ persistence, const-velocity, ridge_blind, outcome-only as references/pool).

## 5. Rules
1. No result from this set may be used to change a model, gate, normaliser, threshold or hyperparameter, nor to repeat the test with another model in the same experiment.
2. A FAIL is reported as FAIL; follow-up research is proposed separately.
3. No training, no WM3-A/C/D, no multi-seed, no ACT+WM, no visual encoder.
4. Artifacts of this evaluation are NOT committed without separate approval.
5. Deviations discovered after the freeze are logged as deviations (what, why, effect); frozen files are not edited silently.

## 6. Known limitations (declared in advance)
- 1 training seed per model => model-to-model differences also contain unmeasured seed variance; the bootstrap covers initial-state sampling only.
- Sim-only, privileged state, scripted-expert-derived variants; the test measures generalisation to new initial states of the same task family and action-variant families, not new tasks or new action types.
- Per-stratum sample sizes are small (10–18 groups); strata results are descriptive.
- Noise variants reuse the development AR(1) noise sequences (variant definition), so the "noise" family tests new initial states, not new noise draws.
