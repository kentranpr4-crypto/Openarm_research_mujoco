# E1 / multi-seed / I-2 — plan, compute ledger, risks and recommendation (Phase 0; DRAFT; nothing is run)

Date: 2026-11-01. Supersedes the compute and sequencing parts of `2026-11-01-multiseed-i2-prereg-draft.md` (committed; left unchanged). Gate v3 is a draft (`2026-11-01-gate-v3-final-draft.md`); I-1 results are untouched.

## 1. Compute ledger (limit: 22 CPU-hours; `wm2_ceiling_cpu_hours` = 22 in `outputs/wm2/model_config.json`, repeated as "WM-3 total ceiling 22 CPU-h shared with earlier stages")
Convention: learned-model training/evaluation = wall time × 3 threads (the configured thread count) ÷ 3600; simulation reported separately (4 workers). Numbers are reconstructed from logs (`fit_seconds`, `total_wall_s`, stdout) — estimates, not metered.
| Item | Wall (s) | CPU-h |
|---|---|---|
| WM-2 dry run | 280 | 0.23 |
| WM-2 final learned run (ridge, 2 MLPs 118 s each, GRU 1 928 s, predictions) | 2 446 | 2.04 |
| WM-2 GRU smoke tests / diagnostics | ≈ 120 (est.) | 0.10 |
| WM-2.1 bit-exact reproduction (MLP, MLP-blind, GRU retrained) + audit scripts | 2 418 + ≈ 100 | 2.11 |
| WM3-B training (first run; per-epoch log lost) incl. ridge + crash | ≈ 2 160 | 1.80 |
| WM3-B reload run (predictions, latency) | 87 | 0.07 |
| WM3-B unit tests (3 runs incl. overfit 400 steps) | ≈ 450 (est.) | 0.40 |
| I-1 predictions (66 s + dry runs 3 × ≈ 100 s) and evaluator (≈ 3 runs × 150 s) | ≈ 800 | 0.67 |
| yaw audit / loss probe / OOF ridge probe / Gate v3 tests | ≈ 500 (est.) | 0.40 |
| **Learned-model ledger total** | | **≈ 7.8 CPU-h (range 7–8.5)** |
| Remaining under 22 CPU-h | | **≈ 14.2 (range 13.5–15)** |
| Simulation, separate (4 workers): WM-1 intervention 639 s, I-1 collection 369 s, contract-v2 replay ≈ 130 s | | ≈ 0.4 + 0.7 + 0.15 ≈ 1.3 |
Whether earlier ACT roll-out/training stages (E0, x-sweep, coverage) and simulation are charged to the same 22 h cannot be told from the repository; they are not included. **I ask you to confirm the accounting convention and whether simulation counts.**

## 2. Inference-only yaw persistence (no training; `yaw_persistence_feasibility.json`)
Variants of the frozen predictions: P0 = persistence yaw for every window; P2 = keep the model's yaw only when the input shows an interaction (box speed > 0.005 m/s, finger/palm contact, or a closing gripper command in the horizon), otherwise persistence. Box, joint, contact, event outputs are untouched.
| K = 20 yaw error (rad, group-macro) | I-1: WM3-B | I-1: C0 | development: WM3-B | development: C0 |
|---|---|---|---|---|
| model yaw | 0.0512 | 0.0458 | 0.0799 | 0.0680 |
| **P0 persistence yaw** | **0.0146** | 0.0146 | 0.0277 | 0.0277 |
| P2 interaction rule | 0.0434 | 0.0376 | 0.0625 | 0.0538 |
| ridge | 0.0147 | | 0.0313 | |
| on windows that truly rotate (|Δyaw| > 0.05; diagnostic) — model / P0 / ridge | 0.171 / 0.172 / 0.153 | 0.162 / 0.172 / 0.153 | 0.165 / 0.156 / 0.142 | 0.157 / 0.156 / 0.142 |
| static input subset (no interaction): model / P0 | 0.029 / 0.0002 | 0.030 / 0.0002 | 0.046 / 0.004 | 0.040 / 0.004 |
Findings: (i) P0 removes the hallucination at no cost (ratio 1 by construction; static error 0.0002 rad) and is **fully feasible**; (ii) the GRUs have essentially **no yaw skill** even on truly rotating windows (WM3-B ≈ persistence; C0 6 % better in I-1, 5 % worse in development) while ridge is 11 % better in both sets — so a yaw-aware model must beat **ridge and P0 on rotating windows**, not just fix the static case; (iii) the interaction rule P2 does not help — the spurious rotation also appears when the input shows interaction. Caveat: strata that use true |Δyaw| are diagnostic; I-1 is development now.

## 3. E1 — ablation matrix (NOT approved; one model family, WM3-B architecture, 20 epochs, same budget, seed-paired)
Three factors are separated; the package arm is labelled a package and supports no causal claim.
| Arm | Data / contract | Yaw handling | Loss | Isolates |
|---|---|---|---|---|
| B0 | v1 (frozen) | v1 | v1 | reference: existing WM3-B (seed 0, no training) + P0 post-hoc (no training) |
| A1 | **v1.1: tilt in radians** (exact repair, new dataset version) | v1 | v1 | tilt-unit repair only |
| A2 | v1 | v1 channel + geometric `yaw_step_valid` mask | masked Huber on the wrapped residual (δ 0.05 rad, robust scale) | yaw loss only |
| A3 | v2 inputs/targets (R6 + rotation-vector targets), tilt column left as in v1 (mixed) | rotation vector | v1 plain MSE (normalised, robust scale for rotvec only if required for convergence — logged) | representation only |
| A4 (package) | v2 full (tilt in rad, R6, rotvec) | masked | masked Huber + rotvec loss | whether the whole package reaches Gate v3 G7; **not a causal test** |
Interactions are not identified by one-factor-at-a-time; if A4 passes but no single arm does, a follow-up with pairs (A1+A2, A2+A3, A1+A3) is the only way to attribute it — decided after the result, not now.
Evaluation: Gate v3 G7 (yaw components accept legacy per-step yaw outputs; orientation components need rotation-vector outputs ⇒ NOT_EVALUABLE for B0/A1/A2), box/joint/contact errors (v2 gates inherited), P0 and ridge as yaw baselines, on the **development** windows and I-1 (development). **Never on I-2.**
## 3a. Cheaper first step — head-only probe E1-lite (needs approval; ≈ 0.2 CPU-h)
Freeze the WM3-B trunk (no checkpoint modification: features computed in a forward pass), train only a new yaw head on (a) the v1 yaw target with MSE (control, expected to reproduce the hallucination) and (b) masked Huber, 3 min each. It tests whether the loss alone can remove spurious rotation given the learned representation, before spending the full-arm budget. Limits: it cannot show that the trunk itself would have learned differently.

## 4. Compute, stop rules and uncertainty
| Stage | Runs | Wall | CPU-h |
|---|---|---|---|
| (optional) E1-lite head probes | 2 | ≈ 10 min | ≈ 0.2 |
| Data: v1.1 (arithmetic on existing arrays) + v2 replay of 4 696 episodes (new dataset version) | — | ≈ 22 min (4 workers) | ≈ 1.5 simulation |
| E1 stage 1: A1–A4, seed 0 | 4 × ≈ 36 min | ≈ 2.4 h (≈ 1.6 h with 2 parallel jobs, estimate) | 7.2 |
| E1 stage 2 (only arms passing the stage-1 rule): baseline retrained at seeds 1–2 + best arm at seeds 1–2 | 4 × ≈ 36 min | ≈ 2.4 h | 7.2 |
| **E1 total (stage 1 + 2)** | | | **≈ 14.4** — equals the whole remaining ledger |
| Paired multi-seed C0 vs WM3-B (8 new runs) | | ≈ 4.5 h | 13.5 — **does not fit** unless the ceiling is raised or E1 is cut |
Stop rules (fixed before any run): (1) per run, abort if the projected wall time > 45 min or any loss is non-finite; (2) stage 1 → stage 2 only for arms meeting ALL of: heading-valid yaw ratio ≤ 1.10 (or ≥ 40 % lower yaw error than B0) on development windows, box K = 20 error not worse than B0 by > 5 % (paired group bootstrap), joint error unchanged; (3) if no single-factor arm (A1–A3) and not A4 meets (2) ⇒ STOP, report "yaw-aware training did not help", keep P0 as the yaw output; (4) cumulative E1 compute hard stop at 12 CPU-h; (5) no hyperparameter search: the listed parameters (δ 0.05, clip 0.2, τ 0.2) are used once; (6) no repetition of the evaluation with other settings.
Uncertainty: seeds {0,1,2}; unit of analysis = initial-state group; paired seed-level differences; two-level bootstrap (groups, then seeds) and a t-interval (df = 2 is very wide — reported as such, not hidden); effects are reported with the within-seed sign pattern.

## 5. I-2 (design; NOT collected; new data, 100 groups)
- 100 new initial-state groups, namespace `eval_seed_label = 7002`, `rng = [20261101, 7002]`, reserved seeds 31–33 never used; Chebyshev distance ≥ 0.003 to all states used in the wm2 dataset, WM-1 manifests, ACT sweeps, **I-1** and I-2's own pilot namespace 7099; poses drawn before any model or prediction is consulted; no adaptive sampling.
- Stratified by initial-state regime: x-stratum (36 / 20 / 20 / 24) × initial-yaw tercile on [−0.20, 0.20] (equal counts within each x-stratum, ±1).
- Rotation/contact coverage: core 25 development variants + two exploratory extension families declared before collection — **lateral grasp offset** (±0.01, ±0.02, ±0.03 m along the TCP y-axis; expected to induce box yaw/tilt on closing) and **extra-deep depth** (+0.05, +0.06 m; more palm contact) — i.e., 33 variants × 100 = 3 300 episodes (≈ 1.1 GB, ≈ 17 min on 4 workers ≈ 1.1 CPU-h simulation). Extension families are outside Gate v3 families unless their true-effect eligibility holds. A sim-only feasibility pilot (≤ 60 episodes on throw-away poses from namespace 7099, no model involved) is required first to check IK feasibility and that the lateral family produces rotation; it needs approval and its data are never evaluated.
- Honest limit: standing-on-end and fast tumbling arise mostly in ACT policy roll-outs (115/122 events), not in scripted variants; the extension families may or may not reach those regimes. Gate v3 reports the support (NOT_EVALUABLE if < 10 groups / 200 windows) rather than assuming it.
- Freeze before collection (collector refuses otherwise): Gate v3 config + unit tests, model checkpoint hashes (all seeds/arms to be evaluated), contract-v2 builder, protocol document, manifest, evaluator.
- Independent from analysed data: I-1 is development; nothing from I-1/E1 development may tune I-2 settings.

## 6. Risk assessment — masking physical tumbling
| Risk | Evidence / likelihood | Impact | Mitigation |
|---|---|---|---|
| Real rotation is masked and the model never learns it | 3 of 19 physical rotations (16 %) have hnorm_min < 0.2; 66 of 1 836 masked steps (3.6 %) carry > 0.1 rad geodesic rotation | yaw supervision lost where the box rotates while nearly on its end | the mask applies to the **yaw** term only; the 3-D rotation-vector target (A3/A4) still supervises those steps; Gate v3 reports tumbling/singular separately with the quaternion geodesic metric; mask never used to hide results |
| Mask threshold sensitivity (0.1 misses 6 singular events, 0.3 flags the same as 0.2 on this sample) | τ chosen from geometry on 394 replayed episodes; full replay would settle | boundary steps mis-masked | report mask coverage per split; sensitivity of arms to τ ∈ {0.1, 0.3} only as a post-hoc diagnostic, not tuned |
| Mask uses future truth | target-time mask `yaw_step_valid`; inputs only see `heading_valid` of the current row (tested: no leakage) | none for inference, risk of misuse in evaluation | masks are loss weights / diagnostic strata only; Gate v3 states this explicitly |
| Safety-relevant failures (box tumbling) become invisible | they are rare in scripted data and frequent in ACT failures | a model could look fine while ignoring tumbling | keep geodesic/tilt supervised and reported; require orientation components once models output rotation vectors; plan ACT roll-out coverage separately |
| Train/test mismatch of mask (train masked, test scored on valid only) | by design the gated yaw metric excludes singular windows too | none beyond reporting | all-window numbers are reported next to the gated ones |

## 7. Recommendation — should E1 be trained?
**Yes, but in this order and with the stop rules above; not the full matrix first.**
1. **P0 (inference-only persistence yaw) is already feasible and is the honest default** for the yaw channel today; it costs nothing and removes the failure. Any learned yaw must beat P0 and ridge on rotating windows to add value — the evidence says the current GRUs add none.
2. **E1-lite head-only probe (≈ 0.2 CPU-h)** first: it tests the loss hypothesis (H1/H2) cheaply. If a masked-Huber head cannot even match P0 on static windows or ridge on rotating windows, the full arms are unlikely to rescue yaw skill from this input set.
3. **E1 stage 1 (A1–A4, seed 0, 7.2 CPU-h)** only if (2) is positive or ambiguous; it is the only way to attribute an effect to loss vs representation vs tilt repair. Data prerequisite: new dataset version (v1.1, v2) built by replay (≈ 1.5 simulation CPU-h).
4. Multi-seed C0 vs WM3-B (13.5 CPU-h) **does not fit** in the remaining ≈ 14 CPU-h together with E1; either raise the ceiling or run it on the adopted configuration only. The box/joint advantage of WM3-B is already tight within the I-1 sample (96 % of groups favour it; SD of paired difference 0.93 mm), so the multi-seed has lower information gain per CPU-h than E1-lite.
5. I-2 last, once, after freezing.
