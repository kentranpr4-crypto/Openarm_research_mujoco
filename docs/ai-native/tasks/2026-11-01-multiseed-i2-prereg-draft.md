# Multi-seed (paired) + I-2 independent evaluation — PREREGISTRATION DRAFT, budget and recommendation

Date: 2026-11-01. Status: **DRAFT; nothing is executed; nothing is frozen; no training, no collection.** Gate v3 is itself a draft (`2026-11-01-gate-v3-spec.md`). I-1 results stay as reported; I-1 is **development** from now on (it has been analysed in detail).

## 1. Questions the experiments answer
Q1 (multi-seed): is WM3-B's box/joint/contact advantage over WM2-C0 (−0.62 mm, −16 % on I-1, one seed) larger than training-seed variance?
Q2 (yaw mechanism, optional E1): does a yaw-loss intervention (mask singular steps, clip/robust loss) remove the spurious rotation without losing the other channels? (root-cause report, hypotheses H1/H2)
Q3 (I-2): do the conclusions hold on initial states that no earlier analysis has seen?

## 2. Multi-seed design (paired)
- Models/configs: WM2-C0 (`outputs/wm2/model_config.json`, GRU, 20 epochs, AdamW 1e-3 cosine, batch 128, best-val selection) and WM3-B (`outputs/wm3/model_config_wm3b.json`), **configs, hash-frozen, unchanged**. Same wm2 train/val windows (hashes in `outputs/wm3/freeze.json`), same normaliser statistics (train-only), same deterministic ridge, same budgets (≤ 45 min per run guard, 3 threads).
- Seeds {0,1,2,3,4}. A seed fixes torch initialisation and the shuffling generator. Seed 0 = the existing frozen checkpoints (`gru_c0.pt` reproduced bit-exactly in WM-2.1; `wm3b.pt`), **not retrained**; seeds 1–4 are new: 4 × 2 = 8 trainings. Pairing = same seed index ⇒ same data order and same train/val data; initialisation streams cannot be matched across architectures (stated limitation). Hybrid = deterministic ridge + C0 per seed (no extra training; not independent of C0 for box/contact).
- Missing per-epoch log of the seed-0 WM3-B is a known limitation (D-1); new runs log every epoch.
- Evaluation: with the frozen Gate v3 (and v2 gates) on (a) development splits + I-1 (exploratory) and (b) I-2 (confirmatory, single look).
- Unit of analysis = initial-state group. Per model-seed: group-macro errors (as in the gates). Seed-level paired difference d_s = E(WM3-B, s) − E(C0, s).

## 3. Preregistered success criteria and uncertainty analysis (primary endpoint: box error at K = 20 on I-2)
- Uncertainty: (i) two-level paired bootstrap — resample groups, then seeds (10 000, seed 0) — of mean(d_s); (ii) t-interval on the five d_s (df = 4); (iii) report seed SD of each model and of d_s.
- S1 "WM3-B is lower-error than C0": d_s < 0 in 5/5 seeds AND both intervals have upper bound < 0.
- S2 "practically relevant": additionally the upper bound of the two-level interval < −0.30 mm (≈ 8 % of C0 error; minimum effect of interest fixed before seeing I-2).
- S3 joint and contact: same rule with joint error and contact_std (secondary).
- S4 orientation: G7 ratio and static hallucination reported per seed (gate verdicts per seed; fraction of seeds PASS); no averaging of verdicts.
- Anything failing S1 is reported as "not established"; no re-run, no new seeds added after the freeze, no switch of endpoint. Staging: seeds 0–2 first (seeds 1–2 new); if the seed SD of d_s makes S1 unreachable (t half-width > |mean d|), stop and report; otherwise run seeds 3–4. The stopping rule is fixed here.
- Multiplicity: one primary endpoint; secondary endpoints are descriptive.

## 4. Optional E1 — yaw-mechanism intervention (separate approval; run BEFORE a final multi-seed if approved)
Arm A (WM3-B architecture, everything else identical): yaw loss on per-step relative yaw with (a) steps flagged singular (body x-axis vertical; needs a flag, see §6) masked out, (b) clipping at ±0.2 rad (Huber), (c) robust scale. Seeds {0,1,2}; evaluated on development splits and I-1 only (**never on I-2**). Decision rule fixed in advance: adopt only if the yaw ratio (G7) falls ≤ 1.10 on development data and box K = 20 error does not worsen by > 5 %. The adopted configuration is then frozen and enters the multi-seed (replacing WM3-B) — otherwise the multi-seed uses the existing WM3-B.

## 5. I-2 protocol (to be frozen before ANY training of new seeds and before collection)
- ≈ 100 new initial-state groups x 25 core variants (= 2 500 episodes, ≈ 0.8 GB, ≈ 12–13 min on 4 workers). Namespace `eval_seed_label = 7002`, `rng = [20261101, 7002]`; reserved seeds 31–33 never used; poses drawn before any model/prediction is looked at; no adaptive sampling.
- Stratification by initial-state regime (fixed counts): x-stratum (36 `test_id` [0.38–0.45], 20 `test_low_heldout` [0.36–0.38], 20 `test_ood_low_strict` [0.34–0.36], 24 `test_ood_high` [0.4501–0.48]) x initial-yaw bin (three equal-width bins on [−0.20, 0.20]) with equal counts per yaw bin within each x-stratum (±1). Contact regime is not selectable before simulation without using a model; it is covered through the registered variants (depth, release) and **reported by outcome stratum** (palm contact yes/no, grasp failure yes/no).
- Optional extension variants (exploratory, outside the gated families unless their true-effect eligibility holds): depth +0.05 / +0.06 m and close-timing ±15 steps, subject to a planner-only IK feasibility pre-check (no episodes, no models). Decision requested; default = none.
- Non-overlap: Chebyshev distance ≥ 0.003 (x, y, yaw) to all previously used states **including the 50 groups of I-1**, the wm2 dataset, the WM-1 manifests and every ACT/sweep episode; the min distance is recorded.
- Hard stops before collection: protocol document, manifest, Gate v3 config + unit-test results, model checkpoint hashes (all seeds), code hashes, evaluator hash written to `protocol_freeze.json`; the collector refuses to start otherwise (as in I-1). Evaluation is a single look by the frozen evaluator; FAIL is reported as FAIL.
- Known coverage gap: box tumbling / standing-on-end events came almost exclusively from ACT roll-out episodes (115/122 events), not from scripted variants (I-1: 8 event windows in 2 groups). I-2 with scripted variants will again exercise the orientation gate mainly through its non-event windows; true tumble coverage would need ACT-policy roll-outs (separate decision, cost not estimated).

## 6. Prerequisites that are NOT yet done (each needs approval)
1. Data contract fix: single-unit tilt (radians) and an explicit "body standing on end" flag in future datasets (the frozen wm2 dataset cannot be changed; a derived v1.1 would need a replay of the 4 696 episodes, ≈ minutes of simulation, no training).
2. Gate v3 decisions listed in the spec (palm hard subset, static-threshold sensitivity).
3. Budget approval below.

## 7. Compute budget (measured from existing logs; 4 cores available)
| Item | Wall | CPU-h (3 threads/run, 4 workers for sim) |
|---|---|---|
| WM3-B training, 1 seed (measured) | 2 137 s | ≈ 1.8 |
| WM2-C0 training, 1 seed (measured) | 1 928 s | ≈ 1.6 |
| Multi-seed, seeds 1–4 x 2 models = 8 runs | ≈ 4.5 h sequential (≈ 3–3.5 h with 2 parallel jobs, estimate) | ≈ 13.5 |
| of which stage 1 (seeds 1–2: 4 runs) | ≈ 2.3 h | ≈ 6.8 |
| E1 yaw arm, WM3-B x 3 seeds | ≈ 1.8 h | ≈ 5.3 |
| I-2 collection (2 500 episodes, 4 workers; I-1: 1 250 in 369 s) | ≈ 12.5 min | ≈ 0.8 |
| I-2 build + prediction + evaluation per model-seed | ≈ 3 min | ≈ 0.15 |
| Whole plan with E1 | ≈ 7–8 h wall | ≈ 20 |
Note: the WM-3 note in the model config states a 22 CPU-h ceiling "shared with earlier stages"; I have not recomputed what was already used, so the headroom must be confirmed before approval.

## 8. Recommendation — highest information gain per cost
1. **Stage 0 (no training, ≈ 1 h of simulation/compute): fix the data contract and freeze Gate v3 + I-2 manifest.** It is cheap and everything later depends on it; it also decides the palm-gate question.
2. **E1 yaw-mechanism experiment (≈ 5.3 CPU-h).** It is the only experiment that tests the causal hypothesis behind the one clear failure (yaw), and its outcome decides which architecture enters the multi-seed; running the multi-seed first would likely be repeated.
3. **Paired multi-seed on the adopted configuration, stage 1 then stage 2 (≈ 13.5 CPU-h).** It converts the single-seed 0.62 mm into a defensible claim.
4. **I-2 once, last** (single look), because it is only valid if nothing is tuned on it.
If you want a single experiment only, E1 (seeds 0–2) has the highest expected information gain: the box/joint advantage already has a tight within-sample CI (SD of per-group paired differences 0.93 mm; 96 % of groups favour WM3-B) whereas the yaw failure is unexplained.
