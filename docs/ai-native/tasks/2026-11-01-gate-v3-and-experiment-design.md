# Gate v3 + future experiment design — PROPOSAL (prospective only; nothing is run, nothing is retro-applied)

Date: 2026-11-01. Status: **DRAFT for review.** Gate v3 is NOT frozen. It will never be applied to the independent evaluation of 2026-11-01 (G3 FAIL stays FAIL; yaw 0.0515 vs 0.0146 stays as reported) nor to any development result. Evidence used for design: the audits in `2026-11-01-yaw-audit.md` and the planning statistics in `outputs/yaw_audit/oof_ridge_probe.json`. Because the 2026-11-01 independent set has now been analysed in detail, **it is a development set from this point on**; confirmation needs new data (§5).

## 0. Design principles (anti-tailoring)
1. A criterion's *form and thresholds* must be justified by (a) physical or task meaning, or (b) properties of the reference baselines / data (not of the candidate), and be fixed before any new training/evaluation.
2. Every gate gets: a negative control that must FAIL, an oracle control that must PASS, and a synthetic unit test, run before freeze.
3. Where the old criterion was unsatisfiable (ceiling) the new one must be *satisfiable by construction* by an oracle, and must still reject the baselines it is meant to beat.
4. Dependent comparisons are declared up front (hybrid shares C0's box/contact heads ⇒ not independent evidence for box endpoints).

## 1. G3 v3 — event discrimination when the reference is near ceiling
Problem: v2 requires `AUROC − AUROC_outcome_only ≥ 0.05`. With outcome-only AUROC 0.969 (close, independent) the maximum possible margin is 0.031, so no model — not even an oracle — can pass.
Proposal (decided by the **reference's** headroom, a data/baseline property):
- Let `h = 1 − AUROC_ref` (headroom). If `h ≥ 0.10`: keep the v2 margin rule (margin ≥ 0.05, CI lower bound ≥ 0.70, AUROC ≥ 0.80).
- If `h < 0.10` (ceiling regime): replace the margin by **error-reduction** rules, all with group-bootstrap lower bounds: (i) Brier skill vs outcome-only `BSS = 1 − Brier/Brier_ref ≥ 0.50` (lower bound > 0.30); (ii) absolute `AUROC ≥ 0.98`, CI lower ≥ 0.97; (iii) ECE ≤ 0.05 (G5 unchanged).
- Add a **hard subset**, defined from the reference only: windows where the outcome-only probability lies in [0.2, 0.8] (its uncertain region); require AUROC ≥ 0.90 there when ≥ 30 pos/neg windows in ≥ 10 groups, else NOT_EVALUABLE. This is where discrimination beyond the trivial reference can be shown.
- Palm keeps the pooled-sets rule; prevalence is ~1 %, so the future dataset must provide more palm positives (§5) or palm stays "evaluable with wide CI".
Controls: outcome-only as candidate must FAIL (BSS = 0); label-shuffle must FAIL; oracle (true labels with small noise) must PASS; checked on synthetic data and on **development** predictions of non-candidate baselines only.

## 2. Orientation gate (new G7) — yaw (and tilt) as angular quantities
- Metric: relative angular displacement `θ_K = wrap(yaw_{t+K} − yaw_t)`; error `|wrap(θ̂_K − θ_K)|` at K = 5, 10, 20, per-group mean → macro mean. Tumbling windows (any true per-step |Δyaw| > π/2) are reported separately and excluded from the gated mean, with their count disclosed.
- Baseline-relative criterion (not an absolute number): `E_yaw(model) ≤ 1.10 × min(E_yaw(persistence), E_yaw(ridge))` at K = 20 with group-bootstrap **upper** bound of the ratio ≤ 1.25; rationale: a world model must not predict rotation worse than "no rotation" by more than a 10 % tolerance, because the true rotation is small and weakly predictable.
- Static-window hallucination: on windows where the true |θ_20| < 0.003 rad, mean `|θ̂_20|` ≤ 0.01 rad (≈ 0.57°, below the sensor-relevant scale for the task).
- Tilt: same ratio rule, reported as secondary until its instability (R² sign flips between sets) is understood.
Controls: persistence passes trivially (ratio 1); a noisy-rotation predictor (std 2× truth) must FAIL; oracle passes.
Existing models would fail this gate (WM3-B/C0 ratio ≈ 3): this is intended and is not tuned.

## 3. Multi-seed design (NOT approved to run)
- Models: WM2-C0 architecture and WM3-B, **same frozen configs** (hashes), seeds {0,1,2,3,4}; seed s fixes data-order generator and dataset; initialisation streams differ by architecture (cannot be paired exactly) — pairing is by seed index and data order, stated as such.
- Cost estimate from the logs: WM3-B 36 min per seed (3 threads), C0 32 min (1928 s) → ≈ 5.7 h CPU sequentially for 5 seeds × 2 models; ≈ 2 h with 3 parallel jobs on 4 cores. Needs separate approval and budget.
- Analysis: per-seed group-macro error; primary endpoint box K = 20 and (new) yaw ratio; two-level bootstrap (resample groups, then seeds) and a t-interval on the 5 paired seed differences (df = 4); decision rule fixed in advance: "WM3-B better" only if the difference is negative in 5/5 seeds and both intervals exclude 0. Report seed SD of each model.
- Staged: run seeds 0–2 first, estimate seed SD, extend to 5 only if the SD makes the endpoint resolvable; the stopping rule is fixed before seed 0.
- Hybrid is not a separate model (deterministic combination of C0 + ridge): evaluated per C0 seed, dependent on C0 for box/contact.

## 4. Ridge-forecast train/test mismatch (out-of-fold design)
Measured (closed-form probe): ridge joint error 0.00159 in-sample vs 0.00164 out-of-fold vs 0.00169 val vs 0.00151 independent; mismatch ≈ 4–7 %. Therefore OOF is a **sensitivity analysis**, not a prerequisite:
- Primary (unchanged): GRU trained on in-sample forecasts (as WM3-B).
- Sensitivity arm WM3-B-OOF: 5 group folds over the train groups (fixed fold assignment, seed 0); GRU trained on out-of-fold forecasts; at inference the full-train ridge; also report inference with the fold-ensemble mean. Only worth running inside the multi-seed experiment (same seeds), judged by paired differences.
- Unit tests: fold groups disjoint; OOF forecast never uses its own windows; train/inference path identical except the ridge weights.

## 5. New independent evaluation (I-2)
- New namespace: `eval_seed_label = 7002`, `rng = [20261101, 7002]` (or the date of freeze); min Chebyshev distance ≥ 0.003 to **all** earlier states **including the 50 groups of I-1** (which is now development).
- Size from planning statistics (I-1: SD of per-group paired box difference = 0.927 mm): groups needed for a 95 % CI half-width of 0.3 / 0.2 / 0.15 mm = 37 / 83 / 147; proposal ≥ 100 groups × 25 variants = 2 500 episodes (~13 min of simulation, ~0.8 GB (I-1: 402 MB for 1 250 episodes)) for half-width ≈ 0.18 mm; lower if the multi-seed design is run first (seed variance also enters).
- Stratification as I-1; additionally enrich palm-contact support (extra depth levels beyond the registered ±, e.g. the deepest variants that remain IK-feasible) and keep explicit tumbling/failure coverage so orientation gate has non-trivial windows. Variant definitions must be frozen before collection; any new variant family is declared as such.
- Procedure: freeze (models, Gate v3 config, protocol, code, manifest hashes) **before** any training or collection; single evaluation; no retuning; FAIL reported as FAIL.

## 6. Decisions requested
1. Approve Gate v3 elements §1–§2 as the basis for a frozen config (with unit tests) — or request changes (thresholds 0.50 / 0.98 / 1.10 / 0.01 are first proposals).
2. Approve (separately) the multi-seed run (§3) with or without the OOF sensitivity arm.
3. Approve (separately) I-2 data collection (§5) after Gate v3 is frozen.
Nothing here has been executed beyond diagnostics; no commit of these documents has been made.
