# Training-coverage experiment — results
**Status: PARTIAL — STOPPED BY PRE-REGISTERED SPEED GUARD (closed by user decision, 2026-10-31).**
Date 2026-10-30 · pre-registration and deviations: `2026-10-30-cov-preregistration.md` (config sha256 `f01f4733…aef3`, unchanged) · scripts `data/collect_demos_region.py`, `data/build_cov_dataset.py`, `data/validate_merge.py`, `eval/cov_run.sh`, `eval/cov_replay.py`, `eval/cov_report.py` · results `outputs/cov/cov_results.json`, `cov_tables.md`, plots `Y1–Y4` (copies in `assets/cov/`).
Run status: **EXT‑C0 seed 3 COMPLETE · EXT‑S1 seed 3 COMPLETE (one approved restart) · VOL‑C0 seed 3 COMPLETE · VOL‑S1 seed 3 STOPPED (speed guard failed twice: 1.112 and 1.135 s/step) · EXT seed 4 NOT RUN (removed from scope, D1).** All verdicts are single‑seed; the full C1–C8 set is **not** declared complete. No policy is adopted as a final/production model; the guard was not relaxed and no further training was run.

## Gates and bookkeeping
| item | result |
|---|---|
| Data | EXT 30/30 and VOL 30/30 expert successes; sync validation (frame counts, state/action/images vs raw, first 60 episodes bit‑identical to V1, split, train‑only stats) passed; unit tests T1–T13/TA–TC pass; 200‑step smokes pass |
| Training health | EXT‑C0, EXT‑S1, VOL‑C0: 5400 steps, no NaN, loss EMA 0.037 / 0.038 / 0.034 (baselines 0.033–0.038), final val L1 0.027 / 0.027 / 0.024 (baselines 0.022–0.030), open‑loop MAE 0.0058 / 0.0060 / 0.0049 rad (baselines 0.0051–0.0052 C0, 0.0060–0.0063 S1) |
| Speed guard | ext_c0 1.030; ext_s1 first attempt 1.226 (my own concurrent replay → killed; restarted once with identical args after your approval, environment fingerprint saved) → 1.043 PASS; vol_c0 1.082 PASS; **vol_s1 1.112 FAIL (stopped)**. Steady training speed 1.08–1.10 s/step (baselines 0.75–0.79 on a faster VM) |
| Wall‑clock | config frozen 04:05 UTC → 13:32 UTC = **9.5 h** used. If you approve a single restart of VOL‑S1: ≈ 97 min training + ≈ 43 min evaluation = **≈ 2.3 h** more ⇒ ≈ 11.8 h + ≈ 0.5 h analysis ≈ 12.3 h total, inside the 14 h stop rule (13.9 h in your earlier message was the *remaining* time for six runs; with four runs the remaining time was ≈ 8.9 h) |
| Baselines | seed‑0 checkpoints re‑hashed unchanged (`63b20d94…`, `c7f95aff…`); no data, checkpoint or log deleted (partial/guard‑trip runs renamed); test seeds 31–33 not used |
| Side effect to disclose | running `policies/test_norm_contract.py` rewrites `outputs/e6b/unit_tests.json` (same tests, same verdicts) |

## Original vs EXT vs VOL (success %; baseline = frozen seeds 0/1/2 with the three values; EXT/VOL one seed)
| | C0 original | C0 EXT | C0 VOL | S1 original | S1 EXT | S1 VOL |
|---|---|---|---|---|---|---|
| **Boundary x 0.34–0.37** (16 poses) | 62 / 50 / 44 (mean 52) | **100** | **19** | 12 / 25 / 44 (mean 27) | **81** | not run |
| – far x 0.34–0.35 | 25 / 12 / 25 | 100 | 0 | 12 / 25 / 38 | 62 | – |
| – near x 0.36–0.37 | 100 / 88 / 62 | 100 | 38 | 12 / 25 / 50 | 100 | – |
| **In original support x 0.38–0.45** (32 poses) | 94 / 100 / 100 | 88 | 91 | 78 / 100 / 100 | 97 | – |
| NOM seed 21 ids 0–19, unperturbed | 19 / 20 / 18 of 20 | 17 | 18 | 16 / 20 / 20 | 20 | – |
| Upper x 0.46–0.48 (12 poses) | 67 / 58 / 100 | 92 | 92 | 100 / 67 / 67 | 75 | – |
| **Grasp-depth error at x 0.34–0.37**, \|a_close − expert\| [cm] | 1.8 / 2.5 / 2.0 | **0.49** | 2.54 | 3.1 / 3.2 / 2.4 | **0.51** | – |
| Grasp error signed (a_close − expert) | −1.8 / −2.5 / −2.0 | +0.46 | −2.37 | −3.1 / −3.2 / −2.4 | −0.23 | – |
| Box–palm contact, x 0.34–0.37 [%] | 25 / 44 / 19 | 0 | 31 | 75 / 75 / 31 | 6 | – |
| Cluster‑flagged failures (of 16) | 6 / 8 / 6 | 0 | 0 | 14 / 11 / 5 | 0 | – |
| Final placement error, x 0.34–0.37 [cm] | 2.4 / 3.0 / 2.8 | 0.55 | 15.8 | 3.4 / 3.3 / 3.8 | 2.0 | – |
| Box slip a_release − a_close [cm] | 1.4 / 0.5 / 0.7 | 0.1 | −5.1 | 0.2 / 0.9 / 1.7 | 1.4 | – |
| **Shortcut index** right_mean / jitter_small | 0.00 ×3 / 0.00 ×3 | 0.06 / 0.00 | 0.00 / 0.00 | 1.06, 1.0, 1.0 / 1.0 ×3 | **1.0 / 1.0** | – |
| Latency per query, median [ms] (cpu, 1 thread; this VM is slower) | 70 | 86 | 82 | 70 | 80 | – |
| Training time [min], s/step | 68–87 min, 0.76–0.97 | 97 min, 1.08 | 99 min, 1.10 | 68–184 min | 98 min, 1.09 | stopped |

## Registered hypotheses (single seed each; "SUPPORTED" is not "confirmed")
| Hyp | Result | Numbers |
|---|---|---|
| C1 boundary success | **SUPPORTED (single seed)**, both arms | EXT‑C0 100 % vs baseline 52 %; EXT‑S1 81 % vs 27 % (≥ +20 pp and ≥ 75 %; each EXT seed ≥ baseline + 10) |
| C2 volume control | **C0 only: coverage‑specific (single seed, exploratory)**; S1 **not testable** | VOL‑C0 19 % ≤ baseline + 10 (52 + 10); VOL gives no gain and is even worse than the baseline seeds |
| C3 grasp position | **SUPPORTED (single seed)**, both arms | \|dev\| 0.49 cm (C0), 0.51 cm (S1) ≤ 1.0 and improved by 1.6 / 2.4 cm vs baseline |
| C4 S1 − C0 trade‑off (1 pair, descriptive) | boundary deficit **persists but smaller** | x 0.34–0.37: −19 pp (baseline −25 pp); in support +9 pp (baseline −5 pp) |
| C5 no in‑support regression | **C0 fails narrowly, S1 holds** | EXT‑C0 88 % vs 98 % (−10.4 pp, margin −10); EXT‑S1 97 % vs 93 % |
| C6 shortcut retained | **SUPPORTED (single seed)** | EXT‑S1 SI 1.0/1.0; EXT‑C0 SI 0.06/0.00 |
| C7 effect is local (x ≥ 0.46) | **C0 fails, S1 holds** — but VOL‑C0 shows the same upper‑side change (92 %), so it is not caused by the added low‑x coverage | EXT‑C0 92 % vs 75 %; EXT‑S1 75 % vs 78 % |
| C8 latency (descriptive) | within ±25 % | ratios 1.23 / 1.15 / 1.17 (slower VM) |
| Overall rule | **not issued** (needs C2 for both arms) | provisional reading below |

## Uncertainty (what these numbers can and cannot support)
- **Across training seeds:** EXT/VOL have one seed per arm, so no seed‑to‑seed spread exists for them; the comparator (baseline) shows the spread: LOW stratum 44–62 % (C0) and 12–44 % (S1). The comparison EXT/VOL vs baseline is between independent seeds, unpaired; the C0/S1 comparison inside EXT is one pair. The VOL‑C0 result (19 %, worse than any baseline C0 seed) shows that boundary behaviour outside the support varies strongly between trainings, which is exactly why a single seed cannot be generalised.
- **Across pose blocks (4 blocks, y/yaw fixed per block):** LOW stratum per block: EXT‑C0 100/100/100/100; EXT‑S1 100/75/75/75 (all misses at x = 0.34; block bootstrap CI [75, 100]); VOL‑C0 25/0/25/25. In‑support: EXT‑C0 75/100/100/75 (the 4 misses are x = 0.44–0.45 in blocks 0 and 3, i.e. yaw 0.0 and +0.2), EXT‑S1 88/100/100/100, VOL‑C0 88/100/88/88. Per‑x cells are 4 poses; differences of one pose are 25 pp.
- **Replay:** 5/180 replays diverge (> 5 mm box error); live success/final‑error unaffected; replay‑derived a_close/slip exclude them.
## Coverage effect — what the evidence says (provisional)
1. **Coverage restores grasp positioning at the old boundary.** With demos at x 0.34–0.38 the policy's grasp depth stays at the expert level (≈ −0.5…−1.8 cm) all the way down to x = 0.34 for both arms (`Y1` bottom), whereas the originals and the VOL control go to −3…−5 cm (`Y1`). The palm contact and the x ≈ −3.5 cm cluster disappear (palm 0/6 %, cluster 0/0 of 16). Adding the same amount of data in the original region (VOL‑C0) did **not** help and did not change the grasp‑depth curve (a_close at 0.34: −4.8 cm; baseline −4.2).
2. **Success at the boundary follows:** C0 100 % (was 52), S1 81 % (was 27); the remaining S1 failures are 3 of 4 poses at the very edge x = 0.34 (final error 3.7–10.5 cm), i.e. S1's breakdown point sits at the edge of its support whereas C0's does not — consistent with the earlier "S1 is shifted ≈ 1 cm inward" observation (exploratory).
3. **VOL‑C0's failure mode differs** (12 of 13 low‑x failures are `no_pick`, box left at its start, mean final error 15.8 cm, slip −5 cm) from the baseline C0 mode (picked, placed 3 cm off). Training on more data from the same region did not remove extrapolation failures but changed their form — consistent with "no support ⇒ unreliable behaviour", not with a data‑volume effect.
4. **Costs / side effects (single seed, exploratory):** both 90‑demo C0 branches show failures at x = 0.44–0.45 (EXT‑C0 4/32, VOL‑C0 3/32 vs 2/96 in the baselines) ⇒ possible in‑support dilution from the fixed 5400 steps (fewer steps per demo) or seed variation; not separable with one seed. EXT‑S1 shows none (1/32).
5. **Reading (single seed, exploratory):** for C0, coverage (C1, C2, C3 hold) rather than data volume accompanies the recovery of the boundary grasp; for S1 only C1 and C3 hold and the volume control is missing, so **nothing is inferred about S1 coverage‑vs‑volume**. The S1 deficit at the new edge x = 0.34 and the in‑support C0 regression are open questions. Shortcut robustness (S1 SI 1.0, C0 SI ≈ 0) is unchanged by coverage.

## Limits
One training seed per EXT/VOL arm (D1), VOL‑S1 missing (guard), thread count and VM speed differ from the baselines (seed‑0 baselines 4 threads; this VM ≈ 25 % slower), fixed compute makes steps per demo lower in the 90‑demo branches (confound shared by EXT and VOL), 4 poses per x per checkpoint, 4 pose blocks only, replay divergence in 5/180, self‑checked only (no independent review). Test seeds 31–33 not used.

## Structure of the evidence (what is registered, observed, interpreted, missing)
**1. Pre-registered hypotheses (rules frozen before any run; verdicts single‑seed; "SUPPORTED" ≠ confirmed).** C1 SUPPORTED for EXT‑C0 and EXT‑S1 (one seed each). C3 SUPPORTED for EXT‑C0 and EXT‑S1 (one seed each). C2 evaluable **for C0 only**: VOL‑C0 19 % ≤ baseline 52 % + 10 ⇒ coverage‑specific for C0; **no C2 verdict for S1 and none is inferred**. C4 one pair, descriptive. C5: EXT‑C0 fails narrowly (−10.4 pp vs margin −10), EXT‑S1 holds. C6 SUPPORTED (single seed). C7: EXT‑C0 fails (and VOL‑C0 shows the same upper‑side change, so it is not caused by the added coverage), EXT‑S1 holds. C8 within ±25 % (slower VM). **The overall rule (COVERAGE EXPLAINS / PARTIAL / NOT EXPLAINED) is not issued** because it needs C2 for both arms.
**2. Observed evidence (tables above).** EXT branches keep the expert's grasp depth (|a_close − expert| 0.49 / 0.51 cm) down to x = 0.34 with 0 cluster failures; the baselines and VOL‑C0 deepen to −3…−5 cm below x = 0.38. Boundary success EXT‑C0 100 % (baseline 52 %), EXT‑S1 81 % (baseline 27 %), VOL‑C0 19 %. Remaining EXT‑S1 misses: 3 of 4 poses at x = 0.34. Failures at x = 0.44–0.45 in both 90‑demo C0 branches (EXT‑C0 4/32, VOL‑C0 3/32 vs 2/96 in baselines). Shortcut indices unchanged (S1 1.0/1.0; C0 ≈ 0). VOL‑C0's low‑x failures are `no_pick` (12/13) instead of the baselines' placed‑3‑cm‑off mode.
**3. Exploratory interpretation (not tested).** Added support at x < 0.38 restores the grasp depth for both arms, and the same data volume from the original region does not (C0); S1's breakdown point sits at the edge of its support (x = 0.34); the in‑support C0 regression at x = 0.44–0.45 may reflect fewer steps per demo at fixed compute or seed variation (not separable with one seed); the VOL‑C0 result shows boundary behaviour outside the support varies strongly between trainings.
**4. Missing comparison.** VOL‑S1 (no checkpoint) ⇒ no coverage‑vs‑volume test and no EXT − VOL difference for S1; EXT − VOL exists for C0 only (EXT‑C0 − VOL‑C0: boundary +81 pp, grasp error −2.05 cm, in‑support −3 pp, HIGH 0 pp; single seed each). No seed‑to‑seed spread for EXT/VOL (EXT seed 4 not run). No paired baselines (independent seeds).
**5. Protocol deviations (registered design unchanged; config sha256 `f01f4733…aef3`).** D1: EXT seed 4 not run (compute, user decision; all branches single‑seed). D2: VOL training starts right after the second EXT evaluation, inspection concurrent (no serious problem found). D3: one approved restart of `vol_s1_seed3` with identical arguments, environment fingerprint saved, nothing concurrent; **failed the guard again (1.135 s/step)**; stopped, not restarted again, guard not relaxed. Also: the ext_s1 first attempt was killed by the guard because of my own concurrent replay (1.226) and restarted once with approval (1.043 pass); running the norm‑contract unit tests rewrites `outputs/e6b/unit_tests.json` with identical verdicts.
**6. Runtime / compute budget / speed‑guard events.** Guard (first 200 steps, ≤ 1.10 s/step): ext_c0 1.030 pass · ext_s1 attempt 1 1.226 fail (my concurrent replay) · ext_s1 restart 1.043 pass · vol_c0 1.082 pass · vol_s1 attempt 1 1.112 fail · vol_s1 restart 1.135 fail (stopped). Training 97–99 min per completed run (1.08–1.10 s/step, vs 0.75–0.97 for the baselines on a faster VM); evaluation ≈ 43 min per checkpoint. Compute used ≈ 9.6 h of the 14 h stop rule (active time, review pauses excluded). All incomplete runs, STOP/guard logs (`outputs/cov/guard.log`, `guard_trip_*.flag`), partial checkpoints and config/dataset hashes are kept (`assets/cov/`).

## Closure
Experiment closed on the user's decision with status PARTIAL — STOPPED BY PRE‑REGISTERED SPEED GUARD. No ACT coverage/normalization experiment continues; no policy is adopted; test seeds 31–33 were never used.
