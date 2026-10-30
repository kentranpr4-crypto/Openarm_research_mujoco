# x-sweep (evaluation only) — results
Date 2026-10-30 · pre-registration `2026-10-30-xsweep-preregistration.md` (config sha256 `6c59292d…596c`, unchanged) · scripts `eval/xsweep_run.py`, `eval/xsweep_run.sh`, `eval/xsweep_replay.py`, `eval/xsweep_report.py` · results `outputs/xsweep/xsweep_results.json`, `xsweep_tables.md`, plots `X1–X5` (copies in `assets/xsweep/`).
Scope kept: evaluation only; 6 frozen checkpoints (C0/S1 × training seeds 0,1,2) + scripted expert; no training, no new demos or seeds, weights/normalization/action adapter/controller/success criteria unchanged; no test seeds 31–33 (poses are explicit; eval seed label 21). Policy episodes: the unchanged `evaluate.py` episode runner with the box pose taken from the registered table.
Status: **DONE (self-checked, not independently reviewed)**. Pre-registered hypotheses are reported first with their frozen rule outcome and an honest reading; everything else is labelled **exploratory**.

## Acceptance criteria
| AC | Result | Evidence |
|---|---|---|
| AC1 expert ≥ 90 % over x ∈ [0.36, 0.47] | **PASS** (48/48; also 4/4 at every x incl. 0.34, 0.35, 0.48) | `AC.expert_success_by_x` |
| AC2 6 × 60/60 episodes | **PASS** | `AC.AC2` |
| AC3 no forbidden seed, config unchanged | **PASS** (sha re-checked by every script) | scripts |
| AC4 replay vs live | live/replay label agreement 359/360; **6/360 replays diverge** (box error > 5 mm: c0_s0 #37 x=0.41 (21 mm, also the one label disagreement), c0_s2 #3 x=0.37 (19 mm), c0_s2 #10 x=0.44 (49 mm), c0_s2 #17 x=0.36 (22 mm), s1_s0 #37 x=0.41 (5.8 mm), s1_s1 #28 x=0.47 (18 mm)). Live success/final-error metrics are unaffected; the 6 are excluded from the replay‑derived a_close and slip means only (palm/cluster flags still include them) | `AC.AC4_diverging` |
| AC5 ≈ 1.5 h | **Overrun, within the stop rule:** 1.83 h wall‑clock (≈ 18.4 min per checkpoint, ≈ 18 s/episode vs the planned 15 s) — below the 2.0 h stop threshold | `timing.jsonl` |

## A. Pre-registered findings (rules frozen before the run)
| Hyp | Rule outcome | Numbers (S1 − C0, paired by training seed, n = 3) | Honest reading |
|---|---|---|---|
| **HA** grasp depth at x ∈ {0.36, 0.37} | **SUPPORTED** | a_close diff −1.7 / −0.9 / −1.5 cm, mean −1.35 cm [−2.43, −0.28], 3/3 negative | S1 grasps ≈ 1.4 cm deeper than C0 at the near edge; CI excludes 0 but is n = 3 |
| **HB** success cost, x ≤ 0.37 | **SUPPORTED** (≥ 2/3 negative, mean ≤ −10 pp) | −50.0 / −25.0 / 0.0 pp, mean −25.0 pp, CI [−87, +37] | passes the rule but the CI is huge and the cost is **not** spread over the whole low range: at x ∈ {0.34, 0.35} both arms collapse (C0 12–25 %, S1 12–38 %; diff +4 pp, 1/3 negative); the cost sits at x ∈ {0.36, 0.37} (diff −54 pp, 3/3 negative: C0 100 / 88 / 62 % vs S1 12 / 25 / 50 %, CI [−149, +41]) |
| **HC** boundary effect, D(x) = a_close(policy) − a_close(expert) | **SUPPORTED for both arms** (S1 3/3, C0 3/3) | S1: D_low −2.6 / −2.6 / −2.2 vs D_in −0.8 / −0.4 / −0.5 cm; C0: D_low −0.9 / −1.7 / −0.7 vs D_in −0.3 / −0.1 / +0.2 cm | both policies go deeper near the lower training boundary; S1 more (and already ≈ 0.5 cm deeper in support) |
| **HD** cluster signature, S1 post‑pick failures at x ≤ 0.37 | **SUPPORTED** | n = 35; cluster‑flagged 86 %, palm contact 74 % | the x ≈ −3.5 cm cluster is reproduced on controlled poses |
| **HE** in‑support (x 0.38–0.45) non‑inferiority | **INCONCLUSIVE** | −15.6 / 0.0 / 0.0 pp, mean −5.2 pp [−27.6, +17.2] | the whole in‑support deficit is one checkpoint (S1 seed 0: 78 %; S1 seeds 1, 2 = 100 % = C0) — seed 0 is the thread‑count‑confounded run |
| HF upper boundary (undirected) | reported, no prediction | see B3 | |

## B. Exploratory findings (not pre-registered; interpret with n = 3 seeds × 4 blocks)
**B1 Training-boundary effect — success vs x (`X1`).** Expert 100 % at all 15 x. In the training region (0.38–0.45): C0 98 % (94/96), S1 93 % (89/96); all 7 S1 failures belong to seed 0 (seeds 1, 2: 100 %). Just below it (LOW_NEAR_EDGE, x = 0.36–0.37) per training seed: C0 100 / 88 / 62 %, S1 12 / 25 / 50 %; pooled per x: C0 75 % at 0.36 and 92 % at 0.37, S1 8 % and 50 %. Both arms collapse at x ≤ 0.35 (C0 17–25 %, S1 8–42 %) with 19 and 18 of 24 episodes failing post‑pick.
**B2 Mechanism vs x (`X2`, `X4`).** Expert a_close = −0.97 … −1.02 cm at every x. C0 follows the expert for x ≥ 0.38 (−0.8 … −1.3) and then falls: −1.6 (0.37) → −2.65 (0.36) → −3.7 (0.35) → −4.2 cm (0.34). S1 is already ≈ 0.5 cm deeper in support (−1.4 … −1.6) and falls earlier: −2.0 (0.38) → −2.8 (0.37) → −4.0 (0.36) → −4.7 (0.35). **S1's curve is approximately C0's shifted ≈ 1 cm toward the workspace centre plus a constant ≈ 0.4–0.5 cm.** Box–palm contact appears when a_close ≲ −3.5: C0 at x ≤ 0.35 (42–75 %), S1 from x = 0.36 (75 %) and 100 % at 0.35; the cluster flag follows the same pattern (S1 11/12 at 0.36, C0 10/12 at 0.35). Final placement error exceeds the 3 cm tolerance exactly where a_close ≲ −3 cm (`X2` right). The failure mechanism of the previous report (deep grasp → box slides back → seated against the palm → placed ≈ 3.5 cm off) therefore **is not specific to S1**: C0 produces the identical signature (19/19 post‑pick failures cluster‑flagged at x ≤ 0.35) once it is pushed far enough outside the training support; S1 reaches that regime ≈ 1 cm earlier.
**B3 Slip and upper boundary.** Slip (a_release − a_close, lift/transport) is small and noisy: S1 +0.8 cm vs C0 in support (CI [−1.5, +3.2]), no clean pattern with x; the cluster arises from the grasp position, not from transport slip (consistent with H2/H3). Above the training region (HF): a_close rises from −0.1 / −0.3 cm (C0 / S1) at x = 0.46 to +2.2 cm at x = 0.48 (TCP behind the box centre, i.e. shallow), both policies lose the pick (C0 33 %, S1 42 % at x = 0.48, mean final error 12 cm; all 17 failures in the 0.46–0.48 stratum are "no_pick") while the expert is 100 %. Policy grasp position is thus **compressed toward the training range on both sides** (too deep below, too shallow above); S1 − C0 differences at x ≥ 0.46 are small and of mixed sign (n = 3, CI ±80 pp).
**B4 Per-checkpoint (`X5`, tables).** At the near edge the S1 deficit is seed‑dependent in size (−88 / −62 / −12 pp) but never positive; S1 seed 2 (50 % at 0.36–0.37) is the closest to C0. S1 seed 0 is also the only checkpoint with an in‑support deficit.

## C. Answers to the requested analyses
- **Grasp‑relative offset at closing:** `X2` left; per‑x and per‑checkpoint tables in `xsweep_tables.md`; frames: a = TCP‑relative (box centre − TCP along the fingers), final errors in world frame relative to the tray centre.
- **Slip, final error, palm/finger contact, success by x, S1 − C0 by x with uncertainty:** `X2`, `X3` (config‑paired block‑bootstrap CI per x, descriptive) and `X4`; per‑x CIs at n = 12 are wide (e.g. x = 0.36 −67 pp [−100, −25], x = 0.37 −42 pp [−75, −8], the other x CIs include 0 or only touch it, e.g. x = 0.42 −25 [−50, 0], x = 0.43 −17 [−42, 0]).
- **Training‑boundary effect:** present and monotone on the lower side for both arms (a_close and success), mirrored on the upper side (shallow grasp, pick failures); S1's breakdown point is ≈ 1 cm further inside the support.
- **Failure mechanism:** deep grasp → palm seating → 3–5 cm placement offset (shared by C0/S1 at their respective thresholds); upper‑side failures are pick failures.

## D. What this changes / does not change
Supported (pre‑registered): S1's near‑edge deficit and its deeper grasp are real in all 3 seeds (sign‑consistent), the cluster signature is reproduced, and in‑support equivalence is not demonstrable at n = 3. **Not shown:** why S1 is shifted (right‑arm‑state channel, capacity, regularisation: untested), nor whether more coverage removes the shift (needs training — not approved). The in‑support difference rests on one thread‑confounded checkpoint.

## Limits
n = 3 training seeds; 4 blocks (y, yaw) only; 4 episodes per checkpoint per x; replay divergence in 6/360 episodes; the grid and blocks were fixed before the run, but the stratum boundaries are the training‑region edge (geometry‑defined); B1–B4 are post‑hoc readings; self‑checked only; wall‑clock 1.83 h (over the 1.5 h estimate, under the 2.0 h stop rule).

## Decision options for you (nothing started)
1. **Training‑side coverage test** (needs approval; ≈ 5 h): extend demo coverage to x ∈ [0.34, 0.38] (the expert is 100 % there) for C0 and S1, 1–2 seeds each, then rerun this exact sweep (config is reusable). If S1's shift vanishes with support, the cost is coverage; if the ≈ 1 cm shift and the ≈ 0.5 cm deeper in‑support offset persist, the right‑arm‑state hypothesis (H1) becomes the next test.
2. **Do not** extend seeds or run a floor sweep on the current data: the in‑support S1 − C0 difference is dominated by one checkpoint and the near‑edge effect is already sign‑consistent.
