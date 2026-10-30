# H2/H3 — grasp geometry of expert vs C0 vs S1, origin of the x ≈ −3.5 cm cluster
Date 2026-10-29 · scripts `eval/h23_replay.py`, `eval/h23_analysis.py` · results `outputs/h23/h23_results.json`, `h23_tables.md` (copies + plots in `assets/h23/`).
Scope kept: no training, no new evaluation protocol, no policy/controller/source changes, success criteria unchanged, test seeds 31–33 untouched, checkpoints untouched. Dev rollouts (E0 seeds 1–2, Stage‑1 seed 0) are **replayed** through the same simulator with the logged executed actions, adding quantities the rollouts did not log (box orientation, finger/palm contacts, TCP frame). The scripted expert is replayed on the same 90 dev configurations (paired).
Status: **DONE (self-checked; exploratory/post‑hoc)**.

## 0. Validity of the replay
- 540/540 policy replays; logged vs replayed task_success agree in 534/540; median box‑trajectory error 0.06 mm, max 3.2 cm (≈ 6 episodes diverge chaotically). The cluster counts re‑derived from the replay (S1 21, C0 2) equal those from the logged data (E0.1), so conclusions do not depend on the 6 diverging episodes.
- Expert replay: 90/90 success on the dev configurations.
## Frames (never mixed below)
WORLD = lab frame (x ≈ fingertip approach direction, tray centre (0.44, 0.30)). TCP‑relative: **a** = box centre − TCP along the fingers (a < 0 ⇒ box centre behind the fingertips, i.e. deeper in the hand), **c** across the jaws, **v** vertical. BOX‑relative: TCP in the box frame (5 cm side = longitudinal). Hand yaw vs box yaw: expert Δψ = −0.0° ± 0.1; C0 −2.1° ± 8.2; S1 −0.1° ± 8.4 (cos 8° = 0.99), so world‑x and along‑finger offsets differ by ≲ 1 % for typical episodes; all numbers below are the quantity named in the column header.
Gripper geometry (TCP frame, open pose, from the collision meshes): fingers span a ∈ [−7.94, +0.55] cm (TCP ≈ fingertip), palm collision front at a = −6.96 cm, inner jaw faces at c = ±4.32 cm; box 5 cm (along fingers when yaw ≈ hand) × 4 cm (across) × 4 cm.

## H2 — expert offset and what the policies do with it
- **Confirmed by construction and by measurement.** `ScriptedExpert.segments()` sets grasp = box centre + 0.01·f ⇒ box centre 1.0 cm behind the TCP. Measured: a_close = **−0.99 ± 0.05 cm** (NOM −0.99, B21 −1.01, B22 −0.98), aim at the tray +0.97 cm (+x), final error 0.07 cm.
- C0 reproduces it (seed means of a_close over all episodes −0.85 / −0.81 / −1.02 cm; Δ vs expert +0.14, +0.18, −0.02 cm). **S1 is deeper in all 3 seeds: −1.39 / −1.16 / −1.38 cm (Δ −0.40, −0.17, −0.39)**, with larger spread (SD ≈ 1.3 cm vs 0.6–1.1).
- Policy grasp depth depends on box x, the expert's does not: slope of a_close vs box x (cm/cm) expert 0.00, C0 0.08, S1 0.20 (Spearman ρ expert −0.45 (tiny range), C0 +0.33, S1 +0.54). ⇒ the policies do not track the expert's constant offset; they drift deeper as x decreases, S1 about 2.5× more than C0.

## H3 — where does the x ≈ −3.5 cm cluster come from?
The cluster = 21 S1 post‑pick failures (final x − x_tray = −3.47 ± 0.21 cm, y ≈ 0) vs 2 for C0 (same definition as E0.1).
**Chain of evidence (grasp position → contact mechanics → release geometry):**
1. **Initial pose.** All 21/21 cluster episodes have box x = 0.3647 ± 0.0034 m, i.e. the near edge of the evaluation region; the training region is x ∈ [0.38, 0.45] (`policies/baseline.py`), so these poses are *outside the training support* (they occur only in B21/B22). No cluster episode has x ≥ 0.38. Yaw does not differ between cluster and other S1 post‑pick episodes (p = 0.33); y slightly (p = 0.02, 0.092 vs 0.104 m).
2. **Grasp position (primary cause).** At closing the cluster has a = **−3.62 ± 0.44 cm** (expert −0.99): fingertips 1.1 cm beyond the box front edge, box rear edge 6.4 cm behind the TCP, i.e. 0.6 cm from the palm (palm front −6.96). Lateral offset is fine (c = +0.19 ± 0.66 cm); world and TCP frames agree. By box‑x bin (a_close, pooled over seeds): x < 0.38: expert −0.98 / C0 −1.27 / **S1 −2.32**; all bins with x ≥ 0.38: S1 −0.2…−1.26, C0 −0.2…−1.06.
3. **Contact mechanics (secondary, amplifies and fixes the position).** In the cluster the box slides further back along the fingers during lift and transport: a −3.62 → −3.79 (hold) → −3.88 (lift) → −4.21 (mid) → **−4.42 ± 0.25 (release)**; successes move the other way (+0.6 to +1.1 cm). At release the box rear edge is at ≈ −6.9 cm = the palm front (−6.96): the box **ends up seated against the palm**, a geometric stop. Direct evidence: finger contact point on the box shifts from −0.27 cm (successes) to −2.66 cm (cluster, hold) and −3.32 cm (mid); box–palm contact force > 0.01 N in 43 % of the cluster episodes between close and release (14 % at the hold instant; the rest sit within the contact margin or touch below the threshold — not resolved); finger force and tilt are normal (47.8 N, 0.3°). This also explains why the cluster is tighter at release (SD 0.25 cm) than at closing (0.44 cm).
4. **Release geometry (not the cause).** TCP aim at release is +0.95 ± 0.06 cm (same as expert/successes), post‑release drift 0.0 cm, c = 0.0 ± 0.1 cm. Final world x = aim (+0.95) + box‑in‑hand (−4.41) = −3.47 cm; correlation of the final x with a_release 0.96 (with a_close 0.75).
**Verdict:** the cluster originates in the **grasp position** (box grasped ≈ 2.6 cm too deep at poses outside the training support), is **locked in by contact mechanics** (box slides back until it hits the palm) and is **not a release artefact**. The box rests flat on the tray base 0.5 cm outside the 3 cm tolerance (success criteria unchanged).
**By‑region effect (new; post‑hoc, per‑seed paired):**
| region | C0 per seed | S1 per seed | S1 − C0 per seed (pp) | paired mean [95 % CI, n=3] |
|---|---|---|---|---|
| x ≥ 0.38 (in training support, 67 ep/seed) | 91.0 / 88.1 / 94.0 | 79.1 / 94.0 / 88.1 | −11.9, +6.0, −6.0 | −4.0 [−26.6, +18.7] |
| x < 0.38 (near edge, outside support, 23 ep/seed) | 78.3 / 52.2 / 78.3 | 65.2 / 34.8 / 60.9 | −13.0, −17.4, −17.4 | −15.9 [−22.2, −9.7] |
Near‑edge cluster rate S1 6/8/7 of 23 per seed vs C0 0/2/0. Pick rates are similar in both arms. So the NOM equality (train region only) and the BOUND deficit of S1 found in E0 are consistent with **an S1‑specific degradation under extrapolation to x < 0.38**, not with a uniform competence loss; in‑support the difference is small and sign‑unstable (seed 0, the thread‑count outlier, carries it).
Other pose dependence: for y outside the training band (0.07–0.13 m) the spread of a_close grows for both arms (up to ±3 cm, mostly shallow/pick failures), but without a systematic cluster; yaw has no effect on grasp depth for either arm (ρ ≈ −0.07/−0.08).

## Evidence labels
| statement | status |
|---|---|
| expert grasps 1.0 cm past the box centre | fact (code + 90 replays) |
| S1 grasps deeper than C0/expert, in all 3 seeds | measured (seed‑level; post‑hoc) |
| cluster boxes start at x ≈ 0.365, outside the training region | measured |
| box ends against the palm at release (rear edge −6.9 vs palm −6.96 cm) | measured geometry; direct contact force in 43 % |
| S1 extrapolates worse than C0 near x < 0.38 | measured (3/3 seeds negative; n = 3) |
| cause of the worse extrapolation (e.g. loss of the right‑arm state channel acting as a phase/position code, or capacity/regularisation) | **hypothesis, untested** |
| policy "range compression" (TCP gain < 1 w.r.t. box x: S1 0.80, C0 0.92, expert 1.00 from the slopes) | derived from slopes; mechanism untested |

## Limits
Replay not bit‑identical (6/540 diverge); dev seeds already seen; n = 3 training seeds; thread‑count confound (seed 0) unchanged; the region split x < 0.38 was chosen *after* seeing the cluster (post‑hoc; it equals the training‑region boundary, a geometry‑defined cut, not a tuned threshold). Palm contact threshold 0.01 N is arbitrary. Self‑checked only.

## Recommendation for the next experiment (none started)
1. **Cheapest and most informative (evaluation‑only, needs your approval as a new eval protocol):** a dense, *dev‑synthetic* pose sweep on the existing 6 checkpoints + expert: box x from 0.34 to 0.48 in 1 cm steps at fixed y = 0.10, yaw = 0, 4 repeats with small yaw jitter; measure a_close(x), pick/place and palm contact. This maps where extrapolation begins per arm/seed (is S1's slope 0.20 vs C0's 0.08 real and monotone?) without training. ≈ 6 checkpoints × ~60 episodes × 15 s ≈ 1.5 h. Test seeds not used.
2. **If confirmed:** pre‑register stratified endpoints (in‑support BOUND_in: x ≥ 0.38; extrapolation BOUND_out: x < 0.38) before any seed extension. Recompute the seed budget separately for BOUND_in (current paired difference −4 pp, CI ±22 pp at n = 3; its between‑seed SD is not yet estimated reliably) and BOUND_out (3/3 seeds already negative, so extra seeds add little there).
3. **Training‑side mechanism test (needs explicit approval, not requested now):** extend the demo region to x ∈ [0.36, 0.38] (expert plans are clean there) for C0 and S1, 1–2 seeds each (≈ 5 h): if the S1 deficit disappears with support, the floor is not the problem and the cost is coverage; if it persists, H1 (right‑arm state channel) stays the lead hypothesis and the right‑arm‑state ablation becomes the next test.
4. **Do not** run more S1/C0 seeds on the current setup, the seed‑0 replicate or a floor sweep before step 1: the new result changes what the seeds should measure.
