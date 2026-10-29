# E0.1 — precision & failure diagnostics on the E0 data (analysis only)
Date 2026-10-29 · script `eval/e0_1_diagnostics.py` (reads existing E0/Stage‑1 dev rollouts; no training, no new rollouts, no test seeds 31–33; E0 report/config untouched) · results `outputs/e0_1/e0_1_results.json` (copy in `assets/e0_1/`).
Status: **DONE (self-checked)**. Everything below is **exploratory/post‑hoc** (E0's pre‑registered verdict R1 supported / R2–R3 inconclusive is unchanged). Thresholds in the failure taxonomy come from geometry (tolerance 3 cm, tray, gripper) and were not tuned on S1/C0 differences.

## 1. Statistical precision
Unit = training seed (n = 3 per arm); success per seed in %. ALL_DEV (NOM+B21+B22, 90 ep/seed) is an extra exploratory endpoint.
| set | C0 per seed | S1 per seed | Welch diff [95 % CI] (df) | paired‑by‑seed diff [CI] (seed diffs) | hier. bootstrap CI | leave‑one‑seed‑out diff |
|---|---|---|---|---|---|---|
| NOM | 96.7 / 100 / 90 | 83.3 / 100 / 100 | −1.1 [−21.0, +18.7] (3.0) | −1.1 [−30.2, +28.0] (−13.3, 0, +10) | [−13.3, +10.0] | +5.0 / −1.7 / −6.7 |
| BOUND | 83.3 / 66.7 / 91.7 | 71.7 / 68.3 / 71.7 | −10.0 [−40.7, +20.7] (2.1) | −10.0 [−37.1, +17.1] (−11.7, +1.7, −20) | [−23.9, +5.0] | −9.2 / −15.8 / −5.0 |
| ALL_DEV | 87.8 / 77.8 / 91.1 | 75.6 / 78.9 / 81.1 | −7.0 [−21.9, +7.9] (2.6) | −7.0 [−24.8, +10.7] (−12.2, +1.1, −10) | [−15.9, +2.6] | −4.4 / −11.1 / −5.6 |
Pairing by seed id is legitimate in design (same seed ⇒ same initial weights and batch order; only normalization differs), but with n = 3 the sign‑flip test cannot go below p = 0.25 (observed 1.0 / 0.5 / 0.5). Cross‑arm seed correlation is unestimable (r = −0.19 NOM, 0.94 BOUND, 0.13 ALL_DEV with n = 3).
**Assumption checks (what they show and cannot show)**
- Normality of seed‑level proportions is untestable at n = 3. Welch df is 2.1–3.0 (not 4), so the CI is wide and fragile: leave‑one‑seed‑out Welch CIs for BOUND have half‑widths of 53–148 pp; the arcsine‑√ transform changes df but not the conclusion.
- Variances are unequal and unstable: BOUND between‑seed SD C0 12.7 pp vs S1 1.9 pp; NOM C0 5.1 vs S1 9.6 (S1 NOM driven by seed 0 = 83.3). Variance ratios flip sign between sets, so there is no stable "S1 is more/less variable" statement. The 95 % CI of a between‑seed SD from 2 df spans a factor ≈ 0.5–6 (e.g. BOUND C0 SD 12.7 → [6.6, 80]).
- Over‑dispersion vs pure binomial (χ² homogeneity across seeds, df 2): C0 BOUND p = 0.002, S1 NOM p = 0.005, S1 BOUND p = 0.90, C0 NOM p = 0.16 ⇒ for most cells seed‑to‑seed variation exceeds episode sampling noise, so treating all 90 episodes as independent would overstate precision.
- Hierarchical bootstrap (seeds then episodes) gives narrower intervals than Welch, but resampling 3 seeds yields ≤ 10 distinct seed sets and under‑estimates between‑seed variance; it is a sensitivity check, not a confirmation. All intervals include 0.
**Sample size (seed‑level model, Welch half‑width; SD scenarios: E0 pooled SD NOM 7.7 / BOUND 9.1 pp; 10 pp; upper‑90 % bound of the SD 23.7 / 28.0 pp)**
| seeds / arm | NOM hw (pooled SD) | BOUND hw (pooled SD) | hw if SD = 10 pp | hw if SD = upper‑90 % | power non‑inf. BOUND, true Δ = 0 / −5 pp (SD 10) |
|---|---|---|---|---|---|
| 3 (now) | 17.5 | 20.6 | 22.7 | 54–64 | 0.22 / 0.11 |
| 5 | 11.2 | 13.3 | 14.6 | 35–41 | 0.51 / 0.26 |
| 6 | 9.9 | 11.7 | 12.9 | 31–36 | 0.62 / 0.32 |
| 8 | 8.3 | 9.8 | 10.7 | 25–30 | 0.78 / 0.45 |
| 10 | 7.2 | 8.6 | 9.4 | 22–26 | 0.88 / 0.55 |
| 15 | 5.8 | 6.8 | 7.5 | 18–21 | 0.98 / 0.74 |
- A ≈ 10 pp half‑width needs ≈ 6 (NOM) – 8 (BOUND) seeds/arm if the E0 SD is right, ≈ 10 if SD = 10 pp, and is not reachable (> 20) if the SD is at its upper bound. The SD estimate itself is the dominant uncertainty; a first batch would also improve it.
- NOM non‑inferiority (margin −10 pp) at true Δ = −5 pp stays under 35 % power even with 20 seeds ⇒ NOM equivalence is not a realistic target; BOUND (margin −15) is the only endpoint a modest extension can settle.
- Figures: `assets/e0_1/E01_1_paired_seeds.png`, `E01_2_ci_halfwidth.png`.

## 2. Failure diagnosis (dev sets NOM+B21+B22, 90 ep/seed, pooled over seeds)
Where the extra S1 failures are: C0 has 17 post‑pick failures vs S1 31; pick‑stage failures are similar. Conditional place rate (place | pick): C0 95.2 / 93.3 / 91.1 %, S1 82.9 / 89.9 / 89.0 % (Welch diff −5.9 pp [−13.7, +1.9], df 3.1; exploratory). Pick rate: C0 92.2 / 83.3 / 100, S1 91.1 / 87.8 / 91.1 (diff −1.9 [−21.3, +17.6]). By set (S1 vs C0 post‑pick failures): NOM 5 vs 4, B21 6 vs 4, B22 20 vs 9 ⇒ the excess is concentrated in B22.
**Evidence 1 — S1's "release" failures are near‑misses with a single signature.** 27/31 lie 3–4 cm from the tray centre (tolerance 3 cm); 21/31 form a tight cluster at x − x_tray = **−3.47 ± 0.20 cm, y ≈ 0.0 ± 0.26 cm** (6/8/7 per seed). C0: 10/17 within 3–4 cm, only 2/17 in that cluster. Final speed 0, box resting on the tray base (z error ≈ 0): they were *placed* correctly in height and rest, but off‑centre. Post‑release drift is small (S1 0.14 cm, C0 0.48 cm) ⇒ not a bouncing/dropping problem. `E01_3_release_diagnosis.png`.
**Evidence 2 — error decomposition (final offset = TCP aim at release + box‑in‑hand offset + drift).**
| | aim (TCP − tray) | in‑hand (box − TCP) at release | of which present at closing | growth after closing | drift |
|---|---|---|---|---|---|
| successes C0 / S1 | 1.1 / 1.0 cm (+x) | 0.74 / 0.77 cm | 1.3 / 1.6 cm | 1.2 / 1.5 cm | 0.04 / 0.05 |
| post‑pick failures C0 | 1.5 cm | 4.3 cm (−y dominated) | 2.1 cm (x −1.9) | 4.5 cm | 0.48 |
| post‑pick failures S1 | 1.0 cm | 4.7 cm (−x dominated, −2.6) | **3.2 cm (x −3.1)** | 3.0 cm | 0.14 |
The aim point is the same in both arms (≈ +1 cm in x, also in successes), so the arms do not differ in where the TCP goes at release; they differ in **where the box sits in the gripper**.
**Evidence 3 — grasp offset along x, seed‑level (all episodes with pick+release, n = 75–90/seed):** mean box‑TCP x offset at closing: C0 −1.03 / −1.15 / −0.99 cm, S1 −1.73 / −1.42 / −1.64 cm; Welch diff of |offset|: +0.53 cm [0.28, 0.78], df 4.0; the two arms do not overlap in any seed. A 0.5 cm systematic shift moves successes toward the 3 cm edge and produces the observed near‑miss cluster. (Post‑hoc; the CI is on 3 seeds and the seed‑0 thread confound applies.)
**Mechanism classes (measured, post‑pick failures):**
| class (rule) | C0 (17) | S1 (31) |
|---|---|---|
| in‑hand offset already present at closing (off‑centre grasp; |ih_close| ≥ |growth|, in‑hand > 1.5 cm) | 2 | **21** |
| in‑hand offset grew after closing (slip during lift/transport) | **11** | 8 |
| post‑release drift > 2 cm | 1 | 0 |
| early release before tray | 1 | 0 |
| mixed / small | 2 | 2 |
By seed S1: seed 0 split (6 off‑centre grasp, 7 slip, 1 mixed), seed 1 purely off‑centre grasp (8/8), seed 2 mostly off‑centre grasp (7/9; 1 slip, 1 mixed). C0 failures are mostly transport slip (seed 0: 4/4, seed 2: 7/8). **So the two arms fail by different mechanisms: C0 loses the box's grip alignment during transport; S1 acquires an off‑centre grasp (≈ −3 cm along x) that survives to the tray.**
**Not verified (hypotheses only):**
- H1 – in S1 the right‑arm state channel becomes a near‑constant (σ_eff = 0.02 rad ⇒ values ≈ 0), whereas in C0 it is a large, deterministic, slightly varying input that may act as a learned time/phase code; removing it could degrade grasp‑position/phase precision in x. Evidence for/against: none yet; needs an ablation (right‑arm state dropped from the observation) or phase‑probing — requires training, not approved.
- H2 – the expert itself produces a ≈ −1 cm grasp offset (successes ≈ −1 cm in both arms) and S1 amplifies it by ≈ 0.5 cm; not checked against expert rollouts.
- H3 – the −3.5 cm cluster is a geometric attractor (box seated against a finger edge); the extremely tight spread (0.2 cm) suggests it, but finger geometry was not inspected.
- Speed/jerk: kinematics of S1 failures are not unusual (carry separation 6.4 cm vs 3.3 cm for successes, mostly from the off‑centre box); no evidence of abnormal transport dynamics.

## 3. Thread‑count deviation D1 (seed 0 = 4 threads; seeds 1–2 = 3 threads)
- **Not identifiable** from current data: thread count is perfectly confounded with seed identity in both arms. A direct test (re‑train seed 0 at 3 threads or seed 1 at 4 threads) needs training — not approved.
- What the data say: C0 seed 0 is inside the range of C0 seeds 1–2 on NOM (96.7 vs 100/90), BOUND (83.3 vs 66.7/91.7); S1 seed 0 is low on NOM (83.3 vs 100/100) but not on BOUND (71.7 vs 68.3/71.7). Open‑loop val MAE (S1 0.00626 vs 0.00612/0.00602 rad; C0 0.00507 vs 0.00523/0.00516) and final training loss EMA (S1 0.0382 vs 0.0353/0.0384; C0 0.0356 vs 0.0334/0.0370) show no seed‑0 outlier (< 5 %). So seed 0 is an outlier in 1 of 4 (arm × {NOM, BOUND}) success cells and 0 of 4 non‑success metrics.
- Primary analysis keeps all 3 seeds (no pre‑registered exclusion rule). Excluding seed 0 is shown only as sensitivity (NOM +5.0, BOUND −9.2 pp, n = 2/arm).
- Cheap way to resolve it (proposal, not started): re‑train seed 0 of both arms at 3 threads (2 runs ≈ 3.4 h incl. eval). This gives a same‑seed, different‑thread replicate = direct estimate of numerics‑induced variation, and makes all 3 seeds homogeneous.

## 4. Estimated compute budget (this VM: 0.75–0.80 s/step train ≈ 70 min; eval ≈ 32 min; ≈ 1.7 h per run incl. eval)
| option | runs | wall‑clock | gain |
|---|---|---|---|
| A. +3 seeds/arm (n=6) | 6 | ≈ 10 h | hw ≈ 10–12 pp if SD as E0 |
| B. +5 seeds/arm (n=8) | 10 | ≈ 17 h | hw ≈ 8–10 pp; BOUND non‑inf. power 0.78–0.86 at Δ=0 |
| C. +7 seeds/arm (n=10) | 14 | ≈ 24 h | hw ≈ 7–9 pp |
| D. Thread‑replicate of seed 0 (both arms) | 2 | ≈ 3.4 h | removes D1 confound |
| E. Mechanism ablation (e.g. right‑arm state dropped or masked, 3 seeds, one arm) | 3 | ≈ 5 h | tests H1 directly |
Wall‑clock assumes this 4‑core VM, one job at a time (parallel jobs oversubscribe the cores). Eval can be cut to NOM+BOUND only (already the case).

## 5. Recommendation (needs your decision; nothing started)
1. **Do not launch the 5–7 extra seeds per arm yet.** With the E0 SD the benefit is real only for BOUND, the SD estimate (2 df) is the biggest uncertainty, NOM equivalence is unreachable, and the failure analysis shows the S1/C0 difference is a **specific mechanism** (off‑centre grasp along x vs transport slip) — more seeds would measure the size of an effect without explaining it.
2. **Run D first (≈ 3.4 h)** if you want a clean 3‑seed baseline, then **E (≈ 5 h)** as a mechanism test with a pre‑registered prediction (if H1 holds, dropping right‑arm state from S1 should leave the shortcut gone, SI ≈ 1, and move the grasp x‑offset toward C0's −1.0 cm; if the offset remains, H1 is falsified). E changes the observation spec, so it needs your explicit approval as a model change.
3. **Then decide on seeds:** if E explains the offset, extra seeds should be run on the *fixed* candidate (3 more seeds ≈ 10 h) rather than on S1 as is. If not, run option B (≈ 17 h) with BOUND as the primary endpoint, pre‑registered.
4. Pre‑register before any new run: seed list, thread count (3 for all), endpoints (BOUND primary, ALL_DEV secondary), grasp x‑offset as secondary mechanism endpoint, exclusion rules (none).

## Limits
Post‑hoc exploratory analysis on dev seeds 21–22 that were already seen; n = 3 seeds per arm; taxonomy thresholds from geometry but chosen after seeing the failure plot; no independent review. Test seeds 31–33 untouched.
