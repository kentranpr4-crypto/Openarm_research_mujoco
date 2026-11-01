# Offline branching replay — REPORT (2026-11-01) — DEVELOPMENT, short-horizon, privileged oracle observer
Status: **DONE (protocol executed as frozen; oracle-headroom stop rule TRIGGERED).** Tự kiểm. Freeze sha256 `f2249bd3229c8f50fac0cfef7749b2d70c10ac1baade71fd71314898c10cfe24` (2026-11-01T08:40:14Z, before any branch result). Raw: `outputs/branching_replay/results/{worker*.json,branching_analysis.json,arrays/}`.
Not closed-loop success, not independent, not Sim2Real. Labels are executed one-chunk (J20) or two-chunk-with-hold (J2) outcomes; they do not measure task success.

## Integration gates (8/8 PASS; first run crashed before reporting — fixed float32/float64 `prev` bug, its CPU is in the ledger)
Snapshot (`mj_copyData` + prev_ctrl + time) restore bit-exact (repeat, after another branch, fresh env instance: max diff 0.0); branch c0_s0 == Stage-1 A0 continuation exactly (box, TCP, executed actions 0.0 at t=40,100); control-history restore + negative control; no future leakage (identical predictions after branches and with poisoned future rows); executed == ActionAdapter chain, applied ctrl == executed; J2 first block == J20 path (1e-6); ids/atomic/guard; metric unit tests.
Note: `mj_setState(mjSTATE_INTEGRATION)` was NOT exact (prototype: up to 0.2 m divergence) and was rejected before the freeze.

## Scale / compute
24 poses, 216 decision states (t=20…180), 648 executed branches, all legal/finite, 0 violations (11 300 gripper-saturation clips, 26 slews). J2 evaluable on 192 states (t≤160). **CPU 753 s = 0.209 CPU-h of the 0.40 cap** (prototype 64 s + tests 129 s incl. crashed run + run 553 s + analysis ≈8 s).

## Oracle headroom (realised J20; headroom = J(default c0_s0) − min over 3 candidates)
Mean 0.0022 [0.0017, 0.0029]; **states with headroom ≥ 0.01: 20/216 (9.3 %) → 90.7 % without headroom > 80 % ⇒ stop rule triggered.** By phase (states with headroom ≥0.01): approach 0/48, grasp 16/24 (mean headroom 0.0109 [0.0082, 0.0134]), carry 3/55, place 1/89.

## WM3-B ranking (pose-bootstrap CIs, 24 poses)
| | value |
|---|---|
| Pairwise accuracy, |ΔJ_true| ≥ 0.01 (87 pairs) | **0.793 [0.716, 0.868]** |
| grasp (56 pairs) / carry (13) / approach (16) / place (2) | 0.875 [0.78, 0.95] / 0.846 [0.75, 1.0] / 0.438 [0.125, 0.75] / 1.0 (n=2) |
| Regret: WM / default / random (J units) | 0.0022 [0.0012, 0.0034] / 0.0022 [0.0017, 0.0029] / 0.0033 [0.0024, 0.0045] |
| WM − default (paired) | **0.0000 [−0.0007, +0.0007]** |
| WM − random (paired) | −0.0011 [−0.0018, −0.0005] |
| Top-1 = oracle best | 76.9 % (WM picks default in 48.6 %) |
| Headroom captured by WM (states with headroom) | 1.5 % overall; **81.6 % in grasp** (regret 0.0020 vs default 0.0109, paired −0.0089 [−0.0119, −0.0058]) |
| Approach | WM worse than default: +0.0043 [+0.0011, +0.0082] |
Normalized regret (n=100 defined states): WM 2.49, default 3.54, random 4.53.

## Secondary
- Margin vs correctness (meaningful pairs): margin <1e-3: 0.50 (n=2); 1e-3–5e-3: 0.75 (4); 5e-3–1e-2: 0.56 (18); **≥1e-2: 0.873 (63)**.
- P3 gating (τ=0.01, fallback c0_s0): covers 48/216 states (22 %); regret 0.0014 [0.0008, 0.0022] vs default 0.0022; **gated − default −0.0008 [−0.0012, −0.0005]** (always-rank − default 0.0000).
- Palm: predicted 53.5 % of branches vs realised 6.5 % (TP 42, FP 305, FN 0, TN 301): the WM over-predicts palm contact; dxy MAE 1.9 mm.
- Candidate diversity (executed L∞ vs default, left arm): median 0.071 rad (p10 0.035, p90 0.143).
- **J2 (hold continuation, 192 states):** pairwise accuracy 0.767 [0.697, 0.852] (90 pairs); regret WM − default +0.0001 [−0.0011, +0.0013]; gated − default −0.0008 [−0.0012, −0.0004]; headroom ≥0.01 in 10.9 % of states. J2 is not better than J20 on the same states (0.793 [0.716, 0.868], WM − default 0.0000).

## Interpretation (what is and is not supported)
- Supported: the WM3-B ranks executed one-chunk outcomes well when the true gap is meaningful (≈0.79 pairwise, 0.87 at predicted margin ≥0.01), but the three C0 candidates almost never differ in J within a chunk (91 % of states), so the achievable gain over the ACT default is tiny. The only phase with real headroom is grasp, where WM ranking captures most of it (−0.0089 regret vs default). Margin gating (τ=0.01) gives a small, CI-excluding regret reduction over the default (−0.0008) and none of the approach-phase loss.
- Not supported: any statement about task success or the Stage-1 carry-phase drops (J20/J2 are chunk-level proxies; these drops occur around step 105 and were not captured by J within the candidate set); that J2 adds value (it did not); effects in poses outside these 24.
- Limits: 24 poses (grasp phase ≈ 1 state per pose, 87 meaningful pairs), same poses as Stage 1, single seed set, single WM3-B checkpoint, oracle observer, simulation only. The stop rule (headroom <0.01 in >80 % of states) was pre-registered and is reported, not reinterpreted.

## Recommendation
Under this protocol (C0×3 ensemble, frozen J20/J2) the WM-ranking line has little headroom to exploit: stop broad ranking studies. If any continuation is wanted, restrict to the one place where signal exists — grasp-phase decisions with margin ≥0.01 — and first obtain candidates with larger behavioural spread (the current set is nearly equivalent in J), then test closed-loop success on new poses with a new pre-registration. P2 (hold-continuation J2) showed no gain and need not be pursued in this form.
