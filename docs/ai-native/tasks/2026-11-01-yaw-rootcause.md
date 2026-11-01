# Yaw root-cause analysis (offline; no training; frozen artifacts untouched)

Date: 2026-11-01. Code: `eval/yaw_rootcause.py` (stages A/B: events + simulator replay), `eval/yaw_lossprobe.py` (stages C/D: loss contribution, alternative scalings; forward pass of frozen checkpoints only). Data: `outputs/yaw_audit/yaw_rootcause_AB.json`, `yaw_events.json`, `yaw_lossprobe.json` (not in git). Historical normalisers, checkpoints, datasets and the I-1 results are unchanged.

## 1. Where the yaw spikes come from (mechanism: PROVEN by replay)
122 per-step events with |Δyaw| > π/2 in 34 of 4 696 usable episodes (0.7 %). Every event was replayed in the simulator (same reset and executed actions as the dataset builders): the replay reproduces the dataset delta in 100 % of cases, so these are **not replay artifacts**; they are deterministic outputs of the `yaw_of(quaternion)` definition. Raw quaternion, angular velocity ω and the body-x-axis verticality were read for each event.

| Category | Events | Evidence | Meaning |
|---|---|---|---|
| Euler-yaw singularity, box at rest | 34 (28 %) | body-x vertical (|R₂₀|≈1.0), tilt = π/2, ω ≈ 1e-8 rad/s, speed ≈ 3e-10 m/s, delta ≈ ±π | heading of a box **standing on its end** is undefined: `atan2(±0,±0)` flips by π with no motion — pure representation artifact |
| Euler-yaw singularity, box moving | 64 (52 %) | body-x vertical, ω_max median 2.9 rad/s, speed 0.08 m/s | box slides/rotates on its end; "yaw" jumps are not a physical rotation about a defined axis |
| Physical fast rotation | 19 (16 %) | ω_max median 27 rad/s, speed 0.64 m/s, tilt small, wrapped delta ≥ π/4 | box spun/thrown by the gripper in failed grasps (ACT roll-outs) |
| ±π wrap crossing | 5 (4 %) | raw |Δ|>π, wrapped |Δ|<π/4, ω ≈ 20 rad/s | small physical rotation seen through the wrapped angle |

Lineage: 115 events from observational ACT roll-out replays (cov 37, e6b 41, xsweep 17, audit_v1 11, e0 6, full_v1 3) and 7 from scripted-intervention episodes (all 7 are singularities). The 314 "spikes" of the earlier audit are the same ~79 train events counted once per overlapping window (stride 5, K = 20 ⇒ ≈ 4 windows per step). **So ≈ 80 % of what looked like spikes is not rotation at all.**

## 2. Data-contract defect found on the way (PROVEN): the tilt channel mixes units
`b[:,5]` (box tilt) is **radians** for 3 650 usable episodes (`data/wm1_sim.py`: wm1 replays and all interventional episodes) and **degrees** for 1 046 usable episodes (`eval/h23_replay.py` uses `np.degrees`: h23/xsweep/cov replays; max 180, 96.8 % of those episodes exceed π). Consequences: the train tilt normaliser (per-step std 0.60) is dominated by the degree episodes; tilt targets of radian episodes (all interventional episodes, including I-1) are ≈ 57× smaller in normalised units; the tilt R² flipped from ≈ 0.1–0.35 (development windows, mixed) to −9…−23 (I-1, all radians). Yaw is in radians in both. The frozen dataset is not modified; this must be fixed in the next data contract (single unit, plus an explicit "body standing on end" flag).

## 3. Contribution of each regime to the yaw loss (PROVEN as a description of the training data and of the frozen model; causal role = HYPOTHESIS)
Regimes are defined by the largest true per-step |Δyaw| in the 20-step window. Normalised by the historical yaw std (0.0529 rad), WM3-B:

| Regime (train windows) | windows | share of yaw target energy | share of WM3-B yaw squared error (in-sample) | mean window SSE: zero predictor / WM3-B / C0 |
|---|---|---|---|---|
| static (< 0.005) | 63.7 % | 0.0 % | 2.4 % | 0.003 / 0.367 / 0.419 |
| small (0.005–0.05) | 31.8 % | 0.4 % | 2.9 % | 0.24 / 0.89 / 0.63 |
| moderate (0.05–0.5) | 4.3 % | 2.5 % | 3.3 % | 11.6 / 7.8 / 11.8 |
| fast (0.5–π/2) | 0.2 % | 3.0 % | 2.3 % | 359 / 135 / 229 |
| spike (> π/2) | **0.15 %** (129 windows) | **94.2 %** | **89.1 %** | 12 360 / 5 795 / 8 111 |

- 129 of 84 559 train windows carry 94 % of the yaw target energy and 89 % of the model's yaw loss (val: 10 windows, 87 % / 73 %).
- On the 99.8 % of windows without spikes the frozen WM3-B is, **in-sample**, about as good as predicting no rotation (SSE 1.08 vs 1.17) but on **val** it is 1.8× worse than zero (2.10 vs 1.14; C0 1.77, ridge 1.76), driven by the moderate regime (22.8 vs 9.0). In the static regime (64 % of windows) it is ~100× worse than zero even in-sample, yet that regime contributes only 2.4 % of the loss. ⇒ the model learned spurious rotation where the loss cannot see it, and the moderate-rotation signal it learned does not generalise.
- Concentration of per-channel target energy in the top 0.1 % of elements: yaw 98 %, tilt 68 %, speed 21 %, x 18 %, a-offset 8 %, y 7 %, z 7 %.

## 4. Offline effect of alternative scalings / representations (computed on targets, no training)
| Variant | spike-band share of energy | static+small share | normalised penalty of a spurious 0.01 rad step |
|---|---|---|---|
| current (std 0.0529) | 90.2 % | 0.5 % | 0.036 |
| robust scale (std excl. spikes, 0.0166) | 90.2 % (scale-invariant) | 0.5 % | 0.37 (10×) |
| wrap(delta) then std | 80.2 % | 1.1 % | 0.07 |
| circular (sin Δ, 1−cos Δ) | 67.9 % | 1.8 % | n/a |
| clip ±0.2 rad then std (0.0092) | 8.8 % | 17.2 % | 1.18 (33×) |
Reading: a pure re-scaling raises the penalty on small spurious rotations ten-fold but leaves the spike share unchanged; circular/wrap-aware forms cannot remove the ±π flips of a **singular** heading (sin π = 0 but 1−cos π = 2); only clipping/masking reduces the spike share. A singular step needs a **mask** (the heading is undefined), not a better angle.

## 5. What is proven, what is hypothesis
| Statement | Status |
|---|---|
| ≈ 80 % of |Δyaw|>π/2 events are Euler-yaw singularities of a box standing on its end; 16 % physical; 4 % ±π crossing | **Proven** (replay, ω, body-x) |
| Tilt channel mixes degrees and radians | **Proven** |
| Spike windows dominate yaw target energy and the model's in-sample yaw loss | **Proven** (description of data and frozen model) |
| The GRUs predict spurious rotation on static windows, worse than zero on held-out non-spike windows | **Proven** (I-1 and val) |
| The imbalance (spike-dominated loss) is the *cause* of the spurious rotation | **Hypothesis** — consistent with the above, not tested (needs a controlled training experiment) |
| Clipping/masking singular steps + circular or robust loss would fix it | **Hypothesis** — offline arithmetic only |
| Yaw is only weakly predictable from the inputs (best R² ≈ +0.07) | Supported (ridge/MLP), but a better-conditioned model could in principle extract more: **open** |
| Absolute wrapped yaw as an *input* adds noise (singular flips) | **Hypothesis**, not examined |
| Capacity sharing between yaw and the other channels | **Hypothesis**, not examined |
