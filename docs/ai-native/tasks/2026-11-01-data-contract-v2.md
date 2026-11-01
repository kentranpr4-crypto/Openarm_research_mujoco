# Data contract v2 — design and validation (Phase 0; nothing official rebuilt, nothing trained)

Date: 2026-11-01. Code: `eval/geometry_v2.py`, `data/contract_v2.py`, `eval/orientation_loss.py`, `eval/contract_v2_validate.py`, `eval/test_contract_v2.py`, `eval/yaw_event_reconcile.py`, `eval/yaw_persistence_feasibility.py`. Data (not in git): `outputs/contract_v2/` (reconciliation, `validation.json`, `replay_sample.npz` = 394 replayed episodes, unit tests, feasibility). The frozen wm2 dataset, its statistics, the baselines, the checkpoints and the I-1 results were **not modified**; any training data change must create a new dataset version with its own manifest and hash.

## 1. The 314 "spikes" vs the 122 events (reconciled exactly; `yaw_event_reconcile.json`)
- Unit of the 314: **window-step elements** of the TRAIN windows with |Y_b yaw| > π/2 (a transition appears in up to 4 windows: stride 5, K = 20). Unit of the 122: **episode transitions** (row τ → τ+1) over all 4 696 usable episodes.
- TRAIN: 79 unique events → 78 events covered by 4 windows + 1 covered by 2 = 78×4 + 2 = **314 elements**, in 129 windows and 19 episodes. 0 events of any split are uncovered by windows; every window element maps to an event of the 122 (all_in_122 = true).
- Split totals: train 79, val 4, test_id 2, test_low_heldout 5, test_ood_low_strict 8, test_ood_high 24 events ⇒ 122; elements 314 / 16 / 8 / 20 / 32 / 96.
- Category of the 79 train events: singular 59 (75 %), physical fast rotation 16 (20 %), ±π crossing 4 (5 %).
So the two numbers describe the same phenomenon at two units; 314 = Σ events × window multiplicity.

## 2. Conventions verified against MuJoCo (replay of 394 episodes, 69 993 steps)
| Check | Result |
|---|---|
| quaternion `xquat` = (w,x,y,z) ↔ `xmat` (row-major, body→world, column j = body axis j in world) | max |R(q) − xmat| = 2.3e-15; orthonormality 1.6e-15; det = 1 within 1.8e-15 |
| contract-v1 yaw = atan2(R10, R00) = heading of the body-x axis on the horizontal plane | reproduces the dataset yaw to 2e-4 rad max (float32 storage); the heading is **undefined when the body-x axis is vertical** (horizontal norm hnorm = √(R00²+R10²) → 0) |
| angular velocity: `cvel[:3]` is the world-frame ω | finite-difference rotation vector of q_{k+1}q_k⁻¹ vs ω: median cosine 0.986 (world) vs 0.972 (body frame); fd/sim norm ratio median 1.7 (a 20 Hz control step spans several physics sub-steps and ω changes within it; exact equality not expected) |
| tilt = arccos(R22) | stored radian episodes match to 2e-7; degree episodes match after ×π/180 to 2e-7 ⇒ **repair is exact to float32** (3 650 episodes unchanged, 1 046 divided by 180/π) |
| box symmetry | the box is a cuboid: the pose is invariant under body-side 180° rotations about x, y, z (group D2). At rest in the singular state the yaw flips by π (median |Δyaw| = 3.14159) while the D2-symmetric geodesic step is ≤ 4e-8 rad ⇒ **the "rotation" is the same pose** |

## 3. Singular discontinuity vs legitimate rotation (geometry only)
Yaw-inconsistency J = |wrap(Δyaw)| − (box-symmetric geodesic step) vs the smaller horizontal norm of the two ends:
| hnorm_min | steps | fraction with J > 0.25 rad |
|---|---|---|
| < 0.01 | 1 735 | 4.1 % (most are at rest in the singular state, J≈0 until noise flips the sign) |
| 0.01–0.05 | 44 | 70 % |
| 0.05–0.10 | 14 | 71 % |
| 0.10–0.15 | 19 | 47 % |
| 0.15–0.20 | 24 | 17 % |
| 0.20–0.30 | 11 | 9 % (max J 0.37) |
| 0.30–0.50 | 55 | 3.6 % |
| ≥ 0.50 | 67 697 | 0.004 % (3 steps; max J 0.46) |
Events: all 98 singular events have hnorm_min ≤ 0.13 (median 0.005; median J 2.97); the 19 physical rotations have median hnorm_min 0.97 (3 of 19 below 0.2); the 5 ±π crossings have median hnorm_min 0.99 and J ≈ 0 (the wrapped delta is consistent with the geodesic rotation).
**Yaw-validity rule (geometric, current row only):** `heading_valid = hnorm ≥ 0.20` (heading amplification ≤ 5×). At τ = 0.20 all 98 singular events are flagged invalid, 0 of 5 ±π crossings, 3 of 19 physical rotations (16 %, boxes tipped to a high tilt while rotating); τ = 0.1 would miss 6 singular events. Step validity for the loss = both ends valid. The rule is not derived from any model output.
**Coverage of the mask:** 0 invalid steps in 62 918 steps of the 360 sampled non-event episodes; 1 836 of 69 993 steps (2.6 %) overall, all inside event episodes (the event episodes are over-sampled here: 34 of 394 vs 34 of 4 696 in the dataset). Under the mask 66 of 1 836 steps (3.6 %) still carry > 0.1 rad of real geodesic rotation — kept supervised by the rotation-vector target (§5).

## 4. Contract v2 (proposal)
Per-row object block `b2` (14; current row only): pos(3), body-x axis(3), body-y axis(3) (continuous 6-D rotation representation, no wrap, no Euler singularity), speed, **tilt in radians**, a-offset, `heading_valid` (hnorm ≥ 0.20), `hnorm`. Targets per step: Δpos(3), world rotation vector of q_{k+1}⊗q_k⁻¹ (3), Δspeed, Δtilt, Δa-offset (object 9), contact block (7, unchanged), events (4, unchanged). Auxiliary arrays (never inputs): legacy wrapped yaw delta, `yaw_step_valid` (target-time mask, uses rows > t), box-symmetric geodesic step. Tilt is also kept as sym-tilt = min(tilt, π − tilt) for metrics.
**Input/target-time separation (tested, `test_contract_v2.py`, 15/15 PASS):** inputs are bitwise identical when every quantity after row t is replaced; the dict of model inputs contains no target-time mask; `row_features` of a row equals the same row computed alone; rotation-vector targets recompose the orientation trajectory to 4e-8 rad on 60 replayed real episodes.

## 5. Loss proposals and ablation hooks (not run)
- `yaw_huber_masked`: Huber (δ = 0.05 rad) on the **wrapped** residual of per-step yaw deltas, scaled by a robust std of valid non-spike steps (0.0166 rad on current train windows; recomputed on the v2 dataset), multiplied by `yaw_step_valid`, normalised by the number of valid steps. Tests: finite; zero gradient on masked steps; invariant to a 2π offset of the target; a 3.1 rad spike costs ≈ 3 % of the MSE-like cost; masked spike contributes 0.
- `rotvec_loss`: MSE or Huber on the 3-D rotation vectors (no singularity, no wrap) — supervises tumbling and singular windows that the yaw loss masks.
- `clip_only`: clip the target to ±0.2 rad (used only to separate clipping from masking if the matrix is extended).
Offline evidence for these choices (target-energy arithmetic, `yaw_lossprobe.json`): clipping ±0.2 cuts the spike share of target energy from 90 % to 9 %; a pure re-scale does not change the share; circular forms leave 68 % because singular flips are ±π.

## 6. Impact on state, targets, loss, normalisation and checkpoints
| Item | v1 (frozen) | v2 | Consequence |
|---|---|---|---|
| input per row | s 8 + b 7 + c 7 + a 8 = 30 | 8 + 14 + 7 + 8 = 37 | encoder input 4×30 = 120 → 148; MLP inputs change as well |
| object target per step | 7 (xyz, yaw, speed, tilt, a) | 9 (xyz, rotvec 3, speed, tilt, a) | decoder head 22 → 24 outputs per step |
| yaw | wrapped Euler delta, loss-dominating spikes | masked/robust auxiliary; orientation via rotvec | yaw metric of Gate v3 G7 computed from quaternions |
| tilt | mixed rad/deg | radians | tilt normaliser recomputed (v1 std 0.60 was dominated by degrees) |
| normalisation | train-only mean/std over all channels | train-only; rotvec and yaw use robust scale; binary flags unscaled; R6 not standardised | new stats file with its own hash |
| **checkpoints** | WM2-C0, WM3-B, MLPs, ridge | **not loadable** on v2 (shape change) and **not valid** on v1.1 (tilt statistics changed) | frozen baselines stay evaluable on v1 windows only; v2 models must be retrained; for a v2-based comparison the baselines must be re-fitted on v2 (cheap for ridge, retraining for GRUs) |
| intermediate "v1.1" | — | v1 with tilt in radians only | same dims; isolates the tilt-unit repair (arm A1) |
Building v2 requires re-simulating the 4 696 episodes (the quaternion is not stored): estimated ≈ 22 min wall on 4 workers (≈ 1.5 CPU-h of simulation) from the measured replay rate (394 episodes in ≈ 2 min). Output would go to a **new** directory `outputs/wm2_v2/dataset` with manifest, index hash, stats hash and the code commit; the wm2 directory stays read-only. Requires separate approval.

## 7. Open items / limits
1. The geometric threshold τ = 0.20 is supported by a 394-episode replay (event episodes over-sampled), not by the full set; the full replay would give the final mask coverage.
2. Sampling of masks near hnorm 0.2–0.5 shows a few physically-large steps with J > 0.25 (e.g. fast rotation of a tilted box); they remain valid and supervised by Huber — a limitation of any yaw-based supervision.
3. Inference-time validity is not knowable by the model from the future; at inference only `heading_valid` of the current row (input) is available. Gate regimes use the true future and are diagnostic.
