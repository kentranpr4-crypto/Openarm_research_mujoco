# Issue: PickPlace box-friction randomisation is partly or wholly ineffective (confirmed, NOT fixed)

Status: **open, no change made**. PickPlace physics, task definition and P0/P1 behaviour are untouched. Tool: `research/supermimic_openarm/p1_pickplace_audit.py` (read-only); result file `outputs/p1_validation/pickplace_audit.json` (2026-11-11).

## Evidence [measured on the compiled model and on real `d.contact` entries]
| Contact pair | MuJoCo rule (priorities) | Effective sliding mu over the task's scale range 0.8-1.2 (box mu = 0.8 x scale) | Effect of the randomisation |
|---|---|---|---|
| box - desk | equal priority 0 -> element-wise `max(box, desk 0.8)` | 0.80 (scale 0.8 and 1.0) ... 0.96 (scale 1.2); intended 0.64-0.96 | **scales <= 1.0 have no effect** (half of the range) |
| box - tray base | same, `max(box, tray 0.8)` | same | same |
| box - red cylinder / green cube | same | same | same |
| **finger - box (the grasp)** | finger geoms have priority 1 > box 0 -> the **finger** parameters win | **1.00 for every scale** (spread 0.0) | **grasp friction is not randomised at all** |
Measured on real contacts after reset: desk contact friction 0.8 / 0.8 / 0.96 at scale 0.8 / 1.0 / 1.2; tray contact after a forced placement: 0.8 / 0.8 / 0.96.

## Why it matters
- The sampled `friction_scale` is recorded in `episode_info` as if it varied the task, but in 60% of the range it changes nothing (analytic table: scales 0.8, 0.9, 1.0 give mu 0.8), and the grasp (the physics that decides pick success) never sees it.
- Any claim of robustness to friction for PickPlace policies is unsupported by the current environment.

## Not affected
P0 reaching (no box contact in the task); P1 pushing (fixed separately with `friction_mode="pair_exact"` and measured: effective mu 0.56-1.04); existing checkpoints and datasets. The P0 compatibility test (reaching and pick_place trajectories bit-identical to git HEAD) passes with the P1 change in place.

## Options (for the owner; none applied)
1. Do nothing and document that PickPlace friction is nominal (mu 0.8 sliding, 1.0 grasp).
2. Sliding pairs only: give the box geoms `priority=1` as in pushing (box-desk/tray then use the box's own mu). Finger-box stays 1.0 (priorities tie at 1, `max(1.0, 0.64-0.96)`).
3. Randomise the grasp on purpose: scale the finger-pad mu (or set box priority 2, which would make the grasp mu 0.64-0.96, i.e. LOWER than the current 1.0). Option 3 changes the task difficulty and needs a re-measured expert / success baseline first.

## Acceptance criteria for a fix (to be agreed before coding)
- effective mu measured from `d.contact` equals the sampled value for every affected pair;
- slide distance of a launched box follows `v^2 / (2 mu g)` within 3% across the range (same test as pushing);
- the grasp success of the scripted pick-place expert is re-measured on a fresh seed set and reported with Wilson 95%;
- P0 reaching trajectories remain bit-identical to the commit before the change.
