# Gate v3.1 — DESIGN ONLY (static 3-D orientation hallucination); prospective; not frozen; Gate v3.0 unchanged

Date: 2026-11-01. Gate v3.0 (`3f2a6ba6…28b1`) is untouched: this is an additional, separate component (proposed name **G8**) so that G7 keeps its frozen meaning. It is NOT applied retroactively to E1-lite-B/C, I-1 or any development result, and no numeric value below has been chosen or adjusted using any model result.

## 1. Why
E1-lite-B showed that G7 (frozen) can PASS while the rotation-vector head still produces 3-D rotation on input-defined static windows (stable-regime symmetric geodesic 0.042 rad vs persistence 0.007 at K = 20): G7 only gates the yaw component on static windows.

## 2. Definition
- Subset S_in3 (input-only, no future truth): box speed ≤ 0.005 m/s in all history rows, no finger contact and palm force ≤ 0.01 N in all history rows, candidate future gripper command stays open (identical to the G7 S_in rule) — **without** the heading-validity restriction, because a 3-D geodesic is valid at singular orientations. Diagnostic subset S_gt3 (true symmetric geodesic over the horizon < 0.003 rad) uses the future truth and is reported only.
- Errors at K = 5, 10, 20 (K = 20 gating), group-macro mean over initial-state groups, reported separately: **full quaternion geodesic** and **D2 symmetry-aware geodesic** (D2 validity for the blue box verified in the Phase-2 report: single uniform cuboid, uniform appearance, task success without orientation term; it is a conservative subgroup). Tilt (sym-tilt, unit-checked) is reported next to them, not gated.
- Baseline: persistence (zero rotation).
- Components (proposed values, to be confirmed before any freeze): (a) E_sym(K=20) ≤ 0.035 rad; (b) group-bootstrap upper bound of (E_sym,model − E_sym,persistence) ≤ 0.02 rad; (c) E_sym,model / max(E_sym,persistence, 0.01 rad) ≤ 3.0. Verdict logic as v3.0 (FAIL > NOT_EVALUABLE; NOT_APPLICABLE excluded).
- Support: NOT_EVALUABLE if S_in3 has < 10 groups or < 200 windows or if the model has no rotation outputs. Support on the development windows (counts only, no model): 15 157 windows in 96 groups (stable 14 714, rotating 298, tumbling 20, singular 125) — evaluable; regimes other than stable are reported but not gated separately.
- Rationale of the proposed values (engineering assumptions, not fitted): 0.035 rad ≈ 2° is the order of magnitude of a camera-based box-pose estimate and is far below the box-yaw range the task tolerates (below); 0.02 rad ≈ 1° bounds the excess over "no rotation"; the ratio floor 0.01 rad keeps (c) finite because the persistence 3-D error on stable windows is ~0.007 rad; ratio 3.0 mirrors G7.

## 3. Negative controls, boundaries and tests required before a freeze (synthetic fixtures with quaternions)
persistence PASS (ratio 1); oracle PASS; noisy-rotation predictor FAIL (a, b, c); yaw-only hallucination FAIL; **tilt-only hallucination FAIL (this is the case G7 misses)**; a D2-equivalent pose gives zero symmetric error but a non-zero full geodesic, both reported; boundaries at 0.035 / 0.02 / 3.0 on both sides; floor handling when persistence error is 0; upper-bound CI can fail while the mean passes; support < 10 groups or < 200 windows ⇒ NOT_EVALUABLE; missing rotation outputs ⇒ NOT_EVALUABLE; FAIL outranks NOT_EVALUABLE; S_in3 independent of future truth; determinism.

## 4. Relevance to grasp / placement task semantics (`outputs/decision_utility/orientation_task_relevance.json`; 2 484 usable interventional episodes, true trajectories, DEVELOPMENT)
- The placement criterion has **no orientation term** (box centre within 0.03 m xy and 0.006 m z of the tray base, at rest).
- Control variant: 100/100 successes over initial yaw up to 0.198 rad; AUROC of |initial yaw| for failure = 0.503 (no association).
- Episode-level association of orientation with failure is weak and non-monotone: AUROC 0.59 for the box rotation during the hold phase and 0.59 for the maximum sym-tilt. In-hold rotation is 0.2–0.5 rad in 75 % of the episodes (normal carrying with 97.6 % success); the lowest in-hold rotation bin (0.05–0.1 rad, n = 70) has 10 % success because the box was never carried — rotation magnitude is confounded with whether the box was lifted at all.
- Decision-utility study (same phase): feeding yaw=persistence instead of the model's yaw into the rollout changed the selected candidate in 74–77 % of the poses for WM3-B yet changed regret by −0.001 [−0.003, 0.001] / +0.001 [0.000, 0.003] — most candidates are outcome-equivalent, so orientation hallucination of this size did not harm the ranking of this task.
Conclusion: the proposed orientation guard is a **consistency / safety guard** (and a prerequisite for tasks that depend on orientation, e.g. regrasp), not a predictor of the present task's success; it should be labelled as such and not be traded against position metrics.

## 5. Not done / needs approval
No freeze, no code in `wm4_gates.py` (G8 would be added as a separate function with its own config hash and unit tests), no application to any model.
