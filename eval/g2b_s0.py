#!/usr/bin/env python3
"""G2b candidate S0 = pure time-shift null (B3).  DRAFT; not executed against the simulator.
S0: at the decision state hold the LAST EXECUTED 16-D target for k = 8 control steps (every held command still passes the clip -> slew chain, so ctrlrange/slew semantics are those of every other arm), then resume ACT normally: the next chunk is queried from the observation AFTER the pause (state + images of that moment) and all
later chunk boundaries shift by k.  The branch horizon is absolute (t_end), so S0 has k fewer ACT steps inside it.  ACT here is the stateless chunk model (ChunkPolicy: no action queue, no temporal ensembling, no action history input); a stateful runner (ACTRunner) would have to be reset - `reset_policy_state` does that and the tests cover it.
IMPORTANT (interpretation): S0 can GENUINELY change the physics (servo error decays, gravity sag, contact settling, a different image/state at resume), so a failed equivalence test of S0 against c0 does not by itself prove a timing bias of the outcome labels; the labeller's own time-shift invariance is tested separately (eval/test_g2b_labeller.py) and G2b compares event-relative physical quantities between S0 and c0 to attribute any difference."""
import numpy as np
DT = 0.05; K_S0 = 8; CHUNK = 20
def slew_chain(lim, raw, prev):
    out = []; p = np.asarray(prev, float)
    for r in raw: c, _, _ = lim.clip_and_slew(r, p, DT); out.append(c); p = c
    return np.array(out)
def reset_policy_state(policy):
    """stateful runners keep an action queue / temporal-ensemble buffer that must not leak across the pause; stateless chunk policies have nothing to reset"""
    inner = getattr(policy, "policy", None)
    if inner is not None and hasattr(inner, "reset"): inner.reset(); return True
    if hasattr(policy, "reset"): policy.reset(); return True
    return False
def s0_query_steps(t_dec, t_end, k=K_S0, chunk=CHUNK): return list(range(t_dec + k, t_end, chunk))
def run_branch(env, policy, lim, prev, t_dec, t_end, k=0, chunk=CHUNK, on_step=None):
    """Execute from the decision step t_dec to the absolute horizon t_end.  k = 0 is the nominal schedule (c0); k > 0 is S0 (pause of k steps first).
    env interface: step(ctrl) (advance one control step), observe() -> dict(state, images); policy interface: chunk(state, images) -> (chunk, 16) raw targets.  Returns (executed commands (t_end - t_dec, 16), list of (step, role))."""
    t = t_dec; prev = np.asarray(prev, float).copy(); exe_all = []; log = []
    def emit(u, role):
        nonlocal t, prev
        if on_step is not None: on_step(t, u, role)
        env.step(u); exe_all.append(np.asarray(u, float).copy()); prev = np.asarray(u, float).copy(); t += 1
    if k > 0:
        reset_policy_state(policy)
        for u in slew_chain(lim, np.tile(prev, (k, 1)), prev):
            if t >= t_end: break
            emit(u, "hold")
        log.append((t, "pause_end"))
    while t < t_end:
        o = env.observe(); raw = policy.chunk(o["state"], o["images"]); log.append((t, "query")); exe = slew_chain(lim, raw, prev)
        for u in exe[:min(chunk, t_end - t)]: emit(u, "act")
    return np.array(exe_all), log
