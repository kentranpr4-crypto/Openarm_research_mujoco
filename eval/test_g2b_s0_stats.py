#!/usr/bin/env python3
"""Static / unit tests for S0 (mock environment + policy, REAL safety limits from the saved actuator arrays; no simulator) and for the paired statistics (synthetic data).  NOT an evaluation.
Run: python3 eval/test_g2b_s0_stats.py -> outputs/g2b_impl/s0_stats_report.json"""
import sys, json, time, traceback, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "eval")): sys.path.insert(0, p)
import g2b_s0 as S0, g2b_stats as ST
from control.safety_limits import SafetyLimits
Z = np.load(ROOT / "outputs/closed_loop_design/stage0_1_chunks.npz"); LIM = SafetyLimits(Z["lo"].astype(float), Z["hi"].astype(float), Z["max_speed"].astype(float)); DT = 0.05
RES = []
def test(name):
    def deco(fn):
        try: ok, ev = fn()
        except Exception as e: ok, ev = False, f"EXCEPTION {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
        RES.append(dict(test=name, result="PASS" if ok else "FAIL", evidence=ev)); print(("PASS " if ok else "FAIL ") + name, "" if ok else json.dumps(ev, default=float)[:300], flush=True); return fn
    return deco
# ------------------------------------------------------------------------------------------------------------------------------ mocks
class MockEnv:
    """time-invariant ideal environment: the observable state is the last executed command"""
    def __init__(self, s0, t0=40): self.state = np.asarray(s0, float).copy(); self.t = t0; self.executed = []        # t = ABSOLUTE control step
    def step(self, u): self.executed.append(np.asarray(u, float).copy()); self.state = np.asarray(u, float).copy(); self.t += 1
    def observe(self): return dict(state=self.state.copy(), images=dict(t=self.t))
class MockPolicy:
    """stateless, depends on the observation only; mode 'smooth' or 'jumpy' (large jumps and out-of-range values to exercise clip/slew)"""
    def __init__(self, mode="smooth"): self.mode = mode; self.calls = []
    def chunk(self, state, images):
        self.calls.append(images["t"]); j = np.arange(1, S0.CHUNK + 1)[:, None]; raw = state[None, :] + 0.004 * j
        if self.mode == "jumpy": raw = state[None, :] + 1.0 * np.sign(np.sin(j * 1.3)) + 0.5 ; raw[:, 7] = 0.2
        return raw
class MockStatefulPolicy(MockPolicy):
    class _Inner:
        def __init__(self): self.resets = 0
        def reset(self): self.resets += 1
    def __init__(self): super().__init__(); self.policy = self._Inner()
PREV = np.clip(0.5 * (LIM.lo + LIM.hi) * 0.3, LIM.lo, LIM.hi); T_DEC, T_END = 40, 160
def run(k, policy=None, env=None, prev=None):
    env = env or MockEnv(PREV if prev is None else prev); policy = policy or MockPolicy(); exe, log = S0.run_branch(env, policy, LIM, PREV if prev is None else prev, T_DEC, T_END, k=k); return env, policy, exe, log
@test("S0: the first 8 executed commands equal the last executed target exactly (all 16 channels incl. gripper and right arm)")
def _():
    env, pol, exe, log = run(8); d = float(np.abs(exe[:8] - PREV[None]).max()); return d == 0.0, dict(max_abs_diff=d)
@test("S0: ACT is first queried AFTER the pause, from the updated observation (t = 48), then every 20 steps; never at t = 40")
def _():
    env, pol, exe, log = run(8); return pol.calls == [48, 68, 88, 108, 128, 148] and S0.s0_query_steps(40, 160) == [48, 68, 88, 108, 128, 148], dict(calls=pol.calls)
@test("c0 (k=0) queries at 40, 60, ... (nominal schedule)")
def _():
    env, pol, exe, log = run(0); return pol.calls == [40, 60, 80, 100, 120, 140], dict(calls=pol.calls)
@test("clip then slew from the last executed command is preserved for hold and resume: every executed command in range and within the slew limit, also for a jumpy / out-of-range policy")
def _():
    worst = 0.0; ok = True
    for mode in ("smooth", "jumpy"):
        env, pol, exe, log = run(8, policy=MockPolicy(mode)); e = np.vstack([PREV[None], exe]); rate = np.abs(np.diff(e, axis=0)) / DT; worst = max(worst, float((rate / LIM.max_speed[None]).max())); ok &= bool(((exe >= LIM.lo - 1e-12) & (exe <= LIM.hi + 1e-12)).all() and (rate <= LIM.max_speed[None] * (1 + 1e-9) + 1e-9).all())
    return ok, dict(max_rate_over_limit=worst)
@test("horizon accounting: both arms execute exactly t_end - t_dec steps; S0 has 8 fewer ACT steps inside the absolute horizon")
def _():
    e0, p0, x0, l0 = run(0); e8, p8, x8, l8 = run(8); return len(x0) == 120 and len(x8) == 120 and len(e8.executed) == 120, dict(n_c0=len(x0), n_s0=len(x8), act_steps_s0=120 - 8)
@test("k=0 equals an independent re-implementation of the nominal schedule (query at the observation, clip->slew chain)")
def _():
    env, pol, exe, log = run(0); prev = PREV.copy(); ref = []; st = PREV.copy()
    for t in range(T_DEC, T_END, S0.CHUNK):
        raw = MockPolicy().chunk(st, dict(t=t)); u = S0.slew_chain(LIM, raw, prev); ref += list(u[:min(S0.CHUNK, T_END - t)]); prev = u[-1]; st = prev
    return np.array_equal(np.array(ref), exe), dict(max_abs_diff=float(np.abs(np.array(ref) - exe).max()))
@test("pure-shift property on an ideal time-invariant environment: S0 executed sequence == c0 sequence delayed by exactly 8 steps (real physics need not satisfy this)")
def _():
    _, _, x0, _ = run(0); _, _, x8, _ = run(8); return bool(np.array_equal(x8[8:], x0[:-8])), dict(max_abs_diff=float(np.abs(x8[8:] - x0[:-8]).max()))
@test("controller/cache state: a stateful runner is reset exactly once before the resume (S0 only); a stateless chunk policy needs no reset")
def _():
    sp = MockStatefulPolicy(); run(8, policy=sp); sp0 = MockStatefulPolicy(); run(0, policy=sp0); stateless = MockPolicy(); run(8, policy=stateless); return sp.policy.resets == 1 and sp0.policy.resets == 0, dict(resets_s0=sp.policy.resets, resets_c0=sp0.policy.resets)
@test("hold uses the actually executed previous command (not the raw policy output): if the previous command was slewed, S0 repeats the slewed value")
def _():
    raw_prev = np.clip(PREV + 3.0, LIM.lo, LIM.hi); ex = S0.slew_chain(LIM, raw_prev[None], PREV)[0]; env, pol, exe, log = run(8, prev=ex); return np.array_equal(exe[:8], np.tile(ex, (8, 1))), dict(max_jump=float(np.abs(ex - PREV).max()))
# ------------------------------------------------------------------------------------------------------------------------------ statistics
def mk(poses, arms): return {a: {p: dict(value=v[0], ambiguous=bool(v[1]) if len(v) > 1 else False) for p, v in zip(poses, vals)} for a, vals in arms.items()}
P40 = list(range(40)); rng = np.random.default_rng(3); C0 = [("TRUE",) if rng.random() < 0.6 else ("FALSE",) for _ in P40]
def arm_from(c0, flips_up=(), flips_down=(), special=None):
    out = [list(x) for x in c0]
    for i in flips_up: out[i] = ["TRUE"]
    for i in flips_down: out[i] = ["FALSE"]
    for i, v in (special or {}).items(): out[i] = list(v)
    return [tuple(x) for x in out]
FULL = lambda **kw: {a: C0 for a in ST.ARMS} | kw
@test("pairing: identical arms give RD = 0 with a degenerate interval; improvements counted per pose")
def _():
    data = mk(P40, FULL()); r = ST.compare(data, "O1", P40, 0.975, B=2000); ups = [i for i, v in enumerate(C0) if v[0] == "FALSE"][:6]; data2 = mk(P40, FULL(O1=arm_from(C0, flips_up=ups))); r2 = ST.compare(data2, "O1", P40, 0.975, B=4000)
    return r["rd"] == 0 and r["ci"] == [0.0, 0.0] and r2["improved"] == 6 and r2["harmed"] == 0 and abs(r2["rd"] - 0.15) < 1e-12 and r2["ci"][0] > 0, dict(identical=r["ci"], rd=r2["rd"], ci=r2["ci"])
@test("one policy for all arms: poses with an unresolved/invalid c0 are dropped from EVERY comparison; attrition is reported, never silent")
def _():
    bad = {3: ("INVALID",), 7: ("CENSORED",), 11: ("NOT_APPLICABLE",)}; data = mk(P40, FULL(c0=arm_from(C0, special=bad))); sets = {}; rep = {}
    for a in ("O1", "O3", "D1", "D2", "H1", "S0"): used, _, _, att = ST.paired_values(data, a, P40); sets[a] = set(used); rep[a] = att["c0_not_resolved"]
    keep, dropped = ST.base_population(data, P40); return all(s == set(keep) for s in sets.values()) and set(dropped) == {3, 7, 11} and set(rep.values()) == {3}, dict(dropped=dropped, n_used=len(keep))
@test("missing/censored/invalid candidate branches: complete-case excludes them with counted reasons; worst-case counts them FALSE, best-case TRUE; ambiguous-but-valid labels keep their value")
def _():
    sp = {1: ("INVALID",), 2: ("CENSORED",), 5: ("TRUE", True)}; arm = arm_from(C0, special=sp); data = mk(P40, FULL(D1=arm)); cc = ST.compare(data, "D1", P40, 0.95, B=1000); wc = ST.compare(data, "D1", P40, 0.95, "worst_case", B=1000); bc = ST.compare(data, "D1", P40, 0.95, "best_case", B=1000)
    return cc["attrition"]["arm_specific"] == {"invalid_excluded": 1, "censored_excluded": 1} and cc["n_poses"] == 38 and wc["n_poses"] == 40 and wc["rd"] <= cc["rd"] <= bc["rd"] + 1e-12 and "ambiguous_excluded" not in cc["attrition"]["arm_specific"], dict(cc=cc["attrition"], rd=[wc["rd"], cc["rd"], bc["rd"]])
@test("all-arms-complete population is a subset of every pairwise complete-case population; ambiguity analyses ordered adverse <= complete-case <= favourable")
def _():
    d = {a: C0 for a in ST.ARMS}; d["O3"] = arm_from(C0, special={9: ("INVALID",)}); d["D1"] = arm_from(C0, special={4: ("FALSE", True), 6: ("TRUE", True)}); data = mk(P40, d)
    u_all, *_ = ST.paired_values(data, "D1", P40, "all_arms_complete"); u_cc, *_ = ST.paired_values(data, "D1", P40, "complete_case"); adv = ST.compare(data, "D1", P40, 0.95, "adverse_ambiguous", B=500)["rd"]; cc = ST.compare(data, "D1", P40, 0.95, B=500)["rd"]; fav = ST.compare(data, "D1", P40, 0.95, "favourable_ambiguous", B=500)["rd"]
    return set(u_all) < set(u_cc) and 9 not in u_all and adv <= cc <= fav, dict(n_all=len(u_all), n_cc=len(u_cc), rd=[adv, cc, fav])
@test("CI levels: primary 97.5 % (wider than 95 %), secondary 95 %, equivalence 90 %; oracle (O1/O3) and deployable (D1/D2/H1/S0) families are tagged and reported separately")
def _():
    ups = [i for i, v in enumerate(C0) if v[0] == "FALSE"][:5]; data = mk(P40, FULL(O1=arm_from(C0, flips_up=ups), D1=arm_from(C0, flips_up=ups[:3]))); rep = ST.full_report(data, P40, B=2000)
    w975 = ST.compare(data, "O1", P40, 0.975, B=4000)["ci"]; w95 = ST.compare(data, "O1", P40, 0.95, B=4000)["ci"]
    return ST.LEVELS == dict(primary=0.975, secondary=0.95, equivalence=0.90) and (w975[1] - w975[0]) >= (w95[1] - w95[0]) and rep["primary"]["O1"]["family"] == "oracle" and rep["primary"]["D1"]["family"] == "deployable" and set(rep["primary"]) == {"O1", "D1"} and rep["primary"]["O1"]["level"] == 0.975 and all(rep["secondary"][a]["level"] == 0.95 for a in ("O3", "D2", "H1")), dict(width975=w975[1] - w975[0], width95=w95[1] - w95[0])
@test("TOST: equivalent iff the 90 % interval lies inside +-0.10 (not the 95 % one); an example exists where the 90 % interval qualifies and the 95 % does not; large shifts are rejected")
def _():
    found = None
    for npl in range(0, 9):
        for nmi in range(0, 9):
            fl = [i for i, v in enumerate(C0) if v[0] == "FALSE"][:npl]; tr = [i for i, v in enumerate(C0) if v[0] == "TRUE"][:nmi]; data = mk(P40, FULL(S0=arm_from(C0, flips_up=fl, flips_down=tr))); r = ST.equivalence_tost(data, P40, B=3000)
            if r["equivalent"] and not (r["ci95"][0] > -0.10 and r["ci95"][1] < 0.10): found = (npl, nmi, r["ci90"], r["ci95"]); break
        if found: break
    big = mk(P40, FULL(S0=arm_from(C0, flips_down=[i for i, v in enumerate(C0) if v[0] == "TRUE"][:8]))); rb = ST.equivalence_tost(big, P40, B=3000); same = ST.equivalence_tost(mk(P40, FULL()), P40, B=1000)
    return found is not None and (not rb["equivalent"]) and same["equivalent"], dict(example_flips_up_down=found[:2] if found else None, ci90=found[2] if found else None, ci95=found[3] if found else None, big_shift_equivalent=rb["equivalent"])
@test("claim rule: improvement is claimed only if SUPPORTED_BETTER AND worst-case point estimate >= 0 AND S0 equivalence holds; verified in both directions")
def _():
    fl = [i for i, v in enumerate(C0) if v[0] == "FALSE"][:12]; trues = [i for i in range(40) if C0[i][0] == "TRUE"]
    bad = mk(P40, FULL(O1=arm_from(C0, flips_up=fl, special={i: ("INVALID",) for i in trues[:16]}))); good = mk(P40, FULL(O1=arm_from(C0, flips_up=fl)))
    rb = ST.full_report(bad, P40, B=2000); rg = ST.full_report(good, P40, B=2000); wb = rb["sensitivity"]["O1"]["worst_case"]["rd"]
    return rb["primary"]["O1"]["verdict"] == "SUPPORTED_BETTER" and wb < 0 and rb["primary"]["O1"]["claim_supported"] is False and rg["primary"]["O1"]["claim_supported"] is True, dict(bad_cc_rd=rb["primary"]["O1"]["rd"], bad_worst_rd=wb, bad_claim=rb["primary"]["O1"]["claim_supported"], good_claim=rg["primary"]["O1"]["claim_supported"])
@test("determinism: the same seed gives the same interval; Monte-Carlo power of the S0 rule at N=40, 5 % discordance is moderate (documented limitation)")
def _():
    data = mk(P40, FULL(O1=arm_from(C0, flips_up=[i for i, v in enumerate(C0) if v[0] == "FALSE"][:4]))); a = ST.compare(data, "O1", P40, 0.975, B=2000, seed=0)["ci"]; b = ST.compare(data, "O1", P40, 0.975, B=2000, seed=0)["ci"]; pw = ST.equivalence_power(40, 0.05, sims=300, B=500)
    return a == b and 0.3 <= pw <= 0.85, dict(power_N40_d005=pw)
rep = dict(verdict="PASS" if all(r["result"] == "PASS" for r in RES) else "FAIL", n_tests=len(RES), results=RES, params=dict(K_S0=S0.K_S0, margin=ST.MARGIN, levels=ST.LEVELS), note="mock/synthetic only; no simulator; no evaluation")
Path(ROOT / "outputs/g2b_impl").mkdir(parents=True, exist_ok=True); (ROOT / "outputs/g2b_impl/s0_stats_report.json").write_text(json.dumps(rep, indent=1, default=float)); print(rep["verdict"], rep["n_tests"])
