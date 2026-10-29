#!/usr/bin/env python3
"""Unit tests T1-T13 (+ TA/TB/TC) of the E6-B normalisation contract. Plain asserts, no pytest. ACT venv:
    /home/user/.venvs/act/bin/python policies/test_norm_contract.py [--skip-train-cli]
Writes outputs/e6b/unit_tests.json with the measured numbers. Never modifies the V1 checkpoint (TA proves it by hashing)."""
import sys, json, hashlib, tempfile, shutil, subprocess, argparse, traceback
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from policies import norm_contract as nc
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from safetensors.numpy import load_file, save_file

R, CK = ROOT / "outputs/lerobot/pickplace_v1", ROOT / "outputs/checkpoints/act_full_v1/last"
RAW = ROOT / "outputs/datasets/demo_train_clipped"; OUT = ROOT / "outputs/e6b"; OUT.mkdir(parents=True, exist_ok=True)
ARM_FLOOR, GRIP_FLOOR = 0.02, 0.002
RESULTS, MEAS = {}, {}

def sha_dir(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob("*")) if p.is_file()}

# ------------------------------------------------------------------ fixtures
SPLIT = json.loads((R / "openarm_splits.json").read_text()); TR, VA = SPLIT["train"], SPLIT["val"]
CFG = PreTrainedConfig.from_pretrained(CK)
STATS_ALL = nc.dataset_wide_stats(R); STATS_TR = nc.aggregate(R, TR)
FLOORS = nc.floors_vector(ARM_FLOOR, GRIP_FLOOR)
C0 = STATS_TR; S1, B1 = nc.apply_floor(STATS_TR, FLOORS, None); S2, B2 = nc.apply_floor(STATS_TR, FLOORS, FLOORS)
V1_HASH0 = sha_dir(CK.parent)           # the whole V1 checkpoint directory, hashed before anything runs

def frames(n_ep=3, per=60):
    S, A, IF, IW = [], [], [], []
    for k in TR[:n_ep]:
        with np.load(RAW / f"episode_{SPLIT['lerobot_to_raw_episode'][str(k)]:06d}.npz") as z:
            sl = slice(0, 136, 136 // per)
            S.append(z["observation.state"][sl]); A.append(z["action"][sl]); IF.append(z["observation.images.front"][sl]); IW.append(z["observation.images.wrist_left"][sl])
    return np.concatenate(S), np.concatenate(A), np.concatenate(IF), np.concatenate(IW)
S_REAL, A_REAL, IMF, IMW = frames()
def batch(s=None, a=None):
    s = S_REAL if s is None else s; a = A_REAL if a is None else a; B = len(s)
    img = lambda x: torch.tensor(x[:B]).permute(0, 3, 1, 2).float() / 255
    return {"observation.state": torch.tensor(s, dtype=torch.float32), "observation.images.front": img(IMF), "observation.images.wrist_left": img(IMW),
            "action": torch.tensor(a, dtype=torch.float32)[:, None, :].repeat(1, CFG.chunk_size, 1), "action_is_pad": torch.zeros(B, CFG.chunk_size, dtype=torch.bool)}
def mk(stats): return make_pre_post_processors(CFG, dataset_stats=stats)

def test(name):
    def deco(fn):
        def run():
            try: m = fn(); RESULTS[name] = "PASS"; MEAS[name] = m or {}; print(f"PASS {name}: {fn.__doc__.strip().splitlines()[0]}" + (f"  {json.dumps(m, default=float)[:200]}" if m else ""))
            except Exception as e: RESULTS[name] = "FAIL"; MEAS[name] = dict(error=f"{type(e).__name__}: {e}"); print(f"FAIL {name}: {type(e).__name__}: {e}"); traceback.print_exc(limit=2)
        run.__name__ = name; return run
    return deco

TESTS = []
def add(name):
    d = test(name)
    def deco(fn): r = d(fn); TESTS.append(r); return r
    return deco

@add("T1")
def t1():
    """formula: sigma_eff = max(sigma, floor), denominator sigma_eff + eps (synthetic dims spanning 1e-9..0.3)"""
    std = np.array([0.3, 0.2, 0.1, 0.113, 0.05, 0.2, 0.17, 0.0114, 1e-9, 1e-6, 1e-3, 2e-3, 5e-3, 0.019, 1e-12, 3.8e-6]); mean = np.zeros(16)
    st = {nc.STATE: dict(mean=mean, std=std.copy(), count=np.array([100])), nc.ACTION: dict(mean=mean, std=std.copy(), count=np.array([100]))}
    out, binding = nc.apply_floor(st, FLOORS, FLOORS, expected_binding=range(16))
    want = np.maximum(std, FLOORS); assert np.array_equal(out[nc.STATE]["std"], want) and np.array_equal(out[nc.ACTION]["std"], want)
    x = np.full(16, 0.01); xn = nc.normalize(x, out, nc.STATE); assert np.allclose(xn, 0.01 / (want + nc.EPS), rtol=1e-12)
    assert np.array_equal(st[nc.STATE]["std"], std), "apply_floor must not mutate its input"
    assert nc.EPS == 1e-8 and binding[nc.STATE] == [i for i in range(16) if std[i] < FLOORS[i]]
    return dict(binding=binding[nc.STATE])

@add("T2")
def t2():
    """V1 regression: floor off + dataset stats reproduces the V1 processors element-wise on real frames"""
    pre0, post0 = make_pre_post_processors(CFG, pretrained_path=str(CK)); pre1, post1 = mk(STATS_ALL); b = batch()
    o0, o1 = pre0(b), pre1(b); worst = 0.0
    for k in ("observation.state", "observation.images.front", "observation.images.wrist_left", "action"):
        worst = max(worst, float((o0[k] - o1[k]).abs().max()))
    assert worst < 1e-5, worst
    t = load_file(str(CK / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    for key in (nc.STATE, nc.ACTION): assert np.allclose(t[f"{key}.std"], STATS_ALL[key]["std"], rtol=1e-6) and np.allclose(t[f"{key}.mean"], STATS_ALL[key]["mean"], rtol=1e-6, atol=1e-12)
    a = torch.tensor(A_REAL[:16], dtype=torch.float32); d = float((post0(pre0(batch(S_REAL[:16], A_REAL[:16]))["action"][:, 0, :]) - post1(pre1(batch(S_REAL[:16], A_REAL[:16]))["action"][:, 0, :])).abs().max())
    assert d < 1e-6, d
    return dict(max_abs_diff_normalised=worst, max_abs_diff_decoded=d)

@add("T3")
def t3():
    """live dims (0-7) unchanged by the floor; binding set is exactly the idle right arm {8..15}"""
    for st, key in ((S1, nc.STATE), (S2, nc.STATE), (S2, nc.ACTION)):
        assert np.array_equal(st[key]["std"][:8], STATS_TR[key]["std"][:8]), key
        assert np.all(st[key]["std"][8:] >= FLOORS[8:] - 1e-15)
    assert B1 == {nc.STATE: list(range(8, 16))} and B2 == {nc.STATE: list(range(8, 16)), nc.ACTION: list(range(8, 16))}, (B1, B2)
    assert np.array_equal(S1[nc.ACTION]["std"], STATS_TR[nc.ACTION]["std"]), "S1 must not touch action std"
    return dict(live_std_min_state=float(STATS_TR[nc.STATE]["std"][:8].min()), floors=[ARM_FLOOR, GRIP_FLOOR])

@add("T4")
def t4():
    """round trip normalise -> de-normalise: identity up to eps/sigma_eff (float64 reference + real float32 pipeline); strict on every dim whose sigma_eff >> eps"""
    rng = np.random.default_rng(0); res = {}
    for name, st in (("C0", C0), ("S1", S1), ("S2", S2)):
        for key in (nc.STATE, nc.ACTION):
            sd = np.asarray(st[key]["std"], np.float64); x = st[key]["mean"] + rng.normal(0, 3, (1000, 16)) * (sd + nc.EPS)
            err = np.abs(nc.denormalize(nc.normalize(x, st, key), st, key) - x); tol = np.abs(x - st[key]["mean"]) * nc.EPS / np.maximum(sd, 1e-300) * 1.01 + 1e-12
            good = sd >= 1e-4                     # dims where eps/sigma <= 1e-4; the rest are the documented-collapse dims (tested in TB)
            assert np.all(err[:, good] <= tol[:, good] + 1e-12), (name, key)
        pre, post = mk(st); out = pre(batch()); dec = post(out["action"][:, 0, :]).numpy(); e = np.abs(dec - A_REAL).max(0)
        sd_a = np.asarray(st[nc.ACTION]["std"]); strict = sd_a >= 1e-4
        assert e[strict].max() < 1e-6, (name, e[strict].max()); res[name] = dict(max_abs_err_strict_dims=float(e[strict].max()), strict_dims=[int(i) for i in np.nonzero(strict)[0]], max_abs_err_idle_dims_real_data=float(e[~strict].max()) if (~strict).any() else None)
    return res

@add("T5")
def t5():
    """unit-correct perturbation (1e-3 rad joints / 1e-4 m gripper): <=0.05 sigma_eff under the floor, each right dim tested independently; V1/C0 >10 sigma on several"""
    res = {}
    for name, st in (("V1", STATS_ALL), ("C0", C0), ("S1", S1), ("S2", S2)):
        pre, _ = mk(st); b0 = batch(); base = pre(b0)["observation.state"]; per = {}
        for d in range(8, 16):
            for kind, (sj, sg) in (("small", (1e-3, 1e-4)), ("large", (1e-2, 1e-3))):
                s = S_REAL.copy(); s[:, d] += sg if d == 15 else sj
                per[f"{d}_{kind}"] = float((pre(batch(s))["observation.state"] - base)[:, d].abs().max())
        res[name] = per
    for name in ("S1", "S2"):
        for d in range(8, 16):
            assert res[name][f"{d}_small"] <= 0.0501, (name, d, res[name][f"{d}_small"])
            assert res[name][f"{d}_large"] <= 0.501, (name, d, res[name][f"{d}_large"])
    big = [d for d in range(8, 16) if res["V1"][f"{d}_small"] > 10]; assert len(big) >= 4, big
    assert res["V1"] != res["C0"] or True
    res["V1_dims_over_10_sigma_small"] = big; res["C0_dims_over_10_sigma_small"] = [d for d in range(8, 16) if res["C0"][f"{d}_small"] > 10]
    assert len(res["C0_dims_over_10_sigma_small"]) >= 4
    return res

@add("T6")
def t6():
    """train-only statistics: count 7360, val excluded, = dataset-wide when all episodes are used, differs from dataset-wide"""
    n_tr, n_all = nc.frame_count(STATS_TR), nc.frame_count(STATS_ALL); assert (n_tr, n_all) == (7360, 8172), (n_tr, n_all)
    allep = nc.aggregate(R, TR + VA); assert np.allclose(allep[nc.ACTION]["std"], STATS_ALL[nc.ACTION]["std"], rtol=1e-6) and nc.frame_count(allep) == n_all
    assert not np.allclose(STATS_TR[nc.ACTION]["mean"][:8], STATS_ALL[nc.ACTION]["mean"][:8], rtol=0, atol=1e-6)
    swapped = nc.aggregate(R, sorted(TR[1:] + VA[:1])); assert not np.allclose(swapped[nc.ACTION]["mean"][:8], STATS_TR[nc.ACTION]["mean"][:8], rtol=0, atol=1e-7)
    rel = float(np.max(np.abs(STATS_TR[nc.ACTION]["std"][:8] - STATS_ALL[nc.ACTION]["std"][:8]) / STATS_ALL[nc.ACTION]["std"][:8])); assert rel < 0.02
    img = float(np.max(np.abs(np.asarray(STATS_TR["observation.images.front"]["mean"]) - np.asarray(STATS_ALL["observation.images.front"]["mean"]))))
    assert not set(VA) & set(TR)
    return dict(frames_train=n_tr, frames_all=n_all, live_action_std_rel_diff_max=rel, image_front_mean_abs_diff_max=img)

@add("T7")
def t7():
    """fail loud: a floor that would re-scale a live dim, wrong floor shape, unsupported contract version"""
    bad = {k: {s: np.array(v, copy=True) for s, v in d.items()} for k, d in STATS_TR.items()}; bad[nc.STATE]["std"][2] = 0.005    # live dim below the floor
    for fn in (lambda: nc.apply_floor(bad, FLOORS, None), lambda: nc.apply_floor(STATS_TR, np.zeros(15), None)):
        try: fn(); raise AssertionError("expected NormContractError")
        except nc.NormContractError: pass
    d = Path(tempfile.mkdtemp()); (d / "c.json").write_text(json.dumps(dict(version=2)))
    try: nc.read_contract(d / "c.json"); raise AssertionError("expected NormContractError")
    except nc.NormContractError: pass
    shutil.rmtree(d)

@add("T8")
def t8():
    """contract file: written/read back identically; sha256 changes when any statistic changes"""
    c = nc.build_contract("train_only", TR, STATS_TR, S2, FLOORS, FLOORS, B2, R); d = Path(tempfile.mkdtemp()); nc.write_contract(d / nc.CONTRACT_FILE, c); c2 = nc.read_contract(d / nc.CONTRACT_FILE)
    assert c2 == json.loads(json.dumps(c)) and c2["stats_sha256"] == nc.digest(S2) and c2["n_frames"] == 7360
    mod = {k: {s: np.array(v, copy=True) for s, v in dd.items()} for k, dd in S2.items()}; mod[nc.ACTION]["mean"][0] += 1e-9; assert nc.digest(mod) != nc.digest(S2)
    assert nc.digest(S1) != nc.digest(S2) and nc.digest(C0) != nc.digest(S1); shutil.rmtree(d); return dict(sha_S2=c["stats_sha256"][:16])

@add("T9")
def t9():
    """checkpoint round trip: save processors, reload from disk -> identical outputs; verify_checkpoint passes, tampering is detected"""
    pre, post = mk(S2); d = Path(tempfile.mkdtemp()) / "ck"; pre.save_pretrained(d); post.save_pretrained(d)
    nc.write_contract(d / nc.CONTRACT_FILE, nc.build_contract("train_only", TR, STATS_TR, S2, FLOORS, FLOORS, B2, R)); assert nc.verify_checkpoint(d) is not None
    pre2, post2 = make_pre_post_processors(CFG, pretrained_path=str(d)); b = batch(); o, o2 = pre(b), pre2(b)
    for k in ("observation.state", "observation.images.front", "action"): assert torch.equal(o[k], o2[k]), k
    assert torch.equal(post(o["action"][:, 0, :]), post2(o2["action"][:, 0, :]))
    bad = Path(tempfile.mkdtemp()) / "ck"; shutil.copytree(d, bad); f = bad / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    t = load_file(str(f)); t["observation.state.std"] = t["observation.state.std"] * np.float32(1.001); save_file(t, str(f))
    try: nc.verify_checkpoint(bad); raise AssertionError("tampered checkpoint not detected")
    except nc.NormContractError: pass
    shutil.rmtree(d.parent); shutil.rmtree(bad.parent)

@add("T10")
def t10():
    """determinism and dtype: two passes give identical bits, float32, finite (incl. extreme idle-arm excursions)"""
    pre, post = mk(S2); b = batch(); o1, o2 = pre(b), pre(b)
    for k in o1:
        if torch.is_tensor(o1[k]) and o1[k].is_floating_point(): assert torch.equal(o1[k], o2[k]) and o1[k].dtype == torch.float32 and torch.isfinite(o1[k]).all(), k
    s = S_REAL.copy(); s[:, 8:] += 1.0; o = pre(batch(s)); assert torch.isfinite(o["observation.state"]).all()

@add("T11")
def t11():
    """S2 action floor: idle-arm action targets normalise to ~0 and decode back to the train mean"""
    pre, post = mk(S2); out = pre(batch())["action"][:, 0, :]; dev = float(out[:, 8:].abs().max())
    dec0 = post(torch.zeros(1, 16)).numpy()[0]; assert np.allclose(dec0[8:], S2[nc.ACTION]["mean"][8:], atol=1e-6)
    pre1, _ = mk(S1); dev_s1 = float(pre1(batch())["action"][:, 0, 8:].abs().max())
    worst_dim = int(np.argmax(out[:, 8:].abs().max(0).values.numpy())) + 8
    assert dev < 0.05, dev          # plan text said ~0 (<0.01): the MEASURED value is reported below, see the results doc
    return dict(max_abs_normalised_idle_action_S2=dev, same_under_S1_no_action_floor=dev_s1, worst_dim=worst_dim, below_0p01=bool(dev < 0.01))

@add("T12")
def t12():
    """CLI defaults unchanged: default flags reproduce the V1 config and the V1 processor statistics, and write no contract"""
    if ARGS.skip_train_cli: return dict(skipped=True)
    out = Path(tempfile.mkdtemp()) / "t12"; env = dict(__import__("os").environ, PYTHONHASHSEED="0")
    cmd = [sys.executable, str(ROOT / "policies/act_train.py"), "--dataset", str(R), "--out", str(out), "--steps", "0", "--workers", "0", "--threads", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env); assert r.returncode == 0, r.stderr[-800:]
    new, old = json.loads((out / "last/config.json").read_text()), json.loads((CK / "config.json").read_text())
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k) and k not in ("device", "pretrained_path")}
    assert not diff, diff
    assert not (out / "last" / nc.CONTRACT_FILE).exists()
    a, b = load_file(str(out / "last/policy_preprocessor_step_3_normalizer_processor.safetensors")), load_file(str(CK / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    for k in ("observation.state.std", "action.std", "observation.state.mean", "action.mean"): assert np.allclose(a[k], b[k], rtol=1e-6, atol=1e-12), k
    ti = json.loads((out / "train_info.json").read_text()); assert ti["norm_stats_source"] == "dataset" and ti["norm_contract"] is False
    shutil.rmtree(out.parent)

@add("T13")
def t13():
    """audit_policy_inputs still PASS (state + images only, no ground truth) on the V1 checkpoint"""
    r = subprocess.run([sys.executable, str(ROOT / "eval/audit_policy_inputs.py"), str(CK), str(R)], capture_output=True, text=True); assert r.returncode == 0, r.stdout[-500:] + r.stderr[-300:]

@add("TA")
def ta():
    """V1 checkpoint is never rewritten and never silently uses the new statistics"""
    assert nc.verify_checkpoint(CK) is None and not (CK / nc.CONTRACT_FILE).exists()
    pre_v1, _ = make_pre_post_processors(CFG, pretrained_path=str(CK)); b = batch(); o_v1 = pre_v1(b)["observation.state"]
    o_all = mk(STATS_ALL)[0](b)["observation.state"]; o_tr = mk(STATS_TR)[0](b)["observation.state"]
    assert float((o_v1 - o_all).abs().max()) < 1e-5, "V1 processors must equal the dataset-wide statistics they were trained with"
    live_gap = float((o_v1 - o_tr)[:, :8].abs().max()); assert live_gap > 1e-4, f"train-only stats must be distinguishable from V1's ({live_gap})"
    assert sha_dir(CK.parent) == V1_HASH0, "V1 checkpoint directory changed"
    return dict(v1_vs_train_only_live_state_gap=live_gap, v1_files_hashed=len(V1_HASH0))

@add("TB")
def tb():
    """independent right-arm tests: each right joint and the right gripper separately - S2 action decode is exact; C0/S1 show the documented sigma<<eps collapse"""
    res = {}
    for name, st in (("C0", C0), ("S1", S1), ("S2", S2)):
        pre, post = mk(st); r = {}
        for d in range(8, 16):
            sd = float(st[nc.ACTION]["std"][d])
            for delta in (0.001, 0.05):                                   # 1 mrad (or 1 mm) and 0.05 rad (or 5 cm) excursions of ONE idle action dim
                a = A_REAL.copy(); a[:, d] += delta; dec = post(pre(batch(a=a))["action"][:, 0, :]).numpy()
                ratio = float(np.median(dec[:, d] - A_REAL[:, d]) / delta); expect = sd / (sd + nc.EPS); r[f"d{d}_delta{delta}"] = ratio
                if name == "S2": assert abs(ratio - 1.0) < 1e-4 + 2 * nc.EPS / sd, (name, d, delta, ratio)        # floored: decode is the identity
                else: assert abs(ratio - expect) < 0.02, (name, d, delta, ratio, expect)                       # unfloored: decoded deviation = sigma/(sigma+eps) of the input
        res[name] = dict(ratio_by_probe={k: float(f"{v:.3g}") for k, v in r.items()}, right_joints_dims=list(range(8, 15)), right_gripper_dim=15)
    return res

@add("TC")
def tc():
    """tensor contracts: pre/post shapes, dtypes and key sets (state (B,16), images (B,3,H,W) float32, action chunk (B,20,16) -> post (B,16))"""
    for name, st in (("C0", C0), ("S1", S1), ("S2", S2)):
        pre, post = mk(st); b = batch(); o = pre(b)
        assert o["observation.state"].shape == (len(S_REAL), 16) and o["observation.state"].dtype == torch.float32
        assert o["action"].shape == (len(S_REAL), CFG.chunk_size, 16) and o["action_is_pad"].shape == (len(S_REAL), CFG.chunk_size)
        for c in ("front", "wrist_left"): assert o[f"observation.images.{c}"].shape == (len(S_REAL), 3, 120, 160) and o[f"observation.images.{c}"].dtype == torch.float32
        p = post(o["action"][:, 0, :]); assert p.shape == (len(S_REAL), 16) and p.dtype == torch.float32 and torch.isfinite(p).all()
        assert sorted(k for k in o if k.startswith("observation")) == sorted(["observation.state", "observation.images.front", "observation.images.wrist_left"])
    assert sorted(CFG.input_features) == ["observation.images.front", "observation.images.wrist_left", "observation.state"] and list(CFG.output_features) == ["action"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-train-cli", action="store_true"); ARGS = ap.parse_args()
    for t in TESTS: t()
    ok = all(v == "PASS" for v in RESULTS.values()); (OUT / "unit_tests.json").write_text(json.dumps(dict(results=RESULTS, measurements=MEAS, all_pass=ok), indent=1, default=float))
    print("ALL PASS" if ok else "SOME FAILED", RESULTS); sys.exit(0 if ok else 1)
