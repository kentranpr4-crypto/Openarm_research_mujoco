#!/usr/bin/env python3
"""E1-lite-B (rotation-vector orientation head) and E1-lite-C (yaw-loss ablations) on the FROZEN original WM3-B trunk — EXPLORATORY diagnostics, not model selection, I-1 not used, not independent evidence.
Common: original trunk weights, original v1 120-D input contract and normaliser (no 148-D v2 input is ever fed to the trunk); trunk features cached once (outputs/e1_lite_cache, not in git) and reused; v2 windows supply targets / masks aligned by (episode id, start t); seed 0; AdamW lr 2e-3, wd 1e-4, cosine, batch 256, 10 epochs; process-CPU guard.
B: head Linear(256->3) x per-axis scale -> per-step world rotation vector (Huber 0.05 rad on rotvec_std_steps_le_0p3 scales; no mask needed); baselines persistence (zero rotation) and a closed-form ridge on the v1 'cond' features -> rotation vectors.  Cap 0.15 CPU-h.
C: yaw head, arms in FIXED order [original, mask_only, huber_only, scale_only, full], identical init / batches; if the cap would be exceeded the run STOPS and reports PARTIAL (arms are never dropped after seeing results).  Cap 0.20 CPU-h.
Usage (ACT venv):  python eval/e1_lite_bc.py B | C"""
import sys, json, time, resource, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "eval"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
MODE = sys.argv[1]; sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm3b_model as B, wm4_gates as G4, geometry_v2 as Gm, orientation_loss as OL, torch.nn as nn
M.logistic_fit = lambda X, y, l2=0.01, iters=200: (np.zeros((X.shape[1], y.shape[1]), np.float32), np.zeros(y.shape[1], np.float32))      # event heads are not used here; skip their LBFGS fit (closed-form Y_s / Y_b / Y_c unchanged)
import os
PRIOR_CPU = float(os.environ.get("PRIOR_CPU", 0))      # CPU already spent by a crashed earlier attempt of the same stage (counted against the same cap)
CAP = {"B": 0.15 * 3600, "C": 0.20 * 3600}[MODE] - PRIOR_CPU; OUT = ROOT / ("outputs/e1_lite_b" if MODE == "B" else "outputs/e1_lite_c"); OUT.mkdir(parents=True, exist_ok=True); CACHE = ROOT / "outputs/e1_lite_cache"; CACHE.mkdir(exist_ok=True)
V2W = ROOT / "outputs/wm2_v2/windows"; DEV = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; SPL = ["train", "val"] + DEV; wrap = Gm.wrap
def cpu(): a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
T0, C0 = time.time(), cpu()
class Stop(Exception): pass
def guard(tag, extra=0.0):
    used = cpu() - C0
    if used + extra > CAP: raise Stop(f"{tag}: {used:.0f}s + {extra:.0f}s > cap {CAP:.0f}s")
    return used
# ---- pre-conditions: Gate v3 frozen, integrity PASS
frz = json.loads((ROOT / "outputs/gates_v3/freeze.json").read_text()); assert G4.load_config(frz["config_sha256"]) is not None, "Gate v3 not frozen"
for v in ("wm2_v1_1", "wm2_v2"): assert json.loads((ROOT / f"outputs/{v}/integrity.json").read_text())["overall"] == "PASS", f"{v} integrity not PASS: STOP"
cfg = G4.CONFIG; torch.set_num_threads(3); torch.manual_seed(0); np.random.seed(0)
tr, va = R.load("train"), R.load("val"); nrm = M.Norm(tr); ridge = M.Ridge("cond").fit(nrm, tr); m = B.WM3B(ridge, nrm); m.model = B.GRUOC(); m.model.load_state_dict(torch.load(ROOT / "outputs/wm3/models/wm3b.pt")); m.model.eval(); guard("fit")
def align(W, sp):
    z = {k: v for k, v in np.load(V2W / f"{sp}.npz").items()}; key = {(str(a), int(t)): i for i, (a, t) in enumerate(zip(z["eid"], z["t"]))}; sel = [(i, key[(str(a), int(t))]) for i, (a, t) in enumerate(zip(W["eid"], W["t"])) if (str(a), int(t)) in key]
    i1 = np.array([a for a, _ in sel]); i2 = np.array([b for _, b in sel]); return i1, {k: v[i2] for k, v in z.items()}
def trunk(W):
    hist, af, rn, raw = m.tensors(W); outs = []
    with torch.no_grad():
        for i in range(0, len(hist), 4096):
            c = m.model.enc(hist[i:i + 4096]); h0 = m.model.init(c).view(-1, 2, 256).transpose(0, 1).contiguous(); x = torch.cat([torch.relu(m.model.ain(af[i:i + 4096])), torch.relu(m.model.sin(rn[i:i + 4096]))], -1); o, _ = m.model.gru(x, h0); outs.append(o.numpy())
    o = np.concatenate(outs); return o, (m.model.out(torch.tensor(o))[..., 3].detach().numpy() * nrm.Yb[1][3] + nrm.Yb[0][3]).astype(np.float32)
D = {}; sanity = {}; cache_hit = all((CACHE / f"{sp}_o.npy").exists() for sp in SPL)
for sp in SPL:
    W = tr if sp == "train" else (va if sp == "val" else R.load(sp)); i1, z = align(W, sp); Wk = {k: (v[i1] if hasattr(v, "__len__") and len(v) == len(W["t"]) else v) for k, v in W.items()}
    if cache_hit: o = np.load(CACHE / f"{sp}_o.npy"); y_old = None
    else: o, y_old = trunk(Wk); np.save(CACHE / f"{sp}_o.npy", o)
    D[sp] = dict(W=Wk, z=z, o=o); guard("features " + sp)
    if y_old is not None: sanity[sp] = dict(n_aligned=int(len(i1)), existing_head_from_cached_features_vs_predict=float(np.abs(y_old - m.predict(Wk)["Y_b"][:, :, 3]).max()))
    D[sp]["y_old"] = y_old if y_old is not None else (m.model.out(torch.tensor(o))[..., 3].detach().numpy() * nrm.Yb[1][3] + nrm.Yb[0][3]).astype(np.float32)
guard("cache")
scales = json.loads((ROOT / "outputs/wm2_v2/dataset/loss_scales.json").read_text()); SC_RV = np.array(scales["rotvec_std_steps_with_sym_geodesic_le_0p3"], np.float32); SC_YAW_R = scales["yaw_delta_std_valid_nonspike"]; SD_ALL = float(nrm.Yb[1][3]); SC = 0.0166
def groupmac(vals, g): u = np.unique(g); return u, np.array([vals[g == x].mean() for x in u])
def bootstrap_ratio(a, b, n=10000, seed=0): rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (n, len(a))); r = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-4); return [float(a.mean() / max(b.mean(), 1e-4)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
res = dict(mode=MODE, prior_crashed_attempt_cpu_seconds_upper_bound=PRIOR_CPU, crash_note="first attempt crashed in the huber_only arm (numpy wrap applied to a torch tensor) before any result was written; fixed to torch wrap; all arms rerun from scratch in the fixed order", gate_v3_config_sha256=frz["config_sha256"], trunk="frozen original WM3-B (v1 120-D contract, original normaliser)", exploratory="not independent evidence; I-1 not used", sanity=sanity, cache_reused=cache_hit)
# =============================================================================================== B
if MODE == "B":
    # representation / frame / reconstruction checks on real episodes (rotation vector = world-frame q_{k+1} (x) conj(q_k))
    z = D["train"]["z"]; rng = np.random.default_rng(0); ii = rng.choice(len(z["t"]), 200, replace=False); mx = 0.0
    for i in ii:
        q = np.load(ROOT / "outputs/wm2_v2/dataset/episodes" / f"{z['eid'][i]}.npz")["q"]; t = int(z["t"][i]); Qr = Gm.compose(z["Q_t"][i], z["Y_rv"][i]); mx = max(mx, float(Gm.geodesic(Qr, q[t + 1:t + 21]).max()))
    res["representation_check"] = dict(frame="world frame", definition="q_{k+1} = exp(rv_k) (x) q_k", reconstruction_max_geodesic_rad_200_train_windows=mx, passed=bool(mx < 1e-5)); assert mx < 1e-5, "rotation-vector reconstruction failed: STOP"
    # ridge baseline (closed form, alpha 10, same standardised 'cond' features as the v1 ridge) -> rotation vectors
    def feats(W): return nrm.feat(W, "cond").astype(np.float64)
    X = feats(D["train"]["W"]); mu, sd = X.mean(0), X.std(0) + 1e-6; X1 = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1); Rg = np.eye(X1.shape[1]) * M.CFG["ridge"]["alpha"]; Rg[-1, -1] = 0; Wt = np.linalg.solve(X1.T @ X1 + Rg, X1.T @ D["train"]["z"]["Y_rv"].reshape(len(X), -1).astype(np.float64)); guard("ridge")
    def ridge_rv(sp): Xs = (feats(D[sp]["W"]) - mu) / sd; return (np.concatenate([Xs, np.ones((len(Xs), 1))], 1) @ Wt).reshape(-1, 20, 3).astype(np.float32)
    class Head(nn.Module):
        def __init__(self): super().__init__(); self.lin = nn.Linear(256, 3); self.register_buffer("sc", torch.tensor(SC_RV))
        def forward(self, o): return self.lin(o) * self.sc
    torch.manual_seed(0); h = Head(); o = torch.tensor(D["train"]["o"]); y = torch.tensor(D["train"]["z"]["Y_rv"]); n = len(y); bs, ep = 256, 10; opt = torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep * (n // bs), 1e-5); g = torch.Generator().manual_seed(0); log = []
    for e in range(ep):
        perm = torch.randperm(n, generator=g); tl = 0.0; nb = 0; gn = 0.0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]; loss = OL.rotvec_loss(h(o[idx]), y[idx], h.sc, delta=0.05); opt.zero_grad(); loss.backward(); gn = max(gn, float(torch.nn.utils.clip_grad_norm_(h.parameters(), 1e9))); opt.step(); sch.step(); tl += loss.item(); nb += 1
        log.append(dict(epoch=e + 1, train_loss=tl / nb, max_grad_norm=gn, finite=bool(np.isfinite(tl)), cpu_s=guard(f"B epoch {e + 1}")))
    h.eval()
    with torch.no_grad(): P = {sp: h(torch.tensor(D[sp]["o"])).numpy() for sp in SPL}
    res["training"] = dict(log=log, loss="Huber 0.05 rad on per-axis scaled rotation-vector residual", params=int(sum(q.numel() for q in h.parameters())))
    def models(sp): return {"persistence": np.zeros((len(D[sp]["z"]["t"]), 20, 3), np.float32), "ridge_rv": ridge_rv(sp), "H_rv": P[sp]}
    Kk = (1, 5, 10, 20); regs = ("stable", "rotating", "singular", "tumbling")
    def tables(sps):
        per = {n: {} for n in ("persistence", "ridge_rv", "H_rv")}; info = {r: [0, set()] for r in regs}
        for sp in sps:
            z = D[sp]["z"]; rg, Qtrue = G4.regimes(z["Q_t"], z["Y_rv"], cfg); g = z["group"]; act = z["active"]; Mm = models(sp)
            for r in regs: mk = act & rg[r]; info[r][0] += int(mk.sum()); info[r][1] |= set(g[mk].tolist())
            for n, rv in Mm.items():
                Qp = G4._traj(z["Q_t"], rv)
                for r in regs:
                    mk = act & rg[r]
                    if not mk.any(): continue
                    for k in Kk:
                        for kind, f in (("sym", Gm.geodesic_sym), ("full", Gm.geodesic)):
                            e = f(Qp[mk][:, k], Qtrue[mk][:, k]); per[n].setdefault((r, k, kind), []).append(groupmac(e, g[mk]))
        out = {}
        for n in per:
            out[n] = {}
            for key, parts in per[n].items(): ids = np.concatenate([p[0] for p in parts]); vals = np.concatenate([p[1] for p in parts]); o_ = np.argsort(ids); per[n][key] = (ids[o_], vals[o_]); out[n]["/".join(map(str, key))] = dict(mean=float(vals.mean()), n_groups=int(len(vals)))
        sup = {r: dict(windows=info[r][0], groups=len(info[r][1]), status=("evaluable" if info[r][0] >= 200 and len(info[r][1]) >= 10 else "below support (10 groups / 200 windows)")) for r in regs}
        ratios = {}
        for r in regs:
            if (r, 20, "sym") in per["H_rv"]:
                a = per["H_rv"][(r, 20, "sym")][1]; refs = {n: per[n][(r, 20, "sym")][1] for n in ("persistence", "ridge_rv")}; best = min(refs, key=lambda n: refs[n].mean()); ratios[r] = dict(best_baseline=best, ratio_sym_K20=bootstrap_ratio(a, refs[best]), support=sup[r]["status"])
        return out, sup, ratios
    for tag, sps in (("validation", ["val"]), ("development", DEV)): t_, s_, r_ = tables(sps); res[tag] = dict(geodesic_error_by_regime_horizon=t_, regime_support=s_, ratio_vs_best_baseline=r_); guard("tables " + tag)
    def g7(sps, model):
        T = {}; grp = {}; act = {}; preds = {n: {} for n in ("persistence", "ridge", "H_rv")}
        for sp in sps:
            z = D[sp]["z"]; T[sp] = dict(Q_t=z["Q_t"], Y_rv=z["Y_rv"], B_hist=z["B_hist"], C_hist=z["C_hist"], A_fut=z["A_fut"]); grp[sp] = z["group"]; act[sp] = z["active"]; Mm = models(sp); preds["persistence"][sp] = dict(Y_rv=Mm["persistence"]); preds["ridge"][sp] = dict(Y_rv=Mm["ridge_rv"]); preds["H_rv"][sp] = dict(Y_rv=Mm["H_rv"])
        return G4.gate_G7(cfg, preds, T, grp, act, model, sets=sps, n_boot=10000)
    res["G7"] = {}
    for tag, sps in (("development", DEV), ("validation", ["val"])):
        r_ = g7(sps, "H_rv"); res["G7"][tag] = dict(verdict=r_["verdict"], components={c["name"]: ("NOT_EVALUABLE" if not c["evaluable"] else ("PASS" if c["passed"] else "FAIL")) + (f" ({c['value']:.4g})" if "value" in c else "") for c in r_["conditions"]}, detail={k: r_["detail"].get(k) for k in ("yaw_primary", "static_input_subset", "orientation_singular_tumbling", "orientation_full_geodesic_reported_not_gating", "regimes_reported", "tumbling_threshold_sensitivity_not_gating", "tilt_reported_only")}); guard("g7 " + tag)
    res["side_effects"] = "trunk and all original heads untouched; only a new rotation-vector head is trained"
# =============================================================================================== C
else:
    ARMS = ["original", "mask_only", "huber_only", "scale_only", "full"]
    class Head(nn.Module):
        def __init__(self): super().__init__(); self.lin = nn.Linear(256, 1)
        def forward(self, o): return self.lin(o).squeeze(-1) * SC
    torch.manual_seed(0); init = Head().state_dict()
    def loss_fn(kind, p, y, v):
        hub = lambda r, d: OL.huber(r, d)
        if kind == "original": return (((p - y) / SD_ALL) ** 2).mean()
        if kind == "mask_only": return ((((p - y) / SD_ALL) ** 2) * v.float()).sum() / v.float().sum().clamp_min(1.0)
        if kind == "huber_only": return hub(OL.wrap(p - y) / SD_ALL, 0.05 / SD_ALL).mean()
        if kind == "scale_only": return (((p - y) / SC_YAW_R) ** 2).mean()
        if kind == "full": return OL.yaw_huber_masked(p, y, v, SC_YAW_R, 0.05)
    o = torch.tensor(D["train"]["o"]); y = torch.tensor(D["train"]["W"]["Y_b"][:, :, 3]); v = torch.tensor(D["train"]["z"]["Y_yaw_valid"]); n = len(y); bs, ep = 256, 10; trained = {}; logs = {}; cost = {}; status = {}
    for kind in ARMS:
        est = 1.3 * max(cost.values()) if cost else 0.0
        try: guard(f"before arm {kind}", est)
        except Stop as e: status[kind] = f"NOT RUN: {e}"; continue
        c_ = cpu(); h = Head(); h.load_state_dict(init); opt = torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep * (n // bs), 1e-5); g = torch.Generator().manual_seed(0); log = []
        for e in range(ep):
            perm = torch.randperm(n, generator=g); tl = 0.0; nb = 0; gn = 0.0
            for i in range(0, n - bs + 1, bs):
                idx = perm[i:i + bs]; loss = loss_fn(kind, h(o[idx]), y[idx], v[idx]); opt.zero_grad(); loss.backward(); gn = max(gn, float(torch.nn.utils.clip_grad_norm_(h.parameters(), 1e9))); opt.step(); sch.step(); tl += loss.item(); nb += 1
            log.append(dict(epoch=e + 1, train_loss=tl / nb, max_grad_norm=gn, finite=bool(np.isfinite(tl))))
        h.eval()
        with torch.no_grad(): trained[kind] = {sp: h(torch.tensor(D[sp]["o"])).numpy() for sp in SPL if sp != "train"}
        logs[kind] = log; cost[kind] = cpu() - c_; status[kind] = "trained"
    res["arms"] = dict(order=ARMS, status=status, train_cpu_s=cost, logs=logs, equivalence_note="under Adam the constant rescaling of a loss (robust scale vs original scale) leaves the update nearly unchanged; scale_only vs original therefore tests exactly this")
    complete = all(s == "trained" for s in status.values()); res["verdict_complete"] = "COMPLETE" if complete else "PARTIAL"
    Kk = (1, 5, 10, 20)
    def models_for(sp):
        W = D[sp]["W"]; d = {"persistence": np.zeros_like(D[sp]["y_old"]), "ridge": ridge.predict(W)["Y_b"][:, :, 3].astype(np.float32), "existing_head": D[sp]["y_old"]}
        for k, v_ in trained.items(): d[k] = v_[sp]
        return d
    names = ["persistence", "ridge", "existing_head"] + [a for a in ARMS if a in trained]
    def tables(sps):
        per = {n: {} for n in names}; info = {}
        for sp in sps:
            z = D[sp]["z"]; rg, Qt = G4.regimes(z["Q_t"], z["Y_rv"], cfg); act = z["active"]; Mm = models_for(sp); g = z["group"]
            for r in ("stable", "rotating", "tumbling", "singular"): mk = act & rg[r]; info.setdefault(r, [0, set()]); info[r][0] += int(mk.sum()); info[r][1] |= set(g[mk].tolist())
            for n in names:
                for r in ("stable", "rotating", "tumbling"):
                    mk = act & rg[r]
                    for k in Kk:
                        if mk.any(): e = np.abs(wrap(wrap(np.cumsum(Mm[n], 1)[mk][:, k - 1]) - G4.yaw_disp(Qt[mk], k))); per[n].setdefault((r, k), []).append(groupmac(e, g[mk]))
        out = {}
        for n in names:
            out[n] = {}
            for key, parts in per[n].items(): ids = np.concatenate([p[0] for p in parts]); vals = np.concatenate([p[1] for p in parts]); o_ = np.argsort(ids); per[n][key] = (ids[o_], vals[o_]); out[n][f"{key[0]}/K={key[1]}"] = dict(mean=float(vals.mean()), n_groups=int(len(vals)))
        return out, per, {r: dict(windows=v[0], groups=len(v[1]), status=("NOT_EVALUABLE (Euler yaw invalid ground truth)" if r == "singular" else ("evaluable" if v[0] >= 200 and len(v[1]) >= 10 else "below support"))) for r, v in info.items()}
    for tag, sps in (("validation", ["val"]), ("development", DEV)):
        t_, per, sup = tables(sps); res[tag] = dict(yaw_error_by_regime_horizon=t_, regime_support=sup)
        if tag == "development":
            rel = {}
            for a in names:
                if a in ("persistence", "ridge", "existing_head"): continue
                rel[a] = {}
                for r in ("stable", "rotating", "tumbling"):
                    vs = per[a][(r, 20)][1]; p0 = per["persistence"][(r, 20)][1]; rd = per["ridge"][(r, 20)][1]; best = min((p0, rd), key=lambda x: x.mean())
                    rel[a][r] = dict(ratio_vs_best_of_P0_ridge=bootstrap_ratio(vs, best), ratio_vs_P0=float(vs.mean() / max(p0.mean(), 1e-4)), ratio_vs_ridge=float(vs.mean() / max(rd.mean(), 1e-4)), frac_groups_worse_than_P0=float((vs > p0).mean()), frac_groups_worse_than_ridge=float((vs > rd).mean()))
            res["development"]["relative_performance"] = rel
        guard("tables " + tag)
    def g7(sps, model):
        T = {}; grp = {}; act = {}; preds = {n: {} for n in names}
        for sp in sps:
            z = D[sp]["z"]; T[sp] = dict(Q_t=z["Q_t"], Y_rv=z["Y_rv"], B_hist=z["B_hist"], C_hist=z["C_hist"], A_fut=z["A_fut"]); grp[sp] = z["group"]; act[sp] = z["active"]
            for n, v_ in models_for(sp).items(): preds[n][sp] = dict(Y_yaw=v_)
        return G4.gate_G7(cfg, preds, T, grp, act, model, sets=sps, n_boot=10000)
    res["G7"] = {}
    for tag, sps in (("development", DEV), ("validation", ["val"])):
        res["G7"][tag] = {}
        for model in ["existing_head"] + [a for a in ARMS if a in trained]:
            r_ = g7(sps, model); res["G7"][tag][model] = dict(verdict=r_["verdict"], components={c["name"]: ("NOT_EVALUABLE" if not c["evaluable"] else ("PASS" if c["passed"] else "FAIL")) + (f" ({c['value']:.4g})" if "value" in c else "") for c in r_["conditions"]}, static=r_["detail"].get("static_input_subset"), yaw_primary=r_["detail"].get("yaw_primary")); guard(f"g7 {tag} {model}")
res["budget"] = dict(cap_cpu_hours=CAP / 3600, cpu_seconds=cpu() - C0, cpu_hours=(cpu() - C0) / 3600, wall_seconds=time.time() - T0)
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))); print(json.dumps(res["budget"]))
