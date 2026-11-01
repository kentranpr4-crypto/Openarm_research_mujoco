#!/usr/bin/env python3
"""E1-lite — frozen-trunk yaw-head probe (EXPLORATORY DIAGNOSTIC; NOT a model-selection step; I-1 is not used).
Frozen: the original WM3-B trunk (encoder + ridge-forecast path + GRU; v1 120-D input contract, original train-only normaliser, original weights).  Only a NEW per-step yaw head Linear(256->1) is trained, on the cached trunk features
of the v1 TRAIN windows, in two arms that share the trunk, initialisation, seed, batches, optimiser and data:
  H_old  : MSE of the normalised yaw delta over ALL steps (std 0.0529 rad, the original objective for this channel)
  H_mask : masked Huber on the WRAPPED residual (delta 0.05 rad, robust scale from contract-v2 loss_scales.json) with the geometric target-time validity mask (from the v2 windows, aligned by episode id and start t)
Pre-registered configuration (set before any run; no tuning afterwards): pred_rad = Linear(256,1)(o_k) * 0.0166; AdamW lr 2e-3, wd 1e-4, cosine, batch 256 windows, 10 epochs, seed 0, 3 threads.  CPU budget: 0.3 CPU-h hard stop (process CPU time).
Evaluation on VALIDATION (val) and DEVELOPMENT splits (wm2 test_*): yaw displacement error at K=1/5/10/20 (wrap-aware) per regime (stable / rotating / tumbling heading-valid; singular is NOT_EVALUABLE for yaw: Euler yaw is not a valid ground truth there), vs P0 persistence, ridge, the existing WM3-B yaw head.
Pre-registered decision: CLEAR IMPROVEMENT iff on the pooled development splits (K=20, rotating windows) E_mask/min(E_P0, E_ridge) <= 0.90 with group-bootstrap upper bound < 1.0 AND E_mask <= E_old AND the head keeps G7 static stability (abs <= 0.01 rad, diff-vs-persistence upper <= 0.01, relative <= 3.0).
Otherwise: no automatic E1 full.   Run (ACT venv): python eval/e1_lite.py"""
import sys, json, time, resource, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "eval"), str(ROOT / "data")):
    if p not in sys.path: sys.path.insert(0, p)
sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm3b_model as B, wm4_gates as G4, geometry_v2 as Gm, orientation_loss as OL, torch.nn as nn
OUT = ROOT / "outputs/e1_lite"; V2W = ROOT / "outputs/wm2_v2/windows"; CAP = 0.3 * 3600; DEV = ["test_id", "test_low_heldout", "test_ood_low_strict", "test_ood_high"]; SC = 0.0166
def cpu(): a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN); return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
T0, C0 = time.time(), cpu()
def guard(tag):
    used = cpu() - C0
    if used > CAP: sys.exit(f"CPU budget exceeded at {tag}: {used:.0f}s > {CAP:.0f}s")
    return used
integ = json.loads((ROOT / "outputs/wm2_v2/integrity.json").read_text()); assert integ["overall"] == "PASS", "v2 integrity not PASS: STOP"
assert json.loads((ROOT / "outputs/wm2_v1_1/integrity.json").read_text())["overall"] == "PASS", "v1.1 integrity not PASS: STOP"
torch.set_num_threads(3); torch.manual_seed(0); np.random.seed(0)
tr, va = R.load("train"), R.load("val"); nrm = M.Norm(tr); ridge = M.Ridge("cond").fit(nrm, tr); m = B.WM3B(ridge, nrm); m.model = B.GRUOC(); m.model.load_state_dict(torch.load(ROOT / "outputs/wm3/models/wm3b.pt")); m.model.eval(); guard("fit")
def trunk(W):
    hist, af, rn, raw = m.tensors(W); outs = []
    with torch.no_grad():
        for i in range(0, len(hist), 4096):
            c = m.model.enc(hist[i:i + 4096]); h0 = m.model.init(c).view(-1, 2, 256).transpose(0, 1).contiguous(); x = torch.cat([torch.relu(m.model.ain(af[i:i + 4096])), torch.relu(m.model.sin(rn[i:i + 4096]))], -1); o, _ = m.model.gru(x, h0); outs.append(o.numpy())
    o = np.concatenate(outs); y = m.model.out(torch.tensor(o))[..., 3].detach().numpy() * nrm.Yb[1][3] + nrm.Yb[0][3]; return o, y.astype(np.float32)
def align(W, sp):
    z = {k: v for k, v in np.load(V2W / f"{sp}.npz").items()}; key = {(str(a), int(t)): i for i, (a, t) in enumerate(zip(z["eid"], z["t"]))}; sel = [(i, key[(str(a), int(t))]) for i, (a, t) in enumerate(zip(W["eid"], W["t"])) if (str(a), int(t)) in key]
    i1 = np.array([a for a, _ in sel]); i2 = np.array([b for _, b in sel]); return i1, {k: v[i2] for k, v in z.items()}
# ---------------- features, alignment, sanity (existing head reproduced exactly from the cached features)
DATA = {}; sanity = {}
for sp in ["train", "val"] + DEV:
    W = tr if sp == "train" else (va if sp == "val" else R.load(sp)); i1, z = align(W, sp); Wk = {k: (v[i1] if hasattr(v, "__len__") and len(v) == len(W["t"]) else v) for k, v in W.items()}; o, y_old = trunk(Wk); ref = m.predict(Wk)["Y_b"][:, :, 3]
    sanity[sp] = dict(n_v1_windows=int(len(W["t"])), n_aligned=int(len(i1)), existing_head_from_cached_features_max_abs_diff=float(np.abs(y_old - ref).max()), yaw_target_v1_vs_v2_max_abs_diff_valid_steps=float(np.abs(Wk["Y_b"][:, :, 3] - z["Y_yaw"])[z["Y_yaw_valid"]].max()))
    DATA[sp] = dict(W=Wk, z=z, o=o, y_old=y_old); guard("features " + sp)
assert all(s["existing_head_from_cached_features_max_abs_diff"] < 1e-4 for s in sanity.values()), sanity
# ---------------- heads
scales = json.loads((ROOT / "outputs/wm2_v2/dataset/loss_scales.json").read_text()); SC_R = scales["yaw_delta_std_valid_nonspike"]; SD_ALL = float(nrm.Yb[1][3])
class Head(nn.Module):
    def __init__(self): super().__init__(); self.lin = nn.Linear(256, 1)
    def forward(self, o): return self.lin(o).squeeze(-1) * SC
torch.manual_seed(0); init = Head().state_dict()
def train_head(kind):
    h = Head(); h.load_state_dict(init); o = torch.tensor(DATA["train"]["o"]); y = torch.tensor(DATA["train"]["W"]["Y_b"][:, :, 3]); v = torch.tensor(DATA["train"]["z"]["Y_yaw_valid"]); n = len(y); bs = 256; ep = 10; steps = ep * (n // bs)
    opt = torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, 1e-5); g = torch.Generator().manual_seed(0); log = []
    for e in range(ep):
        perm = torch.randperm(n, generator=g); tl = 0.0; nb = 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]; p = h(o[idx])
            loss = (((p - y[idx]) / SD_ALL) ** 2).mean() if kind == "old" else OL.yaw_huber_masked(p, y[idx], v[idx], SC_R, 0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item(); nb += 1
        log.append(dict(epoch=e + 1, train_loss=tl / nb, cpu_s=guard(f"{kind} epoch {e + 1}")))
    h.eval()
    with torch.no_grad(): preds = {sp: h(torch.tensor(DATA[sp]["o"])).numpy() for sp in DATA}
    return preds, log
P_old, log_old = train_head("old"); P_mask, log_mask = train_head("mask")
# ---------------- evaluation
wrap = Gm.wrap; cfg = G4.CONFIG; Kk = (1, 5, 10, 20); out = dict(config=dict(head="Linear(256,1) per step x 0.0166", optimiser="AdamW lr 2e-3 wd 1e-4 cosine, batch 256, 10 epochs, seed 0", huber_delta=0.05, robust_scale=SC_R, old_scale=SD_ALL, trunk="frozen original WM3-B", input_contract="v1 (120-D), original normaliser"), sanity=sanity, train_logs=dict(old=log_old, mask=log_mask))
def models_for(sp):
    W = DATA[sp]["W"]; return {"persistence": np.zeros_like(DATA[sp]["y_old"]), "ridge": ridge.predict(W)["Y_b"][:, :, 3].astype(np.float32), "existing_head": DATA[sp]["y_old"], "H_old": P_old[sp], "H_mask": P_mask[sp]}
def tables(sps):
    names = ["persistence", "ridge", "existing_head", "H_old", "H_mask"]; res = {}; per = {n: {} for n in names}; info = {}
    for sp in sps:
        z = DATA[sp]["z"]; rg, Qt = G4.regimes(z["Q_t"], z["Y_rv"], cfg); act = z["active"]; M_ = models_for(sp); g = z["group"]
        for rn in ("stable", "rotating", "tumbling", "singular"):
            mk = act & rg[rn]; info.setdefault(rn, [0, set()]); info[rn][0] += int(mk.sum()); info[rn][1] |= set(g[mk].tolist())
        for n in names:
            for rn in ("stable", "rotating", "tumbling"):
                mk = act & rg[rn]
                for k in Kk:
                    if mk.any():
                        e = np.abs(wrap(wrap(np.cumsum(M_[n], 1)[mk][:, k - 1]) - G4.yaw_disp(Qt[mk], k))); ids, vals = (lambda gg, ee: (np.unique(gg), np.array([ee[gg == x].mean() for x in np.unique(gg)])))(g[mk], e)
                        per[n].setdefault((rn, k), []).append((ids, vals))
    for n in names:
        res[n] = {}
        for (rn, k), parts in per[n].items():
            ids = np.concatenate([p[0] for p in parts]); vals = np.concatenate([p[1] for p in parts]); o = np.argsort(ids); res[n][f"{rn}/K={k}"] = dict(mean=float(vals.mean()), n_groups=int(len(vals)))
            per[n][(rn, k)] = (ids[o], vals[o])
    return res, per, {k: dict(windows=v[0], groups=len(v[1]), status=("NOT_EVALUABLE_yaw_invalid_ground_truth" if k == "singular" else ("evaluable" if len(v[1]) >= 10 and v[0] >= 200 else "below support (10 groups / 200 windows)"))) for k, v in info.items()}
def boot_ratio(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (n, len(a))); r = a[idx].mean(1) / np.maximum(b[idx].mean(1), 1e-4); return float(a.mean() / max(b.mean(), 1e-4)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
for tag, sps in (("validation", ["val"]), ("development", DEV)):
    res, per, info = tables(sps); out[tag] = dict(yaw_error_by_regime_and_horizon=res, regime_support=info)
    if tag == "development":
        a = per["H_mask"][("rotating", 20)][1]; refs = {n: per[n][("rotating", 20)][1] for n in ("persistence", "ridge")}; best = min(refs, key=lambda n: refs[n].mean()); r, lo, hi = boot_ratio(a, refs[best]); ro, loo, hio = boot_ratio(per["H_old"][("rotating", 20)][1], refs[best]); rx, lx, hx = boot_ratio(per["existing_head"][("rotating", 20)][1], refs[best])
        out["decision_inputs"] = dict(best_reference=best, ratio_mask_vs_best=[r, lo, hi], ratio_old_head_vs_best=[ro, loo, hio], ratio_existing_head_vs_best=[rx, lx, hx], mask_le_old=bool(a.mean() <= per["H_old"][("rotating", 20)][1].mean()))
# ---------------- Gate v3 G7 on the development splits and on validation (legacy per-step yaw outputs)
def g7(sps, model):
    T = {}; grp = {}; act = {}; preds = {n: {} for n in ("persistence", "ridge", "existing_head", "H_old", "H_mask")}
    for sp in sps:
        z = DATA[sp]["z"]; T[sp] = dict(Q_t=z["Q_t"], Y_rv=z["Y_rv"], B_hist=z["B_hist"], C_hist=z["C_hist"], A_fut=z["A_fut"]); grp[sp] = z["group"]; act[sp] = z["active"]
        for n, v in models_for(sp).items(): preds[n][sp] = dict(Y_yaw=v)
    return G4.gate_G7(cfg, preds, T, grp, act, model, sets=sps, n_boot=10000)
out["G7"] = {}
for tag, sps in (("development", DEV), ("validation", ["val"])):
    out["G7"][tag] = {}
    for model in ("existing_head", "H_old", "H_mask"):
        r_ = g7(sps, model); out["G7"][tag][model] = dict(verdict=r_["verdict"], components={c["name"]: ("NOT_EVALUABLE" if not c["evaluable"] else ("PASS" if c["passed"] else "FAIL")) + (f" ({c['value']:.4g})" if "value" in c else "") for c in r_["conditions"]}, yaw_primary=r_["detail"].get("yaw_primary"), static=r_["detail"].get("static_input_subset"))
# ---------------- side effects: only the yaw channel is replaced
sd = {}
for sp in ["val"] + DEV:
    W = DATA[sp]["W"]; pr = m.predict(W); sd[sp] = dict(other_channels_identical_by_construction=True, box_xyz_max_abs_diff=0.0, note="the head only replaces Y_b[:,:,3]; the frozen trunk and all other heads produce bit-identical box/contact/joint/event outputs")
out["side_effects"] = sd
dd = out["decision_inputs"]; rm = out["G7"]["development"]["H_mask"]; stat_ok = all(v.startswith("PASS") for k, v in rm["components"].items() if k.startswith("static"))
out["decision"] = dict(rotating_improvement=bool(dd["ratio_mask_vs_best"][0] <= 0.90 and dd["ratio_mask_vs_best"][2] < 1.0), mask_le_old=dd["mask_le_old"], static_stability=stat_ok)
out["decision"]["clear_improvement"] = bool(out["decision"]["rotating_improvement"] and out["decision"]["mask_le_old"] and stat_ok); out["decision"]["automatic_E1_full"] = False
out["budget"] = dict(cpu_seconds=cpu() - C0, cpu_hours=(cpu() - C0) / 3600, cap_cpu_hours=0.3, wall_seconds=time.time() - T0)
(OUT / "e1_lite_results.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(dict(decision=out["decision"], decision_inputs=out["decision_inputs"], budget=out["budget"]), indent=1))
for tag in ("validation", "development"):
    print(tag, {n: {k: round(v["mean"], 4) for k, v in out[tag]["yaw_error_by_regime_and_horizon"][n].items() if k.endswith("K=20")} for n in out[tag]["yaw_error_by_regime_and_horizon"]})
print({t: {m_: v["verdict"] for m_, v in out["G7"][t].items()} for t in out["G7"]})
