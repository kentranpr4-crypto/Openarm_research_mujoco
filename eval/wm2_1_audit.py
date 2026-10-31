#!/usr/bin/env python3
"""WM-2.1 audit (diagnostics only; no training): uses the reproduced WM2-C0 predictions (outputs/wm2_1/preds) and the frozen windows.  Reads only; writes outputs/wm2_1/audit.json and plots A1-A4.
Sections: (1) errors by horizon / variable group / object pose / contact state, (2) ridge-vs-GRU per variable and the post-hoc hybrid (ridge joints + GRU object/contact; no training),
(3) loss decomposition on val, (4) G3 per label (prevalence, positive groups, outcome-only), (5) G4 per action family incl. phase-matched windows.  Run (ACT venv): python eval/wm2_1_audit.py"""
import sys, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "data"))
sys.argv = [sys.argv[0], "none"]
import wm2_run as R, wm2_models as M, wm2_gates as G, wm2_common as C, wm1_windows as W1
PR = ROOT / "outputs/wm2_1/preds"; OUT = ROOT / "outputs/wm2_1"; SP = R.EVAL; MODELS = ["persistence", "ridge", "mlp", "gru"]
def lp(m, sp): z = np.load(PR / f"{m}_{sp}.npz"); return {k: z[k] for k in z.files}
def cum(Y): return np.cumsum(Y, 1)
def macro_by_group(v, g): return G.macro(v, g)
res = {}; Wd = {sp: R.load(sp) for sp in SP}; T = {sp: R.truth(Wd[sp]) for sp in SP}; P = {m: {sp: lp(m, sp) for sp in SP} for m in MODELS}; act = {sp: G.active_mask(Wd[sp]) for sp in SP}; grp = {sp: Wd[sp]["group"] for sp in SP}
# ---- (1) per-horizon x variable
JN = [f"joint{i+1}" for i in range(7)] + ["gripper"]; OB = ["box_x_mm", "box_y_mm", "box_z_mm", "yaw_rad", "speed_mps", "tilt_rad", "aoffset_cm"]; CN = ["finger_force", "n_contacts", "a_contact", "a_valid", "palm_force", "table_force", "tray_force"]; sc = [1000, 1000, 1000, 1, 1, 1, 100]
V = {}
for m in MODELS:
    V[m] = {}
    for sp in SP:
        a = act[sp]; g = grp[sp][a]; V[m][sp] = {}
        for k in (1, 5, 10, 20):
            ds = np.abs(cum(P[m][sp]["Y_s"][a])[:, k - 1] - cum(T[sp]["Y_s"][a])[:, k - 1]); db = np.abs(cum(P[m][sp]["Y_b"][a])[:, k - 1] - cum(T[sp]["Y_b"][a])[:, k - 1]) * np.array(sc); dc = np.abs(P[m][sp]["Y_c"][a][:, k - 1] - T[sp]["Y_c"][a][:, k - 1])
            V[m][sp][f"K={k}"] = dict(joints={n: float(macro_by_group(ds[:, i], g)) for i, n in enumerate(JN)}, object={n: float(macro_by_group(db[:, i], g)) for i, n in enumerate(OB)}, contact={n: float(macro_by_group(dc[:, i], g)) for i, n in enumerate(CN)})
res["errors_by_variable"] = V
# ---- contact-state strata (GRU vs baselines, K=20 box and a-offset)
strata = {}
for sp in ("test_id", "test_ood_low_strict"):
    W = Wd[sp]; a = act[sp]; fc = W["C_hist"][:, -1, 1] > 0; carried = (W["B_hist"][:, -1, 2] > 0.4105); palm = T[sp]["Y_e"][:, 1] > 0; closed = W["A_hist"][:, -1, 7] < 0.028
    for nm, mk in (("finger_contact_at_t", fc), ("no_finger_contact_at_t", ~fc), ("box_carried", carried), ("box_on_table_or_tray", ~carried), ("palm_contact_in_horizon", palm), ("no_palm_contact_in_horizon", ~palm), ("gripper_closed_cmd", closed), ("gripper_open_cmd", ~closed)):
        mm = a & mk
        if len(np.unique(grp[sp][mm])) < 3: continue
        strata.setdefault(sp, {})[nm] = dict(n=int(mm.sum()), groups=int(len(np.unique(grp[sp][mm]))), **{m: {k: float(G.macro(v, grp[sp][mm])) for k, v in G.errors(G.sub(P[m][sp], mm), G.sub(T[sp], mm), 20).items()} for m in MODELS})
res["contact_state_strata_K20"] = strata
# ---- (2) ridge vs GRU per variable + hybrid (ridge joints + GRU object/contact)
win = {}
for sp in SP:
    for part, names in (("joints", JN), ("object", OB), ("contact", CN)):
        for i, n in enumerate(names):
            e = {m: V[m][sp]["K=20"][part][n] for m in ("ridge", "mlp", "gru")}; win.setdefault(sp, {})[f"{part}/{n}"] = dict(e, best=min(e, key=e.get))
res["per_variable_best_model_K20"] = win
cnt = {}
for part in ("joints", "object", "contact"):
    for sp in SP: cnt.setdefault(part, {})[sp] = {m: sum(1 for k, v in win[sp].items() if k.startswith(part) and v["best"] == m) for m in ("ridge", "mlp", "gru")}
res["win_counts_by_group"] = cnt
hyb = {}
for sp in SP:
    Ph = dict(Y_s=P["ridge"][sp]["Y_s"], Y_b=P["gru"][sp]["Y_b"], Y_c=P["gru"][sp]["Y_c"], Y_e=P["gru"][sp]["Y_e"]); a = act[sp]
    hyb[sp] = {k: float(G.macro(v, grp[sp][a])) for k, v in G.errors(G.sub(Ph, a), G.sub(T[sp], a), 20).items()}
res["hybrid_ridge_joints_gru_object_K20"] = hyb
# ---- (3) loss decomposition on val (normalised space)
tr = R.load("train"); nrm = M.Norm(tr); va = R.load("val"); Yn = nrm.targets(va); lossd = {}
for m in ("ridge", "mlp", "gru"):
    p = lp(m, "val"); n = len(Yn); pn = np.concatenate([nrm.n(p["Y_s"], nrm.Ys).reshape(n, -1), nrm.n(p["Y_b"], nrm.Yb).reshape(n, -1), nrm.n(p["Y_c"], nrm.Yc).reshape(n, -1)], 1)
    lossd[m] = dict(joint_mse=float(((pn[:, :160] - Yn[:, :160]) ** 2).mean()), object_mse=float(((pn[:, 160:300] - Yn[:, 160:300]) ** 2).mean()), contact_mse=float(((pn[:, 300:] - Yn[:, 300:]) ** 2).mean()))
    e = va["Y_e"]; pe = np.clip(p["Y_e"], 1e-6, 1 - 1e-6); lossd[m]["event_bce"] = float(-(e * np.log(pe) + (1 - e) * np.log(1 - pe)).mean())
res["val_loss_decomposition_normalised"] = lossd
# ---- (4) G3 per label
lab = ["close", "palm", "lifted", "placed"]; oo = {sp: lp("outcome_only", sp) for sp in SP}; g3 = {}
for sp in SP:
    for j, l in enumerate(lab):
        mk = G.at_risk_mask(Wd[sp], l); y = T[sp]["Y_e"][mk][:, j].astype(int); g = grp[sp][mk]
        if y.sum() == 0 or (1 - y).sum() == 0: g3[f"{l}/{sp}"] = dict(n=int(mk.sum()), pos=int(y.sum()), prevalence=float(y.mean()) if len(y) else None, pos_groups=int(len(np.unique(g[y == 1]))), neg_groups=int(len(np.unique(g[y == 0]))), auroc_gru=None); continue
        W = Wd[sp]; heur = {"close": -W["A_fut"][mk][:, :, 7].min(1), "lifted": W["B_hist"][mk][:, -1, 2], "placed": -np.linalg.norm(W["B_hist"][mk][:, -1, :2] - G.TRAY, axis=1), "palm": W["A_fut"][mk][:, :, 7].min(1) * 0 - np.abs(W["B_hist"][mk][:, -1, 6])}[l]
        g3[f"{l}/{sp}"] = dict(n=int(mk.sum()), pos=int(y.sum()), neg=int((1 - y).sum()), prevalence=float(y.mean()), pos_groups=int(len(np.unique(g[y == 1]))), neg_groups=int(len(np.unique(g[y == 0]))), auroc_gru=G.auroc(y, P["gru"][sp]["Y_e"][mk][:, j]), auroc_ridge=G.auroc(y, P["ridge"][sp]["Y_e"][mk][:, j]), auroc_mlp=G.auroc(y, P["mlp"][sp]["Y_e"][mk][:, j]), auroc_outcome_only=G.auroc(y, oo[sp]["Y_e"][mk][:, j]), auroc_simple_heuristic=G.auroc(y, heur))
res["G3_by_label_split"] = g3
# ---- (5) G4 per family: standard pair windows
pz = np.load(R.WIN / "pairs.npz"); pairs = {k: pz[k] for k in pz.files}; pp = {m: lp(m, "pairs") for m in ("gru", "ridge") if (PR / f"{m}_pairs.npz").exists()}
qa = lambda Pd: (pairs["B_hist"][:, -1, 6] + cum(Pd["Y_b"])[:, 19, 6]) * 100; qt = qa(pairs); fam = pairs["family"]; pid = pairs["pid"]; vid = pairs["vid"]; ctl = {p: i for p, i in zip(pid[vid == 0], np.nonzero(vid == 0)[0])}
dT = np.array([qt[i] - qt[ctl[pid[i]]] for i in range(len(pid))]); g4 = {}
for m, Pd in pp.items():
    qp = qa(Pd); dP = np.array([qp[i] - qp[ctl[pid[i]]] for i in range(len(pid))])
    for f, nm in ((1, "depth"), (2, "timing"), (3, "noise"), (4, "speed"), (5, "release_height")):
        mk = fam == f; sel = mk & (np.abs(dT) >= 0.5); g4.setdefault(m, {})[nm] = dict(n=int(mk.sum()), true_abs_effect_mean_cm=float(np.abs(dT[mk]).mean()), pred_abs_effect_mean_cm=float(np.abs(dP[mk]).mean()), frac_true_effect_ge_0p5cm=float(sel.sum() / max(mk.sum(), 1)), spearman=G.spearman(dP[mk], dT[mk]), sign_agree_on_effect_ge_0p5=float(np.mean(np.sign(dP[sel]) == np.sign(dT[sel]))) if sel.any() else None)
res["G4_by_family_standard_pairs"] = g4
# phase-matched windows: speed (mid-transport), release_height (release-10), timing (close-25) using the GRU weights
idx = C.load_index(); gid = C.group_ids(idx); I = [e for e in idx if e["kind"] == "interventional" and e["split"].startswith("test") and e["replay_ok"]]; byp = {}
for e in I: byp.setdefault(e["pid"], {})[e["idx"]] = e
gru = M.NN("gru"); gru.nrm = nrm; gru.model = M.GRUModel(); gru.model.load_state_dict(torch.load(OUT / "models/gru_c0.pt")); gru.model.eval()
def phase_windows(tfun, fam_id):
    rows = {k: [] for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut", "Y_s", "Y_b", "Y_c", "Y_e", "pid", "vid", "level")}
    for pidx, vs in byp.items():
        c = vs[0]; cl, rl = c["outcomes"]["close_step"], c["outcomes"]["release_step"]
        if cl is None or rl is None: continue
        t = tfun(cl, rl)
        for vid_, e in vs.items():
            if e["family"] not in ("control", {1: "depth", 2: "timing", 3: "noise", 4: "speed", 5: "release_height"}[fam_id]) or not e["replay_ok"]: continue
            ep = C._ep(e); tt = int(min(max(t, W1.H), len(ep["a"]) - 1 - W1.K)); X, Y = W1.get_window(ep, tt)
            for k in ("S_hist", "B_hist", "C_hist", "A_hist", "A_fut"): rows[k].append(X[k])
            for k in ("Y_s", "Y_b", "Y_c", "Y_e"): rows[k].append(Y[k])
            rows["pid"].append(pidx); rows["vid"].append(vid_); rows["level"].append(e["level"] if e["level"] is not None else 0.0)
    return {k: np.asarray(v, np.float32 if k not in ("pid", "vid") else np.int64) for k, v in rows.items()}
PM = {}
for fam_id, nm, tf, quantity in ((4, "speed", lambda cl, rl: (cl + rl) // 2 - 10, "box displacement xy at t+K [mm]"), (5, "release_height", lambda cl, rl: rl - 10, "box z at t+K [mm]"), (2, "timing", lambda cl, rl: cl - 25, "P(close in horizon)"), (1, "depth", lambda cl, rl: cl - 10, "a-offset at t+K [cm]")):
    Wp = phase_windows(tf, fam_id)
    if len(Wp["pid"]) < 10: PM[nm] = dict(note="insufficient windows"); continue
    pr = gru.predict(Wp); pid_, vid_ = Wp["pid"], Wp["vid"]; ctl_ = {p: i for p, i in zip(pid_[vid_ == 0], np.nonzero(vid_ == 0)[0])}
    def Q(Wd_, is_pred):
        if nm == "speed": return np.linalg.norm(cum(Wd_["Y_b"])[:, 19, :2], axis=1) * 1000
        if nm == "release_height": return (Wp["B_hist"][:, -1, 2] + cum(Wd_["Y_b"])[:, 19, 2]) * 1000
        if nm == "timing": return Wd_["Y_e"][:, 0]
        return (Wp["B_hist"][:, -1, 6] + cum(Wd_["Y_b"])[:, 19, 6]) * 100
    qt_, qp_ = Q(Wp, False), Q(pr, True); mk = vid_ != 0; dT_ = np.array([qt_[i] - qt_[ctl_[pid_[i]]] for i in np.nonzero(mk)[0]]); dP_ = np.array([qp_[i] - qp_[ctl_[pid_[i]]] for i in np.nonzero(mk)[0]]); thr = dict(speed=2.0, release_height=2.0, timing=0.5, depth=5.0)[nm] if False else dict(speed=2.0, release_height=2.0, timing=0.5, depth=0.5)[nm]
    sel = np.abs(dT_) >= thr; PM[nm] = dict(quantity=quantity, n=int(mk.sum()), poses=int(len(set(pid_.tolist()))), true_abs_effect_mean=float(np.abs(dT_).mean()), pred_abs_effect_mean=float(np.abs(dP_).mean()), frac_true_effect_ge_threshold=float(sel.mean()), threshold=thr, spearman=G.spearman(dP_, dT_), sign_agree_on_large_effects=float(np.mean(np.sign(dP_[sel]) == np.sign(dT_[sel]))) if sel.any() else None)
res["G4_phase_matched_families_gru"] = PM
(OUT / "audit.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))); print("audit written")
for k in ("win_counts_by_group", "hybrid_ridge_joints_gru_object_K20", "val_loss_decomposition_normalised", "G4_by_family_standard_pairs", "G4_phase_matched_families_gru"): print(k, json.dumps(res[k], default=str)[:1800])
