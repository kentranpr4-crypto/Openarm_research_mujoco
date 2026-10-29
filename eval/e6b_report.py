#!/usr/bin/env python3
"""E6-B stage 0-1 report: comparison table V1 / C0 / S1 / S2, confidence intervals, plots, and the PRE-REGISTERED gates G0-G4.
Thresholds below are copied from docs/ai-native/tasks/2026-10-28-e6b-std-floor-plan.md section 8 and must NOT be edited after results exist.
Inputs (dev seeds 21-22 only): <root>/<arm>/<tag>/act/<cond>/{episodes.jsonl,summary.json,rollouts/*.npz}, <arm>/open_loop_val.json,
checkpoints' train_log.jsonl, unit_tests.json.   Usage (ACT venv): python eval/e6b_report.py [--root outputs/eval/e6b] [--out outputs/e6b]"""
import sys, json, argparse, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from eval.evaluate import wilson
from eval.audit_v1 import events, jerk_stats, ARM

# ---------------------------------------------------------------- PRE-REGISTERED (plan section 8) --------------------------------------------
G1_TRAIN_L1_REL, G1_VAL_L1_REL, G1_IDLE_DEV_RAD = 0.10, 0.10, 2e-3          # train-L1 EMA within +-10% of C0; val-L1 <= 1.10 x C0; S2 idle right-arm command deviation <= 2 mrad (2 mm for the gripper)
G2_SI_MIN, G2_SI_LARGE_INFO = 0.90, 0.80                                     # SI >= 0.90 for right_mean AND jitter_small; jitter_large informational (>= 0.80)
G3_NOM_MIN_OF_30, G3_BOUND_MARGIN_PP, G3_BOUND_CI_LOW_PP = 28, -5.0, -20.0   # NOM >= 28/30; BOUND(S) >= BOUND(C0) - 5 pp (point), paired 95% CI lower bound >= -20 pp
# ------------------------------------------------------------------------------------------------------------------------------------------------
ARMS = {"v1": "V1 (frozen, historical)", "c0": "C0 (train-only stats)", "s1": "S1 (+state floor)", "s2": "S2 (+state+action floor)"}
CKPT = {"v1": ROOT / "outputs/checkpoints/act_full_v1", "c0": ROOT / "outputs/checkpoints/e6b_c0_seed0", "s1": ROOT / "outputs/checkpoints/e6b_s1_seed0", "s2": ROOT / "outputs/checkpoints/e6b_s2_seed0"}
COL = {"v1": "0.55", "c0": "C0", "s1": "C1", "s2": "C3"}
TAGS = {"nom": "nom21", "b21": "bound21", "b22": "bound22", "right_mean": "pert_right_mean", "jit_small": "pert_jit_small", "jit_large": "pert_jit_large", "ind_joints": "ind_joints_small", "ind_grip": "ind_gripper_small"}
COND = {"nom": "nominal", "b21": "unseen_pose", "b22": "unseen_pose", "right_mean": "nominal", "jit_small": "nominal", "jit_large": "nominal", "ind_joints": "nominal", "ind_grip": "nominal"}

def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def load(root, arm, key, with_rollouts=False):
    d = Path(root) / arm / TAGS[key] / "act" / COND[key]
    if not (d / "episodes.jsonl").exists(): return None
    eps = jl(d / "episodes.jsonl")
    if with_rollouts:
        for e in eps:
            f = d / "rollouts" / f"ep{e['episode_index']:04d}.npz"
            if f.exists():
                with np.load(f) as z: e.update({k: z[k] for k in z.files})
    return dict(eps=eps, summary=json.loads((d / "summary.json").read_text()))

def kn(eps, key="task_success"): return sum(bool(e[key]) for e in eps), len(eps)
def paired_diff(a, b, n_boot=20000, seed=0):
    """a, b: aligned 0/1 arrays (same configs). Returns mean difference (pp), bootstrap-percentile 95% CI (pp), McNemar exact p, discordant counts."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1) * 100; fix, brk = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = float(st.binomtest(min(fix, brk), fix + brk, 0.5).pvalue) if fix + brk else 1.0
    return dict(diff_pp=float(d.mean() * 100), ci95_pp=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], mcnemar_p=p, a_only=fix, b_only=brk)
def vec(eps, keyf): return {(e["seed"], e["episode_index"]): float(e["task_success"]) for e in eps}

def gather(root, arm):
    R = dict(arm=arm); D = {k: load(root, arm, k, with_rollouts=True) for k in TAGS}
    if D["nom"] is None: return None
    R["data"] = D; nom = D["nom"]["eps"]; R["nom"] = kn(nom); R["nom_pick"] = kn(nom, "pick_success")
    bound = (D["b21"]["eps"] if D["b21"] else []) + (D["b22"]["eps"] if D["b22"] else []); R["bound"] = kn(bound) if bound else None; R["bound_pick"] = kn(bound, "pick_success") if bound else None
    R["bound_vec"] = [float(e["task_success"]) for e in sorted(bound, key=lambda e: (e["seed"], e["episode_index"]))]
    ids0 = {e["episode_index"] for e in D["right_mean"]["eps"]} if D["right_mean"] else set(); base = [e for e in nom if e["episode_index"] in ids0]
    R["si"] = {}
    for k in ("right_mean", "jit_small", "jit_large"):
        if D[k] is None: continue
        pk, n = kn(D[k]["eps"]); b = kn(base)[0]; R["si"][k] = dict(perturbed=f"{pk}/{n}", unperturbed_same_configs=f"{b}/{len(base)}", SI=(pk / b if b else float("nan")), perturbed_wilson95=wilson(pk, n))
    for k, ids in (("ind_joints", set(range(10))), ("ind_grip", set(range(10)))):
        if D[k] is None: continue
        b = kn([e for e in nom if e["episode_index"] in ids])[0]; pk, n = kn(D[k]["eps"]); R["si"][k] = dict(perturbed=f"{pk}/{n}", unperturbed_same_configs=f"{b}/{len(ids)}", SI=(pk / b if b else float("nan")), perturbed_wilson95=wilson(pk, n))
    # grasp TCP height error at closing, jerk, idle-arm deviation, latency (GT only for diagnostics)
    allr = [e for k in ("nom", "b21", "b22") if D[k] for e in D[k]["eps"] if "action" in e]
    dz, jint, jbnd, jall = [], [], [], []; dev_j, dev_g = 0.0, 0.0
    mu = None
    mfile = CKPT[arm] / "last" / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if mfile.exists():
        from safetensors.numpy import load_file; mu = load_file(str(mfile))["action.mean"]
    for e in allr:
        ev = events(e)
        if ev["t_close"] is not None: dz.append(ev["dz"] * 100)
        js, _, _ = jerk_stats(e["action"], H=20); jall.append(js["all"]); jint.append(js["interior"]); jbnd.append(js["boundary"])
        if mu is not None:
            dev_j = max(dev_j, float(np.abs(e["action"][:, 8:15] - mu[8:15]).max())); dev_g = max(dev_g, float(np.abs(e["action"][:, 15] - mu[15]).max()))
    R["dz_cm"] = np.array(dz); R["dz_summary"] = dict(mean=float(np.mean(dz)), std=float(np.std(dz)), frac_gt_3p4=float(np.mean(np.array(dz) > 3.4)), n=len(dz)) if dz else None
    R["jerk"] = dict(all=float(np.mean(jall)), interior=float(np.mean(jint)), boundary=float(np.mean(jbnd))) if jall else None
    R["idle_dev"] = dict(joints_rad=dev_j, gripper_m=dev_g)
    sm = D["nom"]["summary"]; R["latency_ms_per_query"] = sm.get("latency_ms_per_query"); R["tracking_mean_arm_rad"] = sm.get("tracking_mean_arm_rad")
    ol = Path(root) / arm / "open_loop_val.json"; R["open_loop_val"] = json.loads(ol.read_text()) if ol.exists() else None
    tl = CKPT[arm] / "train_log.jsonl"
    if tl.exists():
        log = jl(tl); v = [x for x in log if "val_l1" in x]; R["train"] = dict(steps=len(log), loss_ema_last=log[-1]["loss_ema"], val_l1_last=v[-1]["val_l1"] if v else None, nan=any(np.isnan(x["loss"]) for x in log)); R["log"] = log
    return R

def gates(res, ut_pass):
    G = {}; c0 = res.get("c0")
    for arm in ("s1", "s2"):
        r = res.get(arm)
        if r is None or c0 is None: continue
        g = {}; g["G0"] = dict(passed=bool(ut_pass), note="unit tests T1-T13+TA/TB/TC and 200-step smoke (this report reads unit_tests.json)")
        t, t0 = r["train"], c0["train"]; rel_tr = t["loss_ema_last"] / t0["loss_ema_last"] - 1; rel_va = t["val_l1_last"] / t0["val_l1_last"] - 1
        idle_ok = (r["idle_dev"]["joints_rad"] <= G1_IDLE_DEV_RAD and r["idle_dev"]["gripper_m"] <= G1_IDLE_DEV_RAD) if arm == "s2" else None
        g["G1"] = dict(passed=bool((not t["nan"]) and abs(rel_tr) <= G1_TRAIN_L1_REL and rel_va <= G1_VAL_L1_REL and (idle_ok in (None, True))),
                       train_l1_ema_rel_to_C0=float(rel_tr), val_l1_rel_to_C0=float(rel_va), idle_dev=r["idle_dev"], idle_gate_applies_to_S2_only=True, idle_ok=idle_ok,
                       caveat="normalised L1 is not strictly comparable across arms with different action scaling; physical-unit val MAE is reported separately (supplementary, not a gate)")
        si = r["si"]; ok2 = all(k in si and si[k]["SI"] >= G2_SI_MIN for k in ("right_mean", "jit_small"))
        g["G2"] = dict(passed=bool(ok2), SI={k: float(v["SI"]) for k, v in si.items()}, jitter_large_info_pass=bool(si.get("jit_large", {}).get("SI", 0) >= G2_SI_LARGE_INFO))
        nk, nn = r["nom"]; pdiff = paired_diff(r["bound_vec"], c0["bound_vec"]) if r["bound_vec"] and c0["bound_vec"] and len(r["bound_vec"]) == len(c0["bound_vec"]) else None
        ok3 = bool(nk >= G3_NOM_MIN_OF_30 and pdiff is not None and pdiff["diff_pp"] >= G3_BOUND_MARGIN_PP and pdiff["ci95_pp"][0] >= G3_BOUND_CI_LOW_PP)
        g["G3"] = dict(passed=ok3, nominal=f"{nk}/{nn}", bound_vs_C0=pdiff)
        g["G4_adopt_candidate"] = bool(all(g[x]["passed"] for x in ("G0", "G1", "G2", "G3"))); G[arm] = g
    if "s1" in G and "s2" in G: G["preferred_if_both_pass"] = "s1" if G["s1"]["G4_adopt_candidate"] else ("s2" if G["s2"]["G4_adopt_candidate"] else None)
    return G

def plots(res, out):
    arms = [a for a in ARMS if a in res]
    # P1 success with Wilson CI
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); w = 0.8 / len(arms)
    for j, (title, getter) in enumerate((("NOM (seed 21, n=30)", lambda r: r["nom"]), ("BOUND (seeds 21+22, n=60)", lambda r: r["bound"]), ("BOUND pick-only", lambda r: r["bound_pick"]))):
        for i, a in enumerate(arms):
            kv = getter(res[a]);
            if not kv: continue
            k, n = kv; lo, hi = wilson(k, n); ax[j].bar(i, 100 * k / n, color=COL[a], yerr=[[100 * k / n - 100 * lo], [100 * hi - 100 * k / n]], capsize=4); ax[j].text(i, 100 * hi + 1, f"{k}/{n}", ha="center", fontsize=8)
        ax[j].set_xticks(range(len(arms))); ax[j].set_xticklabels([a.upper() for a in arms]); ax[j].set_ylim(0, 112); ax[j].set_title(title); ax[j].set_ylabel("success [%] (Wilson 95%)")
    fig.tight_layout(); fig.savefig(out / "P1_success.png", dpi=110); plt.close(fig)
    # P2 shortcut sensitivity
    keys = [("none", "unperturbed"), ("right_mean", "right_mean"), ("jit_small", "jitter small"), ("jit_large", "jitter large"), ("ind_joints", "joints only (small)"), ("ind_grip", "gripper only (small)")]
    fig, ax = plt.subplots(figsize=(11, 4)); w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        r = res[a]; vals, lo, hi = [], [], []
        for k, _ in keys:
            if k == "none": kk, nn = kn([e for e in r["data"]["nom"]["eps"] if e["episode_index"] < 20])
            elif k in r["si"]: kk, nn = map(int, r["si"][k]["perturbed"].split("/"))
            else: kk, nn = 0, 1
            l, h = wilson(kk, nn); vals.append(100 * kk / nn); lo.append(100 * kk / nn - 100 * l); hi.append(100 * h - 100 * kk / nn)
        ax.bar(np.arange(len(keys)) + i * w, vals, w, color=COL[a], label=a.upper(), yerr=[lo, hi], capsize=2)
    ax.set_xticks(np.arange(len(keys)) + w * (len(arms) - 1) / 2); ax.set_xticklabels([n for _, n in keys], fontsize=8); ax.set_ylabel("NOM success under right-arm state perturbation [%]"); ax.legend(); ax.set_title("Shortcut sensitivity (unit-correct perturbations; n=20, joints/gripper-only n=10)")
    fig.tight_layout(); fig.savefig(out / "P2_shortcut.png", dpi=110); plt.close(fig)
    # P3 closing height
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [res[a]["dz_cm"] for a in arms if len(res[a]["dz_cm"])]; ax.boxplot(data, tick_labels=[a.upper() for a in arms if len(res[a]["dz_cm"])], showfliers=True); ax.axhline(2.5, c="gray", ls="--", label="expert 2.5 cm"); ax.axhline(3.4, c="k", ls=":", label="overlap < 1.7 cm")
    ax.set_ylabel("TCP z - box z when closing starts [cm]"); ax.legend(); ax.set_title("Grasp TCP height at closing (NOM + BOUND rollouts)"); fig.tight_layout(); fig.savefig(out / "P3_closing_height.png", dpi=110); plt.close(fig)
    # P4 jerk, P7 latency
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, a in enumerate(arms):
        j = res[a]["jerk"]
        if j: ax[0].bar(i - 0.2, j["interior"], 0.4, color=COL[a]); ax[0].bar(i + 0.2, j["boundary"], 0.4, color=COL[a], alpha=0.45)
        L = res[a]["latency_ms_per_query"]
        if L: ax[1].bar(np.arange(3) + i * w, [L["p50"], L["p95"], L["p99"]], w, color=COL[a], label=a.upper())
    ax[0].set_xticks(range(len(arms))); ax[0].set_xticklabels([a.upper() for a in arms]); ax[0].set_ylabel("mean |jerk| [rad/s^3]  (solid = interior, light = chunk boundary)"); ax[0].axhline(6.7, c="gray", ls="--")
    ax[1].set_xticks(np.arange(3) + w * (len(arms) - 1) / 2); ax[1].set_xticklabels(["p50", "p95", "p99"]); ax[1].set_ylabel("latency per chunk query [ms], CPU 1 thread"); ax[1].legend(); fig.tight_layout(); fig.savefig(out / "P4_jerk_latency.png", dpi=110); plt.close(fig)
    # P5 training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a in arms:
        log = res[a].get("log")
        if not log: continue
        ax[0].plot([x["step"] for x in log], [x["loss_ema"] for x in log], c=COL[a], label=a.upper()); v = [x for x in log if "val_l1" in x]; ax[1].plot([x["step"] for x in v], [x["val_l1"] for x in v], c=COL[a], label=a.upper())
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("train L1 EMA (normalised, arm-specific scaling)"); ax[1].set_xlabel("step"); ax[1].set_ylabel("val L1 (normalised)"); ax[1].legend(); fig.tight_layout(); fig.savefig(out / "P5_training.png", dpi=110); plt.close(fig)

def table(res, G, ut):
    L = []; arms = [a for a in ARMS if a in res]
    def cell(kv): k, n = kv; lo, hi = wilson(k, n); return f"{k}/{n} ({100*k/n:.0f}% [{100*lo:.0f}-{100*hi:.0f}])"
    L.append("| metric | " + " | ".join(ARMS[a] for a in arms) + " |\n|---|" + "---|" * len(arms))
    rows = [("NOM task (seed 21, n=30)", lambda r: cell(r["nom"])), ("BOUND task (seeds 21+22, n=60)", lambda r: cell(r["bound"]) if r["bound"] else "-"), ("BOUND pick", lambda r: cell(r["bound_pick"]) if r["bound_pick"] else "-")]
    for k, nm in (("right_mean", "right_mean"), ("jit_small", "jitter small"), ("jit_large", "jitter large"), ("ind_joints", "right joints only (small)"), ("ind_grip", "right gripper only (small)")):
        rows.append((f"NOM under {nm}: success / SI", lambda r, k=k: (f"{r['si'][k]['perturbed']} / SI={r['si'][k]['SI']:.2f} (unperturbed {r['si'][k]['unperturbed_same_configs']})" if k in r["si"] else "-")))
    rows += [("grasp TCP height at closing, mean±sd [cm] (frac > 3.4 cm)", lambda r: f"{r['dz_summary']['mean']:.2f}±{r['dz_summary']['std']:.2f} ({100*r['dz_summary']['frac_gt_3p4']:.0f}%)" if r["dz_summary"] else "-"),
             ("action jerk all / interior / boundary [rad/s^3]", lambda r: f"{r['jerk']['all']:.1f} / {r['jerk']['interior']:.1f} / {r['jerk']['boundary']:.1f}" if r["jerk"] else "-"),
             ("latency per query p50/p95/p99 [ms]", lambda r: (lambda L_: f"{L_['p50']:.1f}/{L_['p95']:.1f}/{L_['p99']:.1f}" if L_ else "-")(r["latency_ms_per_query"])),
             ("tracking mean arm [rad]", lambda r: f"{r['tracking_mean_arm_rad']:.4f}" if r["tracking_mean_arm_rad"] is not None else "-"),
             ("idle right-arm command deviation: joints [mrad] / gripper [mm]", lambda r: f"{1e3*r['idle_dev']['joints_rad']:.3f} / {1e3*r['idle_dev']['gripper_m']:.3f}"),
             ("train L1 EMA / val L1 (normalised, last)", lambda r: f"{r['train']['loss_ema_last']:.4f} / {r['train']['val_l1_last']:.4f}" if r.get("train") else "-"),
             ("val MAE left arm [rad] / left gripper [mm] (physical, teacher-forced)", lambda r: f"{r['open_loop_val']['mae_left_arm_rad']:.4f} / {1e3*r['open_loop_val']['per_joint_mae'][7]:.2f}" if r.get("open_loop_val") else "-")]
    for nm, f in rows: L.append(f"| {nm} | " + " | ".join(f(res[a]) for a in arms) + " |")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT / "outputs/eval/e6b"); ap.add_argument("--out", type=Path, default=ROOT / "outputs/e6b"); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    res = {arm: r for arm in ARMS if (r := gather(a.root, arm)) is not None}
    ut = json.loads((a.out / "unit_tests.json").read_text())["all_pass"] if (a.out / "unit_tests.json").exists() else False
    G = gates(res, ut); plots(res, a.out); md = table(res, G, ut)
    clean = {arm: {k: v for k, v in r.items() if k not in ("data", "log", "dz_cm", "bound_vec")} for arm, r in res.items()}
    (a.out / "stage1_results.json").write_text(json.dumps(dict(arms=clean, gates=G, thresholds=dict(G1_TRAIN_L1_REL=G1_TRAIN_L1_REL, G1_VAL_L1_REL=G1_VAL_L1_REL, G1_IDLE_DEV_RAD=G1_IDLE_DEV_RAD, G2_SI_MIN=G2_SI_MIN, G3_NOM_MIN_OF_30=G3_NOM_MIN_OF_30, G3_BOUND_MARGIN_PP=G3_BOUND_MARGIN_PP, G3_BOUND_CI_LOW_PP=G3_BOUND_CI_LOW_PP)), indent=1, default=float))
    (a.out / "stage1_table.md").write_text(md); print(md); print(json.dumps(G, indent=1, default=float))

if __name__ == "__main__":
    main()
