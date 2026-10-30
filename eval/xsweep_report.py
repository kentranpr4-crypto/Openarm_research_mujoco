#!/usr/bin/env python3
"""x-sweep step 3: analysis implementing outputs/xsweep/xsweep_config.json (frozen; hypotheses HA-HF and their decision rules are READ from the registered definitions below,
identical to the pre-registration).  Live metrics from the evaluation rollouts, offsets/contacts from the deterministic replay.  Exploratory items are labelled.
Writes outputs/xsweep/xsweep_results.json, xsweep_tables.md, plots X1..X6.  Run (ACT venv): python eval/xsweep_report.py [--root outputs/xsweep]"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, argparse, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
from eval.metrics import task_labels
from eval.evaluate import wilson
EXPECT = "6c59292d6d951673ab813fc50eba38612c01777534ccdad2a6f3c5797b1f596c"
ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT / "outputs/xsweep"); ap.add_argument("--skip-sha-check", action="store_true"); A = ap.parse_args(); OUT = A.root
CFGP = ROOT / "outputs/xsweep/xsweep_config.json"
if hashlib.sha256(CFGP.read_bytes()).hexdigest() != EXPECT and not A.skip_sha_check: sys.exit("x-sweep config modified after registration")
CFG = json.loads(CFGP.read_text()); POSES = {p["idx"]: p for p in CFG["poses"]}; STR = CFG["strata"]; TRAY = np.array([0.44, 0.30]); TRAY_TOP = 0.36 + 0.004; ARMS, SEEDS = ("c0", "s1"), (0, 1, 2)
XS = CFG["x_grid_m"]
def stratum_of(x):
    for k, v in STR.items():
        if any(abs(x - w) < 1e-6 for w in v): return k

def events(R):
    g = R["gripper_cmd"]; box = R["box_pos"]; n = len(g); cl = np.nonzero(g < 0.028)[0]; tc = int(cl[0]) if len(cl) else None
    rise = box[:, 2] - box[0, 2]; tl = int(np.argmax(rise > 0.03)) if (rise > 0.03).any() else None; tr = None
    if tc is not None:
        r = np.nonzero(g[tc:] > 0.035)[0]; tr = tc + int(r[0]) if len(r) else None
    e = dict(close=tc, lift=tl, release=tr)
    if tc is not None: e["hold"] = min(tc + 14, n - 1)
    return e

meta = json.loads((OUT / "replay_meta.json").read_text()); ERR = {(m["agent"], m["idx"]): m["err"] for m in meta["episodes"]}
LIVE = {}
for arm in ARMS:
    for s in SEEDS:
        for l in (OUT / f"eval/{arm}_seed{s}/act/sweep/episodes.jsonl").read_text().splitlines():
            if l.strip(): e = json.loads(l); LIVE[(f"{arm}_s{s}", e["episode_index"])] = e
ROWS = []
for j in meta["jobs"]:
    ag, idx = j["agent"], j["idx"]; pz = POSES[idx]; R = dict(np.load(OUT / f"replay/{ag}_SW_{idx:03d}.npz")); lab = task_labels(R["box_pos"], R["tcp_pos"], R["box_speed"], TRAY, TRAY_TOP); ev = events(R)
    fin = (R["box_pos"][-1, :2] - TRAY)
    row = dict(agent=ag, arm=j["arm"], tseed=j["tseed"], idx=idx, x=pz["x"], block=pz["block"], stratum=stratum_of(pz["x"]), replay_err_m=ERR.get((ag, idx)))
    if j["arm"] == "expert": row.update(success=bool(lab["task_success"]), pick=bool(lab["pick_success"]), place=bool(lab["place_success"]), final_err_m=lab["final_box_to_tray_xy_m"], live=False)
    else:
        L = LIVE[(ag, idx)]; row.update(success=bool(L["task_success"]), pick=bool(L["pick_success"]), place=bool(L["place_success"]), final_err_m=L["final_box_to_tray_xy_m"], live=True, outcome=L["outcome"],
                                         label_agree=bool(lab["task_success"]) == bool(L["task_success"]))
    row["replay_ok"] = (row["replay_err_m"] is None or row["replay_err_m"] <= 0.005) and row.get("label_agree", True)
    tc, tr = ev["close"], ev["release"]; at = lambda t, i: float(R["box_in_tcp"][t, i] * 100) if t is not None else np.nan
    row.update(a_close=at(tc, 2), a_hold=at(ev.get("hold"), 2), a_release=at(tr, 2), c_close=at(tc, 1), slip=(at(tr, 2) - at(tc, 2)) if (tc is not None and tr is not None) else np.nan,
               max_sep_carry=float(np.linalg.norm(R["box_pos"][ev["lift"]:tr] - R["tcp_pos"][ev["lift"]:tr], axis=1).max() * 100) if (ev["lift"] is not None and tr is not None and tr > ev["lift"]) else np.nan,
               palm_contact=bool((R["hand_force"][tc:(tr or len(R["hand_force"]))] > 0.01).any()) if tc is not None else None,
               finger_force_hold=float(R["finger_force"][ev["hold"]]) if tc is not None else np.nan, finger_a_hold=float(R["finger_contact_a"][ev["hold"]] * 100) if tc is not None else np.nan,
               final_x_cm=float(fin[0] * 100), final_y_cm=float(fin[1] * 100))
    post = row["pick"] and not row["success"]; row["post_pick_fail"] = bool(post); row["cluster"] = bool(post and fin[0] < -0.025 and abs(fin[1]) < 0.01)
    ROWS.append(row)
POL = [r for r in ROWS if r["arm"] != "expert"]; EXP = [r for r in ROWS if r["arm"] == "expert"]
def sel(rows, **kw): return [r for r in rows if all((r[k] in v) if isinstance(v, (list, tuple, set)) else r[k] == v for k, v in kw.items())]
def mean(rows, key, only_ok=False):
    v = [r[key] for r in rows if (r["replay_ok"] or not only_ok) and r[key] == r[key] and r[key] is not None]; return float(np.mean(v)) if v else float("nan")
def sdv(rows, key):
    v = [r[key] for r in rows if r[key] == r[key] and r[key] is not None]; return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
def rate(rows, key="success"): return float(np.mean([r[key] for r in rows])) * 100 if rows else float("nan")
def tci(d):
    d = np.asarray(d, float); n = len(d); se = d.std(ddof=1) / np.sqrt(n); t = st.t.ppf(0.975, n - 1); return float(d.mean()), [float(d.mean() - t * se), float(d.mean() + t * se)]
def seed_pairs(xs_f, key):                      # per-seed mean over the selected x values; S1 - C0
    out = []
    for s in SEEDS:
        c = [r for r in POL if r["arm"] == "c0" and r["tseed"] == s and xs_f(r["x"])]; q = [r for r in POL if r["arm"] == "s1" and r["tseed"] == s and xs_f(r["x"])]
        out.append((rate(c) if key == "success" else mean(c, key, only_ok=key in ("a_close", "slip")), rate(q) if key == "success" else mean(q, key, only_ok=key in ("a_close", "slip"))))
    return out
def paired_summary(xs_f, key):
    pr = seed_pairs(xs_f, key); d = [b - a for a, b in pr]; m, ci = tci(d); return dict(per_seed_c0=[a for a, b in pr], per_seed_s1=[b for a, b in pr], diff=d, mean=m, ci95=ci, n_negative=int(sum(x < 0 for x in d)))
res = dict(config_sha256=hashlib.sha256(CFGP.read_bytes()).hexdigest(), n_policy_episodes=len(POL), n_expert_episodes=len(EXP))
# ---- acceptance
exp_main = [r for r in EXP if 0.36 - 1e-6 <= r["x"] <= 0.47 + 1e-6]
res["AC"] = dict(AC1_expert_success_x036_047=[sum(r["success"] for r in exp_main), len(exp_main)], AC1_pass=bool(rate(exp_main) >= 90), expert_success_by_x={f"{x:.2f}": [sum(r["success"] for r in sel(EXP, x=x)), len(sel(EXP, x=x))] for x in XS},
                 AC2_episodes_per_checkpoint={f"{a}_s{s}": len(sel(POL, agent=f"{a}_s{s}")) for a in ARMS for s in SEEDS}, AC2_pass=bool(all(len(sel(POL, agent=f"{a}_s{s}")) == 60 for a in ARMS for s in SEEDS)),
                 AC4_live_vs_replay_label_agreement=[sum(r["label_agree"] for r in POL), len(POL)], AC4_diverging=[dict(agent=r["agent"], idx=r["idx"], x=r["x"], err_m=r["replay_err_m"], label_agree=r.get("label_agree")) for r in ROWS if not r["replay_ok"]])
# ---- per x tables
PX = {}
for x in XS:
    PX[f"{x:.2f}"] = {"expert": dict(n=len(sel(EXP, x=x)), success=rate(sel(EXP, x=x)), a_close=mean(sel(EXP, x=x), "a_close"))}
    for arm in ARMS:
        R = sel(POL, arm=arm, x=x); k = sum(r["success"] for r in R); PX[f"{x:.2f}"][arm] = dict(n=len(R), success=rate(R), wilson=[100 * v for v in wilson(k, len(R))], per_seed=[rate(sel(R, tseed=s)) for s in SEEDS], pick=rate(R, "pick"), a_close=mean(R, "a_close", True), a_close_sd=sdv([r for r in R if r["replay_ok"]], "a_close"),
                                                                    slip=mean(R, "slip", True), final_err_cm=mean(R, "final_err_m") * 100, palm_contact=rate([r for r in R if r["palm_contact"] is not None], "palm_contact"), cluster=int(sum(r["cluster"] for r in R)), finger_a_hold=mean(R, "finger_a_hold", True))
res["per_x"] = PX
# ---- per checkpoint x stratum
PC = {}
for a in ARMS:
    for s in SEEDS:
        ag = f"{a}_s{s}"; PC[ag] = {}
        for k in list(STR) + ["ALL"]:
            R = sel(POL, agent=ag) if k == "ALL" else sel(POL, agent=ag, stratum=k)
            PC[ag][k] = dict(n=len(R), success=rate(R), pick=rate(R, "pick"), a_close=mean(R, "a_close", True), slip=mean(R, "slip", True), final_err_cm=mean(R, "final_err_m") * 100, palm_contact=rate([r for r in R if r["palm_contact"] is not None], "palm_contact"), cluster=int(sum(r["cluster"] for r in R)))
res["per_checkpoint"] = PC
# ---- aggregate by stratum with seed-level paired S1-C0
AG = {}
for k, xs in list(STR.items()) + [("LOW_ALL", STR["LOW_FAR_OOD"] + STR["LOW_NEAR_EDGE"])]:
    f = lambda x, xs=xs: any(abs(x - w) < 1e-6 for w in xs); AG[k] = {key: paired_summary(f, key) for key in ("success", "a_close", "slip", "final_err_m")}
    AG[k]["expert"] = dict(success=rate([r for r in EXP if f(r["x"])]), a_close=mean([r for r in EXP if f(r["x"])], "a_close"))
    for arm in ARMS: AG[k][arm] = dict(success=rate([r for r in POL if r["arm"] == arm and f(r["x"])]), a_close=mean([r for r in POL if r["arm"] == arm and f(r["x"]) ], "a_close", True), slip=mean([r for r in POL if r["arm"] == arm and f(r["x"])], "slip", True),
                                       palm_contact=rate([r for r in POL if r["arm"] == arm and f(r["x"]) and r["palm_contact"] is not None], "palm_contact"), cluster=int(sum(r["cluster"] for r in POL if r["arm"] == arm and f(r["x"]))))
res["aggregate_by_stratum"] = AG
# ---- S1-C0 by x: config-paired differences with within-seed block bootstrap (descriptive)
rng = np.random.default_rng(2026); DX = {}
for x in XS:
    d = []
    for s in SEEDS:
        for b in range(4):
            c = sel(POL, arm="c0", tseed=s, x=x, block=b); q = sel(POL, arm="s1", tseed=s, x=x, block=b)
            if c and q: d.append(float(q[0]["success"]) - float(c[0]["success"]))
    d = np.array(d); bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(4000)] if len(d) else []
    DX[f"{x:.2f}"] = dict(n_pairs=len(d), mean_pp=float(d.mean() * 100) if len(d) else None, boot_ci95_pp=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(d) else None,
                         seed_level_pp=[paired_summary(lambda xx, x=x: abs(xx - x) < 1e-6, "success")["diff"]] if False else [float(np.mean([r["success"] for r in sel(POL, arm="s1", tseed=s, x=x)]) - np.mean([r["success"] for r in sel(POL, arm="c0", tseed=s, x=x)])) * 100 for s in SEEDS])
res["s1_minus_c0_by_x"] = DX
# ---- hypotheses (rules identical to the registration)
H = {}
HA = AG["LOW_NEAR_EDGE"]["a_close"]; H["HA"] = dict(per_seed_diff_cm=HA["diff"], mean_cm=HA["mean"], n_negative=HA["n_negative"], verdict="SUPPORTED" if (HA["n_negative"] >= 2 and HA["mean"] <= -0.5) else "NOT SUPPORTED")
HB = AG["LOW_ALL"]["success"]; H["HB"] = dict(per_seed_diff_pp=HB["diff"], mean_pp=HB["mean"], n_negative=HB["n_negative"], ci95=HB["ci95"], verdict="SUPPORTED" if (HB["n_negative"] >= 2 and HB["mean"] <= -10) else "NOT SUPPORTED")
Dlow, Din = {}, {}
for arm in ARMS:
    Dlow[arm], Din[arm] = [], []
    for s in SEEDS:
        for k, store in (("LOW_NEAR_EDGE", Dlow), ("IN_SUPPORT", Din)):
            xs = STR[k]; pol = mean([r for r in POL if r["arm"] == arm and r["tseed"] == s and r["x"] in xs], "a_close", True); ex = mean([r for r in EXP if r["x"] in xs], "a_close"); store[arm].append(pol - ex)
H["HC"] = {arm: dict(D_low_per_seed=Dlow[arm], D_in_per_seed=Din[arm], n_seeds_low_below_in=int(sum(a < b for a, b in zip(Dlow[arm], Din[arm]))), verdict="SUPPORTED" if sum(a < b for a, b in zip(Dlow[arm], Din[arm])) >= (3 if arm == "s1" else 2) else "NOT SUPPORTED") for arm in ARMS}
fl = [r for r in POL if r["arm"] == "s1" and r["post_pick_fail"] and r["x"] <= 0.37 + 1e-6]
H["HD"] = dict(n_s1_postpick_fail_x_le_0p37=len(fl), cluster_frac=float(np.mean([r["cluster"] for r in fl])) if fl else None, palm_frac=float(np.mean([bool(r["palm_contact"]) for r in fl])) if fl else None,
               verdict=("SUPPORTED" if fl and np.mean([r["cluster"] for r in fl]) >= .5 and np.mean([bool(r["palm_contact"]) for r in fl]) >= .5 else "NOT SUPPORTED"))
HE = AG["IN_SUPPORT"]["success"]; H["HE"] = dict(per_seed_diff_pp=HE["diff"], mean_pp=HE["mean"], ci95=HE["ci95"], verdict=("NON-INFERIOR" if HE["ci95"][0] >= -10 else ("COST EVIDENT" if HE["ci95"][1] < 0 else "INCONCLUSIVE")))
H["HF_exploratory_high_ood"] = dict(success=AG["HIGH_OOD"]["success"], a_close=AG["HIGH_OOD"]["a_close"], expert_success=AG["HIGH_OOD"]["expert"]["success"])
res["hypotheses_preregistered"] = H
# ---- failure mechanism summary at low x (exploratory)
FM = {}
for arm in ARMS:
    for k in ("LOW_FAR_OOD", "LOW_NEAR_EDGE", "IN_SUPPORT", "HIGH_OOD"):
        R = [r for r in POL if r["arm"] == arm and r["stratum"] == k]; F = [r for r in R if not r["success"]]
        FM[f"{arm}/{k}"] = dict(n=len(R), fail=len(F), pick_fail=sum(not r["pick"] for r in F), post_pick_fail=sum(r["post_pick_fail"] for r in F), cluster=sum(r["cluster"] for r in F), palm_contact_in_fail=sum(bool(r["palm_contact"]) for r in F),
                                outcomes={o: sum(r["outcome"] == o for r in F) for o in sorted({r["outcome"] for r in F})})
res["failure_mechanism_exploratory"] = FM
(OUT / "xsweep_results.json").write_text(json.dumps(res, indent=1, default=float))
# ---- plots
xs = np.array(XS); COL = {"c0": "C0", "s1": "C1"}
def shade(ax):
    ax.axvspan(0.38, 0.45, color="gray", alpha=.12); ax.set_xlabel("box x [m] (grey = training region)")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for arm in ARMS:
    m = [PX[f"{x:.2f}"][arm]["success"] for x in XS]; lo = [PX[f"{x:.2f}"][arm]["wilson"][0] for x in XS]; hi = [PX[f"{x:.2f}"][arm]["wilson"][1] for x in XS]
    ax[0].plot(xs, m, "-o", color=COL[arm], label=f"{arm.upper()} (12 ep/x)"); ax[0].fill_between(xs, lo, hi, color=COL[arm], alpha=.12)
    for s in SEEDS: ax[1].plot(xs, [PX[f"{x:.2f}"][arm]["per_seed"][s] for x in XS], color=COL[arm], alpha=.5, lw=1, ls=["-", "--", ":"][s], label=f"{arm.upper()} seed {s}")
ax[0].plot(xs, [PX[f"{x:.2f}"]["expert"]["success"] for x in XS], "k-s", label="expert (4 ep/x)"); shade(ax[0]); shade(ax[1]); ax[0].set_ylabel("task success [%]"); ax[0].set_title("Success vs box x (Wilson 95 %, pooled over 3 seeds × 4 blocks)"); ax[1].set_title("per training seed (4 ep/x/seed)"); ax[0].legend(fontsize=7); ax[1].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(OUT / "X1_success_vs_x.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for arm in ARMS:
    ax[0].errorbar(xs, [PX[f"{x:.2f}"][arm]["a_close"] for x in XS], yerr=[PX[f"{x:.2f}"][arm]["a_close_sd"] for x in XS], color=COL[arm], marker="o", capsize=2, label=arm.upper(), alpha=.85)
    ax[1].plot(xs, [PX[f"{x:.2f}"][arm]["slip"] for x in XS], "-o", color=COL[arm], label=arm.upper()); ax[2].plot(xs, [PX[f"{x:.2f}"][arm]["final_err_cm"] for x in XS], "-o", color=COL[arm], label=arm.upper())
ax[0].plot(xs, [PX[f"{x:.2f}"]["expert"]["a_close"] for x in XS], "k-s", label="expert"); ax[0].set_ylabel("a at closing [cm] (box centre − TCP along fingers)"); ax[1].set_ylabel("slip a_release − a_close [cm]"); ax[2].set_ylabel("mean final box–tray error [cm]"); ax[2].axhline(3, color="k", ls="--", lw=1)
for x_ in ax: shade(x_)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "X2_offset_slip_error_vs_x.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
m = [DX[f"{x:.2f}"]["mean_pp"] for x in XS]; lo = [DX[f"{x:.2f}"]["boot_ci95_pp"][0] for x in XS]; hi = [DX[f"{x:.2f}"]["boot_ci95_pp"][1] for x in XS]
ax[0].errorbar(xs, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt="o-", capsize=2, color="C3"); ax[0].axhline(0, color="k", lw=.8); shade(ax[0]); ax[0].set_ylabel("S1 − C0 success [pp]"); ax[0].set_title("config-paired difference, block bootstrap 95 % (descriptive)")
for s in SEEDS: ax[1].plot(xs, [DX[f"{x:.2f}"]["seed_level_pp"][s] for x in XS], marker="o", label=f"seed {s}")
ax[1].axhline(0, color="k", lw=.8); shade(ax[1]); ax[1].set_title("S1 − C0 per training seed (4 ep/x)"); ax[1].legend(); plt.tight_layout(); plt.savefig(OUT / "X3_s1_minus_c0_vs_x.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for arm in ARMS:
    ax[0].plot(xs, [PX[f"{x:.2f}"][arm]["palm_contact"] for x in XS], "-o", color=COL[arm], label=arm.upper()); ax[1].plot(xs, [PX[f"{x:.2f}"][arm]["cluster"] for x in XS], "-o", color=COL[arm], label=arm.upper()); ax[2].plot(xs, [PX[f"{x:.2f}"][arm]["finger_a_hold"] for x in XS], "-o", color=COL[arm], label=arm.upper())
ax[0].set_ylabel("box–palm contact rate [%]"); ax[1].set_ylabel("cluster-flagged failures (of 12)"); ax[2].set_ylabel("finger contact position at hold, a [cm]")
for x_ in ax: shade(x_)
ax[0].legend(); plt.tight_layout(); plt.savefig(OUT / "X4_palm_cluster_vs_x.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
names = [f"{a}_s{s}" for a in ARMS for s in SEEDS]; Mx = np.array([[rate(sel(POL, agent=n, x=x)) for x in XS] for n in names])
im = ax[0].imshow(Mx, aspect="auto", vmin=0, vmax=100, cmap="viridis"); ax[0].set_xticks(range(len(XS))); ax[0].set_xticklabels([f"{x:.2f}" for x in XS], rotation=90, fontsize=7); ax[0].set_yticks(range(6)); ax[0].set_yticklabels(names); ax[0].set_title("success [%] per checkpoint × x (4 ep each)"); plt.colorbar(im, ax=ax[0])
Ma = np.array([[mean(sel(POL, agent=n, x=x), "a_close", True) for x in XS] for n in names]); im = ax[1].imshow(Ma, aspect="auto", cmap="coolwarm", vmin=-4, vmax=2); ax[1].set_xticks(range(len(XS))); ax[1].set_xticklabels([f"{x:.2f}" for x in XS], rotation=90, fontsize=7); ax[1].set_yticks(range(6)); ax[1].set_yticklabels(names); ax[1].set_title("a_close [cm] per checkpoint × x"); plt.colorbar(im, ax=ax[1])
plt.tight_layout(); plt.savefig(OUT / "X5_checkpoint_heatmaps.png", dpi=130); plt.close()
# ---- tables (markdown)
L = ["### Per x (policies: 12 episodes = 3 seeds × 4 blocks; expert 4)", "| x | expert succ | C0 succ [Wilson] | S1 succ [Wilson] | S1−C0 [pp, boot CI] | a_close exp/C0/S1 [cm] | slip C0/S1 | palm% C0/S1 | cluster C0/S1 |", "|---|---|---|---|---|---|---|---|---|"]
for x in XS:
    p = PX[f"{x:.2f}"]; d = DX[f"{x:.2f}"]; L.append(f"| {x:.2f} | {p['expert']['success']:.0f}% | {p['c0']['success']:.0f}% [{p['c0']['wilson'][0]:.0f},{p['c0']['wilson'][1]:.0f}] | {p['s1']['success']:.0f}% [{p['s1']['wilson'][0]:.0f},{p['s1']['wilson'][1]:.0f}] | {d['mean_pp']:+.0f} [{d['boot_ci95_pp'][0]:+.0f},{d['boot_ci95_pp'][1]:+.0f}] | {p['expert']['a_close']:+.2f} / {p['c0']['a_close']:+.2f} / {p['s1']['a_close']:+.2f} | {p['c0']['slip']:+.2f} / {p['s1']['slip']:+.2f} | {p['c0']['palm_contact']:.0f} / {p['s1']['palm_contact']:.0f} | {p['c0']['cluster']} / {p['s1']['cluster']} |")
L += ["", "### Stratum aggregates, seed-level paired S1 − C0 (n = 3 pairs)", "| stratum | metric | C0 per seed | S1 per seed | diff per seed | mean [95 % CI df=2] | n neg |", "|---|---|---|---|---|---|---|"]
for k, v in AG.items():
    for key in ("success", "a_close", "slip", "final_err_m"):
        q = v[key]; sc = 100 if key == "final_err_m" else 1; L.append(f"| {k} | {key} | {', '.join(f'{u*sc:.1f}' for u in q['per_seed_c0'])} | {', '.join(f'{u*sc:.1f}' for u in q['per_seed_s1'])} | {', '.join(f'{u*sc:+.1f}' for u in q['diff'])} | {q['mean']*sc:+.2f} [{q['ci95'][0]*sc:+.2f}, {q['ci95'][1]*sc:+.2f}] | {q['n_negative']} |")
L += ["", "### Per checkpoint × stratum (success % | a_close cm | palm % | cluster)", "| checkpoint | " + " | ".join(list(STR) + ["ALL"]) + " |", "|---|" + "---|" * (len(STR) + 1)]
for ag, v in PC.items(): L.append(f"| {ag} | " + " | ".join(f"{v[k]['success']:.0f}% / {v[k]['a_close']:+.2f} / {v[k]['palm_contact']:.0f} / {v[k]['cluster']}" for k in list(STR) + ["ALL"]) + " |")
L += ["", "### Pre-registered hypotheses", json.dumps(H, indent=1, default=float), "", "### Acceptance", json.dumps(res["AC"], indent=1, default=float)]
(OUT / "xsweep_tables.md").write_text("\n".join(L)); print("\n".join(L[:30])); print(json.dumps(H, indent=1, default=float))
