#!/usr/bin/env python3
"""Coverage experiment report (implements outputs/cov/cov_config.json; hypotheses C1-C8 and the overall rule are the registered ones; with deviation D1 every EXT/VOL branch has ONE training seed
=> every verdict below is labelled single-seed and nothing is called confirmed).  Baselines = frozen C0/S1 seeds 0,1,2 (outputs/xsweep).  Writes outputs/cov/cov_results.json, cov_tables.md, plots Y1..Y5.
Run (ACT venv, after eval/cov_replay.py): python eval/cov_report.py"""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, hashlib, numpy as np
from pathlib import Path
from scipy import stats as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path: sys.path.insert(0, p)
from eval.metrics import task_labels
EXPECT = "f01f4733023b09da42409b1acf66c114f1890d3bf27ead8dc7860ebc8763aef3"; CFGP = ROOT / "outputs/cov/cov_config.json"
if hashlib.sha256(CFGP.read_bytes()).hexdigest() != EXPECT: sys.exit("coverage config modified after registration")
XCFG = json.loads((ROOT / "outputs/xsweep/xsweep_config.json").read_text()); XS = XCFG["x_grid_m"]; POSES = {p["idx"]: p for p in XCFG["poses"]}; TRAY = np.array([0.44, 0.30]); TRAY_TOP = 0.364
LOW = [0.34, 0.35, 0.36, 0.37]; INS = [round(0.38 + 0.01 * i, 2) for i in range(8)]; HIGH = [0.46, 0.47, 0.48]
STR = {"LOW_FAR": [0.34, 0.35], "LOW_NEAR": [0.36, 0.37], "LOW(0.34-0.37)": LOW, "IN_SUPPORT_ORIG(0.38-0.45)": INS, "HIGH(0.46-0.48)": HIGH}
OUT = ROOT / "outputs/cov"
def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def events(R):
    g = R["gripper_cmd"]; box = R["box_pos"]; n = len(g); cl = np.nonzero(g < 0.028)[0]; tc = int(cl[0]) if len(cl) else None
    rise = box[:, 2] - box[0, 2]; tl = int(np.argmax(rise > 0.03)) if (rise > 0.03).any() else None; tr = None
    if tc is not None:
        r = np.nonzero(g[tc:] > 0.035)[0]; tr = tc + int(r[0]) if len(r) else None
    e = dict(close=tc, lift=tl, release=tr)
    if tc is not None: e["hold"] = min(tc + 14, n - 1)
    return e
def load_run(label, arm, tseed, evdir, repdir, repmeta, live=True):
    ERR = {(m["agent"], m["idx"]): m["err"] for m in repmeta["episodes"]}; agent = f"{label}"; LIVE = {e["episode_index"]: e for e in jl(evdir / "episodes.jsonl")} if live else {}
    rows = []
    for idx, pz in POSES.items():
        f = repdir / f"{agent}_SW_{idx:03d}.npz"
        if not f.exists(): continue
        R = dict(np.load(f)); lab = task_labels(R["box_pos"], R["tcp_pos"], R["box_speed"], TRAY, TRAY_TOP); ev = events(R); fin = R["box_pos"][-1, :2] - TRAY
        L = LIVE.get(idx); row = dict(run=label, arm=arm, tseed=tseed, idx=idx, x=pz["x"], block=pz["block"], err=ERR.get((agent, idx)))
        if L is not None: row.update(success=bool(L["task_success"]), pick=bool(L["pick_success"]), final_err_m=L["final_box_to_tray_xy_m"], lat_q=L.get("latency_ms_queries"), agree=bool(lab["task_success"]) == bool(L["task_success"]))
        else: row.update(success=bool(lab["task_success"]), pick=bool(lab["pick_success"]), final_err_m=lab["final_box_to_tray_xy_m"], lat_q=None, agree=True)
        row["ok"] = (row["err"] is None or row["err"] <= 0.005) and row["agree"]; tc, tr = ev["close"], ev["release"]
        at = lambda t, i: float(R["box_in_tcp"][t, i] * 100) if t is not None else np.nan
        row.update(a_close=at(tc, 2), slip=(at(tr, 2) - at(tc, 2)) if (tc is not None and tr is not None) else np.nan, palm=bool((R["hand_force"][tc:(tr or len(R["hand_force"]))] > 0.01).any()) if tc is not None else None)
        post = row["pick"] and not row["success"]; row["post_pick_fail"] = bool(post); row["cluster"] = bool(post and fin[0] < -0.025 and abs(fin[1]) < 0.01); rows.append(row)
    return rows
RUNS = {}; META = {}
xm = json.loads((ROOT / "outputs/xsweep/replay_meta.json").read_text())
for arm in ("c0", "s1"):
    for s in (0, 1, 2): RUNS[f"BASE-{arm.upper()}"] = RUNS.get(f"BASE-{arm.upper()}", []) + [load_run(f"{arm}_s{s}", arm, s, ROOT / f"outputs/xsweep/eval/{arm}_seed{s}/act/sweep", ROOT / "outputs/xsweep/replay", xm)]
EXPR = []
for idx, pz in POSES.items():
    R = dict(np.load(ROOT / f"outputs/xsweep/replay/expert_SW_{idx:03d}.npz")); ev = events(R); lab = task_labels(R["box_pos"], R["tcp_pos"], R["box_speed"], TRAY, TRAY_TOP)
    EXPR.append(dict(idx=idx, x=pz["x"], block=pz["block"], success=bool(lab["task_success"]), a_close=float(R["box_in_tcp"][ev["close"], 2] * 100)))
EXPA = {r["idx"]: r["a_close"] for r in EXPR}
cm = json.loads((OUT / "replay_meta.json").read_text()) if (OUT / "replay_meta.json").exists() else dict(episodes=[], jobs=[])
for br, k in (("ext_c0", 3), ("ext_s1", 3), ("vol_c0", 3), ("vol_s1", 3)):
    ev_ = OUT / f"eval/{br}_seed{k}/sweep/act/sweep"
    if (OUT / f"eval/{br}_seed{k}.done").exists() and (ev_ / "episodes.jsonl").exists() and (OUT / f"replay/{br}_s{k}_SW_000.npz").exists():
        RUNS[br.upper().replace("_", "-")] = [load_run(f"{br}_s{k}", br.split("_")[1], k, ev_, OUT / "replay", cm)]
GROUPS = [g for g in ("BASE-C0", "BASE-S1", "EXT-C0", "EXT-S1", "VOL-C0", "VOL-S1") if g in RUNS]
def rate(rows): return 100 * float(np.mean([r["success"] for r in rows])) if rows else float("nan")
def sel(rows, xs=None, **kw): return [r for r in rows if (xs is None or any(abs(r["x"] - w) < 1e-6 for w in xs)) and all(r[k] == v for k, v in kw.items())]
def per_seed(g, xs, f=rate): return [f(sel(run, xs)) for run in RUNS[g]]
def absdev(rows): v = [abs(r["a_close"] - EXPA[r["idx"]]) for r in rows if r["ok"] and r["a_close"] == r["a_close"]]; return float(np.mean(v)) if v else float("nan")
def signdev(rows): v = [r["a_close"] - EXPA[r["idx"]] for r in rows if r["ok"] and r["a_close"] == r["a_close"]]; return float(np.mean(v)) if v else float("nan")
res = dict(config_sha256=EXPECT, groups=GROUPS, deviation="D1: EXT/VOL have a single training seed (seed 3); all verdicts are single-seed and exploratory in the sense of the registration")
# ---- tables per group x stratum
T = {}
for g in GROUPS:
    T[g] = {}
    for sname, xs in STR.items():
        ps = per_seed(g, xs); T[g][sname] = dict(success_per_seed=ps, success_mean=float(np.mean(ps)), success_range=[float(min(ps)), float(max(ps))],
                                                   a_close_dev_abs_cm=[absdev(sel(run, xs)) for run in RUNS[g]], a_close_dev_signed_cm=[signdev(sel(run, xs)) for run in RUNS[g]],
                                                   slip_cm=[float(np.nanmean([r["slip"] for r in sel(run, xs) if r["ok"]])) for run in RUNS[g]], final_err_cm=[100 * float(np.mean([r["final_err_m"] for r in sel(run, xs)])) for run in RUNS[g]],
                                                   palm_pct=[100 * float(np.mean([bool(r["palm"]) for r in sel(run, xs) if r["palm"] is not None])) for run in RUNS[g]], cluster=[int(sum(r["cluster"] for r in sel(run, xs))) for run in RUNS[g]],
                                                   pick_pct=[100 * float(np.mean([r["pick"] for r in sel(run, xs)])) for run in RUNS[g]])
res["table"] = T
res["expert"] = {s: dict(success=100 * float(np.mean([r["success"] for r in EXPR if any(abs(r["x"] - w) < 1e-6 for w in xs)])), a_close=float(np.mean([r["a_close"] for r in EXPR if any(abs(r["x"] - w) < 1e-6 for w in xs)]))) for s, xs in STR.items()}
res["per_x"] = {g: {f"{x:.2f}": dict(success=[rate(sel(run, [x])) for run in RUNS[g]], a_close=[float(np.nanmean([r["a_close"] for r in sel(run, [x]) if r["ok"]] or [np.nan])) for run in RUNS[g]], palm=[100 * float(np.mean([bool(r["palm"]) for r in sel(run, [x]) if r["palm"] is not None] or [np.nan])) for run in RUNS[g]], cluster=[int(sum(r["cluster"] for r in sel(run, [x]))) for run in RUNS[g]]) for x in XS} for g in GROUPS}
# per-block (uncertainty across pose blocks) for LOW and IN strata + block bootstrap for single-seed branches
rng = np.random.default_rng(2026); BLK = {}
for g in GROUPS:
    BLK[g] = {}
    for sname in ("LOW(0.34-0.37)", "IN_SUPPORT_ORIG(0.38-0.45)", "HIGH(0.46-0.48)"):
        xs = STR[sname]; bs = [[rate(sel(run, xs, block=b)) for b in range(4)] for run in RUNS[g]]
        boot = []
        for _ in range(4000):
            bb = rng.integers(0, 4, 4); boot.append(np.mean([np.mean([bs[i][b] for b in bb]) for i in range(len(bs))]))
        BLK[g][sname] = dict(per_block_per_seed=bs, block_bootstrap_ci95_of_mean=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
res["across_blocks"] = BLK
# ---- shortcut sensitivity
e0 = json.loads((ROOT / "outputs/e0/e0_results.json").read_text())["B_across_scenarios"]["shortcut"]; SI = {}
for g in GROUPS:
    if g.startswith("BASE"):
        arm = g.split("-")[1].lower(); SI[g] = [dict(seed=s, right_mean=e0[arm][str(s)]["right_mean"]["SI"], jit_small=e0[arm][str(s)]["jit_small"]["SI"], unperturbed=e0[arm][str(s)]["right_mean"]["unperturbed"]) for s in (0, 1, 2)]
    else:
        br = g.lower().replace("-", "_"); d = OUT / f"eval/{br}_seed3"; sm = {}
        for tag in ("unpert20", "pert_right_mean", "pert_jit_small"):
            f = d / tag / "act/nominal/episodes.jsonl"; sm[tag] = sum(json.loads(l)["task_success"] for l in f.read_text().splitlines() if l.strip()) if f.exists() else None
        SI[g] = [dict(seed=3, unperturbed=sm["unpert20"], right_mean_succ=sm["pert_right_mean"], jit_small_succ=sm["pert_jit_small"], right_mean=(sm["pert_right_mean"] / sm["unpert20"]) if sm["unpert20"] else None, jit_small=(sm["pert_jit_small"] / sm["unpert20"]) if sm["unpert20"] else None)]
res["shortcut_SI"] = SI
# ---- training health, open loop, latency
def health(name_path, ol_path):
    L = jl(name_path / "train_log.jsonl"); v = [r["val_l1"] for r in L if "val_l1" in r]; d = dict(steps=len(L), final_loss_ema=L[-1]["loss_ema"], s_per_step=L[-1]["elapsed_s"] / len(L), val_l1_last=v[-1] if v else None, val_l1_min=min(v) if v else None, nan=bool(any(not np.isfinite(r["loss"]) for r in L)))
    if ol_path.exists(): o = json.loads(ol_path.read_text()); d.update(ol_mae_arm_rad=o["mae_arm_rad"], ol_p95=o["p95_arm_rad"], ol_mae_left=o["mae_left_arm_rad"])
    return d
H = {}
for arm in ("c0", "s1"):
    for s in (0, 1, 2):
        ck = ROOT / f"outputs/checkpoints/{'e6b' if s == 0 else 'e0'}_{arm}_seed{s}"; ol = ROOT / (f"outputs/eval/e6b/{arm}/open_loop_val.json" if s == 0 else f"outputs/eval/e0/{arm}_seed{s}/open_loop_val.json"); H[f"base_{arm}_s{s}"] = health(ck, ol)
for g in GROUPS:
    if not g.startswith("BASE"): br = g.lower().replace("-", "_"); H[f"{br}_s3"] = health(ROOT / f"outputs/checkpoints/cov_{br}_seed3", OUT / f"eval/{br}_seed3/open_loop_val.json")
res["training_health_and_open_loop"] = H
LAT = {}
for g in GROUPS:
    q = [x for run in RUNS[g] for r in run if r["lat_q"] for x in r["lat_q"]]; LAT[g] = dict(median_query_ms=float(np.median(q)), p95_query_ms=float(np.percentile(q, 95)), n=len(q))
res["latency_per_query_ms"] = LAT
# ---- hypotheses (registered rules; single-seed labelling)
def bm(g, s): return float(np.mean(per_seed(g, STR[s]))) if g in RUNS else float("nan")
Hh = {}; arms = ("C0", "S1")
if all(f"EXT-{a}" in RUNS for a in arms):
    c1 = {}
    for a in arms:
        base = bm(f"BASE-{a}", "LOW(0.34-0.37)"); ext = per_seed(f"EXT-{a}", LOW); c1[a] = dict(baseline_mean=base, ext=ext, ok=bool(np.mean(ext) >= base + 20 and np.mean(ext) >= 75 and all(e >= base + 10 for e in ext)))
    Hh["C1"] = dict(detail=c1, verdict=("SUPPORTED (single seed)" if all(v["ok"] for v in c1.values()) else "NOT SUPPORTED (single seed)"))
    c3 = {}
    for a in arms:
        base_abs = float(np.mean([absdev(sel(run, LOW)) for run in RUNS[f"BASE-{a}"]])); ext_abs = [absdev(sel(run, LOW)) for run in RUNS[f"EXT-{a}"]]; c3[a] = dict(baseline_abs_dev_cm=base_abs, ext_abs_dev_cm=ext_abs, ok=bool(all(e <= 1.0 for e in ext_abs) and base_abs - np.mean(ext_abs) >= 1.5))
    Hh["C3"] = dict(detail=c3, verdict=("SUPPORTED (single seed)" if all(v["ok"] for v in c3.values()) else "NOT SUPPORTED (single seed)"))
    d4 = per_seed("EXT-S1", LOW)[0] - per_seed("EXT-C0", LOW)[0]; d4i = per_seed("EXT-S1", INS)[0] - per_seed("EXT-C0", INS)[0]
    Hh["C4"] = dict(low_S1_minus_C0_pp=d4, in_support_S1_minus_C0_pp=d4i, baseline_low_S1_minus_C0_pp=bm("BASE-S1", "LOW(0.34-0.37)") - bm("BASE-C0", "LOW(0.34-0.37)"), n_pairs=1, verdict=("S1 boundary deficit within -10 pp after coverage (single pair, descriptive)" if d4 >= -10 else "S1 boundary deficit persists (single pair, descriptive)"))
    Hh["C5"] = {a: dict(baseline=bm(f"BASE-{a}", "IN_SUPPORT_ORIG(0.38-0.45)"), ext=per_seed(f"EXT-{a}", INS)[0], ok=bool(per_seed(f"EXT-{a}", INS)[0] >= bm(f"BASE-{a}", "IN_SUPPORT_ORIG(0.38-0.45)") - 10)) for a in arms}
    Hh["C7"] = {a: dict(baseline=bm(f"BASE-{a}", "HIGH(0.46-0.48)"), ext=per_seed(f"EXT-{a}", HIGH)[0], ok=bool(abs(per_seed(f"EXT-{a}", HIGH)[0] - bm(f"BASE-{a}", "HIGH(0.46-0.48)")) <= 15)) for a in arms}
    Hh["C6"] = {a: dict(SI=SI[f"EXT-{a}"], ok=bool(SI[f"EXT-{a}"][0]["right_mean"] is not None and ((a == "S1" and SI[f"EXT-{a}"][0]["right_mean"] >= .9 and SI[f"EXT-{a}"][0]["jit_small"] >= .9) or (a == "C0" and SI[f"EXT-{a}"][0]["right_mean"] <= .25 and SI[f"EXT-{a}"][0]["jit_small"] <= .25)))) for a in arms}
    Hh["C8"] = {g: dict(median_ms=LAT[g]["median_query_ms"], baseline_ms=LAT[f"BASE-{g.split('-')[1]}"]["median_query_ms"], ratio=LAT[g]["median_query_ms"] / LAT[f"BASE-{g.split('-')[1]}"]["median_query_ms"]) for g in GROUPS if not g.startswith("BASE")}
if all(f"VOL-{a}" in RUNS for a in arms) and "C1" in Hh:
    c2 = {a: dict(baseline=bm(f"BASE-{a}", "LOW(0.34-0.37)"), vol=per_seed(f"VOL-{a}", LOW)[0], ext=per_seed(f"EXT-{a}", LOW)[0]) for a in arms}
    for a in arms: c2[a]["ok"] = bool(c2[a]["vol"] <= c2[a]["baseline"] + 10); c2[a]["volume_confound"] = bool(c2[a]["vol"] >= c2[a]["baseline"] + 20)
    Hh["C2"] = dict(detail=c2, verdict=("COVERAGE-SPECIFIC (single seed, exploratory)" if all(v["ok"] for v in c2.values()) else ("VOLUME CONFOUND evident in at least one arm (single seed)" if any(v["volume_confound"] for v in c2.values()) else "INTERMEDIATE: VOL between +10 and +20 pp in at least one arm")))
    c1v, c2v, c3v = Hh["C1"]["verdict"].startswith("SUPPORTED"), all(v["ok"] for v in c2.values()), Hh["C3"]["verdict"].startswith("SUPPORTED")
    Hh["overall"] = ("COVERAGE EXPLAINS the boundary grasp failure (single-seed evidence)" if (c1v and c2v and c3v) else ("PARTIAL: C1 holds but C2 or C3 fails (single-seed evidence)" if c1v else "NOT EXPLAINED by coverage (C1 fails, single-seed evidence)"))
res["hypotheses"] = Hh
(OUT / "cov_results.json").write_text(json.dumps(res, indent=1, default=float))
# ---- plots
COL = {"BAS": "gray", "EXT": "C3", "VOL": "C2"}; xs = np.array(XS)
fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
for j, a in enumerate(("C0", "S1")):
    for g in GROUPS:
        if not g.endswith(a): continue
        pk = g[:3]; Y = np.array([res["per_x"][g][f"{x:.2f}"]["success"] for x in XS])      # (x, seeds)
        Ya = np.array([res["per_x"][g][f"{x:.2f}"]["a_close"] for x in XS])
        if pk == "BAS": ax[0, j].plot(xs, np.nanmean(Y, 1), color="gray", lw=2, label=f"baseline mean (3 seeds)"); ax[0, j].fill_between(xs, np.nanmin(Y, 1), np.nanmax(Y, 1), color="gray", alpha=.25); ax[1, j].plot(xs, np.nanmean(Ya, 1), color="gray", lw=2); ax[1, j].fill_between(xs, np.nanmin(Ya, 1), np.nanmax(Ya, 1), color="gray", alpha=.25)
        else: ax[0, j].plot(xs, Y[:, 0], "-o", color=COL[pk], label=f"{pk} seed 3"); ax[1, j].plot(xs, Ya[:, 0], "-o", color=COL[pk])
    ax[0, j].plot(xs, [100 * np.mean([r["success"] for r in EXPR if abs(r["x"] - x) < 1e-6]) for x in XS], "k-s", ms=4, label="expert"); ax[1, j].plot(xs, [np.mean([r["a_close"] for r in EXPR if abs(r["x"] - x) < 1e-6]) for x in XS], "k-s", ms=4)
    for r_ in range(2): ax[r_, j].axvspan(0.38, 0.45, color="gray", alpha=.1); ax[r_, j].axvspan(0.34, 0.38, color="C3", alpha=.06)
    ax[0, j].set_title(f"{a}: success vs box x (grey = original support, red tint = added support)"); ax[0, j].set_ylabel("success [%] (4 poses per point)"); ax[1, j].set_ylabel("a_close [cm]"); ax[1, j].set_xlabel("box x [m]"); ax[0, j].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "Y1_success_a_close_vs_x.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for g in GROUPS:
    pk = g[:3]; arm = g[-2:]
    for k, key in enumerate(("palm", "cluster")):
        Y = np.array([res["per_x"][g][f"{x:.2f}"][key] for x in XS], float); ax[k].plot(xs, np.nanmean(Y, 1), color=COL[pk], ls="-" if arm == "C0" else "--", marker="o" if pk != "BAS" else None, label=g)
    ax[2].bar(GROUPS.index(g), T[g]["LOW(0.34-0.37)"]["success_mean"], color=COL[pk], alpha=.8 if arm == "C0" else .5, yerr=[[T[g]["LOW(0.34-0.37)"]["success_mean"] - T[g]["LOW(0.34-0.37)"]["success_range"][0]], [T[g]["LOW(0.34-0.37)"]["success_range"][1] - T[g]["LOW(0.34-0.37)"]["success_mean"]]], capsize=3)
ax[0].set_ylabel("box–palm contact [%]"); ax[1].set_ylabel("cluster-flagged failures (of 4)"); ax[2].set_xticks(range(len(GROUPS))); ax[2].set_xticklabels(GROUPS, rotation=45, fontsize=7); ax[2].set_ylabel("success x in 0.34–0.37 [%] (bars: mean, whiskers: seed range)"); ax[0].legend(fontsize=6)
for k in (0, 1): ax[k].set_xlabel("box x [m]")
plt.tight_layout(); plt.savefig(OUT / "Y2_palm_cluster_lowstratum.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); w = 0.8 / max(1, len(GROUPS))
for i, g in enumerate(GROUPS):
    ax[0].bar(np.arange(len(STR)) + i * w, [T[g][s]["success_mean"] for s in STR], w, color=COL[g[:3]], alpha=.9 if g.endswith("C0") else .5, label=g)
    ax[1].bar(np.arange(len(STR)) + i * w, [np.nanmean(T[g][s]["a_close_dev_signed_cm"]) for s in STR], w, color=COL[g[:3]], alpha=.9 if g.endswith("C0") else .5)
for k, (t_, yl) in enumerate((("success by stratum", "success [%]"), ("grasp-depth error vs expert (a_close − expert), signed", "[cm]"))): ax[k].set_xticks(np.arange(len(STR)) + w * len(GROUPS) / 2); ax[k].set_xticklabels(list(STR), rotation=20, fontsize=7); ax[k].set_title(t_); ax[k].set_ylabel(yl)
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(OUT / "Y3_strata_summary.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, key in enumerate(("right_mean", "jit_small")):
    for i, g in enumerate(GROUPS):
        vals = [s[key] for s in SI[g] if s[key] is not None]; ax[k].scatter([i] * len(vals), vals, color=COL[g[:3]], s=40)
    ax[k].axhline(0.9, color="k", ls=":"); ax[k].axhline(0.25, color="k", ls=":"); ax[k].set_xticks(range(len(GROUPS))); ax[k].set_xticklabels(GROUPS, rotation=45, fontsize=7); ax[k].set_title(f"shortcut index, {key}")
plt.tight_layout(); plt.savefig(OUT / "Y4_shortcut_SI.png", dpi=130); plt.close()
# ---- markdown tables
L = ["### Success / grasp-depth / contact by stratum (per training seed in brackets)", "| group | stratum | success % | pick % | a_close − expert, signed [cm] | |a_close − expert| [cm] | slip [cm] | final err [cm] | palm % | cluster |", "|---|---|---|---|---|---|---|---|---|---|"]
fm = lambda v, d=1: "[" + ", ".join("nan" if x != x else f"{x:.{d}f}" for x in v) + "]"
for g in GROUPS:
    for s in STR: t = T[g][s]; L.append(f"| {g} | {s} | {fm(t['success_per_seed'], 0)} | {fm(t['pick_pct'], 0)} | {fm(t['a_close_dev_signed_cm'], 2)} | {fm(t['a_close_dev_abs_cm'], 2)} | {fm(t['slip_cm'], 2)} | {fm(t['final_err_cm'], 2)} | {fm(t['palm_pct'], 0)} | {t['cluster']} |")
L += ["", "### Shortcut index (success perturbed / unperturbed on the same 20 configurations)", "| group | seed | unperturbed | right_mean SI | jitter_small SI |", "|---|---|---|---|---|"]
for g in GROUPS:
    for s in SI[g]: L.append(f"| {g} | {s['seed']} | {s.get('unperturbed')} | {s['right_mean'] if s['right_mean'] is None else round(s['right_mean'], 2)} | {s['jit_small'] if s['jit_small'] is None else round(s['jit_small'], 2)} |")
L += ["", "### Training health / open-loop (6 original val episodes) / latency", "| run | steps | final loss EMA | val L1 last | s/step | open-loop MAE arm [rad] |", "|---|---|---|---|---|---|"]
for k, v in H.items(): L.append(f"| {k} | {v['steps']} | {v['final_loss_ema']:.4f} | {v['val_l1_last']} | {v['s_per_step']:.2f} | {v.get('ol_mae_arm_rad', float('nan')):.5f} |")
L += ["", "latency per query [ms] (cpu, 1 thread): " + ", ".join(f"{g} {v['median_query_ms']:.0f} (p95 {v['p95_query_ms']:.0f})" for g, v in LAT.items()), "", "### Registered hypotheses (single-seed labelling, deviation D1)", json.dumps(Hh, indent=1, default=float)]
(OUT / "cov_tables.md").write_text("\n".join(L)); print("\n".join(L))
