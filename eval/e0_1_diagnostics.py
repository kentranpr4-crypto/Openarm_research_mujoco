#!/usr/bin/env python3
"""E0.1 diagnostics (analysis only; no training, no new rollouts, dev seeds 21-22 data from E0 / Stage 1).
Part 1 statistical precision (paired-by-seed differences, Welch stability, hierarchical bootstrap, sample-size/CI half-width).
Part 2 failure diagnosis (post-pick 'release' failures: error decomposition aim / in-hand offset / post-release drift, timing, gripper, velocity).
Part 3 thread-count deviation D1 (what is and is not identifiable).
Exploratory: thresholds below are geometry/definition-derived and were NOT tuned on these results. Writes outputs/e0_1/.
Run (ACT venv): python eval/e0_1_diagnostics.py"""
import sys, json, itertools, numpy as np
from pathlib import Path
from scipy import stats as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/e0_1"; OUT.mkdir(parents=True, exist_ok=True)
TRAY = np.array([0.44, 0.30]); PLACE_XY_TOL = 0.03; TRAY_INNER = 0.14; PLACE_Z = 0.36 + 0.004 + 0.02 + 0.025 + 0.003
SETS = {"NOM": ("nom21", "nominal"), "B21": ("bound21", "unseen_pose"), "B22": ("bound22", "unseen_pose")}
ARMS, SEEDS = ("c0", "s1"), (0, 1, 2)
rng = np.random.default_rng(12345)

def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def base(arm, seed): return (ROOT / "outputs/eval/e6b" / arm) if seed == 0 else (ROOT / "outputs/eval/e0" / f"{arm}_seed{seed}")
def load(arm, seed, S):
    tag, cond = SETS[S]; d = base(arm, seed) / tag / "act" / cond; out = []
    for e in jl(d / "episodes.jsonl"):
        f = d / "rollouts" / f"ep{e['episode_index']:04d}.npz"
        with np.load(f) as z: e = dict(e, **{k: z[k] for k in z.files})
        out.append(dict(e, S=S, arm=arm, tseed=seed))
    return out
DATA = {(a, s): sum((load(a, s, S) for S in SETS), []) for a in ARMS for s in SEEDS}

# ------------------------------------------------------------------ Part 1: statistical precision
def prop(arm, seed, sets):
    E = [e for e in DATA[(arm, seed)] if e["S"] in sets]; return np.array([e["task_success"] for e in E], float)
SET_DEF = {"NOM": ("NOM",), "BOUND": ("B21", "B22"), "ALL_DEV": ("NOM", "B21", "B22")}     # ALL_DEV is exploratory (not a pre-registered endpoint)
P = {(S, a): np.array([prop(a, s, SET_DEF[S]).mean() * 100 for s in SEEDS]) for S in SET_DEF for a in ARMS}
N = {S: sum(len([e for e in DATA[("c0", 0)] if e["S"] == x]) for x in SET_DEF[S]) for S in SET_DEF}

def welch(x, y):
    nx, ny = len(x), len(y); vx, vy = x.var(ddof=1), y.var(ddof=1); se = np.sqrt(vx / nx + vy / ny); d = y.mean() - x.mean()
    if se == 0: return d, (d, d), np.inf
    df = se**4 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)); t = st.t.ppf(0.975, df); return d, (d - t * se, d + t * se), df
def paired_t(x, y):
    d = y - x; n = len(d); se = d.std(ddof=1) / np.sqrt(n); t = st.t.ppf(0.975, n - 1); return d.mean(), (d.mean() - t * se, d.mean() + t * se), d
def signflip_p(d):
    obs = abs(d.mean()); vals = [abs(np.mean(np.array(s) * d)) for s in itertools.product([-1, 1], repeat=len(d))]; return float(np.mean(np.array(vals) >= obs - 1e-12))
def hier_boot(S, B=20000, seeds=SEEDS):
    out = []
    for _ in range(B):
        m = {}
        for a in ARMS:
            ss = rng.choice(seeds, len(seeds)); v = []
            for s in ss:
                x = prop(a, s, SET_DEF[S]); v.append(x[rng.integers(0, len(x), len(x))].mean() * 100)
            m[a] = np.mean(v)
        out.append(m["s1"] - m["c0"])
    return np.percentile(out, [2.5, 97.5]).tolist()
HB = {}
stat = {}
for S in SET_DEF:
    x, y = P[(S, "c0")], P[(S, "s1")]; w = welch(x, y); pt = paired_t(x, y)
    loo_w = {f"drop_seed{k}": welch(np.delete(x, k), np.delete(y, k))[:2] for k in range(3)}
    loo_p = {f"drop_seed{k}": float(np.delete(y - x, k).mean()) for k in range(3)}
    xs, ys = np.arcsin(np.sqrt(x / 100)), np.arcsin(np.sqrt(y / 100)); wa = welch(xs, ys)
    # overdispersion: between-seed variance vs binomial expectation (per arm)
    od = {}
    for a in ARMS:
        ks = np.array([prop(a, s, SET_DEF[S]).sum() for s in SEEDS]); n = N[S]; pbar = ks.sum() / (3 * n)
        chi2 = float(((ks - n * pbar) ** 2 / (n * pbar * (1 - pbar))).sum()) if 0 < pbar < 1 else float("nan")
        od[a] = dict(chi2_homog=chi2, df=2, p=float(st.chi2.sf(chi2, 2)) if chi2 == chi2 else None, between_sd_pp=float(P[(S, a)].std(ddof=1)), binom_sd_pp=float(100 * np.sqrt(pbar * (1 - pbar) / n)) if 0 < pbar < 1 else 0.0)
    HB[S] = hier_boot(S)
    sd_ci = {a: [float(P[(S, a)].std(ddof=1) * np.sqrt(2 / st.chi2.ppf(q, 2))) for q in (0.975, 0.025)] for a in ARMS}   # 95% CI for the between-seed SD (chi-square, df=2)
    stat[S] = dict(n_per_seed=N[S], per_seed_c0=x.tolist(), per_seed_s1=y.tolist(), welch=dict(diff=w[0], ci=list(w[1]), df=w[2]), welch_arcsine_sqrt_df=wa[2],
                   paired_by_seed=dict(diff=pt[0], ci=list(pt[1]), seed_diffs=pt[2].tolist(), signflip_p=signflip_p(pt[2]), min_possible_p=0.25), leave_one_seed_out_welch={k: [float(v[0]), [float(v[1][0]), float(v[1][1])]] for k, v in loo_w.items()},
                   leave_one_seed_out_paired_mean={k: v for k, v in loo_p.items()}, variance_ratio_s1_over_c0=float(y.var(ddof=1) / x.var(ddof=1)) if x.var(ddof=1) > 0 else None,
                   corr_c0_s1_across_seeds=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None, overdispersion=od, between_seed_sd_ci95=sd_ci, hierarchical_bootstrap_ci=HB[S])

# sample size / CI half-width: seed-level model, Welch-type two-sample with pooled SD scenarios
def halfwidth(sd, n): return st.t.ppf(0.975, 2 * n - 2) * sd * np.sqrt(2 / n)
def power_noninf(sd, delta, margin, n, B=4000):
    ok = 0
    for _ in range(B):
        x = rng.normal(0, sd, n); y = rng.normal(delta, sd, n); d, ci, _ = welch(x, y); ok += ci[0] >= -margin
    return ok / B
NS = [3, 4, 5, 6, 8, 10, 12, 15, 20]
sizing = {}
for S, margin in (("NOM", 10), ("BOUND", 15)):
    sds = {a: P[(S, a)].std(ddof=1) for a in ARMS}; sd_pool = float(np.sqrt((sds["c0"] ** 2 + sds["s1"] ** 2) / 2))
    sd_hi = float(max(sd_pool, np.sqrt(2 / st.chi2.ppf(0.1, 2)) * sd_pool))        # upper 90% one-sided bound of the SD from 2 df (very wide)
    sizing[S] = dict(margin_pp=margin, sd_pooled_pp=sd_pool, sd_upper90_pp=sd_hi, sd_binomial_floor_pp=float(100 * np.sqrt(0.8 * 0.2 / N[S])), rows=[])
    for n in NS:
        sizing[S]["rows"].append(dict(n=n, hw_sd_pooled=float(halfwidth(sd_pool, n)), hw_sd_upper90=float(halfwidth(sd_hi, n)),
                                      hw_sd_10pp=float(halfwidth(10.0, n)), power_noninf_delta0_sd_pooled=power_noninf(sd_pool, 0, margin, n), power_noninf_delta0_sd10=power_noninf(10.0, 0, margin, n),
                                      power_noninf_delta_neg5_sd10=power_noninf(10.0, -5, margin, n)))
    # smallest n with half-width <= 10 pp and <= margin
    for key in ("hw_sd_pooled", "hw_sd_upper90", "hw_sd_10pp"):
        sizing[S][f"n_for_hw_le_10pp[{key}]"] = next((r["n"] for r in sizing[S]["rows"] if r[key] <= 10), ">20")
stat["_sizing"] = sizing

# ------------------------------------------------------------------ Part 2: failure diagnosis (post-pick failures)
def feats(e):
    a, B, T, bs = e["action"], e["box_pos"], e["tcp_pos"], e["box_speed"]; n = len(a); g = a[:, 7]
    rise = B[:n, 2] - B[0, 2]; sep = np.linalg.norm(B[:n] - T[:n], axis=1); held = (sep < 0.06) & (rise > 0.03)
    t_lift = int(np.argmax(rise > 0.03)) if (rise > 0.03).any() else None
    tc = int(np.argmax(g < 0.028)) if (g < 0.028).any() else None
    re = None
    if tc is not None:
        r = np.nonzero(g[tc:] > 0.035)[0]; re = tc + int(r[0]) if len(r) else None
    d_tray = np.linalg.norm(T[:n, :2] - TRAY, axis=1); arr = np.nonzero((d_tray < 0.025) & (T[:n, 2] < PLACE_Z + 0.015))[0]; arr = int(arr[0]) if len(arr) else None
    f = dict(pick=bool(e["pick_success"]), place=bool(e["place_success"]), success=bool(e["task_success"]), final_xy=float(e["final_box_to_tray_xy_m"]), final_z_err=float(e["final_box_z_err_m"]),
             final_speed=float(e["final_box_speed"]), t_close=tc, t_lift=t_lift, t_reopen=re, t_arrive=arr, steps=n)
    if re is not None and t_lift is not None:
        tcp_rel, box_rel, box_fin = T[re, :2], B[re, :2], B[-1, :2]
        aim = tcp_rel - TRAY; inhand = box_rel - tcp_rel; drift = box_fin - box_rel
        ih_close = (B[tc, :2] - T[tc, :2]) if tc is not None else inhand; slip = inhand - ih_close            # offset present at gripper closing vs growth between closing and release
        lost = np.nonzero(sep[t_lift:] > 0.06)[0]; f["t_box_far"] = None if not len(lost) else int(t_lift + lost[0])
        f.update(aim=aim.tolist(), inhand=inhand.tolist(), drift=drift.tolist(), ih_close=ih_close.tolist(), slip=slip.tolist(), max_sep_carry=float(sep[t_lift:re].max()) if re > t_lift else None, total=(box_fin - TRAY).tolist(), tcp_z_at_release=float(T[re, 2]), tcp_z_minus_place=float(T[re, 2] - PLACE_Z),
                 box_z_at_release=float(B[re, 2] - B[0, 2]), box_speed_at_release=float(bs[re]) if re < len(bs) else None, box_speed_max_after=float(bs[re:].max()),
                 steps_arrive_to_release=None if arr is None else re - arr, release_before_arrival=bool(arr is None or re < arr),
                 carried_frac=float(held[t_lift:re].mean()) if re > t_lift else None, transport_steps=re - t_lift,
                 grip_open_rate=float(np.max(np.diff(g[re - 1:re + 12]) / 0.05)) if len(g[re - 1:re + 12]) > 2 else None,   # m/s of the gripper command right after release
                 box_tcp_height=float(B[re, 2] - T[re, 2]))
    return f
F = {k: [dict(feats(e), S=e["S"], idx=e["episode_index"], tseed=k[1], arm=k[0]) for e in v] for k, v in DATA.items()}
ALLF = [f for v in F.values() for f in v]

def mech(f):
    """Mechanism label for post-pick failures from measured quantities (thresholds from geometry: tray inner half-width 0.07, place tol 0.03)."""
    if not f["pick"] or f["success"]: return None
    if "aim" not in f: return "no release event recorded"
    aim, inh, dr, tot = (np.array(f[k]) for k in ("aim", "inhand", "drift", "total"))
    if f["release_before_arrival"]: return "early release (before reaching tray)"
    if np.linalg.norm(dr) > 0.02: return "post-release drift (box moved >2 cm after opening)"
    if f["tcp_z_minus_place"] > 0.03: return "release from height (TCP >3 cm above placing height)"
    if np.linalg.norm(aim) >= np.linalg.norm(inh) and np.linalg.norm(aim) > 0.02: return "aim error: TCP offset from tray centre at release"
    if np.linalg.norm(inh) > 0.015:                                  # box centre >1.5 cm off the TCP in xy at release (successes: median 0.7 cm)
        return ("in-hand offset, already present at closing (off-centre grasp)" if np.linalg.norm(f["ih_close"]) >= np.linalg.norm(f["slip"]) else "in-hand offset, grew after closing (slip during lift/transport)")
    return "mixed / small components"
for f in ALLF: f["mech"] = mech(f)

def fails(arm, post_only=True): return [f for f in ALLF if f["arm"] == arm and not f["success"] and (f["pick"] or not post_only)]
from collections import Counter
mech_tab = {a: dict(Counter(f["mech"] for f in fails(a))) for a in ARMS}
mech_by_seed = {a: {s: dict(Counter(f["mech"] for f in fails(a) if f["tseed"] == s)) for s in SEEDS} for a in ARMS}
cond_place = {a: {s: [sum(1 for f in F[(a, s)] if f["pick"] and f["place"]), sum(1 for f in F[(a, s)] if f["pick"])] for s in SEEDS} for a in ARMS}
fin = {a: [f["final_xy"] for f in ALLF if f["arm"] == a and f["pick"] and not f["success"]] for a in ARMS}
near_miss = {a: dict(n=len(v), within_3to4cm=int(sum(0.03 <= x < 0.04 for x in v)), within_3to5cm=int(sum(0.03 <= x < 0.05 for x in v)), over_5cm=int(sum(x >= 0.05 for x in v))) for a, v in fin.items()}
pick_fail = {a: {s: sum(1 for f in F[(a, s)] if not f["pick"]) for s in SEEDS} for a in ARMS}
n_ep = {a: {s: len(F[(a, s)]) for s in SEEDS} for a in ARMS}

def vec_stats(arm, key, subset):
    V = np.array([f[key] for f in ALLF if f["arm"] == arm and "aim" in f and subset(f)]);
    return None if len(V) == 0 else dict(n=len(V), mean_xy=V.mean(0).tolist(), mean_norm=float(np.linalg.norm(V, axis=1).mean()), sd_xy=V.std(0).tolist())
decomp = {}
for lab, sub in (("post-pick failures", lambda f: f["pick"] and not f["success"]), ("successes", lambda f: f["success"])):
    decomp[lab] = {a: {k: vec_stats(a, k, sub) for k in ("aim", "inhand", "drift", "total", "ih_close", "slip")} for a in ARMS}
scal = {}
for lab, sub in (("post-pick failures", lambda f: f["pick"] and not f["success"]), ("successes", lambda f: f["success"])):
    scal[lab] = {}
    for a in ARMS:
        R = [f for f in ALLF if f["arm"] == a and "aim" in f and sub(f)]
        scal[lab][a] = {k: (float(np.mean([r[k] for r in R if r.get(k) is not None])), float(np.std([r[k] for r in R if r.get(k) is not None]))) for k in ("max_sep_carry", "tcp_z_minus_place", "box_speed_at_release", "box_speed_max_after", "steps_arrive_to_release", "transport_steps", "carried_frac", "grip_open_rate", "box_tcp_height")} if R else None
# direction of final error (angle of 'total') -- systematic bias?
def mean_resultant(arm):
    V = np.array([f["total"] for f in ALLF if f["arm"] == arm and "aim" in f and f["pick"] and not f["success"]]); ang = np.arctan2(V[:, 1], V[:, 0]); return float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean())), float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))), int(len(V))
rayleigh = {a: mean_resultant(a) for a in ARMS}
by_set = {a: {S: Counter("post-pick" if (f["pick"]) else "pick" for f in ALLF if f["arm"] == a and f["S"] == S and not f["success"]) for S in SETS} for a in ARMS}
by_set = {a: {S: dict(v) for S, v in d.items()} for a, d in by_set.items()}
# conditional place rate S1 vs C0 (per seed) -> same Welch machinery
cp = {a: np.array([100 * cond_place[a][s][0] / cond_place[a][s][1] for s in SEEDS]) for a in ARMS}
cond_stat = dict(per_seed_c0=cp["c0"].tolist(), per_seed_s1=cp["s1"].tolist(), welch=dict(zip(("diff", "ci", "df"), (lambda w: (w[0], list(w[1]), w[2]))(welch(cp["c0"], cp["s1"])))))
pk = {a: np.array([100 * (1 - pick_fail[a][s] / n_ep[a][s]) for s in SEEDS]) for a in ARMS}
pick_stat = dict(per_seed_c0=pk["c0"].tolist(), per_seed_s1=pk["s1"].tolist(), welch=dict(zip(("diff", "ci", "df"), (lambda w: (w[0], list(w[1]), w[2]))(welch(pk["c0"], pk["s1"])))))
# final offset distribution of post-pick episodes (success + fail), per arm
fin_all = {a: [f["final_xy"] for f in ALLF if f["arm"] == a and f["pick"]] for a in ARMS}
ks = st.ks_2samp(fin_all["c0"], fin_all["s1"])
fin_q = {a: np.percentile(fin_all[a], [10, 25, 50, 75, 90]).tolist() for a in ARMS}
fail_stats = dict(post_pick_fail_counts={a: sum(len(fails(a)) for _ in [0]) for a in ARMS}, mech_total=mech_tab, mech_by_seed=mech_by_seed, near_miss=near_miss, conditional_place_rate=cond_stat, pick_rate=pick_stat,
                  cond_place_counts=cond_place, pick_fail_counts=pick_fail, decomposition=decomp, scalars=scal, direction_mean_resultant_len_angle_n=rayleigh, fail_by_set=by_set, final_xy_quantiles_post_pick=fin_q,
                  ks_final_xy_post_pick=dict(stat=float(ks.statistic), p=float(ks.pvalue)))

# ------------------------------------------------------------------ Part 3: thread-count deviation D1 (what is identifiable)
VAL = {}
for a in ARMS:
    for s in SEEDS:
        p = (ROOT / "outputs/eval/e6b" / a / "open_loop_val.json") if s == 0 else (ROOT / "outputs/eval/e0" / f"{a}_seed{s}" / "open_loop_val.json")
        v = json.loads(p.read_text()); ck = ROOT / "outputs/checkpoints" / (f"e6b_{a}_seed0" if s == 0 else f"e0_{a}_seed{s}") / "train_log.jsonl"
        tl = [json.loads(l) for l in ck.read_text().splitlines() if l.strip()] if ck.exists() else []
        last = [r for r in tl if "loss_ema" in r][-1]["loss_ema"] if tl else None
        VAL[f"{a}_seed{s}"] = dict(mae_arm=v["mae_arm_rad"], mae_left=v["mae_left_arm_rad"], p95=v["p95_arm_rad"], final_train_loss_ema=last)
d1 = dict(open_loop_and_train=VAL, note=("Thread count is perfectly confounded with seed identity (seed 0 = 4 threads, seeds 1-2 = 3 threads) in both arms; with 3 seeds the data cannot separate a thread effect from seed-to-seed variation. "
          "Checkable: (i) whether seed 0 is an outlier relative to seeds 1-2 within each arm on non-success metrics; (ii) the pre-registered seed-0-excluded sensitivity (n=2/arm, uninformative); (iii) the Stage-1 benchmark showing thread count changes numerics only (loss diff 8e-2 after 60 steps, then same loss scale). "
          "A direct test (same seed 1 re-trained at 4 threads) would need new training = not approved."))
d1["seed0_vs_others"] = {a: dict(zip(("nom_pp", "bound_pp", "all_dev_pp"), ([float(P[(S, a)][0]) for S in ("NOM", "BOUND", "ALL_DEV")]))) for a in ARMS}
d1["others_mean"] = {a: [float(P[(S, a)][1:].mean()) for S in ("NOM", "BOUND", "ALL_DEV")] for a in ARMS}
d1["primary_analysis_keeps_all_seeds"] = True

res = dict(stat_precision=stat, failure_diagnosis=fail_stats, thread_deviation=d1); (OUT / "e0_1_results.json").write_text(json.dumps(res, indent=1, default=float))

# ------------------------------------------------------------------ plots
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for i, S in enumerate(("NOM", "BOUND", "ALL_DEV")):
    for k in range(3): ax[i].plot([0, 1], [P[(S, "c0")][k], P[(S, "s1")][k]], "-o", color="C0" if k == 0 else "C1" if k == 1 else "C2", label=f"train seed {k}")
    ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(["C0", "S1"]); ax[i].set_title(f"{S}: paired by training seed"); ax[i].set_ylabel("success [%]")
ax[0].legend(); plt.tight_layout(); plt.savefig(OUT / "E01_1_paired_seeds.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, S in enumerate(("NOM", "BOUND")):
    ns = [r["n"] for r in sizing[S]["rows"]]
    for key, lab in (("hw_sd_pooled", "SD = E0 pooled"), ("hw_sd_10pp", "SD = 10 pp"), ("hw_sd_upper90", "SD = upper 90% bound")): ax[j].plot(ns, [r[key] for r in sizing[S]["rows"]], "-o", label=lab)
    ax[j].axhline(10, ls=":", color="k"); ax[j].axhline(sizing[S]["margin_pp"], ls="--", color="r", label=f"margin {sizing[S]['margin_pp']} pp"); ax[j].set_title(f"{S}: expected 95% CI half-width of S1-C0"); ax[j].set_xlabel("training seeds per arm"); ax[j].set_ylabel("half-width [pp]"); ax[j].set_ylim(0, 45)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT / "E01_2_ci_halfwidth.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, col in (("c0", "C0"), ("s1", "C1")):
    V = np.array([f["total"] for f in ALLF if f["arm"] == a and "aim" in f and f["pick"] and not f["success"]]) * 100
    ax[0].scatter(V[:, 0], V[:, 1], s=14, color=col, alpha=.7, label=f"{a.upper()} post-pick failures (n={len(V)})")
    ax[2].hist(np.array(fin_all[a]) * 100, bins=np.linspace(0, 8, 33), alpha=.5, color=col, label=a.upper())
th = np.linspace(0, 2 * np.pi, 100); ax[0].plot(3 * np.cos(th), 3 * np.sin(th), "k--", lw=1, label="3 cm tolerance"); ax[0].set_aspect("equal"); ax[0].set_xlabel("final box x − tray [cm]"); ax[0].set_ylabel("y [cm]"); ax[0].legend(fontsize=7); ax[0].set_title("Final box offset (world frame)")
labs = ["aim", "inhand", "drift"]; w = 0.35
for i, a in enumerate(ARMS):
    ax[1].bar(np.arange(3) + i * w, [decomp["post-pick failures"][a][k]["mean_norm"] * 100 for k in labs], w, color="C0" if a == "c0" else "C1", label=a.upper())
ax[1].set_xticks(np.arange(3) + w / 2); ax[1].set_xticklabels(["TCP aim error\nat release", "box in-hand\noffset", "post-release\ndrift"]); ax[1].set_ylabel("mean |component| [cm]"); ax[1].set_title("Error decomposition (post-pick failures)"); ax[1].legend()
ax[2].axvline(3, color="k", ls="--"); ax[2].set_xlabel("final box-tray distance, all post-pick episodes [cm]"); ax[2].legend(); ax[2].set_title("Placement error distribution")
plt.tight_layout(); plt.savefig(OUT / "E01_3_release_diagnosis.png", dpi=130); plt.close()

# ------------------------------------------------------------------ console summary
def pr(S):
    s = stat[S]; print(f"[{S}] n/seed={s['n_per_seed']} C0={np.round(s['per_seed_c0'],1)} S1={np.round(s['per_seed_s1'],1)} welch={s['welch']['diff']:.1f} CI=[{s['welch']['ci'][0]:.1f},{s['welch']['ci'][1]:.1f}] df={s['welch']['df']:.1f}"
                       f" | paired={s['paired_by_seed']['diff']:.1f} CI=[{s['paired_by_seed']['ci'][0]:.1f},{s['paired_by_seed']['ci'][1]:.1f}] diffs={np.round(s['paired_by_seed']['seed_diffs'],1)} p>={s['paired_by_seed']['signflip_p']} | hboot={np.round(s['hierarchical_bootstrap_ci'],1)}")
for S in SET_DEF: pr(S)
print(json.dumps(dict(mech=mech_tab, near=near_miss, cond=cond_stat, pick=pick_stat), indent=1, default=float))

# ------------------------------------------------------------------ extra: seed-level grasp offset along x at closing (all episodes with a closing event)
from scipy import stats as _st
gx = {a: np.array([np.mean([f["ih_close"][0] * 100 for f in F[(a, s)] if "ih_close" in f]) for s in SEEDS]) for a in ARMS}
gabs = {a: np.array([np.mean([abs(f["ih_close"][0]) * 100 for f in F[(a, s)] if "ih_close" in f]) for s in SEEDS]) for a in ARMS}
gn = {a: [sum(1 for f in F[(a, s)] if "ih_close" in f) for s in SEEDS] for a in ARMS}
w = welch(gabs["c0"], gabs["s1"]); w2 = welch(gx["c0"], gx["s1"])
extra = dict(note="ih_close only exists for episodes with pick (lift) and a release event", n=gn, mean_x_cm=dict(c0=gx["c0"].tolist(), s1=gx["s1"].tolist()), mean_abs_x_cm=dict(c0=gabs["c0"].tolist(), s1=gabs["s1"].tolist()),
             welch_abs_x=dict(diff=w[0], ci=list(w[1]), df=w[2]), welch_x=dict(diff=w2[0], ci=list(w2[1]), df=w2[2]))
res["grasp_offset_seed_level"] = extra; (OUT / "e0_1_results.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(extra, default=float))
