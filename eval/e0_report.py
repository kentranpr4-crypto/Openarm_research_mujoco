#!/usr/bin/env python3
"""E0 report: C0 vs S1 across independent training seeds {0 (Stage 1, reused), 1, 2}. Implements the analysis frozen in outputs/e0/e0_config.json
(thresholds are READ from that file, never hard-coded; the file's sha256 is checked). Dev seeds 21-22 only. Never selects a policy.
Run (ACT venv): python eval/e0_report.py [--out outputs/e0]
Report A = variation across training seeds; Report B = variation across evaluation scenarios; then rules R1-R3, failure analysis, cost."""
import sys, json, hashlib, argparse, itertools, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from eval.evaluate import wilson
from eval.e6b_report import load as load_tag, kn
from eval.audit_v1 import events, jerk_stats, classify, HZ
from eval.e6b_failure_analysis import timeline
EXPECT_SHA = "25f20ea4c887534ddc80c8aedee01e93da05790dac182d1b0a86f4217d545e9d"
ARMS, SEEDS = ("c0", "s1"), (0, 1, 2)
COL = {"c0": "C0", "s1": "C1"}

def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def paths(arm, seed, a):
    """(eval root, eval-arm key, checkpoint dir). Seed 0 = Stage 1 (reused, never modified); seeds 1,2 = E0."""
    if seed == 0: return ROOT / "outputs/eval/e6b", arm, ROOT / f"outputs/checkpoints/e6b_{arm}_seed0"
    return Path(a.e0_eval_root), f"{arm}_seed{seed}", Path(a.ckpt_root) / f"e0_{arm}_seed{seed}"

def sets(root, key, with_rollouts=False): return load_tag(root, key[0], key[1], with_rollouts)
def gather(arm, seed, a):
    root, ek, ck = paths(arm, seed, a); R = dict(arm=arm, seed=seed); L = {}
    for k in ("nom", "b21", "b22", "right_mean", "jit_small"):
        L[k] = load_tag(root, ek, k, with_rollouts=True)
    if L["nom"] is None or L["b21"] is None or L["b22"] is None: return None
    R["sets"] = {k: v for k, v in L.items()}
    cnt = lambda eps, key="task_success": (int(sum(bool(e[key]) for e in eps)), len(eps))
    R["NOM"] = cnt(L["nom"]["eps"]); R["B21"] = cnt(L["b21"]["eps"]); R["B22"] = cnt(L["b22"]["eps"])
    R["BOUND"] = (R["B21"][0] + R["B22"][0], R["B21"][1] + R["B22"][1]); R["NOM_pick"] = cnt(L["nom"]["eps"], "pick_success"); R["BOUND_pick"] = (cnt(L["b21"]["eps"], "pick_success")[0] + cnt(L["b22"]["eps"], "pick_success")[0], 60)
    base = [e for e in L["nom"]["eps"] if e["episode_index"] < 20]; b = cnt(base)[0]; R["unperturbed20"] = b; R["SI"] = {}
    for k in ("right_mean", "jit_small"):
        if L[k] is None: continue
        p = cnt(L[k]["eps"])[0]; R["SI"][k] = dict(perturbed=p, unperturbed=b, SI=(p / b if b else float("nan")), n=len(L[k]["eps"]))
    allr = [e for k in ("nom", "b21", "b22") for e in L[k]["eps"] if "action" in e]; dz, jall, jint, jbnd, trk = [], [], [], [], []; mu = None
    f = ck / "last/policy_preprocessor_step_3_normalizer_processor.safetensors"
    if f.exists():
        from safetensors.numpy import load_file; mu = load_file(str(f))["action.mean"]
    dev_j = dev_g = 0.0
    for e in allr:
        ev = events(e)
        if ev["t_close"] is not None: dz.append(ev["dz"] * 100)
        js, _, _ = jerk_stats(e["action"], H=20); jall.append(js["all"]); jint.append(js["interior"]); jbnd.append(js["boundary"]); trk.append(float(np.abs(e["action"][:, :7] - e["state"][:, :7]).mean()))
        if mu is not None: dev_j = max(dev_j, float(np.abs(e["action"][:, 8:15] - mu[8:15]).max())); dev_g = max(dev_g, float(np.abs(e["action"][:, 15] - mu[15]).max()))
    R["dz_cm"] = dict(mean=float(np.mean(dz)), sd=float(np.std(dz)), frac_gt_3p4=float(np.mean(np.array(dz) > 3.4)), n=len(dz)); R["jerk"] = dict(all=float(np.mean(jall)), interior=float(np.mean(jint)), boundary=float(np.mean(jbnd)))
    R["tracking_mean_arm_rad"] = float(np.mean(trk)); R["idle_dev"] = dict(joints_rad=dev_j, gripper_m=dev_g); R["latency_ms_per_query"] = L["nom"]["summary"].get("latency_ms_per_query")
    # failure taxonomy + transport kinematics (NOM + BOUND), diagnostics only
    cls = {}; kin = {"success": [], "fail": []}; nom_fail_kin = []
    for k in ("nom", "b21", "b22"):
        for e in L[k]["eps"]:
            if "action" not in e: continue
            c = classify(e, events(e)); cls[c] = cls.get(c, 0) + 1
            tl = timeline(e)
            if k == "nom" and tl["t_lift"] is not None and tl["t_arrive"] is not None:
                T = e["tcp_pos"][tl["t_lift"]:tl["t_arrive"] + 1]; v = np.diff(T, axis=0) * HZ; acc = np.diff(v, axis=0) * HZ
                kin["success" if e["task_success"] else "fail"].append((float(np.linalg.norm(acc, axis=1).max()), float(np.linalg.norm(v, axis=1).max())))
    R["failure_classes"] = {k: v for k, v in cls.items() if k != "success"}; R["kin"] = {k: (np.mean(v, axis=0).round(3).tolist() if v else None, len(v)) for k, v in kin.items()}
    tl = ck / "train_log.jsonl"
    if tl.exists():
        log = jl(tl); v = [x for x in log if "val_l1" in x]; R["train"] = dict(steps=len(log), minutes=log[-1]["elapsed_s"] / 60, s_per_step=log[-1]["elapsed_s"] / len(log), loss_ema=log[-1]["loss_ema"], val_l1=v[-1]["val_l1"] if v else None); R["log"] = log
    ol = root / ek / "open_loop_val.json"; R["val_mae_left_arm_rad"] = json.loads(ol.read_text())["mae_left_arm_rad"] if ol.exists() else None
    return R

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); da = a.mean() - b.mean()
    if len(a) < 2 or len(b) < 2: return dict(diff_pp=100 * da, ci95_pp=[float("nan")] * 2, df=float("nan"), p=float("nan"))
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = np.sqrt(va + vb)
    if se == 0: return dict(diff_pp=100 * da, ci95_pp=[100 * da, 100 * da], df=float("inf"), p=0.0 if da != 0 else 1.0)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)); t = st.t.ppf(0.975, df)
    return dict(diff_pp=100 * da, ci95_pp=[100 * (da - t * se), 100 * (da + t * se)], df=float(df), p=float(2 * st.t.sf(abs(da / se), df)))
def permutation_p(a, b):
    x = np.r_[a, b]; n = len(a); obs = abs(np.mean(a) - np.mean(b)); cnt = tot = 0
    for idx in itertools.combinations(range(len(x)), n):
        m = np.zeros(len(x), bool); m[list(idx)] = True; tot += 1; cnt += abs(x[m].mean() - x[~m].mean()) >= obs - 1e-12
    return cnt / tot
def hier_bootstrap(res_by_arm, key, n_boot=10000, seed=0):
    """Resample training seeds with replacement within arm, then episodes within the chosen seed (binomial); CI of mean(S1)-mean(C0) in pp. Sensitivity only."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        m = {}
        for arm in ARMS:
            R = [r for r in res_by_arm[arm]]; ch = [R[i] for i in rng.integers(0, len(R), len(R))]; ps = []
            for r in ch: k, n = r[key]; ps.append(rng.binomial(n, k / n) / n)
            m[arm] = np.mean(ps)
        out.append(100 * (m["s1"] - m["c0"]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def main():
    global cfg
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=ROOT / "outputs/e0"); ap.add_argument("--config", type=Path, default=ROOT / "outputs/e0/e0_config.json")
    ap.add_argument("--e0-eval-root", default=str(ROOT / "outputs/eval/e0")); ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/checkpoints")); ap.add_argument("--skip-sha-check", action="store_true"); a = ap.parse_args()
    sha = hashlib.sha256(a.config.read_bytes()).hexdigest()
    if sha != EXPECT_SHA and not a.skip_sha_check: sys.exit(f"config sha256 {sha} != frozen {EXPECT_SHA}: the pre-registration was modified")
    cfg = json.loads(a.config.read_text()); rules = cfg["analysis"]["rules"]; a.out.mkdir(parents=True, exist_ok=True)
    res = {arm: [r for s in SEEDS if (r := gather(arm, s, a)) is not None] for arm in ARMS}
    n_seeds = {arm: len(v) for arm, v in res.items()}; out = dict(config_sha256=sha, n_seeds=n_seeds, seeds={arm: [r["seed"] for r in v] for arm, v in res.items()})
    # ---------------- Report A: variation across training seeds
    prop = lambda r, key: r[key][0] / r[key][1]; A = {}
    for key, nm in (("NOM", "NOM (seed 21, n=30)"), ("B21", "BOUND seed 21 (n=30)"), ("B22", "BOUND seed 22 (n=30)"), ("BOUND", "BOUND pooled (n=60)"), ("BOUND_pick", "BOUND pick (n=60)"), ("NOM_pick", "NOM pick (n=30)")):
        A[key] = {}
        for arm in ARMS:
            v = [prop(r, key) for r in res[arm]]
            A[key][arm] = dict(label=nm, per_seed={r["seed"]: f"{r[key][0]}/{r[key][1]}" for r in res[arm]}, per_seed_wilson={r["seed"]: [round(x, 3) for x in wilson(*r[key])] for r in res[arm]},
                               mean_pp=100 * float(np.mean(v)) if v else None, sd_pp=100 * float(np.std(v, ddof=1)) if len(v) > 1 else None, range_pp=[100 * min(v), 100 * max(v)] if v else None,
                               binomial_sd_pp=100 * float(np.mean([np.sqrt(p * (1 - p) / res[arm][i][key][1]) for i, p in enumerate(v)])) if v else None)
    out["A_across_training_seeds"] = A
    # ---------------- Report B: variation across evaluation scenarios (+ shortcut)
    B = dict(shortcut={arm: {r["seed"]: r["SI"] for r in res[arm]} for arm in ARMS}, unperturbed20={arm: {r["seed"]: r["unperturbed20"] for r in res[arm]} for arm in ARMS},
             scenario_seed_matrix={arm: {r["seed"]: {k: f"{r[k][0]}/{r[k][1]}" for k in ("NOM", "B21", "B22")} for r in res[arm]} for arm in ARMS})
    out["B_across_scenarios"] = B
    # ---------------- inference + rules
    inf = {}
    for key in ("NOM", "B21", "B22", "BOUND"):
        a_, b_ = [prop(r, key) for r in res["s1"]], [prop(r, key) for r in res["c0"]]
        d = welch(a_, b_); d["permutation_p"] = permutation_p(a_, b_) if len(a_) and len(b_) else None
        a2, b2 = [prop(r, key) for r in res["s1"] if r["seed"] != 0], [prop(r, key) for r in res["c0"] if r["seed"] != 0]; d["excluding_seed0"] = welch(a2, b2)
        if all(len(res[x]) >= 2 for x in ARMS): d["hier_bootstrap_ci95_pp_sensitivity"] = hier_bootstrap(res, key)
        inf[key] = d
    out["inference_S1_minus_C0"] = inf
    si_ok = lambda r: all(k in r["SI"] and r["SI"][k]["SI"] >= 0.90 for k in ("right_mean", "jit_small")); si_c0 = lambda r: all(k in r["SI"] and r["SI"][k]["SI"] <= 0.25 for k in ("right_mean", "jit_small"))
    s1_pass, c0_pass = sum(si_ok(r) for r in res["s1"]), sum(si_c0(r) for r in res["c0"])
    R1 = "SUPPORTED" if (len(res["s1"]) == 3 and len(res["c0"]) == 3 and s1_pass >= 2 and c0_pass == 3) else ("INCOMPLETE" if (len(res["s1"]) < 3 or len(res["c0"]) < 3) else "NOT SUPPORTED / MIXED")
    m = rules["R2_competence_cost"]
    def r2(key, margin):
        d = inf[key]; lo, hi = d["ci95_pp"]
        if np.isnan(lo): return "INCOMPLETE"
        return "NON-INFERIOR" if lo >= margin else ("COST EVIDENT" if hi < 0 else "INCONCLUSIVE")
    R2n, R2b = r2("NOM", m["NOM_margin_pp"]), r2("BOUND", m["BOUND_margin_pp"])
    if "INCOMPLETE" in (R1, R2n, R2b): R3 = "INCOMPLETE: not all seeds available"
    elif R1 == "SUPPORTED" and R2n == "NON-INFERIOR" and R2b == "NON-INFERIOR": R3 = "S1 MAY BE RECOMMENDED FOR FURTHER STUDY as a V2 baseline candidate (owner decides)"
    elif "COST EVIDENT" in (R2n, R2b): R3 = "DO NOT ADOPT S1's floor as-is (competence cost evident)"
    elif R1 == "SUPPORTED": R3 = "INCONCLUSIVE: robustness replicates but competence cost cannot be bounded; more training seeds needed"
    else: R3 = "NO RECOMMENDATION: robustness effect did not replicate cleanly (R1 not supported)"
    out["rules"] = dict(R1=dict(verdict=R1, s1_seeds_passing=f"{s1_pass}/{len(res['s1'])}", c0_seeds_with_SI_le_0p25=f"{c0_pass}/{len(res['c0'])}"), R2=dict(NOM=R2n, BOUND=R2b), R3=R3, selection="none: recommendation only")
    # ---------------- failure analysis + cost
    out["failure_analysis"] = {arm: {r["seed"]: dict(classes=r["failure_classes"], dz=r["dz_cm"], kin_peak_accel_speed_success=r["kin"]["success"], kin_peak_accel_speed_fail=r["kin"]["fail"], jerk=r["jerk"], idle_dev=r["idle_dev"]) for r in res[arm]} for arm in ARMS}
    tfile = ROOT / "outputs/e0/timing.jsonl"; cost = dict(train={}, eval={})
    for arm in ARMS:
        for r in res[arm]:
            if r.get("train"): cost["train"][f"{arm}_seed{r['seed']}"] = dict(minutes=round(r["train"]["minutes"], 1), s_per_step=round(r["train"]["s_per_step"], 3), threads=4 if r["seed"] == 0 else 3, reused=(r["seed"] == 0))
    if tfile.exists():
        ev = {}
        for e in jl(tfile): ev.setdefault(e["run"], {})[e["event"]] = e["t"]
        for run, d in ev.items():
            if "eval_start" in d and "eval_end" in d: cost["eval"][run] = round((d["eval_end"] - d["eval_start"]) / 60, 1)
    out["compute_cost"] = cost; out["per_arm_seed"] = {arm: {r["seed"]: dict(NOM=r["NOM"], BOUND=r["BOUND"], BOUND_pick=r["BOUND_pick"], tracking=r["tracking_mean_arm_rad"], val_mae_left_arm_rad=r["val_mae_left_arm_rad"], latency=r["latency_ms_per_query"], train={k: v for k, v in (r.get("train") or {}).items()}) for r in res[arm]} for arm in ARMS}
    (a.out / "e0_results.json").write_text(json.dumps(out, indent=1, default=float)); plots(res, out, a.out); (a.out / "e0_tables.md").write_text(tables(res, out)); print(tables(res, out)); print(json.dumps(out["rules"], indent=1))

def plots(res, out, od):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for j, (key, ttl) in enumerate((("NOM", "NOM n=30"), ("B21", "BOUND seed21 n=30"), ("B22", "BOUND seed22 n=30"), ("BOUND", "BOUND pooled n=60"))):
        for i, arm in enumerate(ARMS):
            rs = res[arm]; ps = [100 * r[key][0] / r[key][1] for r in rs]; xs = np.full(len(ps), i) + np.linspace(-0.12, 0.12, len(ps))
            for r, x, p in zip(rs, xs, ps):
                lo, hi = wilson(*r[key]); ax[j].errorbar(x, p, yerr=[[p - 100 * lo], [100 * hi - p]], fmt="o", c=COL[arm], capsize=3, ms=5); ax[j].text(x + 0.03, p, f"s{r['seed']}", fontsize=7)
            if ps: ax[j].hlines(np.mean(ps), i - 0.3, i + 0.3, colors="k", lw=1.5)
        ax[j].set_xticks([0, 1]); ax[j].set_xticklabels(["C0", "S1"]); ax[j].set_ylim(0, 105); ax[j].set_title(ttl); ax[j].set_ylabel("success [%] per training seed (Wilson 95%); bar = arm mean")
    fig.tight_layout(); fig.savefig(od / "E0_1_across_seeds.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.35
    for i, arm in enumerate(ARMS):
        for r in res[arm]:
            for jx, k in enumerate(("right_mean", "jit_small")):
                if k in r["SI"]: ax.scatter(jx + (i - 0.5) * 0.25 + 0.03 * r["seed"], r["SI"][k]["SI"], c=COL[arm], s=40, edgecolors="k"); ax.text(jx + (i - 0.5) * 0.25 + 0.03 * r["seed"] + 0.02, r["SI"][k]["SI"], f"s{r['seed']}", fontsize=7)
    ax.axhline(0.9, c="gray", ls="--"); ax.axhline(0.25, c="gray", ls=":"); ax.set_xticks([0, 1]); ax.set_xticklabels(["right_mean", "jitter small"]); ax.set_ylabel("shortcut index SI (C0 blue, S1 orange)"); ax.set_title("R1: SI per training seed")
    fig.tight_layout(); fig.savefig(od / "E0_2_shortcut_SI.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["NOM", "B21", "B22", "BOUND"]
    for i, arm in enumerate(ARMS):
        sd = [out["A_across_training_seeds"][k][arm]["sd_pp"] or 0 for k in keys]; bn = [out["A_across_training_seeds"][k][arm]["binomial_sd_pp"] or 0 for k in keys]
        ax.bar(np.arange(4) + i * 0.38, sd, 0.18, color=COL[arm], label=f"{arm.upper()} between-seed SD"); ax.bar(np.arange(4) + i * 0.38 + 0.19, bn, 0.18, color=COL[arm], alpha=.35, label=f"{arm.upper()} binomial SD (within seed)")
    ax.set_xticks(np.arange(4) + 0.28); ax.set_xticklabels(keys); ax.set_ylabel("SD of success [pp]"); ax.legend(fontsize=7); ax.set_title("Between-seed vs within-seed (sampling) variation")
    fig.tight_layout(); fig.savefig(od / "E0_3_variance.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); labels, bottoms = [], None
    classes = sorted({c for arm in ARMS for r in res[arm] for c in r["failure_classes"]}); x = 0
    for arm in ARMS:
        for r in res[arm]:
            b = 0
            for ci, c in enumerate(classes): v = r["failure_classes"].get(c, 0); ax.bar(x, v, 0.7, bottom=b, color=f"C{ci}", label=c[:38] if x == 0 else None); b += v
            labels.append(f"{arm.upper()} s{r['seed']}"); x += 1
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30); ax.set_ylabel("failures (NOM + BOUND 90 episodes)"); ax.legend(fontsize=6); ax.set_title("Failure classes per training seed")
    fig.tight_layout(); fig.savefig(od / "E0_4_failures.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for arm in ARMS:
        for r in res[arm]:
            if r.get("log"):
                v = [x for x in r["log"] if "val_l1" in x]; ax[0].plot([x["step"] for x in v], [x["val_l1"] for x in v], c=COL[arm], alpha=.8, ls="-" if r["seed"] == 0 else "--"); ax[1].plot([x["step"] for x in r["log"]], [x["loss_ema"] for x in r["log"]], c=COL[arm], alpha=.8, ls="-" if r["seed"] == 0 else "--")
    ax[0].set_ylabel("val L1 (normalised)"); ax[1].set_ylabel("train L1 EMA"); ax[1].set_yscale("log"); ax[0].set_xlabel("step"); ax[1].set_xlabel("step"); ax[0].set_title("C0 blue / S1 orange; seed 0 solid, new seeds dashed")
    fig.tight_layout(); fig.savefig(od / "E0_5_training.png", dpi=110); plt.close(fig)

def tables(res, out):
    L = ["### Report A — variation across training seeds (success, per seed; mean ± SD across seeds; Wilson CIs in e0_results.json)\n", "| set | arm | per-seed k/n | mean [pp] | between-seed SD [pp] | range [pp] | within-seed binomial SD [pp] |\n|---|---|---|---|---|---|---|"]
    for key in ("NOM", "B21", "B22", "BOUND", "BOUND_pick"):
        for arm in ARMS:
            d = out["A_across_training_seeds"][key][arm]
            if d["mean_pp"] is None: continue
            L.append(f"| {d['label']} | {arm.upper()} | " + ", ".join(f"s{k}: {v}" for k, v in d["per_seed"].items()) + f" | {d['mean_pp']:.1f} | {'' if d['sd_pp'] is None else format(d['sd_pp'], '.1f')} | {d['range_pp'][0]:.0f}–{d['range_pp'][1]:.0f} | {d['binomial_sd_pp']:.1f} |")
    L += ["\n### Report B — evaluation scenarios and shortcut sensitivity (per training seed)\n", "| arm | seed | NOM | BOUND21 | BOUND22 | unperturbed (ids 0-19) | right_mean (SI) | jitter small (SI) |\n|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for r in res[arm]:
            si = r["SI"]; f = lambda k: f"{si[k]['perturbed']}/{si[k]['n']} ({si[k]['SI']:.2f})" if k in si else "-"
            L.append(f"| {arm.upper()} | {r['seed']} | {r['NOM'][0]}/{r['NOM'][1]} | {r['B21'][0]}/{r['B21'][1]} | {r['B22'][0]}/{r['B22'][1]} | {r['unperturbed20']}/20 | {f('right_mean')} | {f('jit_small')} |")
    L += ["\n### Inference S1 − C0 (per-seed proportions; Welch 95% CI; exact permutation p; excluding seed 0 in brackets)\n", "| set | diff [pp] | Welch CI95 [pp] | df | permutation p | excl. seed 0: diff [CI] |\n|---|---|---|---|---|---|"]
    for key, d in out["inference_S1_minus_C0"].items():
        e = d["excluding_seed0"]; L.append(f"| {key} | {d['diff_pp']:+.1f} | [{d['ci95_pp'][0]:+.1f}, {d['ci95_pp'][1]:+.1f}] | {d['df']:.1f} | {d['permutation_p'] if d['permutation_p'] is None else round(d['permutation_p'], 3)} | {e['diff_pp']:+.1f} [{e['ci95_pp'][0]:+.1f}, {e['ci95_pp'][1]:+.1f}] |")
    L += ["\n### Rules (frozen)\n", f"R1 {out['rules']['R1']} · R2 NOM {out['rules']['R2']['NOM']} · R2 BOUND {out['rules']['R2']['BOUND']} · **R3 {out['rules']['R3']}**"]
    return "\n".join(L)

if __name__ == "__main__":
    main()
