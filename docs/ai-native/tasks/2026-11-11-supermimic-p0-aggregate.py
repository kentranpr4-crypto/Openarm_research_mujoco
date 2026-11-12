#!/usr/bin/env python3
"""Aggregates the SuperMimic P0 independent evaluation (READ-ONLY: reads json files, writes one new report file).
Inputs: per-run directories containing eval/evaluate_best_*.json and eval/evaluate_latest_*.json (written by `research.supermimic_openarm.evaluate`) and, optionally, the results.json of
`research.supermimic_openarm.visualize --no-render --episodes 100` for dwell / stay metrics.  Applies the PRE-REGISTERED decision rule of the plan; thresholds are constants below and must not be edited after results are seen.
Usage: python3 2026-11-11-supermimic-p0-aggregate.py --runs RUN_DIR [RUN_DIR ...] [--viz-json results.json] --out report.json [--expected-n 100 --seed-base 9900000]"""
import argparse, glob, json, math, statistics, sys
from pathlib import Path
T = dict(per_seed_min_success=0.95, pooled_min_success=0.97, median_final_error_m_max=0.005, min_frac_longest_dwell_ge_50=0.95, min_mean_stay_fraction_after_first=0.95, latest_flag_below=0.90)      # PRE-REGISTERED
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [round(max(0, c - h), 4), round(min(1, c + h), 4)]
def newest(run, stem):
    f = sorted(glob.glob(str(Path(run) / "eval" / f"evaluate_{stem}_*.json")))
    return json.loads(Path(f[-1]).read_text()) | {"_file": f[-1]} if f else None
def summarize(j, n_exp, base):
    rows = j["episodes"]; seeds = [r["seed"] for r in rows]; assert len(rows) == n_exp, f"{j['_file']}: {len(rows)} episodes, expected {n_exp}"; assert seeds == [base + i for i in range(n_exp)], f"{j['_file']}: seed list differs from the planned list"
    k = sum(r["success"] for r in rows); errs = [r["final_goal_error_m"] for r in rows]
    return dict(file=j["_file"], checkpoint=j.get("checkpoint"), iteration=j.get("checkpoint_iteration"), n=len(rows), n_success=k, success_rate=k / len(rows), wilson95=wilson(k, len(rows)), median_final_error_m=statistics.median(errs), max_final_error_m=max(errs),
                failing_seeds=[r["seed"] for r in rows if not r["success"]], wm_rollout_rmse_norm=j.get("wm_rollout_rmse_norm"), latency_ms=j.get("actor_inference_latency_ms"))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--viz-json", nargs="*", default=[]); ap.add_argument("--out", required=True); ap.add_argument("--expected-n", type=int, default=100); ap.add_argument("--seed-base", type=int, default=9_900_000); a = ap.parse_args()
    if Path(a.out).exists(): sys.exit(f"refusing to overwrite {a.out}")
    res = {}; fails = []
    for run in a.runs:
        name = Path(run).parent.name; res[name] = {}
        for stem in ("best", "latest"):
            j = newest(run, stem)
            if j is None: fails.append(f"{name}: no evaluate_{stem}_*.json"); continue
            res[name][stem] = summarize(j, a.expected_n, a.seed_base)
    viz = {}
    for vj in a.viz_json:
        v = json.loads(Path(vj).read_text())
        for lab, m in v["models"].items():
            eps = m["episodes"]; viz[lab] = dict(n=len(eps), frac_longest_dwell_ge_50=sum(e["longest_dwell_steps"] >= 50 for e in eps) / len(eps), mean_stay_fraction_after_first=(statistics.mean(e["stay_fraction_after_first"] for e in eps if e["stay_fraction_after_first"] is not None) if any(e["stay_fraction_after_first"] is not None for e in eps) else None), inconsistent_flags=m["inconsistent_success_flags"], seeds=v["eval_seeds"][:2] + ["..."])
    best = [r["best"] for r in res.values() if "best" in r]; tot_k = sum(b["n_success"] for b in best); tot_n = sum(b["n"] for b in best); checks = {}
    checks["P1 each best.pt >= 95/100"] = all(b["success_rate"] >= T["per_seed_min_success"] for b in best) and len(best) == len(a.runs)
    checks["P2 pooled best.pt >= 0.97"] = bool(tot_n) and tot_k / tot_n >= T["pooled_min_success"]
    checks["S1 median final error <= 5 mm for every best.pt"] = all(b["median_final_error_m"] <= T["median_final_error_m_max"] for b in best) and len(best) == len(a.runs)
    if viz:
        checks["S2 >= 95% episodes with longest dwell >= 50 steps (each model)"] = all(v["frac_longest_dwell_ge_50"] >= T["min_frac_longest_dwell_ge_50"] for v in viz.values())
        checks["S3 mean stay fraction after first hold >= 0.95 (each model)"] = all(v["mean_stay_fraction_after_first"] is not None and v["mean_stay_fraction_after_first"] >= T["min_mean_stay_fraction_after_first"] for v in viz.values())
    else: checks["S2/S3"] = "NOT_RUN (no --viz-json)"
    prim = checks["P1 each best.pt >= 95/100"] and checks["P2 pooled best.pt >= 0.97"]; sec = all(v is True for k, v in checks.items() if k.startswith(("S1", "S2", "S3")))
    verdict = "PASS" if (prim and sec) else ("PARTIAL" if prim else "FAIL")
    if fails or any(v == "NOT_RUN (no --viz-json)" for v in checks.values()): verdict = "INCOMPLETE" if verdict == "PASS" else verdict
    flags = [f"{n}: latest.pt success {r['latest']['success_rate']:.2f} < {T['latest_flag_below']}" for n, r in res.items() if "latest" in r and r["latest"]["success_rate"] < T["latest_flag_below"]]
    rep = dict(thresholds_pre_registered=T, expected=dict(n=a.expected_n, seed_base=a.seed_base), per_run=res, pooled_best=dict(n_success=tot_k, n=tot_n, success_rate=(tot_k / tot_n if tot_n else None), wilson95=(wilson(tot_k, tot_n) if tot_n else None),
               caveat="the three runs share the same training goal sequence (train_seed_base), so pooling does not add independent replicates of the training-data variance"), visual_metrics=viz, checks=checks, missing=fails, latest_flags=flags, verdict=verdict)
    Path(a.out).write_text(json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("pooled_best", "checks", "latest_flags", "verdict", "missing")}, indent=1)); return 0
if __name__ == "__main__": sys.exit(main())
