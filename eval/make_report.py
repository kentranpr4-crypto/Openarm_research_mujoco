#!/usr/bin/env python3
"""Merge expert / ACT evaluation summaries into one markdown report with success AND failed rollouts listed.
Usage: python eval/make_report.py outputs/eval/<run> [more runs ...]   (reads <run>/summary_{expert,act}.json and per-condition episodes.jsonl)"""
import sys, json
from pathlib import Path

def fmt_rate(s, m): lo, hi = s[f"{m}_ci95"]; return f"{100*s[f'{m}_rate']:.0f}% [{100*lo:.0f}-{100*hi:.0f}]"

def main(runs):
    out = []
    for run in map(Path, runs):
        S = {a: json.loads((run / f"summary_{a}.json").read_text()) for a in ("expert", "act") if (run / f"summary_{a}.json").exists()}
        if not S: continue
        out.append(f"## {run.name}\n"); cfg = next(iter(S.values()))["config"]
        out.append(f"n per condition = {cfg['n']}, seed = {cfg['seed']}, horizon = {cfg['horizon']} steps @ {cfg['hz']} Hz, render profile = {cfg['render_profile']}. Rates are k/n with Wilson 95% CI in brackets.\n")
        out.append("| condition | agent | n | pick | place | task | tracking mean (rad) | mean abs jerk | latency ms p50/p95/p99 (per step) |\n|---|---|---|---|---|---|---|---|---|")
        for cond in next(iter(S.values()))["conditions"]:
            for a, d in S.items():
                s = d["conditions"].get(cond)
                if not s: continue
                lat = s.get("latency_ms_per_step"); lat = f"{lat['p50']:.1f}/{lat['p95']:.1f}/{lat['p99']:.1f}" if lat else "-"
                out.append(f"| {cond} | {a} | {s['n']} | {fmt_rate(s,'pick')} | {fmt_rate(s,'place')} | {fmt_rate(s,'task')} | {s['tracking_mean_arm_rad']:.4f} | {s['mean_abs_jerk']:.2f} | {lat} |")
        out.append("\n### Rollout outcomes (successes and failures)\n")
        for a, d in S.items():
            for cond in d["conditions"]:
                ep = run / a / cond / "episodes.jsonl"
                if not ep.exists(): continue
                recs = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
                bad = [r for r in recs if r["outcome"] != "success"]
                out.append(f"- **{a}/{cond}**: {len(recs)-len(bad)}/{len(recs)} success; outcomes {d['conditions'][cond]['outcomes']}")
                for r in bad[:12]:
                    out.append(f"  - ep {r['episode_index']} `{r['outcome']}` box=({r['box_xy'][0]:.3f},{r['box_xy'][1]:.3f}) yaw={r['box_yaw']:+.2f} max_rise={r['max_rise_m']:.3f} m, tray_dxy={r['final_box_to_tray_xy_m']:.3f} m  video: `{(run/a/cond/'videos').as_posix()}/ep{r['episode_index']:04d}_{r['outcome']}.mp4`")
        out.append("")
    text = "\n".join(out); (Path(runs[0]) / "report.md").write_text(text); print(text)

if __name__ == "__main__":
    main(sys.argv[1:])
