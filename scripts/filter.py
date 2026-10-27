#!/usr/bin/env python3
"""Filter records.jsonl by topic/priority -> data/shortlist.md. Usage: filter.py [topic ...]  (default: openarm humanoid bimanual)"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
want = set(sys.argv[1:]) or {"openarm","humanoid","bimanual"}
recs = [json.loads(l) for l in open(ROOT/"data/records.jsonl")]
hit = [r for r in recs if want & set(r["_topics"]) or r["category"]=="A_platform"]
hit.sort(key=lambda r:(r["priority"], r["category"], r["name"]))
rows = ["| Pri | Name | Topics | License | Pushed | Fetch |","|---|---|---|---|---|---|"]
for r in hit:
    g = r.get("_gh") or {}
    rows.append(f"| {r['priority']} | [{r['name']}]({r['url']}) | {','.join(r['_topics']) or '-'} | {r['license'] or 'unknown'} | {(g.get('pushed_at') or 'unknown')[:10]} | {r['_fetch'][:20]} |")
(ROOT/"data/shortlist.md").write_text(f"# Shortlist ({', '.join(sorted(want))}) — {len(hit)}/{len(recs)}\n\n"+"\n".join(rows)+"\n")
print(len(hit), "of", len(recs), "kept")
