#!/usr/bin/env python3
"""Collect public metadata for sources.csv -> data/records.jsonl.
Rule: missing = null (unknown), never 0. Reported vs local results are not mixed here.
Optional GITHUB_TOKEN raises the API rate limit (60/h unauthenticated)."""
import csv, json, os, re, sys, urllib.request, urllib.error, datetime, html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TODAY = datetime.date.today().isoformat()
SCHEMA = ["name","url","category","priority","source_kind","license","hardware_support","camera_views",
 "state_dim","action_dim","action_units","control_rate_hz","observation_horizon","action_horizon",
 "tokenizer_or_encoder","trainable_blocks","loss","dataset","training_compute","inference_nfe",
 "latency_hw","simulator","evaluation_metrics","released_code","released_weights","reproducibility",
 "known_limitations","checked_date","open_questions"]
TOPICS = {"openarm":r"open[\s_-]?arm","humanoid":r"humanoid","bimanual":r"bimanual|bi-manual|dual[\s-]arm|aloha",
          "vla":r"\bvla\b|vision-language-action","world_model":r"world model|world-model|jepa|dreamer"}

def get(url, accept=None):
    h = {"User-Agent":"openarm-humanoid-intel/0.1"}
    if accept: h["Accept"] = accept
    if os.environ.get("GITHUB_TOKEN") and "api.github.com" in url:
        h["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
        return r.read().decode("utf-8","replace")

def github(owner, repo):
    meta = json.loads(get(f"https://api.github.com/repos/{owner}/{repo}"))
    readme = ""
    for br in (meta.get("default_branch") or "main", "main", "master"):
        try: readme = get(f"https://raw.githubusercontent.com/{owner}/{repo}/{br}/README.md"); break
        except Exception: pass
    (ROOT/"data/raw"/f"{owner}__{repo}.json").write_text(json.dumps(meta, indent=1))
    lic = (meta.get("license") or {}).get("spdx_id")
    return {"license":lic if lic not in (None,"NOASSERTION") else None, "released_code":True,
            "simulator":None, "_gh":{k:meta.get(k) for k in ("stargazers_count","pushed_at","archived","description","topics","default_branch")}}, readme

def web(url):
    txt = get(url)
    title = re.search(r"<title[^>]*>(.*?)</title>", txt, re.S|re.I)
    body = html.unescape(re.sub(r"<[^>]+>"," ",re.sub(r"(?s)<(script|style).*?</\1>","",txt)))
    return {"_title": title.group(1).strip() if title else None}, body

def main():
    prev = {}
    f = ROOT/"data/records.jsonl"
    if f.exists():  # resume: keep records that already fetched ok
        prev = {r["url"]:r for r in map(json.loads, open(f)) if r.get("_fetch")=="ok"}
    out = []
    for row in csv.DictReader(open(ROOT/"data/sources.csv")):
        if row["url"] in prev: out.append(prev[row["url"]]); continue
        rec = {k:None for k in SCHEMA}; rec.update(row); rec["checked_date"] = TODAY
        text = ""
        try:
            m = re.match(r"https://github\.com/([^/]+)/([^/#?]+)", row["url"])
            if m: extra, text = github(m[1], m[2])
            elif row["source_kind"] in ("web","arxiv","hf"): extra, text = web(row["url"])
            else: extra = {}
            rec.update(extra)
            rec["_fetch"] = "ok"
        except Exception as e:
            rec["_fetch"] = f"error: {type(e).__name__} {e}"
        low = (text + " " + row["name"] + " " + json.dumps(rec.get("_gh") or {})).lower()
        rec["_topics"] = [t for t,p in TOPICS.items() if re.search(p, low)]
        out.append(rec); print(rec["_fetch"][:40].ljust(40), row["name"], file=sys.stderr)
    with open(ROOT/"data/records.jsonl","w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"{len(out)} records -> data/records.jsonl")

if __name__ == "__main__": main()
