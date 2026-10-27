#!/usr/bin/env python3
"""hygiene.py · Registry-hygiene toolkit (generic, config-driven) · v1.2

Tách lõi generic của phương pháp Phase A khỏi layout/từ vựng riêng. Áp cho BẤT KỲ
registry YAML nào (fact = một mapping có id), chỉ cần một file config mô tả từ vựng.

Sáu việc, đều deterministic, không gọi LLM:
  guard   · tìm fact chưa-verified_primary mà vẫn phơi `value` như sự thật cứng (A3)
  fill    · gắn provenance cho fact còn thiếu (suy từ status, hoặc evidence)
  split   · đổi `value` → `claim` + `verified_value: null` cho fact chưa verified
  access  · (v1.2) kiểm access_level + phát hiện field leak giữa các mức L1/L2/L4
  ethical · (v1.2) quét block_patterns đạo đức · flag record cần escalate
  falsify · (v1.2) kiểm cổng FALSIFICATION · tier bắt buộc phải có câu falsification

  python hygiene.py guard   <registry_dir> --config <cfg.yaml> [-v]
  python hygiene.py fill    <registry_dir> --config <cfg.yaml> [--apply]
  python hygiene.py split   <registry_dir> --config <cfg.yaml> [--apply]
  python hygiene.py access  <registry_dir> --config <cfg.yaml>
  python hygiene.py ethical <registry_dir> --config <cfg.yaml>
  python hygiene.py falsify <registry_dir> --config <cfg.yaml>

Exit code: 0 = sạch · 2 = có vi phạm (mọi lệnh kiểm). fill/split chỉ ghi khi --apply.
Không chứa dữ liệu hay đường dẫn cứng. Xem README.md + examples/ để chỉnh config.
"""
import sys, os, re, glob, shutil, datetime
import yaml as _yaml
from ruamel.yaml import YAML


def load_config(path):
    cfg = _yaml.safe_load(open(path, encoding="utf-8"))
    cfg["_primary_re"] = re.compile("|".join(cfg.get("primary_evidence_patterns", [])) or r"(?!x)x",
                                    re.IGNORECASE)
    k = cfg.setdefault("keys", {})
    k.setdefault("id", "id"); k.setdefault("status", "status"); k.setdefault("value", "value")
    k.setdefault("claim", "claim"); k.setdefault("verified_value", "verified_value")
    k.setdefault("evidence", "evidence"); k.setdefault("provenance", "provenance_level")
    return cfg


def derive_provenance(fact, cfg, explain=False):
    """Fallback chain (thứ tự cứng, không đổi):
    1. status có trong status_provenance  → mức đã map
    2. status có nhưng NGOÀI map          → default_provenance
    3. không status, evidence khớp regex  → primary_provenance
    4. không status, evidence không khớp  → default_provenance
    explain=True trả (mức, lý_do) để debug config."""
    k = cfg["keys"]
    st = str(fact.get(k["status"], "")).strip()
    smap = cfg.get("status_provenance", {})
    if st in smap:
        r = (smap[st], f"status_map[{st!r}]")
    elif st:
        r = (cfg.get("default_provenance", "claim"), f"status {st!r} NGOÀI map → default")
    else:
        ev = str(fact.get(k["evidence"], ""))
        m = cfg["_primary_re"].search(ev)
        if m:
            r = (cfg.get("primary_provenance", "verified_primary"), f"evidence khớp {m.group(0)!r}")
        else:
            r = (cfg.get("default_provenance", "claim"), "không status · evidence không khớp → default")
    return r if explain else r[0]


def keeps_value(fact, cfg):
    return derive_provenance(fact, cfg) == cfg.get("primary_provenance", "verified_primary")


def iter_facts(registry_dir, cfg):
    k = cfg["keys"]
    skip = cfg.get("skip_files", [])
    for path in sorted(glob.glob(os.path.join(registry_dir, "*.yaml"))):
        if any(s in os.path.basename(path) for s in skip):
            continue
        data = _yaml.safe_load(open(path, encoding="utf-8")) or {}
        for key in cfg.get("fact_keys", ["facts"]):
            for fact in (data.get(key) or []):
                if isinstance(fact, dict) and k["id"] in fact:
                    yield os.path.basename(path), path, key, fact


def _report(title, registry_dir, cfgname, rows, clean_msg, tail=None):
    print("═" * 60); print(f"{title} · {registry_dir} · config {cfgname}"); print("═" * 60)
    if not rows:
        print(clean_msg); return 0
    for r in rows:
        print("  ! " + r)
    print(f"\n{len(rows)} vi phạm." + (f" {tail}" if tail else ""))
    return 2


# ─── guard ───
def guard(registry_dir, cfg, cfgname, verbose=False):
    k = cfg["keys"]
    out = []
    for fname, _, _, fact in iter_facts(registry_dir, cfg):
        pl, why = derive_provenance(fact, cfg, explain=True)
        if verbose:
            print(f"  · {fact[k['id']]:42s} → {pl:18s} via {why} ({fname})")
        if pl == cfg.get("primary_provenance", "verified_primary"):
            continue
        if k["value"] in fact and fact[k["value"]] is not None:
            out.append(f"{fact[k['id']]:38s} status={str(fact.get(k['status'], '')):8s} "
                       f"value={fact[k['value']]!r} ({fname})")
    return _report("A3 GUARD", registry_dir, cfgname, out, "Sạch · 0 vi phạm.",
                   tail=f"chạy: hygiene.py split {registry_dir} --config {cfgname} --apply")


# ─── shared writer for fill/split ───
def _rewrite(registry_dir, cfg, transform, label, apply):
    k = cfg["keys"]
    yaml = YAML(); yaml.preserve_quotes = True; yaml.width = 4096
    planned, touched = [], {}
    skip = cfg.get("skip_files", [])
    for path in sorted(glob.glob(os.path.join(registry_dir, "*.yaml"))):
        if any(s in os.path.basename(path) for s in skip):
            continue
        data = yaml.load(open(path, encoding="utf-8"))
        if not data:
            continue
        changed = False
        for key in cfg.get("fact_keys", ["facts"]):
            for fact in (data.get(key) or []):
                if isinstance(fact, dict) and k["id"] in fact:
                    if transform(fact, planned, os.path.basename(path)):
                        changed = True
        if changed:
            touched[path] = data
    print("═" * 60); print(f"{label} · {'APPLY' if apply else 'DRY-RUN'} · {registry_dir}"); print("═" * 60)
    if not planned:
        print("Không có gì để đổi."); return 0
    for line in planned[:15]:
        print("  · " + line)
    if len(planned) > 15:
        print(f"  ... +{len(planned)-15} nữa")
    if not apply:
        print("\nDRY-RUN. Thêm --apply để ghi (sẽ backup)."); return 0
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = os.path.join(registry_dir, "_backup", f"{ts}_{label.lower().replace(' ','_')}")
    os.makedirs(bdir, exist_ok=True)
    for path, data in touched.items():
        shutil.copy2(path, os.path.join(bdir, os.path.basename(path)))
        yaml.dump(data, open(path, "w", encoding="utf-8"))
    print(f"\nĐÃ GHI {len(touched)} file. Backup: {bdir}")
    print(f"Restore: cp {bdir}/*.yaml {registry_dir}/")
    return 0


def fill(registry_dir, cfg, apply):
    k = cfg["keys"]
    def t(fact, planned, fname):
        if fact.get(k["provenance"]) is not None:
            return False
        pl = derive_provenance(fact, cfg)
        fact[k["provenance"]] = pl
        planned.append(f"{fact[k['id']]} → {pl} ({fname})")
        return True
    return _rewrite(registry_dir, cfg, t, "FILL PROVENANCE", apply)


def split(registry_dir, cfg, apply):
    k = cfg["keys"]
    def t(fact, planned, fname):
        if keeps_value(fact, cfg) or k["value"] not in fact:
            return False
        new = type(fact)()
        for kk, vv in fact.items():
            if kk == k["value"]:
                new[k["claim"]] = vv
                new[k["verified_value"]] = None
            elif kk == k["verified_value"]:
                continue
            else:
                new[kk] = vv
        fact.clear(); fact.update(new)
        planned.append(f"{fact[k['id']]}: {k['value']} → {k['claim']} ({fname})")
        return True
    return _rewrite(registry_dir, cfg, t, "CLAIM SPLIT", apply)


# ─── v1.2 gates ───
def _record_text(fact):
    parts = []
    def walk(v):
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(fact)
    return "\n".join(parts)


def access(registry_dir, cfg, cfgname):
    """Mỗi record phải có access_level hợp lệ; field thuộc mức nhạy hơn không được
    xuất hiện ở record mức thấp hơn (L1 leak vào L4 = BLOCK)."""
    k = cfg["keys"]
    ac = cfg.get("access_check") or {}
    field = ac.get("field", "access_level")
    levels = ac.get("levels", [])
    forbid = ac.get("forbidden_fields", {})
    out = []
    for fname, _, _, fact in iter_facts(registry_dir, cfg):
        fid = fact[k["id"]]
        lv = fact.get(field)
        if lv is None:
            out.append(f"{fid:42s} thiếu {field} ({fname})"); continue
        if levels and lv not in levels:
            out.append(f"{fid:42s} mức lạ {lv!r} · hợp lệ: {levels} ({fname})"); continue
        for f in (forbid.get(lv) or []):
            if f in fact and fact[f] is not None:
                out.append(f"{fid:42s} LEAK · field {f!r} cấm ở mức {lv} ({fname})")
    return _report("ACCESS GATE", registry_dir, cfgname, out, "Sạch · mọi record đúng mức access.",
                   tail="BLOCK export cho tới khi gỡ leak.")


def ethical(registry_dir, cfg, cfgname):
    """Quét block_patterns trên toàn bộ text của record. Match = flag, escalate."""
    k = cfg["keys"]
    eg = cfg.get("ethical_review_gate") or {}
    pats = [re.compile(p, re.IGNORECASE) for p in eg.get("block_patterns", [])]
    escalate = eg.get("escalate", "CEO")
    out = []
    for fname, _, _, fact in iter_facts(registry_dir, cfg):
        text = _record_text(fact)
        for p in pats:
            m = p.search(text)
            if m:
                out.append(f"{fact[k['id']]:42s} khớp /{p.pattern}/ · '{m.group(0)[:40]}' ({fname})")
    return _report("ETHICAL GATE", registry_dir, cfgname, out, "Sạch · không record nào chạm block_patterns.",
                   tail=f"Escalate {escalate} trước khi giữ/loại record.")


def falsify(registry_dir, cfg, cfgname):
    """Cổng FALSIFICATION: record thuộc required_tiers phải có câu falsification không rỗng."""
    k = cfg["keys"]
    fg = cfg.get("falsification_gate") or {}
    tier_key = fg.get("tier_key", "tier")
    req = fg.get("required_tiers", [])
    field = fg.get("field", "falsification")
    out = []
    for fname, _, _, fact in iter_facts(registry_dir, cfg):
        t = str(fact.get(tier_key, "")).strip()
        if req and t not in req:
            continue
        v = fact.get(field)
        if not (isinstance(v, str) and v.strip()):
            out.append(f"{fact[k['id']]:42s} tier={t or '?':3s} thiếu {field} ({fname})")
    return _report("FALSIFICATION GATE", registry_dir, cfgname, out,
                   "Sạch · mọi record bắt buộc đều có falsification.",
                   tail="Record không nêu được điều-kiện-sai thì chưa đủ chuẩn vào registry.")


CMDS = ("guard", "fill", "split", "access", "ethical", "falsify")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in CMDS:
        sys.exit(f"usage: hygiene.py <{'|'.join(CMDS)}> <registry_dir> --config <cfg.yaml> [--apply] [-v]")
    cmd, reg = sys.argv[1], sys.argv[2]
    cfgpath = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hygiene.yaml")
    cfg = load_config(cfgpath)
    cfgname = os.path.basename(cfgpath)
    apply = "--apply" in sys.argv
    if cmd == "guard":
        return guard(reg, cfg, cfgname, verbose="-v" in sys.argv)
    if cmd == "fill":
        return fill(reg, cfg, apply)
    if cmd == "split":
        return split(reg, cfg, apply)
    if cmd == "access":
        return access(reg, cfg, cfgname)
    if cmd == "ethical":
        return ethical(reg, cfg, cfgname)
    return falsify(reg, cfg, cfgname)


if __name__ == "__main__":
    sys.exit(main())
