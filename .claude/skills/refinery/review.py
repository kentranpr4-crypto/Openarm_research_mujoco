#!/usr/bin/env python3
"""review.py — TIP-NEWS-REALTIME Task 4 · cổng NGƯỜI cho dòng R (refined-fact).

Auto-draft claim đậu ở domains/<d>/staging.jsonl (đã qua gate refinery SPAN/tier nhưng CHƯA vào
claims.jsonl). review cho người xem evidence_span + snapshot rồi DUYỆT/LOẠI từng claim. Duyệt =
human_verified:true + verified_by → chuyển sang claims.jsonl. Đây là 'mắt người' bắt quote-out-of-context
mà gate cấu-trúc mù (Contract §1.4). Export chặn auto-draft chưa duyệt bằng R_EXPORT_UNVERIFIED.

  python3 review.py <domain> list
  python3 review.py <domain> approve <idx> --by "Tên người duyệt"
  python3 review.py <domain> reject  <idx>
"""
import json, sys, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _write(p, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("cmd", choices=["list", "approve", "reject"])
    ap.add_argument("idx", nargs="?", type=int)
    ap.add_argument("--by", default="")
    a = ap.parse_args()

    d = HERE / "domains" / a.domain
    if not d.exists():
        sys.exit(f"domain không tồn-tại: {a.domain}")
    stag, claims = d / "staging.jsonl", d / "claims.jsonl"
    rows = _load(stag)

    if a.cmd == "list":
        if not rows:
            print(f"staging {a.domain}: rỗng (0 claim chờ duyệt)"); return
        print(f"staging {a.domain}: {len(rows)} claim auto-draft chờ NGƯỜI xác ngữ-cảnh")
        for i, c in enumerate(rows):
            cap = c.get("capture") or {}
            print(f"  [{i}] {str(c.get('entity','?'))[:52]} · {c.get('field')} = {str(c.get('value'))[:38]!r}")
            print(f"      span={str(c.get('evidence_span'))[:64]!r} · tier {c.get('tier')} · snap {cap.get('snapshot')}")
        return

    if a.idx is None or not (0 <= a.idx < len(rows)):
        sys.exit("idx không hợp-lệ (xem 'list')")
    c = rows.pop(a.idx)
    if a.cmd == "approve":
        if not a.by:
            sys.exit("approve cần --by \"Tên người duyệt\" (claim danh-tính trách-nhiệm, không trống)")
        c["human_verified"] = True
        c["verified_by"] = a.by
        cl = _load(claims); cl.append(c); _write(claims, cl)
        print(f"✓ DUYỆT [{a.idx}] → claims.jsonl · verified_by={a.by!r} · còn {len(rows)} trong staging")
    else:
        print(f"✗ LOẠI [{a.idx}] khỏi staging · còn {len(rows)}")
    _write(stag, rows)


if __name__ == "__main__":
    main()
