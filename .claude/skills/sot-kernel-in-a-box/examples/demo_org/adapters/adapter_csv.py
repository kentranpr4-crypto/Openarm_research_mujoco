#!/usr/bin/env python3
"""adapter_csv.py · Adapter MẪU: CSV → registry/facts.yaml (khâu 1 · máy xay).

Adapter là mảnh DUY NHẤT mỗi tổ chức tự viết, vì file gốc mỗi nơi một kiểu
(Excel giá, Word chính sách, export ERP...). Hợp đồng của một adapter:

  INPUT : file trong source_canonical/ (KHÔNG BAO GIỜ sửa file gốc)
  OUTPUT: registry/<ten>.yaml dạng {facts: [{id, key, value, unit, status, evidence, source}, ...]}
  LUẬT  : - mỗi fact PHẢI có id duy nhất + status (theo từ vựng sot.hygiene.yaml)
          - KHÔNG bịa giá trị · KHÔNG đổi đơn vị âm thầm · thiếu thì để trống
          - deterministic: chạy 2 lần cùng input → cùng output
          - sau khi chạy adapter: hygiene fill + split sẽ ép A3 (INT/DRAFT mất value)

CSV mẫu cần cột: id,key,value,unit,status,evidence
  python adapter_csv.py <org_root> <ten_file.csv>     # đọc source_canonical/<ten_file.csv>
"""
import sys, os, csv
import yaml


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: adapter_csv.py <org_root> <ten_file.csv>")
    root = os.path.abspath(sys.argv[1])
    name = sys.argv[2]
    src = os.path.join(root, "source_canonical", name)
    if not os.path.exists(src):
        sys.exit(f"FAIL: không thấy {src}")

    facts, seen = [], set()
    with open(src, encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            fid = (row.get("id") or "").strip()
            if not fid:
                sys.exit(f"FAIL dòng {i}: thiếu id — adapter không bịa id.")
            if fid in seen:
                sys.exit(f"FAIL dòng {i}: id trùng: {fid}")
            seen.add(fid)
            fact = {"id": fid,
                    "key": (row.get("key") or "").strip(),
                    "value": _num(row.get("value")),
                    "unit": (row.get("unit") or "").strip(),
                    "status": (row.get("status") or "").strip(),
                    "evidence": (row.get("evidence") or "").strip(),
                    "source": {"file": f"source_canonical/{name}", "row": i}}
            if not fact["status"]:
                sys.exit(f"FAIL dòng {i}: thiếu status — mọi fact phải khai mức tin cậy.")
            facts.append(fact)

    out = os.path.join(root, "registry",
                       os.path.splitext(name)[0].lower().replace(" ", "_") + ".yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    yaml.safe_dump({"generated_by": "adapter_csv.py", "source": name, "facts": facts},
                   open(out, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print(f"ADAPTER OK · {len(facts)} fact → {os.path.relpath(out, root)}")
    print("Tiếp theo (bắt buộc): hygiene fill --apply → hygiene split --apply → sot_check")
    return 0


def _num(v):
    v = (v or "").strip()
    if v == "":
        return None
    try:
        return int(v) if float(v) == int(float(v)) else float(v)
    except ValueError:
        return v


if __name__ == "__main__":
    sys.exit(main())
