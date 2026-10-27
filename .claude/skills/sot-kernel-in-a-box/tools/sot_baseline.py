#!/usr/bin/env python3
"""sot_baseline.py · Vân tay file gốc (hash baseline) · generic, config-driven.

Khâu 2 của phương pháp SOT: mỗi file gốc (source canonical) có một dấu vân tay
sha256. File gốc bị sửa mà registry chưa cập nhật → CHANGED → FAIL. Nhờ đó
"file gốc và kho sự thật không bao giờ lệch nhau trong im lặng".

  python sot_baseline.py <org_root> init            # tạo baseline lần đầu
  python sot_baseline.py <org_root> check           # đối chiếu (0 sạch nội dung · 1 nguồn mới · 2 fail)
  python sot_baseline.py <org_root> accept [--only <ten_file>]   # nhận bản hiện tại làm baseline mới (có backup)

<org_root> là thư mục tổ chức/dự án chứa sot.yaml. Không hardcode đường dẫn,
không chứa dữ liệu của tổ chức nào. Exit: 0 = sạch · 1 = warn (NEW) · 2 = fail (CHANGED/MISSING).
"""
import sys, os, glob, json, hashlib, shutil, datetime
import yaml


def load_cfg(root):
    p = os.path.join(root, "sot.yaml")
    if not os.path.exists(p):
        sys.exit(f"FAIL: không thấy {p} — org_root phải chứa sot.yaml (xem project_template/)")
    cfg = yaml.safe_load(open(p, encoding="utf-8"))
    paths = cfg.get("paths", {})
    cfg["_source_dir"] = os.path.join(root, paths.get("source_dir", "source_canonical"))
    cfg["_hashes"] = os.path.join(root, paths.get("hashes_file", "registry/source_hashes.yaml"))
    cfg["_backup_dir"] = os.path.join(root, paths.get("backup_dir", "registry/_backup"))
    return cfg


def scan_sources(cfg):
    out = {}
    base = cfg["_source_dir"]
    for p in sorted(glob.glob(os.path.join(base, "**", "*"), recursive=True)):
        rel = os.path.relpath(p, base)
        if not os.path.isfile(p) or os.path.basename(p).startswith(".") or rel == "README.md":
            continue
        b = open(p, "rb").read()
        out[rel] = {
            "sha256": hashlib.sha256(b).hexdigest()[:16],
            "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
            "size": len(b),
        }
    return out


def read_baseline(cfg):
    if not os.path.exists(cfg["_hashes"]):
        return None
    return (yaml.safe_load(open(cfg["_hashes"], encoding="utf-8")) or {}).get("sources", {})


def write_baseline(cfg, sources, note):
    os.makedirs(os.path.dirname(cfg["_hashes"]), exist_ok=True)
    data = {"updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "note": note, "sources": sources}
    yaml.safe_dump(data, open(cfg["_hashes"], "w", encoding="utf-8"),
                   allow_unicode=True, sort_keys=True)


def cmd_init(cfg, root):
    if read_baseline(cfg) is not None and "--force" not in sys.argv:
        sys.exit("FAIL: baseline đã tồn tại. Dùng 'check'/'accept', hoặc --force nếu chắc chắn làm lại.")
    src = scan_sources(cfg)
    if not src:
        sys.exit(f"FAIL: {cfg['_source_dir']} chưa có file gốc nào.")
    write_baseline(cfg, src, "init")
    print(f"BASELINE INIT · {len(src)} file gốc đã lấy vân tay → {os.path.relpath(cfg['_hashes'], root)}")
    return 0


def cmd_check(cfg, root):
    base = read_baseline(cfg)
    if base is None:
        sys.exit("FAIL: chưa có baseline. Chạy: sot_baseline.py <org_root> init")
    cur = scan_sources(cfg)
    changed = [n for n in base if n in cur and cur[n]["sha256"] != base[n]["sha256"]]
    missing = [n for n in base if n not in cur]
    new = [n for n in cur if n not in base]
    drift = [n for n in base if n in cur and cur[n]["sha256"] == base[n]["sha256"]
             and cur[n]["mtime"] != base[n]["mtime"]]
    print("═" * 60); print(f"BASELINE CHECK · {os.path.relpath(cfg['_source_dir'], root)}"); print("═" * 60)
    for n in changed:
        print(f"  ! CHANGED  {n}  {base[n]['sha256']} → {cur[n]['sha256']}  (re-atomize + accept)")
    for n in missing:
        print(f"  ! MISSING  {n}  (file gốc biến mất)")
    for n in new:
        print(f"  ⚠ NEW      {n}  (chưa có vân tay — atomize rồi accept)")
    for n in drift:
        print(f"  ⚠ DRIFT    {n}  (mtime đổi, nội dung y nguyên — accept để re-baseline)")
    if not (changed or missing or new):
        if drift:
            print(f"INFO · {len(drift)} mtime khác nhưng nội dung khớp (copy/unzip không làm mất hiệu lực).")
        print(f"Sạch · {len(base)} file gốc khớp vân tay.")
        return 0
    if changed or missing:
        print(f"\nFAIL · {len(changed)} changed + {len(missing)} missing. Nghi thức: cập nhật registry từ "
              f"file gốc mới (adapter/atomize) → 'accept' → 'check' lại.")
        return 2
    print(f"\nWARN · {len(new)} nguồn mới chưa có baseline.")
    return 1


def cmd_accept(cfg, root):
    base = read_baseline(cfg) or {}
    cur = scan_sources(cfg)
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    if only:
        if only not in cur:
            sys.exit(f"FAIL: '{only}' không có trong {cfg['_source_dir']}")
        base[only] = cur[only]
        newbase = base
    else:
        newbase = cur
    if os.path.exists(cfg["_hashes"]):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bdir = os.path.join(cfg["_backup_dir"], f"{ts}_baseline")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(cfg["_hashes"], os.path.join(bdir, os.path.basename(cfg["_hashes"])))
        print(f"Backup baseline cũ → {os.path.relpath(bdir, root)}")
    write_baseline(cfg, newbase, f"accept{' --only ' + only if only else ''}")
    print(f"BASELINE ACCEPT · {len(newbase)} file gốc · vân tay mới đã ghi.")
    return 0


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in ("init", "check", "accept"):
        sys.exit("usage: sot_baseline.py <org_root> <init|check|accept> [--only <file>] [--force]")
    root = os.path.abspath(sys.argv[1])
    cfg = load_cfg(root)
    return {"init": cmd_init, "check": cmd_check, "accept": cmd_accept}[sys.argv[2]](cfg, root)


if __name__ == "__main__":
    sys.exit(main())
