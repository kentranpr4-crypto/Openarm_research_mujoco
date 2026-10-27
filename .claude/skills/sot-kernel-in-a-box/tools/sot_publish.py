#!/usr/bin/env python3
"""sot_publish.py · Cửa xuất hàng (publish gate) · generic.

Khâu 4 của phương pháp SOT. Cửa kiểm tại điểm copy file; không thay phân quyền hệ thống hoặc review nội dung.
Token được tạo sau khi sot_check PASS và gắn digest của registry + inputs tại thời
điểm đó. Registry đổi một byte sau khi kiểm → token hết hiệu lực → publish bị chặn.

  python sot_publish.py <org_root> gate                 # kiểm; PASS → ghi token; FAIL → xoá token
  python sot_publish.py <org_root> status               # token còn hiệu lực không
  python sot_publish.py <org_root> publish <src> <dst>  # copy CHỈ khi token hợp lệ + ghi log

Exit: 0 = OK · 2 = BLOCKED.
"""
import sys, os, json, glob, hashlib, subprocess, datetime, shutil, getpass
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
from sot_baseline import scan_sources, load_cfg as load_baseline_cfg


def load_cfg(root):
    p = os.path.join(root, "sot.yaml")
    if not os.path.exists(p):
        sys.exit(f"FAIL: không thấy {p}")
    cfg = yaml.safe_load(open(p, encoding="utf-8"))
    paths = cfg.get("paths", {})
    cfg["_registry_dir"] = os.path.join(root, paths.get("registry_dir", "registry"))
    cfg["_token"] = os.path.join(root, paths.get("token_file", "index/.publish_token"))
    cfg["_log"] = os.path.join(root, paths.get("verify_log", "index/verify_log.jsonl"))
    cfg["_digest_exclude"] = cfg.get("digest_exclude", ["source_hashes"])
    cfg["_root"] = root
    return cfg


def registry_digest(cfg):
    """Digest toàn bộ YAML registry (đổi 1 byte → đổi digest). Loại file baseline."""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(cfg["_registry_dir"], "*.yaml"))):
        if any(s in os.path.basename(p) for s in cfg["_digest_exclude"]):
            continue
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def check_token(cfg):
    if not os.path.exists(cfg["_token"]):
        return False, "không có token (chưa qua gate)"
    try:
        tok = json.load(open(cfg["_token"]))
    except (ValueError, OSError):
        return False, "token hỏng — chạy gate lại"
    if not isinstance(tok, dict) or tok.get("check") != "PASS":
        return False, "token không có kết quả PASS"
    if tok.get("digest") != registry_digest(cfg):
        return False, "registry đã đổi sau khi kiểm (token hết hiệu lực — chạy gate lại)"
    if tok.get("input_digest") != input_digest(cfg):
        return False, "nguồn/config/adapter/baseline đã đổi sau gate hoặc token cũ — kiểm lại"
    return True, f"token hợp lệ · gate lúc {tok.get('timestamp')}"


def input_digest(cfg):
    """Bind canonical contents, baseline, configuration and local adapter code.

    A token is an integrity marker, not an authorization credential or proof of
    semantic correctness of arbitrary exported documents.
    """
    root = cfg["_root"]
    baseline_cfg = load_baseline_cfg(root)
    sources = {name: row["sha256"] for name, row in scan_sources(baseline_cfg).items()}
    h = hashlib.sha256(json.dumps(sources, sort_keys=True).encode())
    paths = [os.path.join(root, "sot.yaml"),
             os.path.join(root, cfg.get("hygiene_config", "sot.hygiene.yaml")),
             baseline_cfg["_hashes"]]
    paths += sorted(glob.glob(os.path.join(root, "adapters", "**", "*.py"), recursive=True))
    for path in paths:
        h.update(os.path.relpath(path, root).encode())
        h.update(open(path, "rb").read() if os.path.isfile(path) else b"MISSING")
    return h.hexdigest()


def cmd_gate(cfg, root):
    before = (registry_digest(cfg), input_digest(cfg))
    rc = subprocess.run([sys.executable, os.path.join(HERE, "sot_check.py"), root]).returncode
    after = (registry_digest(cfg), input_digest(cfg))
    if before != after:
        rc = 2
        print("FAIL: dữ liệu thay đổi trong lúc kiểm — chạy lại trên bản ổn định")
    if rc == 0:
        os.makedirs(os.path.dirname(cfg["_token"]), exist_ok=True)
        json.dump({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                   "digest": after[0], "input_digest": after[1], "check": "PASS"},
                  open(cfg["_token"], "w"))
        print(f"\nPUBLISH GATE PASS · token đã ghi · digest {registry_digest(cfg)}")
        return 0
    if os.path.exists(cfg["_token"]):
        os.remove(cfg["_token"])
    print(f"\nPUBLISH GATE FAIL (check exit {rc}) · token đã xoá · KHÔNG được publish")
    return 2


def cmd_status(cfg):
    ok, msg = check_token(cfg)
    print(("PUBLISH OK · " if ok else "PUBLISH BLOCKED · ") + msg)
    return 0 if ok else 2


def cmd_publish(cfg, src, dst):
    ok, msg = check_token(cfg)
    if not ok:
        print(f"PUBLISH BLOCKED · {msg}")
        print("→ Chạy: sot_publish.py <org_root> gate  (sau khi check PASS) rồi thử lại.")
        return 2
    if not os.path.exists(src):
        print(f"FAIL: nguồn không tồn tại: {src}")
        return 2
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    shutil.copy2(src, dst)
    entry = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
             "user": getpass.getuser(), "event": "publish",
             "src": os.path.basename(src), "dst": dst, "digest": registry_digest(cfg)}
    os.makedirs(os.path.dirname(cfg["_log"]), exist_ok=True)
    with open(cfg["_log"], "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Đã publish: {src} → {dst} (digest {entry['digest']} · đã ghi log)")
    return 0


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in ("gate", "status", "publish"):
        sys.exit("usage: sot_publish.py <org_root> <gate|status|publish> [src dst]")
    root = os.path.abspath(sys.argv[1])
    cfg = load_cfg(root)
    cmd = sys.argv[2]
    if cmd == "gate":
        return cmd_gate(cfg, root)
    if cmd == "status":
        return cmd_status(cfg)
    if len(sys.argv) < 5:
        sys.exit("usage: sot_publish.py <org_root> publish <src> <dst>")
    return cmd_publish(cfg, sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    sys.exit(main())
