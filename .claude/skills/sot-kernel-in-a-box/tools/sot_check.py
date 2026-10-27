#!/usr/bin/env python3
"""sot_check.py · Một điểm vào nghi thức kiểm định · generic.

Khâu 3 của phương pháp SOT: một lệnh chạy hết các phép kiểm và ghi nhật ký.
Gồm: (1) baseline check (vân tay file gốc), (2) A3 guard (hygiene.py guard —
fact chưa verify không được phơi value), (3) append verify_log.jsonl.

  python sot_check.py <org_root>

Exit: 0 = PASS · 1 = WARN · 2 = FAIL (không được publish).
Cắm vào cron/CI: chạy định kỳ, kỷ luật không phụ thuộc trí nhớ con người.
"""
import sys, os, json, subprocess, getpass, datetime
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_cfg(root):
    p = os.path.join(root, "sot.yaml")
    if not os.path.exists(p):
        sys.exit(f"FAIL: không thấy {p}")
    return yaml.safe_load(open(p, encoding="utf-8"))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: sot_check.py <org_root>")
    root = os.path.abspath(sys.argv[1])
    cfg = load_cfg(root)
    paths = cfg.get("paths", {})
    registry_dir = os.path.join(root, paths.get("registry_dir", "registry"))
    hygiene_cfg = os.path.join(root, cfg.get("hygiene_config", "sot.hygiene.yaml"))
    log = os.path.join(root, paths.get("verify_log", "index/verify_log.jsonl"))

    print(f"════════════ SOT CHECK · {cfg.get('org', os.path.basename(root))} · "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M} ════════════")
    b = subprocess.run([sys.executable, os.path.join(HERE, "sot_baseline.py"), root, "check"]).returncode
    print()
    g = subprocess.run([sys.executable, os.path.join(HERE, "hygiene.py"), "guard",
                        registry_dir, "--config", hygiene_cfg]).returncode

    worst = 2 if any(rc not in (0, 1) for rc in (b, g)) else max(b, g)
    overall = "FAIL" if worst == 2 else ("WARN" if worst == 1 else "PASS")
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "baseline_exit": b,
        "a3_guard_exit": g,
        "overall": overall,
    }
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    msg = {"PASS": "PASS · sẵn sàng gate/publish", "WARN": "WARN · có drift/cảnh báo, xem trên",
           "FAIL": "FAIL · KHÔNG được publish, sửa trước"}[overall]
    print(f"\n════════════ OVERALL: exit {worst} · {msg} · đã ghi verify_log ════════════")
    return worst


if __name__ == "__main__":
    sys.exit(main())
