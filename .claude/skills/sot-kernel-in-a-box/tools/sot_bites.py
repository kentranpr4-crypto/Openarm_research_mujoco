#!/usr/bin/env python3
"""sot_bites.py · Bộ răng: chứng minh mọi cổng của SOT biết cắn · generic.

Doctrine: "Tin cổng sau khi thấy nó cắn." Script này copy org ra bản tạm,
tiêm từng loại lỗi, xác nhận cổng tương ứng trả exit 2, rồi vứt bản tạm.
Một cổng chưa được chứng cắn thì chưa tin.

  python sot_bites.py <org_root>

Sáu răng:
  B1 SOURCE_TAMPER · sửa file gốc          → sot_baseline check phải FAIL (2)
  B2 A3_EXPOSE     · fact chưa-verify phơi value → hygiene guard + gate phải FAIL (2)
  B3 STALE_TOKEN   · registry đổi sau gate → publish phải BLOCKED (2)
  B4 NO_TOKEN      · chưa gate             → publish phải BLOCKED (2)
  B5 SOURCE_AFTER_GATE · nguồn đổi sau gate → status phải BLOCKED (2)
  B6 POLICY_AFTER_GATE · policy đổi sau gate → status phải BLOCKED (2)

Exit 0 = tất cả răng cắn · 2 = có răng KHÔNG cắn (đừng tin hệ này).
"""
import sys, os, glob, shutil, subprocess, tempfile
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(*args):
    return subprocess.run(list(args), capture_output=True, text=True).returncode


def fresh_copy(org_root):
    tmp = tempfile.mkdtemp(prefix="sot_bite_")
    dst = os.path.join(tmp, "org")
    shutil.copytree(org_root, dst)
    return tmp, dst


def find_one_source(dst):
    cfg = yaml.safe_load(open(os.path.join(dst, "sot.yaml"), encoding="utf-8"))
    sdir = os.path.join(dst, cfg.get("paths", {}).get("source_dir", "source_canonical"))
    files = [p for p in glob.glob(os.path.join(sdir, "**", "*"), recursive=True)
             if os.path.isfile(p) and not os.path.basename(p).startswith(".")
             and os.path.basename(p) != "README.md"]
    return files[0]


def find_registry_yaml(dst):
    cfg = yaml.safe_load(open(os.path.join(dst, "sot.yaml"), encoding="utf-8"))
    rdir = os.path.join(dst, cfg.get("paths", {}).get("registry_dir", "registry"))
    return [p for p in glob.glob(os.path.join(rdir, "*.yaml"))
            if "source_hashes" not in os.path.basename(p)][0]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: sot_bites.py <org_root>")
    org = os.path.abspath(sys.argv[1])
    results = []

    # Điều kiện tiên quyết: org sạch (gate PASS được) — bites chỉ có nghĩa trên nền sạch
    tmp, dst = fresh_copy(org)
    pre = run(PY, os.path.join(HERE, "sot_publish.py"), dst, "gate")
    shutil.rmtree(tmp)
    if pre != 0:
        sys.exit("FAIL: org chưa sạch (gate không PASS) — sửa org trước rồi mới chạy bites.")

    # B1 SOURCE_TAMPER
    tmp, dst = fresh_copy(org)
    with open(find_one_source(dst), "ab") as f:
        f.write(b"\n# tampered")
    rc = run(PY, os.path.join(HERE, "sot_baseline.py"), dst, "check")
    results.append(("B1 SOURCE_TAMPER  · sửa file gốc → baseline check", rc == 2))
    shutil.rmtree(tmp)

    # B2 A3_EXPOSE
    tmp, dst = fresh_copy(org)
    reg = find_registry_yaml(dst)
    data = yaml.safe_load(open(reg, encoding="utf-8"))
    key = next(k for k in data if isinstance(data[k], list))
    data[key].append({"id": "BITE.A3.EXPOSED", "key": "fact tiêm thử",
                      "value": 999, "status": "INT", "source": "bite"})
    yaml.safe_dump(data, open(reg, "w", encoding="utf-8"), allow_unicode=True)
    rc_gate = run(PY, os.path.join(HERE, "sot_publish.py"), dst, "gate")
    results.append(("B2 A3_EXPOSE      · fact INT phơi value → gate", rc_gate == 2))
    shutil.rmtree(tmp)

    # B3 STALE_TOKEN
    tmp, dst = fresh_copy(org)
    run(PY, os.path.join(HERE, "sot_publish.py"), dst, "gate")          # PASS, có token
    reg = find_registry_yaml(dst)
    with open(reg, "a", encoding="utf-8") as f:
        f.write("\n# registry doi sau gate\n")
    rc = run(PY, os.path.join(HERE, "sot_publish.py"), dst, "status")
    results.append(("B3 STALE_TOKEN    · registry đổi sau gate → publish", rc == 2))
    shutil.rmtree(tmp)

    # B4 NO_TOKEN
    tmp, dst = fresh_copy(org)
    cfgd = yaml.safe_load(open(os.path.join(dst, "sot.yaml"), encoding="utf-8"))
    tok = os.path.join(dst, cfgd.get("paths", {}).get("token_file", "index/.publish_token"))
    if os.path.exists(tok):
        os.remove(tok)
    rc = run(PY, os.path.join(HERE, "sot_publish.py"), dst, "status")
    results.append(("B4 NO_TOKEN       · chưa gate → publish", rc == 2))
    shutil.rmtree(tmp)

    # B5: source changed after a successful gate, without changing registry.
    tmp, dst = fresh_copy(org)
    run(PY, os.path.join(HERE, "sot_publish.py"), dst, "gate")
    with open(find_one_source(dst), "ab") as f:
        f.write(b"\n# changed after gate")
    rc = run(PY, os.path.join(HERE, "sot_publish.py"), dst, "status")
    results.append(("B5 SOURCE_AFTER_GATE · nguồn đổi sau gate → chặn", rc == 2))
    shutil.rmtree(tmp)

    # B6: a changed hygiene policy invalidates the previously issued token.
    tmp, dst = fresh_copy(org)
    run(PY, os.path.join(HERE, "sot_publish.py"), dst, "gate")
    cfgd = yaml.safe_load(open(os.path.join(dst, "sot.yaml"), encoding="utf-8"))
    hcfg = os.path.join(dst, cfgd.get("hygiene_config", "sot.hygiene.yaml"))
    with open(hcfg, "a", encoding="utf-8") as f:
        f.write("\n# policy changed\n")
    rc = run(PY, os.path.join(HERE, "sot_publish.py"), dst, "status")
    results.append(("B6 POLICY_AFTER_GATE · config đổi sau gate → chặn", rc == 2))
    shutil.rmtree(tmp)

    print("─" * 60)
    ok = True
    for name, bit in results:
        print(f"  {'✓ CẮN' if bit else '✗ KHÔNG CẮN'}  {name}")
        ok = ok and bit
    print("─" * 60)
    print("BITE SUITE: TẤT CẢ RĂNG CẮN ✓" if ok else "BITE SUITE: CÓ RĂNG KHÔNG CẮN ✗ — đừng tin hệ này")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
