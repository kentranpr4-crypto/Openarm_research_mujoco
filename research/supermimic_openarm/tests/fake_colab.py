#!/usr/bin/env python3
"""TEST DOUBLE of the Colab CLI: executes the commands locally (the machine plays the VM; remote paths are real local paths).  Used only by tests/run_t4_driver_selftest.py.  It prints a canned Tesla T4 `nvidia-smi` for the bare command.  It is NOT the real CLI and proves nothing about Colab."""
import os, shutil, subprocess, sys
from pathlib import Path
PY = os.environ.get("FAKE_COLAB_PYTHON", sys.executable)
def main():
    a = sys.argv[1:]
    if not a: return 2
    cmd = a[0]
    if cmd == "sessions": print("openarm-test\tT4\tactive"); return 0
    if cmd == "new": print("created"); return 0
    if cmd == "stop": Path(os.environ.get("FAKE_COLAB_STOPPED", "/tmp/fake_colab_stopped")).write_text("stopped"); print("stopped"); return 0
    if cmd == "upload": Path(a[-1]).parent.mkdir(parents=True, exist_ok=True); shutil.copy(a[-2], a[-1]); return 0
    if cmd == "download": Path(a[-1]).parent.mkdir(parents=True, exist_ok=True); shutil.copy(a[-2], a[-1]); return 0
    if cmd == "ssh":
        line = a[a.index("--") + 1:]; text = line[-1] if line[:2] == ["bash", "-lc"] else " ".join(line)
        if text.strip() == "nvidia-smi": print("| Tesla T4 | 00000000:00:04.0 Off | 0MiB / 15360MiB |"); return 0
        r = subprocess.run(["bash", "-c", text.replace("python3 ", PY + " ")], capture_output=True, text=True); sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); return r.returncode
    if cmd == "exec":
        f = a[a.index("-f") + 1]; env = dict(os.environ); i = 0
        while i < len(a):
            if a[i] == "--env": k, v = a[i + 1].split("=", 1); env[k] = v; i += 2
            else: i += 1
        r = subprocess.run([PY, f], env=env, capture_output=True, text=True); sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); return r.returncode
    print("unsupported", a, file=sys.stderr); return 2
if __name__ == "__main__": sys.exit(main())
