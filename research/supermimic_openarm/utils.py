"""Seeding, reproducibility metadata, atomic IO, hashing.  Standard library + numpy (+ torch lazily)."""
import hashlib, json, os, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
PROTECTED_SEEDS = frozenset({31, 32, 33})          # reserved ACT test seeds of the existing study: never used here
def assert_seeds_allowed(*seeds):
    bad = [s for s in seeds if int(s) in PROTECTED_SEEDS]
    if bad: raise ValueError(f"seed(s) {bad} are reserved ACT test seeds (31-33) and must not be used")
def seed_everything(seed):
    assert_seeds_allowed(seed); random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def git_info():
    def run(*a):
        try: return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception: return ""
    sha = run("rev-parse", "HEAD") or None
    dirty = run("status", "--porcelain", "--", "research/supermimic_openarm")
    return dict(commit=sha, package_dirty=bool(dirty), package_dirty_files=dirty.splitlines()[:50])
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""): h.update(c)
    return h.hexdigest()
def atomic_write_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
def atomic_write_json(path, obj): atomic_write_bytes(path, (json.dumps(obj, indent=1, default=_jd) + "\n").encode())
def _jd(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    raise TypeError(type(o))
def repro_metadata(extra=None):
    md = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git=git_info(), python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__)
    try:
        import torch; md.update(torch=torch.__version__, torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    except Exception as e: md["torch"] = f"unavailable: {e}"
    try: import mujoco; md["mujoco"] = mujoco.__version__
    except Exception as e: md["mujoco"] = f"unavailable: {e}"
    if extra: md.update(extra)
    return md
def rng_state():
    import torch
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(), cuda=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
def set_rng_state(s):
    import torch
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"].cpu() if hasattr(s["torch"], "cpu") else s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])
