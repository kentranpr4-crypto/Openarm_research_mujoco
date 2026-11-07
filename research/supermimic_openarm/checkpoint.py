"""Checkpoint IO with integrity verification.  A manifest (sha256 + size) is written atomically next to the checkpoint files;
`load_checkpoint(verify=True)` refuses a file whose hash does not match."""
import io, json, time
from pathlib import Path
import torch
from .utils import atomic_write_bytes, atomic_write_json, sha256_file
class CheckpointError(RuntimeError): pass
def _manifest_path(d): return Path(d) / "manifest.json"
def read_manifest(d):
    p = _manifest_path(d); return json.loads(p.read_text()) if p.is_file() else {}
def save_checkpoint(ckpt_dir, name, payload):
    d = Path(ckpt_dir); d.mkdir(parents=True, exist_ok=True); buf = io.BytesIO(); torch.save(payload, buf); atomic_write_bytes(d / name, buf.getvalue())
    m = read_manifest(d); m[name] = dict(sha256=sha256_file(d / name), bytes=(d / name).stat().st_size, saved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), iteration=payload.get("iteration")); atomic_write_json(_manifest_path(d), m); return m[name]
def verify_checkpoint(path):
    p = Path(path); 
    if not p.is_file(): raise CheckpointError(f"checkpoint missing: {p}")
    m = read_manifest(p.parent).get(p.name)
    if m is None: raise CheckpointError(f"no manifest entry for {p.name}: integrity cannot be verified")
    if sha256_file(p) != m["sha256"]: raise CheckpointError(f"checkpoint corrupted (sha256 mismatch): {p}")
    return m
def load_checkpoint(path, device="cpu", verify=True):
    if verify: verify_checkpoint(path)
    try: return torch.load(path, map_location=device, weights_only=False)
    except Exception as e: raise CheckpointError(f"cannot read checkpoint {path}: {type(e).__name__}: {e}")
