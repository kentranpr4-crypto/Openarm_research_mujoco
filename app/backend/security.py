"""Read-only artifact access.  Every file the API touches is resolved here: relative paths only, inside the repository, no '..', no symlink escape, no writes, no execution.
Episode / experiment IDs coming from HTTP are never used as paths; they are validated with SAFE_ID and mapped to files by the catalog."""
import re, json, hashlib, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]                              # code checkout (sim/, control/, eval/ ...)
SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,96}$")
ALLOWED_TOP = ("outputs", "docs", "eval", "sim", "control", "policies", "data", "scripts")        # directories that may be read (existence / JSON / hash only)
# Artifact roots are searched in order; the first root in which the relative path exists wins.  The default is the repository itself.
# A demo bundle (or any directory with the same relative layout, e.g. <root>/outputs/...) is added with OPENARM_STUDIO_DATA_ROOT=<dir> or `--data-root <dir>`.
ROOTS = [REPO]; EXCLUSIVE = False
def set_data_root(path, exclusive=False):
    """put an external, read-only artifact root in front of the repository (None resets to the repository only)"""
    global ROOTS, EXCLUSIVE
    _JSON.clear(); EXCLUSIVE = bool(exclusive and path)             # exclusive: outputs/ is read ONLY from the data root (the repository's own outputs/ is ignored)
    if not path: ROOTS = [REPO]; return
    r = Path(path).expanduser().resolve()
    ROOTS = [r] + ([REPO] if r != REPO.resolve() else [])
_env = os.environ.get("OPENARM_STUDIO_DATA_ROOT")
class ArtifactUnavailable(Exception):
    """explicit, user-visible reason why an artifact cannot be served"""
def resolve(rel):
    if not isinstance(rel, str) or not rel or "\x00" in rel or rel.startswith(("/", "\\")) or ".." in Path(rel).parts: raise ArtifactUnavailable(f"path not allowed: {rel!r}")
    top = Path(rel).parts[0]
    if top not in ALLOWED_TOP: raise ArtifactUnavailable(f"directory not allow-listed: {top}")
    first = None
    for root in (ROOTS[:1] if (EXCLUSIVE and top == "outputs") else ROOTS):
        rr = root.resolve(); rp = (root / rel).resolve(); first = first or rp
        if rr not in rp.parents and rp != rr: raise ArtifactUnavailable(f"path escapes the artifact root: {rel}")
        if rp.exists(): return rp
    return first
def exists(rel):
    try: return resolve(rel).exists()
    except ArtifactUnavailable: return False
_JSON = {}
if _env: set_data_root(_env, exclusive=os.environ.get("OPENARM_STUDIO_EXCLUSIVE") == "1")
def read_json(rel):
    p = resolve(rel)
    if not p.is_file(): raise ArtifactUnavailable(f"artifact not found: {rel}")
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _JSON:
        try: _JSON[key] = json.loads(p.read_text())
        except Exception as e: raise ArtifactUnavailable(f"artifact unreadable or corrupted: {rel} ({type(e).__name__})")
    return _JSON[key]
def sha256(rel, max_bytes=256 * 2**20):
    p = resolve(rel)
    if not p.is_file(): raise ArtifactUnavailable(f"artifact not found: {rel}")
    if p.stat().st_size > max_bytes: return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def doc_contains(rel, needle):
    """True iff a text document contains a string (used to verify historical numbers that exist only in reports)"""
    p = resolve(rel)
    return p.is_file() and needle in p.read_text(errors="replace")
def safe_id(s):
    if not isinstance(s, str) or not SAFE_ID.match(s): raise ArtifactUnavailable(f"invalid identifier: {s!r}")
    return s
