#!/usr/bin/env python3
"""Export the lab scene as a stand-alone MJCF that opens in `python -m mujoco.viewer --mjcf=<file>` (or any MuJoCo app).
Mesh/texture directories are rewritten RELATIVE to the output folder, so the file stays portable inside the repo
(it references sim/openarm_mujoco/v1/meshes and sim/assets; it does not copy or modify them).
Usage: python export_scene.py [--out DIR] [--verify]"""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import sys, re, argparse, numpy as np, mujoco
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab_scene

def export_xml(out_dir=None, name="lab_scene.xml"):
    out_dir = Path(out_dir or lab_scene.OUTDIR).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    spec = lab_scene.build_lab_spec()
    spec.compile()                                  # validates with the in-repo dirs first
    xml = spec.to_xml()
    mesh_rel = os.path.relpath(HERE / "openarm_mujoco/v1/meshes", out_dir); tex_rel = os.path.relpath(lab_scene.ASSETS, out_dir)
    xml, n1 = re.subn(r'meshdir="[^"]*"', f'meshdir="{mesh_rel}"', xml)
    xml, n2 = re.subn(r'texturedir="[^"]*"', f'texturedir="{tex_rel}"', xml)
    assert n1 == 1 and n2 == 1, f"expected one meshdir and one texturedir attribute, got {n1}, {n2}"
    path = out_dir / name; path.write_text(xml)
    return path

def verify(path):
    """Round trip: load the exported file with plain MjModel and compare against the in-memory scene."""
    m0, d0 = lab_scene.compile_lab()
    m1 = mujoco.MjModel.from_xml_path(str(path)); d1 = mujoco.MjData(m1); lab_scene.reset_home(m1, d1)
    txt = Path(path).read_text()
    r = dict(counts_equal=all(getattr(m0, k) == getattr(m1, k) for k in ("nq", "nv", "nu", "nbody", "ngeom", "ncam", "nlight", "nmat", "ntex", "nmesh")),
             counts={k: (getattr(m0, k), getattr(m1, k)) for k in ("nq", "nv", "nu", "ngeom", "ncam", "nlight")},
             geom_xpos_max_diff=float(np.abs(d0.geom_xpos - d1.geom_xpos).max()), qpos_max_diff=float(np.abs(d0.qpos - d1.qpos).max()),
             absolute_paths_in_xml=[ln.strip() for ln in txt.splitlines() if ("/home/" in ln or "/tmp/" in ln)][:3])
    # tolerance 1e-5 m: to_xml prints ~7 significant digits, so poses round-trip to ~1e-6
    r["ok"] = r["counts_equal"] and r["geom_xpos_max_diff"] < 1e-5 and r["qpos_max_diff"] < 1e-5 and not r["absolute_paths_in_xml"]
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    p = export_xml(a.out); print("exported", p)
    if a.verify:
        r = verify(p); print(r); sys.exit(0 if r["ok"] else 1)
