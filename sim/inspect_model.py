#!/usr/bin/env python3
"""Audit an MJCF's dimensions and actuator semantics before writing an action adapter.
Usage: inspect_model.py scene.xml ... ; prints nq/nv/nu, qpos/qvel layout per joint, and per-actuator ctrl meaning."""
import sys, numpy as np, mujoco
J = mujoco.mjtJoint; G = mujoco.mjtGain; B = mujoco.mjtBias; D = mujoco.mjtDyn; T = mujoco.mjtTrn
def nm(m,t,i): return mujoco.mj_id2name(m,t,i)
def kind(m,i):
    g,b = G(m.actuator_gaintype[i]), B(m.actuator_biastype[i])
    gp,bp = m.actuator_gainprm[i],m.actuator_biasprm[i]
    if g==G.mjGAIN_FIXED and b==B.mjBIAS_NONE: return "motor (ctrl = force/torque scale)"
    if g==G.mjGAIN_FIXED and b==B.mjBIAS_AFFINE:
        if bp[2]==0 and bp[1]==0: return "affine, no pos/vel term"
        if np.isclose(gp[0],-bp[1]) and bp[2]==0: return f"position servo (ctrl = target pos, kp={gp[0]:g}, kv={-bp[2]:g})"
        if np.isclose(gp[0],-bp[1]): return f"position servo (ctrl = target pos, kp={gp[0]:g}, kv={-bp[2]:g})"
        if bp[1]==0 and np.isclose(gp[0],-bp[2]): return f"velocity servo (ctrl = target vel, kv={gp[0]:g})"
        return f"affine gain={gp[:1]} bias={bp[:3]}"
    return f"{g.name}/{b.name}"
for path in sys.argv[1:]:
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
    print(f"\n=== {path}\nnq={m.nq} nv={m.nv} nu={m.nu} na={m.na} ngeom={m.ngeom} ncam={m.ncam} timestep={m.opt.timestep}")
    print(f"nq==nv: {m.nq==m.nv}  (free/ball joints present: {any(m.jnt_type[j] in (J.mjJNT_FREE,J.mjJNT_BALL) for j in range(m.njnt))})")
    print("\nqpos/qvel layout (joint, type, qposadr, dofadr, range):")
    for j in range(m.njnt):
        r = m.jnt_range[j] if m.jnt_limited[j] else None
        print(f"  {nm(m,mujoco.mjtObj.mjOBJ_JOINT,j):28s} {J(m.jnt_type[j]).name[6:]:6s} q[{m.jnt_qposadr[j]}] v[{m.jnt_dofadr[j]}] range={None if r is None else r.round(3).tolist()}")
    print("\nctrl layout (index, actuator, drives, ctrl meaning, ctrlrange, forcerange, dyn):")
    for i in range(m.nu):
        tgt = nm(m,mujoco.mjtObj.mjOBJ_JOINT,m.actuator_trnid[i,0]) if m.actuator_trntype[i]==T.mjTRN_JOINT else f"{T(m.actuator_trntype[i]).name}:{m.actuator_trnid[i,0]}"
        cr = m.actuator_ctrlrange[i].round(3).tolist() if m.actuator_ctrllimited[i] else "UNLIMITED"
        fr = m.actuator_forcerange[i].round(1).tolist() if m.actuator_forcelimited[i] else "-"
        print(f"  u[{i:2d}] {nm(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i):18s} -> {tgt:28s} {kind(m,i)}  ctrlrange={cr} force={fr} dyn={D(m.actuator_dyntype[i]).name[6:]}")
    # does ctrlrange sit inside the joint range? (only meaningful for position servos)
    for i in range(m.nu):
        if m.actuator_trntype[i]==T.mjTRN_JOINT and m.actuator_ctrllimited[i]:
            j=m.actuator_trnid[i,0]
            if m.jnt_limited[j] and "position" in kind(m,i):
                lo,hi=m.actuator_ctrlrange[i]; jl,jh=m.jnt_range[j]
                if lo<jl-1e-4 or hi>jh+1e-4: print(f"  WARN ctrlrange [{lo:.3f},{hi:.3f}] exceeds joint range [{jl:.3f},{jh:.3f}] for u[{i}]")
