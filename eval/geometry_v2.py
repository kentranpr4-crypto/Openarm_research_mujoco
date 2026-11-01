#!/usr/bin/env python3
"""Orientation geometry for data contract v2 / Gate v3 (pure numpy, no simulator).  Conventions (verified against MuJoCo in eval/contract_v2_validate.py):
quaternion q = (w, x, y, z) (MuJoCo xquat); R = xmat reshaped (3,3) maps body -> world, column j = body axis j expressed in the world; world z is up.
Euler 'yaw' of the data contract v1 = atan2(R[1,0], R[0,0]) = heading of the body x-axis projected on the horizontal plane; it is UNDEFINED when the body x-axis is vertical (horizontal norm hnorm = sqrt(R00^2+R10^2) -> 0).
The box is a cuboid: its pose is invariant under the D2 group (identity and 180-degree rotations about the three body axes), applied on the BODY side: q ~ q (x) s."""
import numpy as np
D2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
def qnorm(q): q = np.asarray(q, float); return q / np.linalg.norm(q, axis=-1, keepdims=True)
def qmul(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float)); aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)
def qconj(q): q = np.asarray(q, float); return q * np.array([1, -1, -1, -1.0])
def quat_to_R(q):
    q = qnorm(q); w, x, y, z = np.moveaxis(q, -1, 0); R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))
def yaw_euler(q): R = quat_to_R(q); return np.arctan2(R[..., 1, 0], R[..., 0, 0])
def hnorm(q): R = quat_to_R(q); return np.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
def tilt_rad(q): R = quat_to_R(q); return np.arccos(np.clip(R[..., 2, 2], -1, 1))
def tilt_sym(q): t = tilt_rad(q); return np.minimum(t, np.pi - t)
def geodesic(q1, q2): d = np.abs(np.sum(qnorm(q1) * qnorm(q2), -1)); return 2 * np.arccos(np.clip(d, 0, 1))
def geodesic_sym(q1, q2): return np.min(np.stack([geodesic(q1, qmul(q2, s)) for s in D2], 0), 0)
def rotvec_from_quat(q):
    q = qnorm(q); q = np.where(q[..., :1] < 0, -q, q); v = q[..., 1:]; n = np.linalg.norm(v, axis=-1, keepdims=True); ang = 2 * np.arctan2(n, q[..., :1]); return np.where(n > 1e-12, v / np.maximum(n, 1e-12) * ang, 2 * v)
def quat_from_rotvec(r):
    r = np.asarray(r, float); a = np.linalg.norm(r, axis=-1, keepdims=True); h = a / 2; s = np.where(a > 1e-12, np.sin(h) / np.maximum(a, 1e-12), 0.5); return np.concatenate([np.cos(h), r * s], -1)
def step_rotvecs(Q):
    """Q (..., T+1, 4) -> world-frame relative rotation vectors of q_{k+1} (x) conj(q_k): (..., T, 3)."""
    return rotvec_from_quat(qmul(Q[..., 1:, :], qconj(Q[..., :-1, :])))
def compose(q0, rv):
    """q0 (...,4), rv (..., K, 3) per-step world-frame rotation vectors -> quaternions of rows 1..K: (..., K, 4)."""
    q = qnorm(q0); out = []
    for k in range(rv.shape[-2]): q = qnorm(qmul(quat_from_rotvec(rv[..., k, :]), q)); out.append(q)
    return np.stack(out, -2)
def wrap(x): return np.arctan2(np.sin(x), np.cos(x))
