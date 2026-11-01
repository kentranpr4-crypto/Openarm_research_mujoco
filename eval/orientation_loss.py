#!/usr/bin/env python3
"""Mask-aware / robust orientation losses for the E1 ablations (DESIGN + UNIT TESTS ONLY; nothing is trained).  torch implementation, all inputs are per-step quantities of shape (B, K, ...).
  yaw_huber_masked : Huber on the WRAPPED residual of per-step yaw deltas, divided by a robust scale, multiplied by the target-time validity mask (both ends heading-valid), normalised by the number of valid steps.
  rotvec_loss      : MSE (optionally Huber) between predicted and true per-step world rotation vectors (3-D; no Euler singularity, no wrap), divided by a robust scale; supervises tumbling/singular windows.
  clip_only        : legacy MSE after clipping the target to +-c (arm used to separate masking from clipping if needed).
Parameters (pre-registered proposals): Huber delta 0.05 rad, clip 0.2 rad, robust scale = std of VALID, non-spike per-step yaw deltas of the TRAIN windows (computed once, stored with the dataset version)."""
import torch
def wrap(x): return torch.atan2(torch.sin(x), torch.cos(x))
def huber(r, delta): a = r.abs(); return torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
def yaw_huber_masked(pred, true, valid, scale, delta=0.05):
    """pred, true: (B,K) per-step yaw deltas [rad]; valid: (B,K) bool/float; scale: robust std [rad].  Residual is wrapped, so a 2*pi representation offset costs nothing."""
    r = wrap(pred - true) / scale; l = huber(r, delta / scale) * valid.float(); return l.sum() / valid.float().sum().clamp_min(1.0)
def rotvec_loss(pred, true, scale, delta=None):
    r = (pred - true) / scale; l = (0.5 * r * r) if delta is None else huber(r, delta / scale); return l.sum(-1).mean()
def clip_only(pred, true, scale, c=0.2): return (((pred - true.clamp(-c, c)) / scale) ** 2).mean()
