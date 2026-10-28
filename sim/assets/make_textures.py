#!/usr/bin/env python3
"""Generate the lab's procedural textures (deterministic, seed fixed). Run: .venv/bin/python assets/make_textures.py
Outputs 512x512 PNGs next to this file. Appearance only - no physical meaning."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(7)

def noise(sigma, size=512):
    a = Image.fromarray((rng.random((size, size)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    a = np.asarray(a, np.float32); return (a - a.mean()) / (a.std() + 1e-6)

def save(name, rgb):
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(HERE / name)

def tint(base, field, amp):
    return np.asarray(base, np.float32)[None, None, :] + field[..., None] * amp

# polished concrete floor: mid-grey with large soft mottling + fine grain
save("floor_concrete.png", tint((150, 152, 156), noise(40) * 0.7 + noise(3) * 0.3, 9))
# painted wall: warm off-white with faint roller texture
save("wall_paint.png", tint((222, 224, 226), noise(25), 3))
# black laminate bench top: near black with fine speckle (matches the dark desk in the user's photo)
save("table_laminate.png", tint((26, 27, 29), noise(1.0) * 0.7 + noise(12) * 0.3, 5))
# brushed aluminium tray
st = np.tile(noise(0.8, 512)[:1, :], (512, 1)) * 0.5 + noise(2) * 0.5
save("tray_metal.png", tint((176, 180, 186), st, 7))
print("textures written to", HERE)
