#!/usr/bin/env bash
# Reports driver / CUDA runtime / PyTorch build.  Exit 0 if CUDA is usable by PyTorch, 1 otherwise.  Read-only.
source "$(dirname "$0")/_common.sh"
echo "== NVIDIA driver (kernel module + nvidia-smi; installed by YOU / your admin, never by these scripts)"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader; nvidia-smi | sed -n 3,4p | grep -o "CUDA Version: [0-9.]*" || true
else echo "nvidia-smi not found: no NVIDIA driver visible to this shell"; fi
echo "== PyTorch build and runtime"
py - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA runtime bundled in the wheel:", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
ok = torch.cuda.is_available(); print("torch.cuda.is_available():", ok)
if ok:
    p = torch.cuda.get_device_properties(0); f, t = torch.cuda.mem_get_info(); print(f"device: {p.name} | compute capability {p.major}.{p.minor} | VRAM {t/2**30:.1f} GiB total, {f/2**30:.1f} GiB free | bf16 supported: {torch.cuda.is_bf16_supported()}")
    x = torch.randn(1024, 1024, device="cuda"); print("matmul on GPU OK:", bool(torch.isfinite(x @ x).all()))
else: print("CUDA unusable. Typical causes: (1) driver too old for the CUDA 12.8 wheel (needs a recent R570+ driver), (2) CPU-only torch wheel, (3) container without --gpus all.")
raise SystemExit(0 if ok else 1)
PY
echo "(driver = kernel-side; CUDA runtime = libraries inside the PyTorch wheel; a system CUDA toolkit (nvcc) is NOT needed)"
