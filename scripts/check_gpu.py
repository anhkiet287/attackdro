"""Sanity check: confirm PyTorch sees the RTX 5070 Ti and can run on it.

Run inside WSL2 after installing the cu128 PyTorch build:
    python scripts/check_gpu.py

Expect:
    CUDA available: True
    Device: NVIDIA GeForce RTX 5070 Ti
    Compute capability: (12, 0)   <-- sm_120 (Blackwell)
    Matmul on GPU: OK
"""

import torch


def main() -> None:
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[!] CUDA not available. Check SETUP.md:")
        print("    - NVIDIA driver installed on Windows host")
        print("    - cu128 PyTorch build installed inside WSL2")
        return

    print(f"CUDA (torch)    : {torch.version.cuda}")
    print(f"Device          : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capab.  : {cap}  (expect (12, 0) for Blackwell)")

    # Real work on the GPU to confirm kernels actually run on sm_120.
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"Matmul on GPU   : OK  (result sum={y.sum().item():.1f})")


if __name__ == "__main__":
    main()
