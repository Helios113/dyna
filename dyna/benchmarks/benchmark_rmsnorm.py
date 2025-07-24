import torch
import time
from flash_attn.ops.triton.layer_norm import RMSNorm
try:
    from flash_attn.ops.triton.layer_norm import RMSNorm as FlashRMSNorm
except ImportError:
    FlashRMSNorm = None

try:
    from torch.nn import RMSNorm as TorchRMSNorm
except ImportError:
    TorchRMSNorm = None

def benchmark(norm_cls, name, x, n_iter=100):
    norm = norm_cls(x.shape[-1]).to(x.device)
    # Warmup
    for _ in range(10):
        _ = norm(x)
    torch.cuda.synchronize() if x.is_cuda else None

    start = torch.cuda.Event(enable_timing=True) if x.is_cuda else None
    end = torch.cuda.Event(enable_timing=True) if x.is_cuda else None

    if x.is_cuda:
        start.record()
        for _ in range(n_iter):
            _ = norm(x)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / n_iter  # ms
    else:
        t0 = time.time()
        for _ in range(n_iter):
            _ = norm(x)
        elapsed = (time.time() - t0) * 1000 / n_iter  # ms

    print(f"{name}: {elapsed:.3f} ms per iteration")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, D = 32, 512, 2048
    x = torch.randn(B, T, D, device=device)

    if FlashRMSNorm is not None:
        benchmark(FlashRMSNorm, "FlashAttn RMSNorm", x)
    else:
        print("FlashAttn RMSNorm not available.")

    if TorchRMSNorm is not None:
        benchmark(TorchRMSNorm, "Torch RMSNorm", x)
    else:
        print("Torch RMSNorm not available (requires PyTorch >= 2.1).")

if __name__ == "__main__":
    main()
