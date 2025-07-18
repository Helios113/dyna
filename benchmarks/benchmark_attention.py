import torch
import torch.nn.functional as F
import time
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

def benchmark_attention(batch_size=1024, seq_len=2048, n_heads=8, head_dim=64, device="cuda", steps=1000):
    # Use fp16 for compatibility with FlashAttention
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        dtype = torch.bfloat16
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device)  # no masking

        # Warmup
        for _ in range(3):
            _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=None, dropout_p=0.0, is_causal=False)
            _ = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

        torch.cuda.synchronize()
        # Benchmark flash_attn_func
        t0 = time.time()
        for _ in range(steps):
            out1 = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        torch.cuda.synchronize()
        t1 = time.time()
        avg_flash = (t1 - t0) / steps

        # Benchmark scaled_dot_product_attention
        t2 = time.time()
        for _ in range(steps):
            out2 = F.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=None, dropout_p=0.0, is_causal=False)
        torch.cuda.synchronize()
        t3 = time.time()
        avg_sdpa = (t3 - t2) / steps

        print(f"flash_attn_func avg time: {avg_flash:.6f}s")
        print(f"scaled_dot_product_attention avg time: {avg_sdpa:.6f}s")
        print(f"Max abs diff: {(out1-out2).abs().max().item():.6e}")

if __name__ == "__main__":
    benchmark_attention()
