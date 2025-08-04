import torch
import torch.nn.functional as F
import time
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

def benchmark_attention(batch_size=32, seq_len=2048, n_heads=8, head_dim=64, device="cuda", steps=100):
    dtype = torch.bfloat16
    
    # Create tensors with correct dimensions
    # For PyTorch SDPA: (batch_size, n_heads, seq_len, head_dim)
    q_sdpa = torch.randn(batch_size, n_heads, seq_len//2, head_dim, device=device, dtype=dtype)
    k_sdpa = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v_sdpa = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # For FlashAttention: (batch_size, seq_len, n_heads, head_dim)
    q_flash = q_sdpa.transpose(1, 2).contiguous()  # (batch_size, seq_len//2, n_heads, head_dim)
    k_flash = k_sdpa.transpose(1, 2).contiguous()
    v_flash = v_sdpa.transpose(1, 2).contiguous()

    # Since q and k have different sequence lengths, we cannot use causal attention
    is_causal = False
    
    print(f"Benchmarking with batch_size={batch_size}, seq_len_q={seq_len//2}, seq_len_k={seq_len}, n_heads={n_heads}, head_dim={head_dim}")
    print(f"SDPA tensor shapes: q={q_sdpa.shape}, k={k_sdpa.shape}")
    print(f"FlashAttention tensor shapes: q={q_flash.shape}, k={k_flash.shape}")
    print(f"Using causal attention: {is_causal}")

    # Warm up
    with torch.no_grad():
        _ = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0, causal=False)
        
        _ = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, dropout_p=0.0, is_causal=is_causal)

    torch.cuda.synchronize()

    # Benchmark FlashAttention
    t0 = time.time()
    with torch.no_grad():
        for _ in range(steps):
            _ = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0, causal=is_causal)
    torch.cuda.synchronize()
    t1 = time.time()
    avg_flash = (t1 - t0) / steps

    # Benchmark PyTorch SDPA with FlashAttention backend
    t2 = time.time()
    with torch.no_grad():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            for _ in range(steps):
                _ = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, dropout_p=0.0, is_causal=is_causal)
    torch.cuda.synchronize()
    t3 = time.time()
    avg_sdpa_flash = (t3 - t2) / steps

    # Benchmark PyTorch SDPA with Math backend (fallback)
    t4 = time.time()
    with torch.no_grad():
        with sdpa_kernel([SDPBackend.MATH]):
            for _ in range(steps):
                _ = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, dropout_p=0.0, is_causal=is_causal)
    torch.cuda.synchronize()
    t5 = time.time()
    avg_sdpa_math = (t5 - t4) / steps

    print(f"\nResults:")
    print(f"FlashAttention (direct)     avg time: {avg_flash:.6f}s")
    print(f"PyTorch SDPA (Flash backend) avg time: {avg_sdpa_flash:.6f}s")
    print(f"PyTorch SDPA (Math backend)  avg time: {avg_sdpa_math:.6f}s")
    print(f"\nSpeedup vs Math backend:")
    print(f"FlashAttention (direct):     {avg_sdpa_math/avg_flash:.2f}x")
    print(f"PyTorch SDPA (Flash):        {avg_sdpa_math/avg_sdpa_flash:.2f}x")

if __name__ == "__main__":
    benchmark_attention()
