import torch
import time
from custom_rotary import RotaryEmbedding

def rope_pytorch(q, k, cos, sin):
    # Standard RoPE implementation for [batch, seq, nheads, head_dim]
    head_dim = q.shape[-1]
    half = head_dim // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    # Fix broadcasting: [1, seq, 1, half]
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot

def generate_rope_cache(seq_len, head_dim, device):
    # Generate cos/sin cache as in FlashAttention
    half = head_dim // 2
    pos = torch.arange(seq_len, device=device).float()
    freq = 10000 ** (-torch.arange(0, half, device=device).float() / half)
    freqs = torch.einsum("i,j->ij", pos, freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin

def main():
    device = "cuda"
    batch = 16
    seq_len = 2048
    nheads = 8
    head_dim = 64
    dtype = torch.float16
    q = torch.randn(batch, seq_len, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, nheads, head_dim, device=device, dtype=dtype)
    cos, sin = generate_rope_cache(seq_len, head_dim, device)

    # Use FlashAttention RotaryEmbedding for RoPE
    rotary_emb = RotaryEmbedding(dim=head_dim, base=10000, device=device)

    for _ in range(5):
        # Warmup
        q_rot, k_rot = rope_pytorch(q, k, cos, sin)
        q_rot_flash, k_rot_flash = rotary_emb(q, k, max_seqlen=seq_len)
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        q_rot, k_rot = rope_pytorch(q, k, cos, sin)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"PyTorch RoPE: {(t1-t0)*1000:.2f} ms")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        q_rot_flash, k_rot_flash = rotary_emb(q,k, max_seqlen=seq_len)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"FlashAttention RoPE: {(t1-t0)*1000:.2f} ms")

    # Benchmark backward pass
    q_pt = q.clone().detach().requires_grad_()
    k_pt = k.clone().detach().requires_grad_()
    cos_pt = cos.clone().detach()
    sin_pt = sin.clone().detach()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        q_rot, k_rot = rope_pytorch(q_pt, k_pt, cos_pt, sin_pt)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()
        q_pt.grad.zero_()
        k_pt.grad.zero_()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"PyTorch RoPE backward: {(t1-t0)*1000:.2f} ms")

    q_flash = q.clone().detach().requires_grad_()
    k_flash = k.clone().detach().requires_grad_()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        q_rot_flash, k_rot_flash = rotary_emb(q_flash, k_flash, max_seqlen=seq_len)
        loss = q_rot_flash.sum() + k_rot_flash.sum()
        loss.backward()
        q_flash.grad.zero_()
        k_flash.grad.zero_()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"FlashAttention RoPE backward: {(t1-t0)*1000:.2f} ms")

    # Compare outputs
    q_rot_pt, k_rot_pt = rope_pytorch(q, k, cos, sin)
    q_rot_flash,k_rot_flash= rotary_emb(q, k, max_seqlen=seq_len)

    print("Compare q_rot:", torch.allclose(q_rot_pt, q_rot_flash, atol=1e-5))
    print("Compare k_rot:", torch.allclose(k_rot_pt, k_rot_flash, atol=1e-5))
    print("Max abs diff q_rot:", (q_rot_pt - q_rot_flash).abs().max().item())
    print("Max abs diff k_rot:", (k_rot_pt - k_rot_flash).abs().max().item())

if __name__ == "__main__":
    main()
    main()
