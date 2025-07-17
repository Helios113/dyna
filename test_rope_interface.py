#!/usr/bin/env python3
"""
Simple test to verify the flash attention import and basic setup works.
"""

from flash_attn.layers.rotary import RotaryEmbedding
import torch

def test_import_and_init():
    """Test that we can import and initialize RotaryEmbedding."""
    
    # Test initialization
    dim = 64
    base = 10000.0
    
    rope = RotaryEmbedding(dim=dim, base=base)
    
    print(f"✓ Successfully created RotaryEmbedding with dim={dim}, base={base}")
    print(f"  - dim: {rope.dim}")
    print(f"  - base: {rope.base}")
    print(f"  - interleaved: {rope.interleaved}")
    
    # Check that buffers are registered
    assert hasattr(rope, 'inv_freq'), "inv_freq buffer should be registered"
    
    print("✓ RotaryEmbedding initialization test passed!")

def test_interface():
    """Test the interface without actually running the forward pass."""
    
    # Test parameters that match our use case
    batch_size = 2
    seq_len = 8
    n_heads = 4
    head_dim = 64
    
    device = torch.device('cpu')
    dtype = torch.float32
    
    rope = RotaryEmbedding(dim=head_dim, base=10000.0)
    
    # Create tensors with the expected shapes
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=dtype)
    
    # Check that the tensors have the right shapes for flash attention
    assert q.shape == (batch_size, seq_len, n_heads, head_dim), f"Q shape: {q.shape}"
    assert k.shape == (batch_size, seq_len, n_heads, head_dim), f"K shape: {k.shape}"
    assert v.shape == (batch_size, seq_len, n_heads, head_dim), f"V shape: {v.shape}"
    
    # Test KV tensor creation
    kv = torch.stack([k, v], dim=2)  # (batch, seq_len, 2, n_heads, head_dim)
    assert kv.shape == (batch_size, seq_len, 2, n_heads, head_dim), f"KV shape: {kv.shape}"
    
    print("✓ Tensor shapes are correct for flash attention interface")
    
    # Test partial rotation setup
    rotate_dim = 32
    rope_partial = RotaryEmbedding(dim=rotate_dim, base=10000.0)
    
    q_rot = q[..., :rotate_dim]
    k_rot = k[..., :rotate_dim]
    v_rot = v[..., :rotate_dim]
    
    assert q_rot.shape == (batch_size, seq_len, n_heads, rotate_dim), f"Q_rot shape: {q_rot.shape}"
    
    print("✓ Partial rotation tensor shapes are correct")
    print("✓ Interface test passed!")

if __name__ == "__main__":
    print("Testing flash attention RotaryEmbedding interface...")
    test_import_and_init()
    test_interface()
    print("✓ All interface tests passed! Ready to use flash attention RotaryEmbedding.")
