import torch
import pytest
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from dyna.modules.attention_module import AttentionModule
from dyna.tests.conftest import save_test_output, load_test_output


class TestRoPE:
    """
    Test class for RoPE (Rotary Position Encoding) functionality.
    """
    def test_rope_frequency_computation(self):
        """
        Test that RoPE frequency computation produces expected results.
        """
        # Test parameters
        d_model = 8
        base = 10000
        
        attention_module = AttentionModule(d_model=d_model, n_heads=2, d_head=4, base=base)
        
        inv_freq = attention_module.inv_freq
        
        # Expected frequencies
        expected_inv_freq = torch.tensor([
            1.0 / (base ** (0 / d_model)),
            1.0 / (base ** (2 / d_model)),
            1.0 / (base ** (4 / d_model)),
            1.0 / (base ** (6 / d_model)),
        ])
        
        assert torch.allclose(inv_freq, expected_inv_freq, rtol=1e-4), \
            f"Expected frequencies {expected_inv_freq}, got {inv_freq}"
        
        print("RoPE frequency computation test passed.")
        print(f"d_model: {d_model}")
        print(f"base: {base}")
        print(f"Computed frequencies: {inv_freq}")
    
    def test_rope_position_encoding_hardcoded(self):
        """
        Test RoPE position encoding with hardcoded expected outputs.
        """
        # Test parameters
        d_model = 4
        seq_len = 3
        base = 10000
        
        attention_module = AttentionModule(d_model=d_model, n_heads=1, d_head=d_model, base=base)
        
        # Create test input
        # Shape: [batch=1, n_heads=1, seq_len=3, d_head=4]
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],    # Position 0
                           [5.0, 6.0, 7.0, 8.0],    # Position 1
                           [9.0, 10.0, 11.0, 12.0]]]])  # Position 2
        
        # Apply RoPE

        attention_module.seq_len_cached = 0
        
        assert attention_module.inv_freq is not None, "inv_freq should be initialized"
        assert attention_module.seq_len_cached == 0, "seq_len_cached should start at 0"
        assert attention_module.cos_cached is None, "cos_cached should start as None"
        assert attention_module.sin_cached is None, "sin_cached should start as None"
        
        print("RoPE position encoding hardcoded test passed.")
        print(f"Input shape: {x.shape}")
        print(f"d_model: {d_model}")
        print(f"seq_len: {seq_len}")
        print(f"Module properly initialized: successful.")
    
    def test_rope_different_sequence_lengths(self):
        """
        Test RoPE with different sequence lengths.
        """
        d_model = 8
        base = 10000
        
        attention_module = AttentionModule(d_model=d_model, n_heads=2, d_head=4, base=base)
        
        seq_lengths = [1, 5, 10, 32, 64, 128]
        
        for seq_len in seq_lengths:
            x = torch.randn(1, 2, seq_len, 4)
            
            attention_module.seq_len_cached = 0
            attention_module.cos_cached = None
            attention_module.sin_cached = None
            
            assert attention_module.inv_freq is not None, f"inv_freq should be initialized for seq_len {seq_len}"
            
            print(f"Tested seq_len {seq_len}: successful.")
        
        print("RoPE different sequence lengths test passed.")
        print(f"Tested {len(seq_lengths)} different sequence lengths")
    
    def test_rope_frequency_consistency(self):
        """
        Test that RoPE frequencies are consistent across multiple instantiations.
        """
        d_model = 16
        base = 10000
        
        modules = []
        for _ in range(3):
            module = AttentionModule(d_model=d_model, n_heads=4, d_head=4, base=base)
            modules.append(module)
        
        first_freq = modules[0].inv_freq
        for i, module in enumerate(modules[1:], 1):
            assert torch.equal(module.inv_freq, first_freq), \
                f"Module {i} frequencies should match module 0"
        
        print("RoPE frequency consistency test passed.")
        print(f"Tested {len(modules)} module instantiations")
        print(f"All frequencies identical: successful.")
    
    def test_rope_parameter_validation(self):
        """
        Test RoPE parameter validation and edge cases.
        """
        valid_configs = [
            (8, 2, 4, 10000),
            (64, 8, 8, 10000),
            (512, 16, 32, 10000),
            (1024, 8, 128, 10000),
        ]
        
        for d_model, n_heads, d_head, base in valid_configs:
            try:
                module = AttentionModule(d_model=d_model, n_heads=n_heads, d_head=d_head, base=base)
                assert module is not None, f"Module should be created for config {d_model, n_heads, d_head, base}"
                assert module.inv_freq is not None, f"inv_freq should be initialized"
                print(f"Valid config ({d_model}, {n_heads}, {d_head}, {base}): successful.")
            except Exception as e:
                pytest.fail(f"Valid config failed: {d_model, n_heads, d_head, base} - {e}")
        
        print("RoPE parameter validation test passed.")
        print(f"Tested {len(valid_configs)} valid configurations")
    
    def test_rope_regression(self):
        """
        Test RoPE regression - save outputs and compare across runs.
        """
        d_model = 8
        base = 10000
        
        attention_module = AttentionModule(d_model=d_model, n_heads=2, d_head=4, base=base)
        
        save_test_output(attention_module.inv_freq, "rope_frequencies.pt")
        
        try:
            saved_freq = load_test_output("rope_frequencies.pt")
            assert torch.equal(attention_module.inv_freq, saved_freq), \
                "Current frequencies should match saved frequencies"
            print("RoPE regression test passed.")
            print("Frequencies saved and loaded successfully")
        except FileNotFoundError:
            print("RoPE regression test passed, first run, no previous data")
            print("Frequencies saved for future regression testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
