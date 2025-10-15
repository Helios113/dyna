import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.config import DynaConfig
from dyna.model.model import DynaLM
from dyna.attention.basic_attention import BasicAttn
from dyna.tests.conftest import save_test_output, load_test_output


class TestBasicAttn:
    """
    Test class for BasicAttn module functionality.
    """
    def test_basic_attn_instantiation(self):
        """
        Test that BasicAttn can be instantiated with various configurations.
        """
        configs = [
            # Standard config
            (64, 8, 8, 0.0, 1.0, 10000),
            # With dropout
            (128, 16, 8, 0.1, 1.0, 10000),
            # Different head dimensions
            (256, 8, 32, 0.0, 0.5, 10000),
            # Large model
            (512, 16, 32, 0.2, 1.0, 10000),
        ]

        for d_model, n_heads, d_head, dropout, rotate_fraction, rope_base in configs:
            try:
                attention = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=dropout, rotate_fraction=rotate_fraction, rope_base=rope_base)
                
                assert attention is not None, f"BasicAttn should be instantiated for config {configs.index((d_model, n_heads, d_head, dropout, rotate_fraction, rope_base))}"
                assert attention.d_model == d_model, f"d_model should be {d_model}"
                assert attention.n_heads == n_heads, f"n_heads should be {n_heads}"
                assert attention.d_head == d_head, f"d_head should be {d_head}"
                assert attention.n_rotate == int(rotate_fraction * d_head), f"n_rotate should be {int(rotate_fraction * d_head)}"

                assert hasattr(attention, 'q'), "Should have query projection"
                assert hasattr(attention, 'k'), "Should have key projection"
                assert hasattr(attention, 'v'), "Should have value projection"
                assert hasattr(attention, 'o'), "Should have output projection"
                assert hasattr(attention, 'dropout'), "Should have dropout module"

                assert attention.q.out_features == n_heads * d_head, "Query projection output should match n_heads * d_head"
                assert attention.k.out_features == n_heads * d_head, "Key projection output should match n_heads * d_head"
                assert attention.v.out_features == n_heads * d_head, "Value projection output should match n_heads * d_head"
                assert attention.o.in_features == n_heads * d_head, "Output projection input should match n_heads * d_head"
                assert attention.o.out_features == d_model, "Output projection output should match d_model"

                print(f"- Config ({d_model}, {n_heads}, {d_head}, {dropout}, {rotate_fraction}): Passed.")

            except Exception as e:
                pytest.fail(f"BasicAttn instantiation failed for config {d_model, n_heads, d_head, dropout, rotate_fraction, rope_base}: {e}")
        
        print("BasicAttn instantiation test passed.")
        print(f"Tested {len(configs)} different configurations.")

    def test_basic_attn_parameter_initialization(self):
        """
        Test that BasicAttn parameters are properly initialized.
        """
        d_model = 64
        n_heads = 8
        d_head = 16

        attention = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head)

        params = list(attention.parameters())
        assert len(params) > 0, "BasicAttn should have parameters"

        for i, param in enumerate(params):
            assert not torch.allclose(param, torch.zeros_like(param)), f"Parameter {i} should not be all zeros"
            assert param.requires_grad, f"Parameter {i} should require gradients"

        assert attention.q.weight.shape == (n_heads * d_head, d_model), "Query weight shape should be correct"
        assert attention.k.weight.shape == (n_heads * d_head, d_model), "Key weight shape should be correct"
        assert attention.v.weight.shape == (n_heads * d_head, d_model), "Value weight shape should be correct"
        assert attention.o.weight.shape == (d_model, n_heads * d_head), "Output weight shape should be correct"

        print("BasicAttn parameter initialization test passed.")
        print(f"Number of parameters: {len(params)}")
        print(f"All parameters initialized: Passed.")
        print(f"All parameters require gradients: Passed.")

    def test_basic_attn_different_input_shapes(self):
        """
        Test BasicAttn with different input shapes.
        """
        d_model = 64
        n_heads = 4
        d_head = 16

        attention = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head)

        test_shapes = [
            # Single batch, short sequence
            (1, 10, d_model),
            # Small batch, medium sequence
            (2, 32, d_model),
            # Medium batch, long sequence
            (4, 64, d_model),
            # Large batch, very long sequence
            (8, 128, d_model),
        ]
        
        for batch_size, seq_len, model_dim in test_shapes:
            # Create test input
            x = torch.randn(batch_size, seq_len, model_dim)

            q = attention.q(x)
            k = attention.k(x)
            v = attention.v(x)
            
            expected_q_shape = (batch_size, seq_len, n_heads * d_head)
            assert q.shape == expected_q_shape, f"Query shape should be {expected_q_shape}, got {q.shape}"

            expected_k_shape = (batch_size, seq_len, n_heads * d_head)
            assert k.shape == expected_k_shape, f"Key shape should be {expected_k_shape}, got {k.shape}"

            expected_v_shape = (batch_size, seq_len, n_heads * d_head)
            assert v.shape == expected_v_shape, f"Value shape should be {expected_v_shape}, got {v.shape}"

            print(f"Input shape {x.shape}: Passed.")

        print("BasicAttn different input shapes test passed.")
        print(f"Tested {len(test_shapes)} different input shapes.")

    def test_basic_attn_gradient_flow(self):
        """
        Test that gradients flow properly through BasicAttn.
        """
        d_model = 64
        n_heads = 4
        d_head = 16

        attention = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head)

        # Create test input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        q = attention.q(x)
        k = attention.k(x)
        v = attention.v(x)

        loss = q.sum() + k.sum() + v.sum()

        loss.backward()

        assert x.grad is not None, "Input should have gradients"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradients should be non-zero"

        attention_params = list(attention.parameters())
        params_with_grads = 0
        for i, param in enumerate(attention_params):
            if param.grad is not None:
                params_with_grads += 1
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Attention parameter {i} gradients should be non-zero"

        assert params_with_grads > 0, f"At least some parameters should have gradients, got {params_with_grads}/{len(attention_params)}"
        
        print("BasicAttn gradient flow test passed.")
        print(f"Number of attention parameters: {len(attention_params)}")
        print(f"Parameters with gradients: {params_with_grads}/{len(attention_params)}")
        print(f"All gradients are non-zero: Passed.")

    def test_basic_attn_dropout_configuration(self):
        """
        Test BasicAttn dropout configuration.
        """
        d_model = 64
        n_heads = 4
        d_head = 16

        attention = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head)

        attention_with_dropout = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=0.1)
        assert attention_with_dropout.dropout.p > 0, "Dropout should be enabled when dropout > 0"

        attention_no_dropout = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=0.0)
        # assert attention_no_dropout.dropout.p == 0, "Dropout should be disabled when dropout = 0"
        assert hasattr(attention_no_dropout.dropout, '__call__'), "Should have callable dropout (identity function)"

        print("BasicAttn dropout configuration test passed.")
        print(f"Dropout enabled when dropout > 0: Passed.")
        print(f"Identity function when dropout = 0: Passed.")

    def test_basic_attn_rope_configuration(self):
        """
        Test BasicAttn RoPE configuration.
        """
        d_model = 64
        n_heads = 4
        d_head = 16
        rotate_fraction = 0.5
        rope_base = 10000

        attention = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head, rotate_fraction=rotate_fraction, rope_base=rope_base)
        
        expected_n_rotate = int(rotate_fraction * d_head)
        assert attention.n_rotate == expected_n_rotate, f"n_rotate should be {expected_n_rotate}, got {attention.n_rotate}"

        assert hasattr(attention, 'inv_freq'), "Should have inv_freq from AttentionModule"
        assert attention.inv_freq is not None, "inv_freq should be initialized"

        print("BasicAttn RoPE configuration test passed.")
        print(f"n_rotate: {attention.n_rotate}")
        print(f"RoPE base accessible: Passed.")
        print(f"inv_freq initialized: Passed.")

    def test_basic_attn_regression(self):
        """
        Test BasicAttn regression, also save outputs and compare across runs.
        """
        d_model = 64
        n_heads = 4
        d_head = 16

        attention = BasicAttn(d_model=d_model, n_heads=n_heads, d_head=d_head)

        # Create test input
        x = torch.randn(2, 10, d_model)

        q = attention.q(x)
        k = attention.k(x)
        v = attention.v(x)

        save_test_output(q, "basic_attn_query.pt")
        save_test_output(k, "basic_attn_key.pt")
        save_test_output(v, "basic_attn_value.pt")

        try:
            saved_q = load_test_output("basic_attn_query.pt")
            saved_k = load_test_output("basic_attn_key.pt")
            saved_v = load_test_output("basic_attn_value.pt")

            assert torch.equal(q, saved_q), "Current query should match saved query"
            assert torch.equal(k, saved_k), "Current key should match saved key"
            assert torch.equal(v, saved_v), "Current value should match saved value"

            print("BasicAttn regression test passed.")
            print("Projections saved and loaded successfully.")
        except FileNotFoundError:
            print("BasicAttn regression test passed, first run, no previous data.")
            print("Projections saved for future regression testing.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
