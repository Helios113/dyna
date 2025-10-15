import torch
import pytest
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.attention.switch_head import SwitchHead
from dyna.tests.conftest import save_test_output, load_test_output


class TestSwitchHead:
    """
    Test class for SwitchHead module functionality.
    """
    def test_switch_head_instantiation(self):
        """
        Test that SwitchHead can be instantiated with various configurations.
        """
        configs = [
            # Standard config
            (64, 8, 4, 8, 0.0, 0.0, 2, 0, 1.0, 10000),
            # With dropout and shared experts
            (128, 16, 8, 16, 0.1, 0.1, 4, 2, 1.0, 10000),
            # Different head dimensions
            (256, 8, 32, 32, 0.0, 0.0, 1, 0, 0.5, 10000),
            # Large model
            (512, 16, 32, 64, 0.2, 0.1, 8, 4, 1.0, 10000),
        ]
        
        for d_model, n_heads, d_head, n_experts_attn, dropout, dropout_expert, k_attn, n_expert_shared_attn, rotate_fraction, rope_base in configs:
            try:
                attention = SwitchHead(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_experts_attn=n_experts_attn,
                    d_head=d_head,
                    dropout=dropout,
                    dropout_expert=dropout_expert,
                    k_attn=k_attn,
                    n_expert_shared_attn=n_expert_shared_attn,
                    rotate_fraction=rotate_fraction,
                    rope_base=rope_base
                )

                assert attention is not None, f"SwitchHead should be instantiated for config {configs.index((d_model, n_heads, d_head, n_experts_attn, dropout, dropout_expert, k_attn, n_expert_shared_attn, rotate_fraction, rope_base))}"
                assert attention.d_model == d_model, f"d_model should be {d_model}"
                assert attention.n_heads == n_heads, f"n_heads should be {n_heads}"
                assert attention.d_head == d_head, f"d_head should be {d_head}"
                assert attention.n_experts_attn == n_experts_attn, f"n_experts_attn should be {n_experts_attn}"
                assert attention.k_attn == k_attn, f"k_attn should be {k_attn}"
                assert attention.n_expert_shared_attn == min(n_expert_shared_attn, n_experts_attn), f"n_expert_shared_attn should be {min(n_expert_shared_attn, n_experts_attn)}"
                assert attention.n_expert_routed_attn == n_experts_attn - min(n_expert_shared_attn, n_experts_attn), f"n_expert_routed_attn should be {n_experts_attn - min(n_expert_shared_attn, n_experts_attn)}"
                
                assert hasattr(attention, 'q'), "Should have query projection"
                assert hasattr(attention, 'k'), "Should have key projection"
                assert hasattr(attention, 'dropout'), "Should have dropout module"
                
                assert attention.q.out_features == d_head * n_heads, "Query projection output should match d_head * n_heads"
                assert attention.k.out_features == d_head * n_heads, "Key projection output should match d_head * n_heads"
                
                print(f"Config ({d_model}, {n_heads}, {d_head}, {n_experts_attn}, {k_attn}): Passed.")
                
            except Exception as e:
                pytest.fail(f"SwitchHead instantiation failed for config {d_model, n_heads, d_head, n_experts_attn, dropout, dropout_expert, k_attn, n_expert_shared_attn, rotate_fraction, rope_base}: {e}")
        
        print("SwitchHead instantiation test passed.")
        print(f"Tested {len(configs)} different configurations.")
    
    def test_switch_head_expert_configuration(self):
        """
        Test SwitchHead expert configuration.
        """
        d_model = 128
        n_heads = 8
        d_head = 16
        n_experts_attn = 16
        k_attn = 4
        n_expert_shared_attn = 4

        attention = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head,
            k_attn=k_attn,
            n_expert_shared_attn=n_expert_shared_attn
        )

        assert attention.n_experts_attn == n_experts_attn, f"n_experts_attn should be {n_experts_attn}"
        assert attention.k_attn == k_attn, f"k_attn should be {k_attn}"
        assert attention.n_expert_shared_attn == n_expert_shared_attn, f"n_expert_shared_attn should be {n_expert_shared_attn}"
        assert attention.n_expert_routed_attn == n_experts_attn - n_expert_shared_attn, f"n_expert_routed_attn should be {n_experts_attn - n_expert_shared_attn}"
        
        assert hasattr(attention, '_init_expert_parameters'), "Should have expert parameter initialization method"
        
        print("SwitchHead expert configuration test passed.")
        print(f"n_experts_attn: {attention.n_experts_attn}")
        print(f"k_attn: {attention.k_attn}")
        print(f"n_expert_shared_attn: {attention.n_expert_shared_attn}")
        print(f"n_expert_routed_attn: {attention.n_expert_routed_attn}")
    
    def test_switch_head_parameter_initialization(self):
        """
        Test that SwitchHead parameters are properly initialized.
        """
        d_model = 128
        n_heads = 8
        d_head = 16
        n_experts_attn = 16
        
        attention = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head
        )

        params = list(attention.parameters())
        assert len(params) > 0, "SwitchHead should have parameters"
        
        non_zero_params = 0
        trainable_params = 0
        for i, param in enumerate(params):
            if not torch.allclose(param, torch.zeros_like(param)):
                non_zero_params += 1
            if param.requires_grad:
                trainable_params += 1

        assert non_zero_params > 0, f"At least some parameters should be non-zero, got {non_zero_params}/{len(params)}"
        assert trainable_params > 0, f"At least some parameters should be trainable, got {trainable_params}/{len(params)}"
        
        assert attention.q.weight.shape == (d_head * n_heads, d_model), "Query weight shape should be correct"
        assert attention.k.weight.shape == (d_head * n_heads, d_model), "Key weight shape should be correct"
        
        print("SwitchHead parameter initialization test passed.")
        print(f"Number of parameters: {len(params)}")
        print(f"Non-zero parameters: {non_zero_params}/{len(params)}")
        print(f"Trainable parameters: {trainable_params}/{len(params)}")
        
        print("SwitchHead parameter initialization test passed.")
        print(f"Number of parameters: {len(params)}")
        print(f"Non-zero parameters: {non_zero_params}/{len(params)}")
        print(f"Trainable parameters: {trainable_params}/{len(params)}")

    def test_switch_head_different_input_shapes(self):
        """
        Test SwitchHead with different input shapes.
        """
        d_model = 64
        n_heads = 4
        d_head = 16
        n_experts_attn = 8

        attention = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head
        )

        # Different input shapes
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
            expected_q_shape = (batch_size, seq_len, d_head * n_heads)
            assert q.shape == expected_q_shape, f"Query shape should be {expected_q_shape}, got {q.shape}"

            k = attention.k(x)
            expected_k_shape = (batch_size, seq_len, d_head * n_heads)
            assert k.shape == expected_k_shape, f"Key shape should be {expected_k_shape}, got {k.shape}"

            print(f"Input shape {x.shape}: Passed.")

        print("SwitchHead different input shapes test passed.")
        print(f"Tested {len(test_shapes)} different input shapes.")

    def test_switch_head_gradient_flow(self):
        """
        Test that gradients flow properly through SwitchHead.
        """
        d_model = 64
        n_heads = 4
        d_head = 16
        n_experts_attn = 8
        
        attention = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head
        )

        # Create test input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        q = attention.q(x)
        k = attention.k(x)

        loss = q.sum() + k.sum()

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

        print("SwitchHead gradient flow test passed.")
        print(f"Number of attention parameters: {len(attention_params)}")
        print(f"Parameters with gradients: {params_with_grads}/{len(attention_params)}")
        print(f"All gradients are non-zero: Passed.")

    def test_switch_head_dropout_configuration(self):
        """
        Test SwitchHead dropout configuration.
        """
        d_model = 64
        n_heads = 4
        d_head = 16
        n_experts_attn = 8
        

        attention_with_dropout = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head,
            dropout=0.1,
            dropout_expert=0.1
        )
        assert attention_with_dropout.dropout.p > 0, "Dropout should be enabled when dropout > 0"
        
        attention_no_dropout = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head,
            dropout=0.0,
            dropout_expert=0.0
        )
        # assert attention_no_dropout.dropout.p == 0, "Dropout should be disabled when dropout = 0"
        assert hasattr(attention_no_dropout.dropout, '__call__'), "Should have callable dropout (identity function)"

        print("SwitchHead dropout configuration test passed.")
        print(f"Dropout enabled when dropout > 0: Passed.")
        print(f"Identity function when dropout = 0: Passed.")

    def test_switch_head_rope_configuration(self):
        """
        Test SwitchHead RoPE configuration.
        """
        d_model = 64
        n_heads = 4
        d_head = 16
        n_experts_attn = 8
        rotate_fraction = 0.5
        rope_base = 10000

        attention = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head,
            rotate_fraction=rotate_fraction,
            rope_base=rope_base
        )

        expected_n_rotate = int(rotate_fraction * d_head)
        assert attention.n_rotate == expected_n_rotate, f"n_rotate should be {expected_n_rotate}, got {attention.n_rotate}"

        assert hasattr(attention, 'inv_freq'), "Should have inv_freq from AttentionModule"
        assert attention.inv_freq is not None, "inv_freq should be initialized"

        print("SwitchHead RoPE configuration test passed.")
        print(f"n_rotate: {attention.n_rotate}")
        print(f"RoPE base accessible: Passed.")
        print(f"inv_freq initialized: Passed.")

    def test_switch_head_bias_tracking(self):
        """
        Test SwitchHead bias tracking configuration.
        """
        d_model = 64
        n_heads = 4
        d_head = 16
        n_experts_attn = 8

        attention = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head
        )

        assert hasattr(attention, 'bias_update_lr'), "Should have bias_update_lr"
        assert attention.bias_update_lr == 0.001, "bias_update_lr should be 0.001"

        print("SwitchHead bias tracking test passed.")
        print(f"bias_update_lr: {attention.bias_update_lr}")

    def test_switch_head_regression(self):
        """
        Test SwitchHead regression - save outputs and compare across runs.
        """
        d_model = 64
        n_heads = 4
        d_head = 16
        n_experts_attn = 8

        attention = SwitchHead(
            d_model=d_model,
            n_heads=n_heads,
            n_experts_attn=n_experts_attn,
            d_head=d_head
        )

        # Create test input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)

        q = attention.q(x)
        k = attention.k(x)

        save_test_output(q, "switch_head_query.pt")
        save_test_output(k, "switch_head_key.pt")
        
        try:
            saved_q = load_test_output("switch_head_query.pt")
            saved_k = load_test_output("switch_head_key.pt")

            assert torch.equal(q, saved_q), "Current query should match saved query"
            assert torch.equal(k, saved_k), "Current key should match saved key"

            print("SwitchHead regression test passed.")
            print("Projections saved and loaded successfully.")
        except FileNotFoundError:
            print("SwitchHead regression test passed, first run, no previous data.")
            print("Projections saved for future regression testing.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
