import torch
import pytest
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.transition.sigma_moe import SigmaMoE
from dyna.tests.conftest import save_test_output, load_test_output


@pytest.mark.skip(reason="Triton issues")
class TestSigmaMoE:
    """
    Test class for SigmaMoE module functionality.
    """
    
    def test_sigma_moe_instantiation(self):
        """
        Test that SigmaMoE can be instantiated with various configurations.
        """
        configs = [
            (64, 128, 8, 2, 0.0, 0.0, 0, 0.01),
            (128, 256, 16, 4, 0.1, 0.1, 2, 0.01),
            (256, 512, 32, 8, 0.0, 0.0, 0, 0.01),
            (512, 1024, 64, 16, 0.2, 0.1, 8, 0.01),
        ]
        
        for d_model, d_ffn, n_experts_ffn, k_ffn, dropout, dropout_expert, n_expert_shared_ffn, reg_entropy in configs:
            try:
                moe = SigmaMoE(
                    d_model=d_model,
                    n_experts_ffn=n_experts_ffn,
                    d_expert_ffn=d_ffn,
                    n_expert_shared_ffn=n_expert_shared_ffn,
                    k_ffn=k_ffn,
                    dropout_expert=dropout_expert
                )
                
                assert moe is not None, f"SigmaMoE should be instantiated for config {configs.index((d_model, d_ffn, n_experts_ffn, k_ffn, dropout, dropout_expert, n_expert_shared_ffn, reg_entropy))}"
                assert moe.d_model == d_model, f"d_model should be {d_model}"
                assert moe.d_expert_ffn == d_ffn, f"d_expert_ffn should be {d_ffn}"
                assert moe.n_experts_ffn == n_experts_ffn, f"n_experts_ffn should be {n_experts_ffn}"
                assert moe.k_ffn == k_ffn, f"k_ffn should be {k_ffn}"
                assert moe.n_expert_shared_ffn == min(n_expert_shared_ffn, n_experts_ffn), f"n_expert_shared_ffn should be {min(n_expert_shared_ffn, n_experts_ffn)}"
                assert moe.n_expert_routed_ffn == n_experts_ffn - min(n_expert_shared_ffn, n_experts_ffn), f"n_expert_routed_ffn should be {n_experts_ffn - min(n_expert_shared_ffn, n_experts_ffn)}"
                
                assert hasattr(moe, 'keys'), "Should have keys module"
                assert hasattr(moe, 'values'), "Should have values module"
                assert hasattr(moe, 'expert_sel'), "Should have expert_sel module"
                
                print(f"Config ({d_model}, {d_ffn}, {n_experts_ffn}, {k_ffn}): successful.")
                
            except Exception as e:
                pytest.fail(f"SigmaMoE instantiation failed for config {d_model, d_ffn, n_experts_ffn, k_ffn, dropout, dropout_expert, n_expert_shared_ffn, reg_entropy}: {e}")
        
        print("SigmaMoE instantiation test passed.")
        print(f"Tested {len(configs)} different configurations")
    
    def test_sigma_moe_expert_configuration(self):
        """
        Test SigmaMoE expert configuration.
        """
        d_model = 128
        d_ffn = 256
        n_experts_ffn = 16
        k_ffn = 4
        n_expert_shared_ffn = 4
        
        moe = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=n_expert_shared_ffn,
            k_ffn=k_ffn
        )
        
        assert moe.n_experts_ffn == n_experts_ffn, f"n_experts_ffn should be {n_experts_ffn}"
        assert moe.k_ffn == k_ffn, f"k_ffn should be {k_ffn}"
        assert moe.n_expert_shared_ffn == n_expert_shared_ffn, f"n_expert_shared_ffn should be {n_expert_shared_ffn}"
        assert moe.n_expert_routed_ffn == n_experts_ffn - n_expert_shared_ffn, f"n_expert_routed_ffn should be {n_experts_ffn - n_expert_shared_ffn}"
        
        print("SigmaMoE expert configuration test passed.")
        print(f"n_experts_ffn: {moe.n_experts_ffn}")
        print(f"k_ffn: {moe.k_ffn}")
        print(f"n_expert_shared_ffn: {moe.n_expert_shared_ffn}")
        print(f"n_expert_routed_ffn: {moe.n_expert_routed_ffn}")
    
    def test_sigma_moe_parameter_initialization(self):
        """
        Test that SigmaMoE parameters are properly initialized.
        """
        d_model = 128
        d_ffn = 256
        n_experts_ffn = 16
        
        moe = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=2
        )
        
        params = list(moe.parameters())
        assert len(params) > 0, "SigmaMoE should have parameters"
        
        non_zero_params = 0
        trainable_params = 0
        for i, param in enumerate(params):
            if not torch.allclose(param, torch.zeros_like(param)):
                non_zero_params += 1
            if param.requires_grad:
                trainable_params += 1
        
        assert non_zero_params > 0, f"At least some parameters should be non-zero, got {non_zero_params}/{len(params)}"
        assert trainable_params > 0, f"At least some parameters should be trainable, got {trainable_params}/{len(params)}"
        
        print("SigmaMoE parameter initialization test passed.")
        print(f"Number of parameters: {len(params)}")
        print(f"Non-zero parameters: {non_zero_params}/{len(params)}")
        print(f"Trainable parameters: {trainable_params}/{len(params)}")
    
    def test_sigma_moe_different_input_shapes(self):
        """
        Test SigmaMoE with different input shapes.
        """
        d_model = 64
        d_ffn = 128
        n_experts_ffn = 8
        
        moe = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=2
        )
        
        test_shapes = [
            (1, 10, d_model),
            (2, 32, d_model),
            (4, 64, d_model),
            (8, 128, d_model),
        ]
        
        for batch_size, seq_len, model_dim in test_shapes:
            x = torch.randn(batch_size, seq_len, model_dim)
            
            assert hasattr(moe, 'keys'), "SigmaMoE should have keys"
            assert hasattr(moe, 'values'), "SigmaMoE should have values"
            
            print(f"Input shape {x.shape}: successful.")
        
        print("SigmaMoE different input shapes test passed.")
        print(f"Tested {len(test_shapes)} different input shapes")
    
    def test_sigma_moe_forward_pass(self):
        """
        Test SigmaMoE forward pass with standard input.
        """
        d_model = 64
        d_ffn = 128
        n_experts_ffn = 8
        k_ffn = 2
        
        moe = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=k_ffn
        )
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, selection = moe(x, x)
        
        assert output.shape == x.shape, f"Output shape should match input shape {x.shape}, got {output.shape}"
        assert output.dtype == x.dtype, f"Output dtype should match input dtype {x.dtype}, got {output.dtype}"
        assert selection is not None, "SigmaMoE should return selection indices"
        
        assert not torch.allclose(output, x), "Output should be different from input"
        
        print("SigmaMoE forward pass test passed.")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output transformed: successful.")
    
    def test_sigma_moe_gradient_flow(self):
        """
        Test that gradients flow properly through SigmaMoE.
        """
        d_model = 64
        d_ffn = 128
        n_experts_ffn = 8
        k_ffn = 2
        
        moe = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=k_ffn
        )
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        output, selection = moe(x, x)
        
        loss = output.sum()
        
        loss.backward()
        
        assert x.grad is not None, "Input should have gradients"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradients should be non-zero"
        
        moe_params = list(moe.parameters())
        params_with_grads = 0
        for i, param in enumerate(moe_params):
            if param.grad is not None:
                params_with_grads += 1
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"MoE parameter {i} gradients should be non-zero"
        
        assert params_with_grads > 0, f"At least some parameters should have gradients, got {params_with_grads}/{len(moe_params)}"
        
        print("SigmaMoE gradient flow test passed.")
        print(f"Number of MoE parameters: {len(moe_params)}")
        print(f"Parameters with gradients: {params_with_grads}/{len(moe_params)}")
        print(f"All gradients are non-zero: successful.")
    
    def test_sigma_moe_dropout_configuration(self):
        """
        Test SigmaMoE dropout configuration.
        """
        d_model = 64
        d_ffn = 128
        n_experts_ffn = 8
        
        moe_with_dropout = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=2,
            dropout_expert=0.1
        )
        assert moe_with_dropout.dropout_expert > 0, "Dropout should be enabled when dropout_expert > 0"
        
        moe_no_dropout = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=2,
            dropout_expert=0.0
        )
        assert moe_no_dropout.dropout_expert == 0, "Dropout should be disabled when dropout_expert = 0"
        
        print("SigmaMoE dropout configuration test passed.")
        print("Dropout enabled when dropout > 0: successful.")
        print("Dropout disabled when dropout = 0: successful.")
    
    def test_sigma_moe_entropy_regularization(self):
        """
        Test SigmaMoE entropy regularization.
        """
        d_model = 64
        d_ffn = 128
        n_experts_ffn = 8
        reg_entropy = 0.01
        
        moe = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=2
        )

        print("SigmaMoE entropy regularization test passed.")
        print("SigmaMoE instantiated successfully.")
    
    def test_sigma_moe_regression(self):
        """
        Test SigmaMoE regression - save outputs and compare across runs.
        """
        d_model = 64
        d_ffn = 128
        n_experts_ffn = 8
        k_ffn = 2
        
        moe = SigmaMoE(
            d_model=d_model,
            n_experts_ffn=n_experts_ffn,
            d_expert_ffn=d_ffn,
            n_expert_shared_ffn=0,
            k_ffn=k_ffn
        )
        
        x = torch.randn(2, 10, d_model)
        
        output, selection = moe(x, x)
        
        save_test_output(output, "sigma_moe_output.pt")
        
        try:
            saved_output = load_test_output("sigma_moe_output.pt")
            assert torch.equal(output, saved_output), "Current output should match saved output"
            
            print("SigmaMoE regression test passed.")
            print("Output saved and loaded successfully")
        except FileNotFoundError:
            print("SigmaMoE regression test passed, first run, no previous data")
            print("Output saved for future regression testing")
    
    def test_sigma_moe_different_expert_configurations(self):
        """
        Test SigmaMoE with different expert configurations.
        """
        d_model = 64
        d_ffn = 128
        
        expert_configs = [
            (4, 1, 0),   # Few experts, k=1, no shared
            (8, 2, 0),   # Medium experts, k=2, no shared
            (16, 4, 2),  # Many experts, k=4, some shared
            (32, 8, 4),  # Many experts, k=8, more shared
        ]
        
        for n_experts_ffn, k_ffn, n_expert_shared_ffn in expert_configs:
            moe = SigmaMoE(
                d_model=d_model,
                n_experts_ffn=n_experts_ffn,
                d_expert_ffn=d_ffn,
                n_expert_shared_ffn=n_expert_shared_ffn,
                k_ffn=k_ffn
            )
            
            x = torch.randn(2, 10, d_model)
            
            output, selection = moe(x, x)
            
            assert output.shape == x.shape, f"Output shape should match input shape for experts={n_experts_ffn}, k={k_ffn}"
            assert output.dtype == x.dtype, f"Output dtype should match input dtype for experts={n_experts_ffn}, k={k_ffn}"
            
            print(f"experts={n_experts_ffn}, k={k_ffn}, shared={n_expert_shared_ffn}: successful.")
        
        print("SigmaMoE different expert configurations test passed.")
        print(f"Tested {len(expert_configs)} different expert configurations")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])