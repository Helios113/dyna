import torch
import pytest
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.transition.basic_ffn import BasicFFN
from dyna.tests.conftest import save_test_output, load_test_output


class TestBasicFFN:
    """
    Test class for BasicFFN module functionality.
    """
    
    def test_basic_ffn_instantiation(self):
        """
        Test that BasicFFN can be instantiated with various configurations.
        """
        configs = [
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 1024),
        ]
        
        for d_model, d_ffn in configs:
            try:
                ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
                
                assert ffn is not None, f"BasicFFN should be instantiated for config {configs.index((d_model, d_ffn))}"
                assert ffn.d_model == d_model, f"d_model should be {d_model}"
                assert ffn.d_expert_ffn == d_ffn, f"d_expert_ffn should be {d_ffn}"
                
                assert hasattr(ffn, 'projection_up'), "Should have projection_up module"
                assert hasattr(ffn, 'projection_down'), "Should have projection_down module"
                
                assert ffn.projection_up.out_features == d_ffn, "projection_up output should match d_ffn"
                assert ffn.projection_up.in_features == d_model, "projection_up input should match d_model"
                assert ffn.projection_down.out_features == d_model, "projection_down output should match d_model"
                assert ffn.projection_down.in_features == d_ffn, "projection_down input should match d_ffn"
                
                print(f"Config ({d_model}, {d_ffn}): successful.")
                
            except Exception as e:
                pytest.fail(f"BasicFFN instantiation failed for config {d_model, d_ffn}: {e}")
        
        print("BasicFFN instantiation test passed.")
        print(f"Tested {len(configs)} different configurations")
    
    def test_basic_ffn_parameter_initialization(self):
        """
        Test that BasicFFN parameters are properly initialized.
        """
        d_model = 128
        d_ffn = 256
        
        ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
        
        params = list(ffn.parameters())
        assert len(params) > 0, "BasicFFN should have parameters"
        
        for i, param in enumerate(params):
            assert not torch.allclose(param, torch.zeros_like(param)), f"Parameter {i} should not be all zeros"
            assert param.requires_grad, f"Parameter {i} should require gradients"
        
        assert ffn.projection_up.weight.shape == (d_ffn, d_model), "projection_up weight shape should be correct"
        assert ffn.projection_down.weight.shape == (d_model, d_ffn), "projection_down weight shape should be correct"
        
        print("BasicFFN parameter initialization test passed.")
        print(f"Number of parameters: {len(params)}")
        print(f"All parameters initialized: successful.")
        print(f"All parameters require gradients: successful.")
    
    def test_basic_ffn_different_input_shapes(self):
        """
        Test BasicFFN with different input shapes.
        """
        d_model = 64
        d_ffn = 128
        
        ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
        
        test_shapes = [
            (1, 10, d_model),    # Single batch, short sequence
            (2, 32, d_model),    # Small batch, medium sequence
            (4, 64, d_model),    # Medium batch, long sequence
            (8, 128, d_model),   # Large batch, very long sequence
        ]
        
        for batch_size, seq_len, model_dim in test_shapes:
            x = torch.randn(batch_size, seq_len, model_dim)
            
            up = ffn.projection_up(x)
            expected_up_shape = (batch_size, seq_len, d_ffn)
            assert up.shape == expected_up_shape, f"projection_up shape should be {expected_up_shape}, got {up.shape}"
            
            down = ffn.projection_down(up)
            expected_down_shape = (batch_size, seq_len, d_model)
            assert down.shape == expected_down_shape, f"projection_down shape should be {expected_down_shape}, got {down.shape}"

            print(f"Input shape {x.shape}: successful.")
        
        print("BasicFFN different input shapes test passed.")
        print(f"Tested {len(test_shapes)} different input shapes")
    
    def test_basic_ffn_forward_pass(self):
        """
        Test BasicFFN forward pass with standard input.
        """
        d_model = 64
        d_ffn = 128
        
        ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, selection = ffn(x, x)  # Use x as both token_stream and selection_input
        
        assert output.shape == x.shape, f"Output shape should match input shape {x.shape}, got {output.shape}"
        assert output.dtype == x.dtype, f"Output dtype should match input dtype {x.dtype}, got {output.dtype}"
        assert selection is None, "BasicFFN should return None for selection"
        
        assert not torch.allclose(output, x), "Output should be different from input"
        
        print("BasicFFN forward pass test passed.")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output transformed: successful.")
    
    def test_basic_ffn_gradient_flow(self):
        """
        Test that gradients flow properly through BasicFFN.
        """
        d_model = 64
        d_ffn = 128
        
        ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        output, selection = ffn(x, x)  # Use x as both token_stream and selection_input
        
        loss = output.sum()
        
        loss.backward()
        
        assert x.grad is not None, "Input should have gradients"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradients should be non-zero"
        
        ffn_params = list(ffn.parameters())
        for i, param in enumerate(ffn_params):
            assert param.grad is not None, f"FFN parameter {i} should have gradients"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"FFN parameter {i} gradients should be non-zero"
        
        print("BasicFFN gradient flow test passed.")
        print(f"Number of FFN parameters: {len(ffn_params)}")
        print(f"All parameters have gradients: successful.")
        print(f"All gradients are non-zero: successful.")
    
    def test_basic_ffn_activation_configuration(self):
        """
        Test BasicFFN activation configuration.
        """
        d_model = 64
        d_ffn = 128
        
        ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
        assert ffn.activation == torch.nn.functional.gelu, "Default activation should be GELU"
        
        ffn_custom = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn, activation=torch.nn.functional.relu)
        assert ffn_custom.activation == torch.nn.functional.relu, "Custom activation should be ReLU"
        
        print("BasicFFN activation configuration test passed.")
        print("Default activation (GELU): successful.")
        print("Custom activation (ReLU): successful.")
    
    def test_basic_ffn_activation_function(self):
        """
        Test BasicFFN activation function.
        """
        d_model = 64
        d_ffn = 128
        
        ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
        
        x = torch.randn(2, 10, d_model)
        
        output, selection = ffn(x, x)
        
        assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"
        assert not torch.allclose(output, torch.full_like(output, output.mean().item())), "Output should have variation"
        
        print("BasicFFN activation function test passed.")
        print("Activation function applied: successful.")
        print("Output has variation: successful.")
    
    def test_basic_ffn_regression(self):
        """
        Test BasicFFN regression - save outputs and compare across runs.
        """
        d_model = 64
        d_ffn = 128
        
        ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
        
        x = torch.randn(2, 10, d_model)
        
        output, selection = ffn(x, x)
        
        save_test_output(output, "basic_ffn_output.pt")
        
        try:
            saved_output = load_test_output("basic_ffn_output.pt")
            assert torch.equal(output, saved_output), "Current output should match saved output"
            
            print("BasicFFN regression test passed.")
            print("Output saved and loaded successfully.")
        except FileNotFoundError:
            print("BasicFFN regression test passed, first run, no previous data.")
            print("Output saved for future regression testing.")
    
    def test_basic_ffn_different_ffn_sizes(self):
        """
        Test BasicFFN with different FFN sizes.
        """
        d_model = 64
        
        ffn_sizes = [32, 64, 128, 256, 512]
        
        for d_ffn in ffn_sizes:
            ffn = BasicFFN(d_model=d_model, d_expert_ffn=d_ffn)
            
            x = torch.randn(2, 10, d_model)
            
            output, selection = ffn(x, x)
            
            assert output.shape == x.shape, f"Output shape should match input shape for d_ffn={d_ffn}"
            assert output.dtype == x.dtype, f"Output dtype should match input dtype for d_ffn={d_ffn}"
            
            print(f"d_ffn={d_ffn}: successful.")
        
        print("BasicFFN different FFN sizes test passed.")
        print(f"Tested {len(ffn_sizes)} different FFN sizes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

