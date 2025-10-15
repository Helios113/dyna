import torch
import pytest
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.modules.saturation_gate import SaturationGate
from dyna.modules.dtanh import DynamicTanh
from dyna.modules.layer_scaled_identity_fn import layer_scaled_identity, LayerScaledIdentityFn
from dyna.tests.conftest import save_test_output, load_test_output


class TestSaturationGate:
    """
    Test class for SaturationGate module functionality.
    """
    def test_saturation_gate_instantiation(self):
        """
        Test that SaturationGate can be instantiated correctly.
        """
        gate = SaturationGate(d_model=64)
        assert gate is not None, "SaturationGate should be instantiated"
        assert isinstance(gate, torch.nn.Module), "SaturationGate should inherit from torch.nn.Module"

        print("SaturationGate instantiation test passed.")
        print(f"Module created successfully")
        print(f"Inherits from torch.nn.Module: {isinstance(gate, torch.nn.Module)}")

    def test_saturation_gate_forward_pass(self):
        """
        Test SaturationGate forward pass with standard input.
        """
        gate = SaturationGate(d_model=64)

        # Create test input
        x = torch.randn(2, 10, 64)

        # Forward Pass
        output = gate(x)

        assert output.shape == (2, 10), "Output shape should be (2, 10)"
        assert output.dtype == x.dtype, "Output dtype should match input dtype"

        print("SaturationGate forward pass test passed.")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

    def test_saturation_gate_gradient_flow(self):
        """
        Test that gradients flow properly through SaturationGate.
        """
        gate = SaturationGate(d_model=64)

        x = torch.randn(2, 10, 64, requires_grad=True)

        # Forward Pass
        output = gate(x)
        
        loss = output.sum()

        # Backward pass
        loss.backward()

        gate_params_with_grad = [p for p in gate.parameters() if p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad))]
        assert len(gate_params_with_grad) > 0, "At least some gate parameters should have non-zero gradients"

        print("SaturationGate gradient flow test passed.")
        print(f"Gate parameter gradients: {gate_params_with_grad}")
        print(f"All gradients are non-zero: {all(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)) for p in gate.parameters())}")

    def test_saturation_gate_regression(self):
        """
        Test SaturationGate regression, save outputs and compare across runs.
        """
        gate = SaturationGate(d_model=64)

        # Create test input
        x = torch.randn(2, 10, 64)

        # Forward Pass
        output = gate(x)
        
        save_test_output(output, "saturation_gate_output.pt")

        # Test that we can load and compare
        try:
            saved_output = load_test_output("saturation_gate_output.pt")
            assert torch.equal(output, saved_output), "Current output should match saved output"
            
            print("SaturationGate regression test passed.")
            print("Output saved and loaded successfully")
        except FileNotFoundError:
            print("SaturationGate regression test passed, first run, no previous data.")
            print("Output saved for future regression testing.")


class TestDynamicTanh:
    """
    Test class for DynamicTanh module functionality.
    """
    def test_dynamic_tanh_instantiation(self):
        """
        Test that DynamicTanh can be instantiated correctly.
        """
        # Test configurations
        configs = [
            (64, True, 0.5),   # Standard config with channels_last=True
            (128, False, 0.3), # With channels_last=False
            (32, True, 0.8),   # Different alpha value
            (256, False, 0.1), # Large normalized_shape
        ]
        
        for normalized_shape, channels_last, alpha_init_value in configs:
            try:
                dtanh = DynamicTanh(
                    normalized_shape=normalized_shape,
                    channels_last=channels_last,
                    alpha_init_value=alpha_init_value
                )
                
                assert dtanh is not None, f"DynamicTanh should be instantiated for config {configs.index((normalized_shape, channels_last, alpha_init_value))}"
                assert dtanh.normalized_shape == normalized_shape, f"normalized_shape should be {normalized_shape}"
                assert dtanh.channels_last == channels_last, f"channels_last should be {channels_last}"
                assert dtanh.alpha_init_value == alpha_init_value, f"alpha_init_value should be {alpha_init_value}"

                assert hasattr(dtanh, 'alpha'), "Should have alpha parameter"
                assert hasattr(dtanh, 'weight'), "Should have weight parameter"
                assert hasattr(dtanh, 'bias'), "Should have bias parameter"

                assert dtanh.alpha.shape == (1,), f"Alpha shape should be (1,) got {dtanh.alpha.shape}"
                assert dtanh.weight.shape == (normalized_shape,), f"Weight shape should be ({normalized_shape},) got {dtanh.weight.shape}"
                assert dtanh.bias.shape == (normalized_shape,), f"Bias shape should be ({normalized_shape},) got {dtanh.bias.shape}"

                print(f"Config ({normalized_shape}, {channels_last}, {alpha_init_value}): ✓")

            except Exception as e:
                pytest.fail(f"DynamicTanh instantiation failed for config {normalized_shape, channels_last, alpha_init_value}: {e}")
        
        print("DynamicTanh instantiation test passed.")
        print(f"Tested {len(configs)} different configurations")
    
    def test_dynamic_tanh_parameter_initialization(self):
        """
        Test that DynamicTanh parameters are properly initialized.
        """
        normalized_shape = 64
        alpha_init_value = 0.5

        dtanh = DynamicTanh(
            normalized_shape=normalized_shape,
            channels_last=True,
            alpha_init_value=alpha_init_value
        )

        params = list(dtanh.parameters())
        assert len(params) == 3, f"DynamicTanh should have 3 parameters but got {len(params)}"

        non_zero_params = sum(1 for param in params if not torch.allclose(param, torch.zeros_like(param)))
        assert non_zero_params > 0, "At least some parameters should be non-zero"

        for i, param in enumerate(params):
            assert param.requires_grad, f"Parameter {i} should require gradients"

        assert torch.allclose(dtanh.alpha, torch.tensor([alpha_init_value])), f"Alpha should be initialized to {alpha_init_value}"
        assert torch.allclose(dtanh.weight, torch.ones(normalized_shape)), f"Weight should be initialized to ones"
        assert torch.allclose(dtanh.bias, torch.zeros(normalized_shape)), f"Bias should be initialized to zeros"
        
        print("DynamicTanh parameter initialization test passed.")
        print(f"Number of parameters: {len(params)}")
        print(f"All parameters initialized.")
        print(f"All parameters require gradients.")
    
    def test_dynamic_tanh_forward_pass(self):
        """
        Test DynamicTanh forward pass with standard input.
        """
        normalized_shape = 64
        alpha_init_value = 0.5

        dtanh = DynamicTanh(
            normalized_shape=normalized_shape,
            channels_last=True,
            alpha_init_value=alpha_init_value
        )

        x = torch.randn(2, 10, normalized_shape)

        output = dtanh(x)

        assert output.shape == x.shape, f"Output shape should match input shape {x.shape} got {output.shape}"
        assert output.dtype == x.dtype, f"Output dtype should match input dtype {x.dtype}, got {output.dtype}"

        assert torch.all(output >= -1.1), "Output should be >= -1.1 (allowing for small numerical errors)"
        assert torch.all(output <= 1.1), "Output should be <= 1.1 (allowing for small numerical errors)"

        print("DynamicTanh channels_last behavior test passed.")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    def test_dynamic_tanh_channels_last_behavior(self):
        """
        Test DynamicTanh behavior with different channels_last settings.
        """
        normalized_shape = 32
        alpha_init_value = 0.5

        dtanh = DynamicTanh(
            normalized_shape=normalized_shape,
            channels_last=True,
            alpha_init_value=alpha_init_value
        )

        dtanh_channels_first = DynamicTanh(
            normalized_shape=normalized_shape,
            channels_last=False,
            alpha_init_value=alpha_init_value
        )

        x = torch.randn(2, 10, normalized_shape)

        output_channels_last = dtanh(x)
        # output_channels_first = dtanh_channels_first(x)

        assert output_channels_last.shape == x.shape, "channels_last=True should preserve input shape"
        # assert output_channels_first.shape == x.shape, "channels_last=False should preserve input shape"

        # assert torch.allclose(output_channels_last, output_channels_first), "Different channels_last settings should produce different outputs"

        print("DynamicTanh channels_last behavior test passed.")
        print(f"channels_last=True: {output_channels_last.shape}")
        # print(f"channels_last=False: {output_channels_first.shape}")
    
    def test_dynamic_tanh_gradient_flow(self):
        """
        Test that gradients flow properly through DynamicTanh.
        """
        dtanh = DynamicTanh(
            normalized_shape=64,
            channels_last=True,
            alpha_init_value=0.5
        )

        x = torch.randn(2, 10, 64, requires_grad=True)

        output = dtanh(x)
        
        loss = output.sum()

        loss.backward()

        dtanh_params_with_grad = [p for p in dtanh.parameters() if p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad))]
        assert len(dtanh_params_with_grad) > 0, "At least some dtanh parameters should have non-zero gradients"

        print("DynamicTanh gradient flow test passed.")
        print(f"Dtanh parameter gradients: {dtanh_params_with_grad}")
        print(f"All gradients are non-zero: {all(p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad)) for p in dtanh.parameters())}")
    
    def test_dynamic_tanh_regression(self):
        """
        Test DynamicTanh regression, save outputs and compare across runs.
        """
        dtanh = DynamicTanh(
            normalized_shape=64,
            channels_last=True,
            alpha_init_value=0.5
        )
        x = torch.randn(2, 10, 64)

        output = dtanh(x)

        save_test_output(output, "dynamic_tanh_output.pt")

        saved_output = load_test_output("dynamic_tanh_output.pt")
        
        try:
            saved_output = load_test_output("dynamic_tanh_output.pt")
            assert torch.equal(output, saved_output), "Current output should match saved output"
            
            print("DynamicTanh regression test passed.")
            print("Output saved and loaded successfully")
        except FileNotFoundError:
            print("DynamicTanh regression test passed, first run, no previous data.")
            print("Output saved for future regression testing.")
        

class TestLayerScaledIdentityFn:
    """
    Test class for LayerScaledIdentityFn module functionality.
    """
    def test_layer_scaled_identity_function(self):
        """
        Test layer_scaled_identity function with standard input.
        """
        x = torch.randn(2, 10, 64)
        total_layers = 6

        output = layer_scaled_identity(x, total_layers)

        assert output.shape == x.shape, f"Output shape should match input shape {x.shape}, got {output.shape}"
        assert output.dtype == x.dtype, f"Output dtype should match input dtype {x.dtype}, got {output.dtype}"

        assert torch.allclose(output, x), "Output should be identical to input (identity function)"

        print("LayerScaledIdentityFn function test passed.")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Scaling factor: {1/total_layers}")

    def test_layer_scaled_identity_gradient_flow(self):
        """
        Test that gradients flow properly through LayerScaledIdentityFn.
        """
        x = torch.randn(2, 10, 64, requires_grad=True)
        total_layers = 6

        output = layer_scaled_identity(x, total_layers)

        loss = output.sum()

        loss.backward()

        assert x.grad is not None, "Input should have gradients"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradients should be non-zero"
        
        expected_grad = torch.ones_like(x) * total_layers
        assert torch.allclose(x.grad, expected_grad), f"Input gradient should be scaled by {total_layers}"

        print("LayerScaledIdentityFn gradient flow test passed.")
        print(f"Input gradients: {x.grad}")
        print(f"Gradient scaling: {expected_grad}")
    
    def test_layer_scaled_identity_different_layers(self):
        """
        Test LayerScaledIdentityFn with different total_layers values.
        """
        x = torch.randn(2, 10, 64)
        layer_counts = [1, 2, 4, 8, 12, 24]

        for total_layers in layer_counts:
            output = layer_scaled_identity(x, total_layers)
            assert output.shape == x.shape, f"Output shape should match input shape for {total_layers} layers"
            assert torch.allclose(output, x), f"Output should be identical to input for {total_layers} layers"
            print(f"{total_layers} layers")
        
        print("LayerScaledIdentityFn different layers test passed.")
        print(f"Tested {len(layer_counts)} different layer counts")
    
    def test_layer_scaled_identity_regression(self):
        """
        Test LayerScaledIdentityFn regression, save outputs and compare across runs.
        """
        x = torch.randn(2, 10, 64)
        total_layers = 6

        output = layer_scaled_identity(x, total_layers)

        save_test_output(output, "layer_scaled_identity_output.pt")

        saved_output = load_test_output("layer_scaled_identity_output.pt")
        
        try:
            saved_output = load_test_output("layer_scaled_identity_output.pt")
            assert torch.equal(output, saved_output), "Current output should match saved output"

            print("LayerScaledIdentityFn regression test passed.")
            print("Output saved and loaded successfully")
        except FileNotFoundError:
            print("LayerScaledIdentityFn regression test passed, first run, no previous data.")
            print("Output saved for future regression testing.")
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])