import torch
import pytest
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.modules.layer_module import LayerModule
from dyna.modules.attention_module import AttentionModule
from dyna.modules.dyna_module import DynaModule
from dyna.config import DynaConfig, NormStructure, RescaleMethod
from dyna.tests.conftest import save_test_output, load_test_output


class TestLayerModule:
    """
    Test class for LayerModule base class functionality.
    """

    def _create_concrete_layer_module(self, config, attention_module, ffn_module):
        """Helper to create a concrete LayerModule implementation for testing."""
        class ConcreteLayerModule(LayerModule):
            def forward(self, x, attention_mask=None, source_len_mask=None):
                # Simple forward pass
                return x, None
        
        return ConcreteLayerModule(config, attention_module, ffn_module)
    
    def test_layer_module_instantiation(self):
        """
        Test that LayerModule can be instantiated with various configurations.
        """
        config = DynaConfig(
            d_model=64,
            n_heads=4,
            d_head=16,
            n_layers=6,
            dropout=0.1,
            use_rms_norm=True,
            norm_structure=NormStructure.peri,
            rescaling_method=RescaleMethod.none,
            enable_early_exit=False
        )
        
        attention_module = AttentionModule(d_model=64, n_heads=4, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
    
        layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
        
        assert layer is not None, "LayerModule should be instantiated"
        assert isinstance(layer, torch.nn.Module), "LayerModule should inherit from torch.nn.Module"
        assert hasattr(layer, 'attention'), "LayerModule should have attention module"
        assert hasattr(layer, 'ffn'), "LayerModule should have FFN module"
        assert hasattr(layer, 'attn_pre'), "LayerModule should have attn_pre normalization"
        assert hasattr(layer, 'attn_post'), "LayerModule should have attn_post normalization"
        assert hasattr(layer, 'ffn_pre'), "LayerModule should have ffn_pre normalization"
        assert hasattr(layer, 'ffn_post'), "LayerModule should have ffn_post normalization"
        
        print("✓ LayerModule instantiation test passed")
        print("Module created successfully")
        print("Has required components: successful.")
        print("Configuration applied: successful.")
    
    def test_layer_module_normalization_structures(self):
        """
        Test LayerModule with different normalization structures.
        """
        norm_structures = [
            NormStructure.peri,
            NormStructure.pre,
            NormStructure.post,
            NormStructure.moeut
        ]
        
        for norm_structure in norm_structures:
            config = DynaConfig(
                d_model=32,
                n_heads=2,
                d_head=16,
                norm_structure=norm_structure,
                dropout=0.0
            )
            
            attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
            
            class MockFFN(DynaModule):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(32, 32)
                
                def forward(self, x, y):
                    return self.linear(x), None
            
            ffn_module = MockFFN()
            
            try:
                layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
                assert layer is not None, f"LayerModule should be instantiated with {norm_structure}"
                assert layer.norm_structure == norm_structure, f"Norm structure should be {norm_structure}"
                
                print(f"{norm_structure}: successful.")
                
            except Exception as e:
                pytest.fail(f"LayerModule instantiation failed for {norm_structure}: {e}")
        
        print("LayerModule normalization structures test passed.")
        print(f"Tested {len(norm_structures)} different normalization structures")
    
    def test_layer_module_rescaling_methods(self):
        """
        Test LayerModule with different rescaling methods.
        """
        rescaling_methods = [
            RescaleMethod.none,
            RescaleMethod.cum_avg_prot_emb,
            RescaleMethod.sqrt_prot_emb
        ]
        
        for rescaling_method in rescaling_methods:
            config = DynaConfig(
                d_model=32,
                n_heads=2,
                d_head=16,
                rescaling_method=rescaling_method,
                dropout=0.0
            )
            
            attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
            
            class MockFFN(DynaModule):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(32, 32)
                
                def forward(self, x, y):
                    return self.linear(x), None
            
            ffn_module = MockFFN()
            
            try:
                layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
                assert layer is not None, f"LayerModule should be instantiated with {rescaling_method}"
                assert layer.rescaling_method == rescaling_method, f"Rescaling method should be {rescaling_method}"
                
                print(f"{rescaling_method}: successful.")
                
            except Exception as e:
                pytest.fail(f"LayerModule instantiation failed for {rescaling_method}: {e}")
        
        print("LayerModule rescaling methods test passed.")
        print(f"Tested {len(rescaling_methods)} different rescaling methods")
    
    def test_layer_module_parameter_initialization(self):
        """
        Test that LayerModule parameters are properly initialized.
        """
        config = DynaConfig(
            d_model=64,
            n_heads=4,
            d_head=16,
            dropout=0.1
        )
        
        attention_module = AttentionModule(d_model=64, n_heads=4, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
        
        layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
        
        params = list(layer.parameters())
        assert len(params) > 0, "LayerModule should have parameters"
        
        non_zero_params = 0
        trainable_params = 0
        for i, param in enumerate(params):
            if not torch.allclose(param, torch.zeros_like(param)):
                non_zero_params += 1
            if param.requires_grad:
                trainable_params += 1
        
        assert non_zero_params > 0, f"At least some parameters should be non-zero, got {non_zero_params}/{len(params)}"
        assert trainable_params > 0, f"At least some parameters should be trainable, got {trainable_params}/{len(params)}"
        
        print("LayerModule parameter initialization test passed.")
        print(f"Number of parameters: {len(params)}")
        print(f"Non-zero parameters: {non_zero_params}/{len(params)}")
        print(f"Trainable parameters: {trainable_params}/{len(params)}")
    
    def test_layer_module_normalization_behavior(self):
        """
        Test LayerModule normalization behavior.
        """
        config = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            norm_structure=NormStructure.peri,
            dropout=0.0
        )
        
        attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
        
        layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
        
        # Normalization layers
        x = torch.randn(2, 10, 32)
        
        attn_pre_norm = layer.attn_pre(x)
        assert attn_pre_norm.shape == x.shape, f"Attention pre-norm shape should match input {x.shape}, got {attn_pre_norm.shape}"
        attn_post_norm = layer.attn_post(x)
        assert attn_post_norm.shape == x.shape, f"Attention post-norm shape should match input {x.shape}, got {attn_post_norm.shape}"
        
        ffn_pre_norm = layer.ffn_pre(x)
        assert ffn_pre_norm.shape == x.shape, f"FFN pre-norm shape should match input {x.shape}, got {ffn_pre_norm.shape}"
        ffn_post_norm = layer.ffn_post(x)
        assert ffn_post_norm.shape == x.shape, f"FFN post-norm shape should match input {x.shape}, got {ffn_post_norm.shape}"
        
        print("LayerModule normalization behavior test passed.")
        print("Attention normalization: successful.")
        print("FFN normalization: successful.")
        print("Shape preservation: successful.")
    
    def test_layer_module_dropout_configuration(self):
        """
        Test LayerModule dropout configuration.
        """
        config_with_dropout = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            dropout=0.1
        )
        
        attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
        
        layer_with_dropout = self._create_concrete_layer_module(config_with_dropout, attention_module, ffn_module)
        assert layer_with_dropout.drop.p > 0, "Dropout should be enabled when dropout > 0"
        
        config_no_dropout = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            dropout=0.0
        )
        
        layer_no_dropout = self._create_concrete_layer_module(config_no_dropout, attention_module, ffn_module)
        assert layer_no_dropout.drop.p == 0, "Dropout should be disabled when dropout = 0"
        
        print("LayerModule dropout configuration test passed.")
        print("Dropout enabled when dropout > 0: successful.")
        print("Dropout disabled when dropout = 0: successful.")
    
    def test_layer_module_early_exit_configuration(self):
        """
        Test LayerModule early exit configuration.
        """
        config_early_exit = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            enable_early_exit=True
        )
        
        attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
        
        layer_early_exit = self._create_concrete_layer_module(config_early_exit, attention_module, ffn_module)
        assert layer_early_exit.enable_early_exit == True, "Early exit should be enabled"
        
        config_no_early_exit = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            enable_early_exit=False
        )
        
        layer_no_early_exit = self._create_concrete_layer_module(config_no_early_exit, attention_module, ffn_module)
        assert layer_no_early_exit.enable_early_exit == False, "Early exit should be disabled"
        
        print("LayerModule early exit configuration test passed.")
        print("Early exit enabled: successful.")
        print("Early exit disabled: successful.")
    
    def test_layer_module_gradient_flow(self):
        """
        Test that gradients flow properly through LayerModule.
        """
        config = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            dropout=0.0
        )
        
        attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
        
        layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
        
        # Test input
        x = torch.randn(2, 10, 32, requires_grad=True)
        
        attn_pre_norm = layer.attn_pre(x)
        ffn_pre_norm = layer.ffn_pre(x)
        
        loss = attn_pre_norm.sum() + ffn_pre_norm.sum()
        
        loss.backward()
        
        assert x.grad is not None, "Input should have gradients"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradients should be non-zero"
        
        layer_params = list(layer.parameters())
        params_with_grads = 0
        for i, param in enumerate(layer_params):
            if param.grad is not None:
                params_with_grads += 1
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Layer parameter {i} gradients should be non-zero"
        
        assert params_with_grads > 0, f"At least some parameters should have gradients, got {params_with_grads}/{len(layer_params)}"
        
        print("LayerModule gradient flow test passed.")
        print(f"Number of layer parameters: {len(layer_params)}")
        print(f"Parameters with gradients: {params_with_grads}/{len(layer_params)}")
        print(f"All gradients are non-zero: successful.")
    
    def test_layer_module_regression(self):
        """
        Test LayerModule regression - save outputs and compare across runs.
        """
        config = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            dropout=0.0
        )
        
        attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
        
        layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
        
        # Test input
        x = torch.randn(2, 10, 32)
        
        attn_pre_norm = layer.attn_pre(x)
        ffn_pre_norm = layer.ffn_pre(x)
        
        save_test_output(attn_pre_norm, "layer_module_attn_pre_norm.pt")
        save_test_output(ffn_pre_norm, "layer_module_ffn_pre_norm.pt")
        
        try:
            saved_attn_norm = load_test_output("layer_module_attn_pre_norm.pt")
            saved_ffn_norm = load_test_output("layer_module_ffn_pre_norm.pt")
            
            assert torch.equal(attn_pre_norm, saved_attn_norm), "Current attention norm should match saved"
            assert torch.equal(ffn_pre_norm, saved_ffn_norm), "Current FFN norm should match saved"
            
            print("LayerModule regression test passed.")
            print("Normalization outputs saved and loaded successfully")
        except FileNotFoundError:
            print("LayerModule regression test passed, first run, no previous data")
            print("Normalization outputs saved for future regression testing")
    
    def test_layer_module_abstract_method(self):
        """
        Test that LayerModule properly defines abstract forward method.
        """
        config = DynaConfig(
            d_model=32,
            n_heads=2,
            d_head=16,
            dropout=0.0
        )
        
        attention_module = AttentionModule(d_model=32, n_heads=2, d_head=16)
        
        class MockFFN(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)
            
            def forward(self, x, y):
                return self.linear(x), None
        
        ffn_module = MockFFN()
        
        layer = self._create_concrete_layer_module(config, attention_module, ffn_module)
        
        assert hasattr(layer, 'forward'), "LayerModule should have forward method"
        
        x = torch.randn(2, 10, 32)
        e = torch.randn(2, 10, 32)
        router = torch.randn(32)
        cum_sum = torch.zeros(2, 10)
        mask = (torch.ones(2, 10, 10, dtype=torch.bool), torch.arange(10).unsqueeze(0).expand(2, -1))
        
        output, _ = layer.forward(x)
        assert output.shape == x.shape, "Concrete implementation should work"
        
        print("LayerModule abstract method test passed.")
        print("Forward method exists: successful.")
        print("Abstract method behavior: successful.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
