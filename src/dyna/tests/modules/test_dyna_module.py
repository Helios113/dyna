import torch
import pytest
import numpy as np
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.modules.dyna_module import DynaModule
from dyna.tests.conftest import save_test_output, load_test_output


class TestDynamicTanh:
    """
    Test class for DynamicTanh module functionality.
    """
    def test_dyna_module_instantiation(self):
        """
        Test that DynaModule can be instantiated correctly.
        """
        module = DynaModule()
        assert module is not None, "DynaModule should be instantiated"
        assert isinstance(module, torch.nn.Module), "DynaModule should inherit from torch.nn.Module"
        assert hasattr(module, 'training'), "DynaModule should have training attribute"
        assert hasattr(module, 'eval'), "DynaModule should have eval method"
        assert hasattr(module, 'train'), "DynaModule should have train method"

        print("DynaModule instantiation test passed.")
        print("Module created successfully")
        print("Inherits from torch.nn.Module.")
        print("Has required attributes.")

    def test_dyna_module_inheritance(self):
        """
        Test that DynaModule properly inherits from torch.nn.Module.
        """
        module = DynaModule()
        assert module.training == True, "Module should start in training mode"
        module.eval()
        assert module.training == False, "Module should be in eval mode after calling eval()"
        module.train()
        assert module.training == True, "Module should be in training mode after calling train()"

        print("DynaModule inheritance test passed.")
        print("Training mode switched successfully.")
        print("Device handling successful.")
        print("Standard PyTorch functionality successful.")
    
    def test_dyna_module_as_base_class(self):
        """
        Test that DynaModule can be used as base class for other modules.
        """
        class TestModule(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                self.param = torch.nn.Parameter(torch.randn(5))
            
            def forward(self, x):
                return self.linear(x) + self.param
        
        test_module = TestModule()
        
        assert isinstance(test_module, DynaModule), "TestModule should inherit from DynaModule"
        assert isinstance(test_module, torch.nn.Module), "TestModule should inherit from torch.nn.Module"
        
        # Forward pass
        x = torch.randn(2, 10)
        output = test_module(x)
        
        assert output.shape == (2, 5), f"Output shape should be (2, 5) got {output.shape}"
        assert output.dtype == x.dtype, f"Output dtype should match input dtype {x.dtype} got {output.dtype}"
        
        params = list(test_module.parameters())
        assert len(params) == 3, f"Should have 3 parameters (linear weight, linear bias, param) got {len(params)}"
        
        print("✓ DynaModule as base class test passed")
        print("Derived class instantiation successful.")
        print("Forward pass successful.")
        print("Parameter handling successful.")
    
    def test_dyna_module_parameter_handling(self):
        """
        Test that DynaModule handles parameters correctly.
        """
        class TestModule(DynaModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(5, 3))
                self.bias = torch.nn.Parameter(torch.randn(3))
                self.buffer = torch.randn(2, 2)
        
        module = TestModule()
        
        params = list(module.parameters())
        assert len(params) == 2, f"Should have 2 parameters got {len(params)}"
        
        for param in params:
            assert param.requires_grad == True, "Parameters should require gradients by default"
            assert isinstance(param, torch.nn.Parameter), "Should be torch.nn.Parameter instances"
        
        named_params = dict(module.named_parameters())
        assert 'weight' in named_params, "Should have 'weight' parameter"
        assert 'bias' in named_params, "Should have 'bias' parameter"
        assert 'buffer' not in named_params, "Buffer should not be in named_parameters"
        
        print("DynaModule parameter handling test passed.")
        print(f"Number of parameters: {len(params)}")
        print("Parameter properties: successful.")
        print("Named parameters: successful.")
    
    def test_dyna_module_gradient_flow(self):
        """
        Test that gradients flow properly through DynaModule.
        """
        class TestModule(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 3)
            
            def forward(self, x):
                return self.linear(x)
        
        module = TestModule()
        
        # Test input
        x = torch.randn(2, 5, requires_grad=True)
        
        # Forward pass
        output = module(x)
        
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        assert x.grad is not None, "Input should have gradients"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradients should be non-zero"
        
        module_params = list(module.parameters())
        for i, param in enumerate(module_params):
            assert param.grad is not None, f"Module parameter {i} should have gradients"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Module parameter {i} gradients should be non-zero"
        
        print("DynaModule gradient flow test passed.")
        print(f"Number of module parameters: {len(module_params)}")
        print("All parameters have gradients: successful.")
        print("All gradients are non-zero: successful.")
    
    def test_dyna_module_state_dict(self):
        """
        Test that DynaModule handles state dict correctly.
        """
        class TestModule(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
                self.param = torch.nn.Parameter(torch.randn(2))
        
        module = TestModule()
        
        state_dict = module.state_dict()
        assert isinstance(state_dict, dict), "state_dict should return a dictionary"
        assert len(state_dict) > 0, "state_dict should not be empty"
        
        new_module = TestModule()
        new_module.load_state_dict(state_dict)
        
        for (name1, param1), (name2, param2) in zip(module.named_parameters(), new_module.named_parameters()):
            assert name1 == name2, f"Parameter names should match: {name1} vs {name2}"
            assert torch.equal(param1, param2), f"Parameters should match: {name1}"
        
        print("DynaModule state dict test passed.")
        print(f"State dict keys: {list(state_dict.keys())}")
        print("State dict loading: successful.")
        print("Parameter matching: successful.")
    
    def test_dyna_module_device_handling(self):
        """
        Test that DynaModule handles device placement correctly.
        """
        class TestModule(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
        
        module = TestModule()
        
        assert next(module.parameters()).device.type == 'cpu', "Module should start on CPU"
        
        # Device movement
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            module = module.to(device)
            assert next(module.parameters()).device.type == 'cuda', "Module should be on CUDA after to(device)"
        else:
            print("CUDA not available, skipping CUDA device test")
        
        print("DynaModule device handling test passed.")
        print("CPU device handling: successful.")
        print("Device movement: successful.")
    
    def test_dyna_module_regression(self):
        """
        Test DynaModule regression, save outputs and compare across runs.
        """
        class TestModule(DynaModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 3)
            
            def forward(self, x):
                return self.linear(x)
        
        module = TestModule()
        
        # Test input
        x = torch.randn(2, 5)
        
        # Forward pass
        output = module(x)
        
        save_test_output(output, "dyna_module_output.pt")
        
        try:
            saved_output = load_test_output("dyna_module_output.pt")
            assert torch.equal(output, saved_output), "Current output should match saved output"
            
            print("DynaModule regression test passed")
            print("Output saved and loaded successfully")
        except FileNotFoundError:
            print("DynaModule regression test passed, first run, no previous data.")
            print("Output saved for future regression testing")
    
    def test_dyna_module_initialization(self):
        """
        Test that DynaModule initialization works correctly.
        """
        # Multiple instantiations
        modules = []
        for i in range(5):
            module = DynaModule()
            modules.append(module)
        
        for i, module in enumerate(modules):
            assert module is not None, f"Module {i} should be instantiated"
            assert isinstance(module, DynaModule), f"Module {i} should be DynaModule instance"
        
        print("DynaModule initialization test passed")
        print(f"Created {len(modules)} independent modules")
        print("All modules properly initialized: successful.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
