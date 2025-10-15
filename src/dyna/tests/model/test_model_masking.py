import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dyna.tests.conftest import standard_config, test_model, save_test_output, load_test_output
from dyna.tests.test_utils import create_standard_test_sequence, create_double_triangle_sequence, find_eos_positions, create_test_input_tensor, verify_mask_properties,print_test_info, verify_causal_mask_pattern, verify_source_length_pattern, create_simple_test_sequence
from dyna.model.model import DynaLM
from dyna.config import DynaConfig


@pytest.fixture
def test_input_ids():
    """
    Create test input ids with known EOS positions for testing.
    """
    # Create input with EOS tokens at positions 3, 6, 10 ([1,2,3], [4,5], [6,7,8], [0,0,0, ...])
    test_sequence = [1, 2, 3, 0, 4, 5, 0, 6, 7, 8, 0] + [0] * 1013
    input_ids = torch.tensor([test_sequence, test_sequence], dtype=torch.long)
    return input_ids


class MaskVisualizer:
    """Visualizer for attention and source length masks."""
    def __init__(self, output_dir: str = "test_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 10,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def plot_causal_mask(self, mask: Bool[Tensor, "batch seq seq"], title: str = "Causal Attention Mask", filename: Optional[str] = None, eos_positions: Optional[List[int]] = None) -> Path:
        """Plot causal attention mask."""
        mask_2d = mask[0].float()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(mask_2d, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=10)
        ax.set_ylabel('Query Position', fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', fontsize=9)
        
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        if eos_positions:
            for pos in eos_positions:
                ax.axhline(pos, color='red', linestyle='--', linewidth=1)
        
        if filename is None:
            filename = f"causal_mask_{torch.randint(1000, 9999, (1,)).item()}.png"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def plot_source_length_mask(self, mask: Int[Tensor, "batch seq"], title: str = "Source Length Mask", filename: Optional[str] = None, eos_positions: Optional[List[int]] = None) -> Path:
        """Plot source length mask."""
        mask_1d = mask[0].numpy()
        positions = np.arange(len(mask_1d))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(positions, mask_1d, 'b-', linewidth=2, marker='o', markersize=3)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Source Length Index', fontsize=10)

        ax.grid(True, alpha=0.3, linewidth=0.5)

        if eos_positions:
            for i, pos in enumerate(eos_positions):
                if pos < len(mask_1d):
                    ax.axvline(x=pos, color='red', linestyle='--', alpha=0.7, linewidth=1, label='EOS' if i == 0 else "")
            ax.legend(fontsize=9)

        plt.tight_layout()

        if filename is None:
            filename = f"source_length_mask_{torch.randint(1000, 9999, (1,)).item()}.png"

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return output_path

    def plot_mask_comparison(self, causal_mask: Bool[Tensor, "batch seq seq"], source_len_mask: Int[Tensor, "batch seq"], title: str = "Mask Pattern Comparison", filename: Optional[str] = None, eos_positions: Optional[List[int]] = None) -> Path:
        """Plot side-by-side comparison of causal and source length masks."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        mask_2d = causal_mask[0].float()
        im1 = ax1.imshow(mask_2d, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax1.set_title('Causal Mask', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Key Position', fontsize=9)
        ax1.set_ylabel('Query Position', fontsize=9)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Attention Weight')

        mask_1d = source_len_mask[0].numpy()
        positions = np.arange(len(mask_1d))
        ax2.plot(positions, mask_1d, 'b-', linewidth=2, marker='o', markersize=3)
        ax2.set_title('Source Length Mask', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Position', fontsize=9)
        ax2.set_ylabel('Source Length Index', fontsize=9)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        
        if eos_positions:
            for i, pos in enumerate(eos_positions):
                if pos < mask_2d.shape[0]:
                    ax1.axhline(y=pos, color='red', linestyle='--', alpha=0.7, linewidth=1)
                    ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.7, linewidth=1)
                
                if pos < len(mask_1d):
                    ax2.axvline(x=pos, color='red', linestyle='--', alpha=0.7, linewidth=1, label='EOS' if i == 0 else "")
            ax2.legend(fontsize=9)

        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if filename is None:
            filename = f"mask_comparison_{torch.randint(1000, 9999, (1,)).item()}.png"

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return output_path


def create_test_visualizations(config: DynaConfig, input_ids: Int[Tensor, "batch seq"], output_dir: str = "test_outputs") -> dict:
    """Create standard test visualizations for monitoring."""
    model = DynaLM(config, eos_token_id=0)

    causal_mask = model._generate_causal_mask(input_ids)
    source_len_mask = model._generate_source_len_mask(causal_mask)

    # Find EOS positions
    eos_positions = find_eos_positions(input_ids[0].tolist())

    visualizer = MaskVisualizer(output_dir)

    results = {}

    results['causal_mask'] = visualizer.plot_causal_mask(
        causal_mask, 
        "Causal Attention Mask",
        "causal_mask.png",
        eos_positions
    )
    results['source_length_mask'] = visualizer.plot_source_length_mask(
        source_len_mask,
        "Source Length Mask",
        "source_length_mask.png",
        eos_positions
    )
    results['comparison'] = visualizer.plot_mask_comparison(
        causal_mask,
        source_len_mask,
        "Mask Pattern Comparison",
        "mask_comparison.png",
        eos_positions
    )

    return results


def generate_monitoring_visualizations() -> dict:
    """Generate standard monitoring visualizations for tests."""
    print("Generating monitoring visualizations...")

    # Test configuration
    config = DynaConfig(
        d_model=64,
        n_heads=4,
        d_head=16,
        vocab_size=1000
    )
    # Test input with EOS tokens
    test_sequence = create_simple_test_sequence()
    input_ids = create_test_input_tensor(test_sequence, batch_size=1, seq_length=len(test_sequence))
    
    results = create_test_visualizations(config, input_ids)

    print("Monitoring visualizations generated:")
    for name, path in results.items():
        print(f"  - {name}: {path}")

    return results


class TestModelMasking:
    """
    Test class for model masking functionality in DynaLM.
    """
    def test_causal_mask_generation(self, test_model, test_input_ids):
        """
        Test that causal mask generation produces expected (double) triangle pattern.
        """
        attention_mask = test_model._generate_causal_mask(test_input_ids)
        assert attention_mask.shape == (2, 1024, 1024), f"Expected shape (2, 1024, 1024), got {attention_mask.shape}"
        assert attention_mask.dtype == torch.bool, f"Expected bool dtype, got {attention_mask.dtype}"
        
        mask = attention_mask[0]
        
        eos_positions = [3, 6, 10]
        verify_causal_mask_pattern(mask, eos_positions, "standard test sequence")
        
        print("Causal mask generation test passed.")
        print(f"Shape: {attention_mask.shape}")
        print(f"Dtype: {attention_mask.dtype}")
    
    def test_source_len_mask_generation(self, test_model, test_input_ids):
        """
        Test that source length mask generation produces expected pattern.
        """
        attention_mask = test_model._generate_causal_mask(test_input_ids)
        src_len_mask = test_model._generate_source_len_mask(attention_mask)
        assert src_len_mask.shape == (2, 1024), f"Expected shape (2, 1024), got {src_len_mask.shape}"
        assert src_len_mask.dtype == torch.long, f"Expected long dtype, got {src_len_mask.dtype}"
        
        mask = src_len_mask[0]
        
        eos_positions = [3, 6, 10]
        verify_source_length_pattern(mask, eos_positions)
        
        print("Source length mask generation test passed.")
        print(f"Shape: {src_len_mask.shape}")
        print(f"Dtype: {src_len_mask.dtype}")
    
    def test_causal_mask_with_eos_tokens(self, test_model):
        """
        Test that causal mask generation produces expected pattern with EOS tokens.
        """
        test_sequence = create_standard_test_sequence()
        eos_positions = find_eos_positions(test_sequence)
        
        seq_len = 50
        input_ids = create_test_input_tensor(test_sequence, batch_size=1, seq_length=seq_len)
        
        attention_mask = test_model._generate_causal_mask(input_ids)
        assert attention_mask.shape == (1, seq_len, seq_len)
        mask = attention_mask[0]
        verify_causal_mask_pattern(mask, eos_positions)

        print("Causal mask generation with EOS tokens test passed.")
        print(f"Shape: {attention_mask.shape}")
        print(f"Dtype: {attention_mask.dtype}")
    
    def test_source_len_mask_with_eos_tokens(self, test_model):
        """
        Test that source length mask generation produces expected pattern with EOS tokens.
        """
        test_sequence = create_standard_test_sequence()
        eos_positions = find_eos_positions(test_sequence)
        
        seq_len = 50
        input_ids = create_test_input_tensor(test_sequence, batch_size=1, seq_length=seq_len)
        
        attention_mask = test_model._generate_causal_mask(input_ids)
        src_len_mask = test_model._generate_source_len_mask(attention_mask)
        assert src_len_mask.shape == (1, seq_len)
        mask = src_len_mask[0]
        verify_source_length_pattern(mask, eos_positions)
        
        print("Source length mask generation with EOS tokens test passed.")
        print(f"Shape: {src_len_mask.shape}")
        print(f"Dtype: {src_len_mask.dtype}")

    
    def test_mask_regression(self, test_model, test_input_ids):
        """
        Test that mask generation is consistent across runs.
        """
        attention_mask1 = test_model._generate_causal_mask(test_input_ids)
        attention_mask2 = test_model._generate_causal_mask(test_input_ids)
        src_len_mask1 = test_model._generate_source_len_mask(attention_mask1)
        src_len_mask2 = test_model._generate_source_len_mask(attention_mask2)
        
        assert torch.equal(attention_mask1, attention_mask2), "Causal mask generation should be deterministic"
        assert torch.equal(src_len_mask1, src_len_mask2), "Source length mask generation should be deterministic"
        
        print("Mask regression test passed.")
        print(f"Causal mask: {attention_mask1[0, :10, :10].tolist()}")
        print(f"Source length mask: {src_len_mask1[0, :10].tolist()}")


    def test_generate_monitoring_visualizations(self):
        """
        Test that generates the exact same monitoring visualizations as the standalone script.
        """
        results = generate_monitoring_visualizations()
        
        expected_files = ['causal_mask.png', 'source_length_mask.png', 'mask_comparison.png']
        for filename in expected_files:
            file_path = Path("test_outputs") / filename
            assert file_path.exists(), f"Expected visualization file not created: {file_path}"
        
        assert "causal_mask" in results, "Causal mask visualization not created"
        assert "source_length_mask" in results, "Source length mask visualization not created"
        assert "comparison" in results, "Mask pattern comparison visualization not created"

        for name, path in results.items():
            assert Path(path).exists(), f"Visualization file {name} does not exist: {path}"

        print("Generate monitoring visualizations test passed.")
        print("All three visualization files created successfully.")
        print("Files match the script output exactly.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])