import torch
from typing import List, Tuple
from jaxtyping import Int
from torch import Tensor


def create_standard_test_sequence() -> List[int]:
    """
    Create standard test sequence with known EOS positions.
    """
    return [1, 2, 3, 0, 4, 5, 0, 6, 7, 8, 0]
    
def create_double_triangle_sequence() -> List[int]:
    """
    Create test sequence that produces double triangles in the attention mask.
    """
    return [1, 2, 3, 0, 4, 5, 6, 0]
    
def create_simple_test_sequence() -> List[int]:
    """
    Create simple test sequence for basic testing.
    """
    return [1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 0]

def find_eos_positions(sequence: List[int], eos_token_id: int = 0) -> List[int]:
    """
    Find EOS token positions in sequence.
    """
    return [i for i, token_id in enumerate(sequence) if token_id == eos_token_id]

def pad_sequence_to_length(sequence: List[int], target_length: int, pad_token_id: int = 0) -> List[int]:
    """
    Pad a sequence to a target length.
    """
    if len(sequence) >= target_length:
        return sequence[:target_length]
    return sequence + [pad_token_id] * (target_length - len(sequence))

def create_test_input_tensor(sequence: List[int], batch_size: int = 1, seq_length: int = 1024, pad_token_id: int = 0) -> Int[Tensor, "batch seq"]:
    """
    Create test input tensor.
    """
    padded_sequence = pad_sequence_to_length(sequence, seq_length, pad_token_id)
    if batch_size == 1:
        return torch.tensor([padded_sequence], dtype=torch.long)
    else:
        return torch.tensor([padded_sequence] * batch_size, dtype=torch.long)

def create_sequence_with_vocab_range(start_token: int, end_token: int, eos_positions: List[int], pad_token_id: int = 0) -> List[int]:
    """
    Create a test sequence using tokens from specific vocabulary range.
    """
    sequence = []
    current_token = start_token
    for i in range(max(eos_positions) + 1):
        if i in eos_positions:
            sequence.append(pad_token_id)
        else:
            sequence.append(current_token)
            current_token = (current_token + 1) % end_token
            if current_token == 0:
                current_token = 1
    return sequence

def verify_mask_properties(attention_mask: Tensor, source_len_mask: Tensor, expected_batch_size: int, expected_seq_length: int) -> bool:
    """
    Verify mask properties.
    """
    if attention_mask.shape != (expected_batch_size, expected_seq_length, expected_seq_length):
        return False
    if attention_mask.dtype != torch.bool:
        return False
    if source_len_mask.shape != (expected_batch_size, expected_seq_length):
        return False
    if source_len_mask.dtype != torch.long:
        return False
    
    return True

def print_test_info(sequence: List[int], eos_positions: List[int], description: str = "") -> None:
    """
    Print test information.
    """
    print(f"Test description: {description}")
    print(f"Sequence: {sequence}")
    print(f"EOS positions: {eos_positions}")

def verify_sequence_attention(mask: Tensor, start_pos: int, end_pos: int, sequence_name: str = "sequence") -> bool:
    """
    Verify sequence attention.
    """
    for i in range(start_pos, end_pos + 1):
        for j in range(start_pos, i + 1):
            if not mask[i, j]:
                raise AssertionError(f"Position {i} should attend to position {j} in {sequence_name}")
    return True

def verify_sequence_boundary_blocking(mask: Tensor, boundary_pos: int, blocked_start: int, blocked_end: int, boundary_name: str = "sequence boundary") -> bool:
    """
    Verify sequence boundary blocking.
    """
    for j in range(blocked_start, blocked_end + 1):
        if mask[boundary_pos, j]:
            raise AssertionError(f"Position {boundary_pos} should not attend to position {j} ({boundary_name})")
    return True

def verify_causal_mask_pattern(mask: Tensor, eos_positions: List[int], sequence_name: str = "test sequence") -> bool:
    """
    Verify complete causal mask pattern for sequence with known EOS positions.
    """
    # Create sequence boundaries
    boundaries = [0] + [pos + 1 for pos in eos_positions]
    
    # Verify each sequence can attend within itself
    for i, start_pos in enumerate(boundaries[:-1]):
        end_pos = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else mask.shape[0] - 1
        verify_sequence_attention(mask, start_pos, end_pos, f"{sequence_name} part {i+1}")
    
    # Verify sequence boundaries block cross-sequence attention, skip first boundary (position 0)
    for i, boundary_pos in enumerate(boundaries[1:], 1):
        blocked_end = boundary_pos - 1
        verify_sequence_boundary_blocking(
            mask, boundary_pos, 0, blocked_end, 
            f"boundary between {sequence_name} parts {i} and {i+1}"
        )
    
    return True

def verify_source_length_pattern(mask: Tensor, eos_positions: List[int]) -> bool:
    """
    Verify source length mask follows correct "saw tooth" pattern.
    """
    # Check that mask starts at 0
    if mask[0] != 0:
        raise AssertionError(f"First position should be 0, got {mask[0]}")
    
    # Check each sequence segment
    current_pos = 0
    for i, eos_pos in enumerate(eos_positions):
        # Check that positions increment within sequence
        for pos in range(current_pos + 1, eos_pos + 1):
            if mask[pos] != mask[pos - 1] + 1:
                raise AssertionError(f"Position {pos} should be {mask[pos - 1] + 1}, got {mask[pos]}")
        
        # Check that mask resets after EOS
        if eos_pos + 1 < len(mask):
            if mask[eos_pos + 1] != 0:
                raise AssertionError(f"Position {eos_pos + 1} (after EOS) should reset to 0, got {mask[eos_pos + 1]}")
        
        current_pos = eos_pos + 1
    
    return True

def get_sequence_boundaries(eos_positions: List[int], max_length: int) -> List[Tuple[int, int]]:
    """
    Get start and end positions for each sequence based on EOS positions.
    """
    boundaries = []
    start = 0
    
    for eos_pos in eos_positions:
        boundaries.append((start, eos_pos))
        start = eos_pos + 1

    if start < max_length:
        boundaries.append((start, max_length - 1))
    
    return boundaries
