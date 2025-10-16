import torch


def generate_standard_inputs():
    # --- 1. Define your parameters ---
    vocab_size = 49152  # The upper bound for the random integers (exclusive)
    tensor_shape = (32, 1024)  # The dimensions of your tensor
    seed = 42  # The seed for reproducibility

    # --- 2. Generate the reproducible tensor ---
    torch.manual_seed(seed)
    regenerated_tensor = torch.randint(0, vocab_size, tensor_shape)

    rows, cols = tensor_shape

    for i in range(rows):
        # For each row, decide how many zeros to insert
        # Let's say between 1 and about half the columns
        num_zeros = torch.randint(low=1, high=4, size=(1,)).item()

        # Get a random permutation of column indices and pick the first 'num_zeros'
        # torch.randperm() is great for getting unique random indices
        indices_to_zero = torch.randperm(cols)[:num_zeros]

        # Use the chosen indices to place the zeros in the current row
        regenerated_tensor[i, indices_to_zero] = 0
    return regenerated_tensor
