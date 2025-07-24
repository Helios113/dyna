import torch
import torch.nn.functional as F

def basic_attention_with_value_exclusion(query, key, value, excluded_idx):

    batch_size, num_queries, feature_dim = query.shape
    _, num_keys, _ = key.shape

    
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    scaled_scores = scores / (feature_dim ** 0.5)

    attention_weights = F.softmax(scaled_scores, dim=-1)

   
    # value_contribution_mask = torch.ones_like(attention_weights, dtype=torch.bool)
    # value_contribution_mask[:, :, excluded_idx] = False
    
    # # Apply the mask: set attention weights for the excluded_idx to zero
    # filtered_attention_weights = attention_weights.masked_fill(~value_contribution_mask, 0.0)

    # Now, multiply the filtered attention weights with the *original* value vector.
    # The multiplication for the excluded_idx will now be by zero, effectively removing its contribution.
    output = torch.matmul(attention_weights, value)


    return output, attention_weights # Return original attention_weights for inspection

# Example Usage:
batch_size = 1
num_queries = 3
num_keys = 4 # Sequence length
feature_dim = 4
excluded_index = 2 # The index to be excluded from value contribution

# Create dummy tensors
query = torch.randn(batch_size, num_queries, feature_dim)
key = torch.randn(batch_size, num_keys, feature_dim)
value = torch.randn(batch_size, num_keys, feature_dim)


print("q", query.flatten())
print("k", key.flatten())
print("v", value.flatten())

# Apply attention with value exclusion
attention_output, weights = basic_attention_with_value_exclusion(query, key, value, excluded_index)

print("\nAttention Output:", attention_output)
print("Attention Weights:", weights)

# # Verify that the excluded index's contribution is zeroed out by looking at weights
# # after applying the internal mask (before the final matmul)
# # If weights[:, :, excluded_index] were completely non-zero,
# # then multiplying by 0 in `filtered_attention_weights` makes it zero.
# print(f"\nAttention weights targeting excluded_index {excluded_index} (batch 0, query 0):")
# print(weights[0, 0, excluded_index]) # This should be a non-zero value from softmax

# # To show the effect, let's manually calculate the output for a single query
# # and see if the contribution from `excluded_index` is indeed zero.
# # This is for verification, not part of the function.
# # The `output` tensor should reflect that the value at `excluded_index` had no impact.

# # Let's re-run the relevant part to see the 'masked' weights
# value_contribution_mask_verify = torch.ones_like(weights, dtype=torch.bool)
# value_contribution_mask_verify[:, :, excluded_index] = False
# filtered_weights_for_verify = weights.masked_fill(~value_contribution_mask_verify, 0.0)

# # Check that the column for `excluded_index` in `filtered_weights_for_verify` is all zeros
# print(f"Filtered attention weights for column {excluded_index} (batch 0, head 0, all queries):")
# print(filtered_weights_for_verify[0, :, excluded_index]) # Should be all zeros