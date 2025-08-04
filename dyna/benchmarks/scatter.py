import torch
import time
x = torch.rand(16, 1024, 1024).cuda()
cum_sum = torch.rand(16,1024).cuda()
skip_mask = cum_sum < 0.5
lengths = skip_mask.sum(dim=1)  # number of valid entries per batch
max_len = lengths.max()


t1 = time.time()

positions = torch.zeros(x.shape[0], max_len, dtype=torch.long, device=x.device)

        # compact batch
out = torch.zeros(x.shape[0], max_len, x.shape[-1], device=x.device)
 
for i in range(10000):
 
    batch_idx, seq_idx = skip_mask.nonzero(as_tuple=True)
# Create a mapping for efficient packing
    if batch_idx.numel() > 0:
        # Count valid tokens per batch
        batch_counts = torch.bincount(batch_idx, minlength=x.shape[0])
        cumsum_counts = torch.cumsum(torch.cat([torch.tensor([0], device=x.device), batch_counts[:-1]]), dim=0)
        
        # Create output indices
        local_indices = torch.arange(batch_idx.numel(), device=x.device) - cumsum_counts[batch_idx]
        valid_mask = local_indices < max_len
        
        if valid_mask.any():
            # Pack efficiently
            out[batch_idx[valid_mask], local_indices[valid_mask]] = x[batch_idx[valid_mask], seq_idx[valid_mask]]
            positions[batch_idx[valid_mask], local_indices[valid_mask]] = seq_idx[valid_mask]

t2 = time.time()

print(t2-t1)


t1 = time.time()


positions = torch.zeros(x.shape[0], max_len, dtype=torch.long, device=x.device)

        # compact batch
out1 = torch.zeros(x.shape[0], max_len, x.shape[-1], device=x.device)

idx = 0
# Exit condition needs to be here?
# If elements of a batch are empty we can start to remove them one by one.
# Don't forget the masking still doesn't work.
# Ideally we can do something about this. -- packing instead of padding?

for i in range(10000):
    for i in range(x.shape[0]):
        idx = skip_mask[i].nonzero(as_tuple=False).squeeze(-1)  # shape (n,)
        n = idx.numel()
        if n > 0:
            out1[i, :n] = x[i, idx]
            positions[i, :n] = idx

t2 = time.time()

print(t2-t1)

print(torch.allclose(out,out1))


