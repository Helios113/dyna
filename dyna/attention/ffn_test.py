import torch
import torch.nn as nn
import time

class SimpleFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (bs, seqlen, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Example usage:
if __name__ == "__main__":
    bs, seqlen, dmodel, dff = 16, 1024, 1024, 4096
    x = torch.randn(bs, seqlen, dmodel)
    ffn = SimpleFFN(d_model=dmodel, d_ff=dff)
    # out = ffn(x)
    # print(out)  # should be (bs, seqlen, dmodel)
    # indeces = torch.randint(0, 3, (bs, 1))  # Randomly select 10 indices to zero out
    # mask = torch.ones_like(x, dtype=torch.bool)
    # mask[:, indeces, :] = False
    # x = x[mask].view(bs, -1, dmodel)
    # out = ffn(x)
    # print(out)

    print("Benchmarking FFN methods...")
    for i in range(seqlen + 1):
        # Method 1: Zero out elements
        x1 = x.clone()
        indeces = torch.randint(0, seqlen, (bs, i))
        x1[:, indeces, :] = 0
        start = time.time()
        out1 = ffn(x1)
        t1 = time.time() - start

        # Method 2: Mask out elements
        x2 = x.clone()
        mask = torch.ones_like(x2, dtype=torch.bool)
        indeces = torch.randint(0, seqlen, (bs, i))
        mask[:, indeces, :] = False
        start = time.time()
        x2_masked = x2[mask].view(bs, -1, dmodel)
        out2 = ffn(x2_masked)
        index_expanded = indeces.unsqueeze(-1).expand(-1, -1, dmodel)

        # Perform scatter_add
        x2.scatter_add_(dim=1, index=index_expanded, src=x2_masked)
        t2 = time.time() - start

        print(f"i={i}: zero-out={t1:.6f}s, mask-out={t2:.6f}s")
