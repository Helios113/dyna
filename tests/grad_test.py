import torch

from dyna.attention.basic_attention import BasicAttn

B, S = 2, 32
d_model, n_heads, d_head = 256, 4, 64
x = torch.randn(B, S, d_model)  # input
mask = torch.ones(B, 1, S, S, dtype=torch.bool)
seq = torch.arange(S).expand(B, S)

attn = BasicAttn(d_model, n_heads, d_head)
print("requires_grad:", attn.q.weight.requires_grad, attn.k.weight.requires_grad)

# Forward with grad tracking
y, _ = attn(x, x, x, mask, seq)
print("y.requires_grad:", y.requires_grad)

# Explicit grad query
g = torch.autograd.grad(
    y.pow(2).mean(), attn.q.weight, retain_graph=True, allow_unused=True
)
print("autograd.grad(q.weight) is None?", g[0] is None)

(y.pow(2).mean()).backward()
print(
    "backward grad norm q:",
    (attn.q.weight.grad.norm().item() if attn.q.weight.grad is not None else None),
)
