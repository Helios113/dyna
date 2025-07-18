import torch
import time
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

def benchmark(loss_fn, input, target, n_iter=100, is_tuple=False):
    torch.cuda.synchronize()
    # Forward timing
    start = time.time()
    for _ in range(n_iter):
        loss_fn(input, target)
    torch.cuda.synchronize()
    forward_time = (time.time() - start) / n_iter

    # Backward timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        input.grad = None
        loss = loss_fn(input, target)
        if is_tuple:
            loss = loss[0]

        loss.mean().backward()
    torch.cuda.synchronize()
    backward_time = (time.time() - start) / n_iter

    return forward_time, backward_time

batch_size = 1024
vocab_size = 50257
device = "cuda"

input = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16, requires_grad=True)
target = torch.randint(0, vocab_size, (batch_size,), device=device)

# Torch loss
torch_loss_fn = torch.nn.CrossEntropyLoss()
input_torch = input.clone().detach().requires_grad_()
target_torch = target.clone().detach()
torch_forward, torch_backward = benchmark(torch_loss_fn, input_torch, target_torch, is_tuple=False)

input_flash = input.clone().detach().requires_grad_()
target_flash = target.clone().detach()
flash_forward, flash_backward = benchmark(cross_entropy_loss, input_flash, target_flash, is_tuple=True)

print(f"Torch CrossEntropyLoss avg forward time: {torch_forward:.6f} s")
print(f"Torch CrossEntropyLoss avg backward time: {torch_backward:.6f} s")
print(f"FlashAttn CrossEntropyLoss avg forward time: {flash_forward:.6f} s")
print(f"FlashAttn CrossEntropyLoss avg backward time: {flash_backward:.6f} s")
