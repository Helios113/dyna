import torch


class LayerScaledIdentityFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, total_layers: int):
        # Save values needed for backward
        ctx.total_layers = total_layers
        return input.clone()  # return unchanged

    @staticmethod
    def backward(ctx, grad_output):
        # Compute scaling factor
        grad_input = grad_output * ctx.total_layers
        return grad_input, None


def layer_scaled_identity(x, total_layers: int):
    return LayerScaledIdentityFn.apply(x, total_layers)
