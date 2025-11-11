import torch


class RMSNorm(torch.nn.RMSNorm):
    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """RMSNorm Class."""
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.eps,
        ).to(dtype=x.dtype)
