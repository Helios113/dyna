"""Module for CVMMSel and CVMM Triton kernels.

This module is taken as it is from: https://github.com/RobertCsordas/moeut
All credits to their authors.
We have although slightly modified the code to fit our needs:
 - linting errors
 - support to FSDP
 - additional optimized configurations for our models
"""

# ruff: noqa: N803, N806, E731, ANN001
# type: ignore[reportMissingParameterType]
# pyright: ignore[reportMissingParameterType]

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch
import triton
import triton.language as tl

# Based on https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py
# torch.compile() fixes by Julian Büchel <jub@zurich.ibm.com>, based on https://github.com/pytorch/pytorch/issues/115344


def is_cvmm_available() -> bool:
    """Return True if CUDA is available.

    Returns
    -------
    bool
        True when CUDA is available.
    """
    return bool(torch.cuda.is_available())


# Safe wrappers to avoid import-time Triton driver initialization errors in CPU envs
T = TypeVar("T", bound=Callable[..., Any])


def _safe_autotune(  # pyright: ignore[reportUnusedFunction]
    *args: object,
    **kwargs: object,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap triton.autotune and avoid driver init errors at import time.

    Returns
    -------
    Callable[[T], T]
        A decorator that applies triton.autotune when available; otherwise
        returns the function unchanged.
    """

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        try:
            return triton.autotune(*args, **kwargs)(fn)  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            return fn

    return _decorator


def _safe_jit(fn: Callable[..., Any]) -> Callable[..., Any]:  # pyright: ignore[reportUnusedFunction]
    """Wrap triton.jit and avoid driver init errors at import time.

    Returns
    -------
    Callable[..., Any]
        The jitted function when possible, else the original function.
    """
    try:
        return triton.jit(fn)  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return fn


def cvmm_kernel(  # noqa: PLR0913, PLR0914, PLR0917, PLR0915
    # Pointers to matrices
    a_ptr: torch.Tensor,
    b_ptr: torch.Tensor,
    c_ptr: torch.Tensor,
    index_ptr: torch.Tensor,
    sel_ptr: torch.Tensor,
    out_index_ptr: torch.Tensor,
    # Matrix dimensions
    M: int,
    N: int,
    K: int,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase
    # `a_ptr` by to get the element one row down (A has M rows).
    stride_am: int,
    stride_ak: int,
    stride_bo: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    stride_index: int,
    stride_sel: int,
    stride_out_index: int,
    out_index_is_none: tl.constexpr,
    float32: tl.constexpr,
    allow_tf32: tl.constexpr,
    compute_is_bf16: tl.constexpr,
    out_bf16: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    """Kernel for computing the matmul C = A x B.

    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    """
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.

    sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
    sel_last = tl.load(
        sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel,
    )

    # Tile offsets (no modulo) and bounds masks
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    am_in_bounds = offs_am < M
    bn_in_bounds = offs_bn < N

    # Per-row expert id for the tile (only valid where am_in_bounds)
    sel_all = tl.load(
        sel_ptr + stride_sel * offs_am,
        mask=am_in_bounds,
        other=0,
    )

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    remap_offs_am = tl.load(
        index_ptr + stride_index * offs_am,
        mask=am_in_bounds,
        other=0,
    )

    # Precompute output row remap (if any) and output base pointers;
    # invariant w.r.t. expert
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if out_index_is_none:
        remap_offs_cm = remap_offs_am
    else:
        remap_offs_cm = tl.load(
            out_index_ptr + stride_out_index * offs_am,
            mask=am_in_bounds,
            other=0,
        )
    c_ptrs_base = (
        c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
    )
    bounds_c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Hoist base pointers and k-tiles outside expert loop
    a_base_ptrs = a_ptr + remap_offs_am[:, None] * stride_am
    b_base_ptrs = b_ptr + (offs_bn[None, :] * stride_bn)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    tile_has_rows = tl.sum(am_in_bounds, axis=0) > 0
    tile_has_cols = tl.sum(bn_in_bounds, axis=0) > 0
    if tile_has_rows & tile_has_cols:
        for matrix_id in range(sel_first, sel_last + 1):
            # Skip experts that have no rows in this tile
            tile_has_expert = tl.sum((sel_all == matrix_id) & am_in_bounds, axis=0) > 0
            if tile_has_expert:
                # Create offset pointers
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                tl.multiple_of(offs_k, 32)
                a_ptrs = a_base_ptrs + offs_k[None, :] * stride_ak
                b_ptrs = (
                    b_base_ptrs
                    + matrix_id * stride_bo
                    + offs_k[:, None] * stride_bk
                )
                # Triton compiler hints for vectorization (already called above)

                # -----------------------------------------------------------
                # Iterate to compute a block of the C matrix.
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for k in range(k_tiles):
                    k_mask_row = offs_k[None, :] < K - k * BLOCK_SIZE_K
                    k_mask_col = offs_k[:, None] < K - k * BLOCK_SIZE_K
                    # Apply row/col bounds on loads to avoid OOB and redundant traffic
                    a = tl.load(
                        a_ptrs,
                        mask=(am_in_bounds[:, None] & k_mask_row),
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=(k_mask_col & bn_in_bounds[None, :]),
                        other=0.0,
                    )
                    if compute_is_bf16:
                        a = a.to(tl.bfloat16)
                        b = b.to(tl.bfloat16)
                    else:
                        a = a.to(tl.float16)
                        b = b.to(tl.float16)
                    accumulator += tl.dot(a, b, allow_tf32=allow_tf32)
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk

                if float32:
                    c = accumulator
                else:
                    c = accumulator.to(tl.bfloat16 if out_bf16 else tl.float16)
                c_ptrs = c_ptrs_base
                # Fast path: if all sel in this tile are the same expert, skip eq mask
                if sel_first == sel_last:
                    c_mask = bounds_c_mask
                else:
                    c_mask = bounds_c_mask & (sel_all[:, None] == matrix_id)
                tl.store(c_ptrs, c, mask=c_mask)


@_safe_jit
def cvmm_backward_kernel3(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
    # Pointers to matrices
    a_ptr: torch.Tensor,
    b_ptr: torch.Tensor,
    c_ptr: torch.Tensor,
    index_ptr: torch.Tensor,
    sel_ptr: torch.Tensor,
    out_index_ptr: torch.Tensor,
    # Matrix dimensions
    M: int,
    N: int,
    K: int,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase
    # `a_ptr` by to get the element one row down (A has M rows).
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_co: int,
    stride_cm: int,
    stride_cn: int,
    stride_index: int,
    stride_sel: int,
    stride_out_index: int,
    out_index_is_none: tl.constexpr,
    float32_out: tl.constexpr,
    allow_tf32: tl.constexpr,
    op_float16: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    K_BLOCKS: tl.constexpr,
) -> None:


    # ----------------------------------------------------------
    # Tile offsets and bounds masks (avoid modulo)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    am_in_bounds = offs_cm < M
    bn_in_bounds = offs_cn < N

    # Base pointers hoisted per tile
    a_ptrs_this = a_ptr + offs_cm[:, None] * stride_am
    b_ptrs_this = b_ptr + offs_cn[None, :] * stride_bn

    # Precompute C tile base (matrix_id-dependent offset added inside loop)
    c_tile_base = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_bounds_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # (Removed old commented-out work partitioning logic)

    block_start_index = k_block_id * BLOCK_SIZE_K * K_BLOCKS
    block_end_index = min(block_start_index + BLOCK_SIZE_K * K_BLOCKS, K) - 1

    first_mat = tl.load(sel_ptr + stride_sel * block_start_index)
    last_mat = tl.load(sel_ptr + stride_sel * block_end_index)

    # Skip tiles with no rows/cols in bounds
    tile_has_rows = tl.sum(am_in_bounds, axis=0) > 0
    tile_has_cols = tl.sum(bn_in_bounds, axis=0) > 0
    if tile_has_rows & tile_has_cols:
        for matrix_index in range(first_mat, last_mat + 1):
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # Binary search for [start_i, end_i) range of this expert in the K-window
            start_i = block_start_index
            end_i = block_end_index + 1
            while start_i < end_i:
                middle = (start_i + end_i) // 2
                middle_matrix = tl.load(sel_ptr + middle * stride_sel)
                if middle_matrix < matrix_index:
                    start_i = middle + 1
                else:
                    end_i = middle

            start_i2 = start_i
            end_i = block_end_index + 1
            while start_i2 < end_i:
                middle = (start_i2 + end_i) // 2
                middle_matrix = tl.load(sel_ptr + middle * stride_sel)
                if middle_matrix <= matrix_index:
                    start_i2 = middle + 1
                else:
                    end_i = middle

            end_i = start_i2
            count = end_i - start_i

            if count > 0:
                block_mem_indices_f_base = start_i + tl.arange(0, BLOCK_SIZE_K)

                # Full tiles and tail for fewer masked loads
                full_iters = count // BLOCK_SIZE_K
                tail = count % BLOCK_SIZE_K

                # Process full tiles without sel_ok masks
                for it in range(full_iters):
                    block_mem_indices_f = block_mem_indices_f_base + it * BLOCK_SIZE_K
                    # block_mem_indices_f < end_i is true for full tiles by construction
                    block_mem_indices = block_mem_indices_f  # within [0, K)

                    a_index = tl.load(index_ptr + stride_index * block_mem_indices)
                    if out_index_is_none:
                        b_index = a_index
                    else:
                        b_index = tl.load(
                            out_index_ptr + stride_out_index * block_mem_indices,
                        )

                    a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                    b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk

                    a = tl.load(a_ptrs, mask=am_in_bounds[:, None], other=0.0)
                    b = tl.load(b_ptrs, mask=bn_in_bounds[None, :], other=0.0)

                    if op_float16:
                        a = a.to(tl.float16)
                        b = b.to(tl.float16)
                    accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

                # Tail iteration with mask if needed
                if tail > 0:
                    block_mem_indices_f = (
                        block_mem_indices_f_base + full_iters * BLOCK_SIZE_K
                    )
                    block_mem_indices = block_mem_indices_f
                    a_index = tl.load(
                        index_ptr + stride_index * block_mem_indices,
                        mask=(block_mem_indices_f < end_i),
                        other=0,
                    )
                    if out_index_is_none:
                        b_index = a_index
                    else:
                        b_index = tl.load(
                            out_index_ptr + stride_out_index * block_mem_indices,
                            mask=(block_mem_indices_f < end_i),
                            other=0,
                        )
                    sel_ok = block_mem_indices_f < end_i

                    a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                    b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk

                    a = tl.load(
                        a_ptrs,
                        mask=(am_in_bounds[:, None] & sel_ok[None, :]),
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=(sel_ok[:, None] & bn_in_bounds[None, :]),
                        other=0.0,
                    )
                    if op_float16:
                        a = a.to(tl.float16)
                        b = b.to(tl.float16)
                    accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

                c = accumulator if float32_out else accumulator.to(tl.float16)

                # Write back C (per expert)
                c_ptrs = c_ptr + stride_co * matrix_index + c_tile_base
                if USE_ATOMICS:
                    tl.atomic_add(c_ptrs, c, mask=c_bounds_mask)
                else:
                    tl.store(c_ptrs, c, mask=c_bounds_mask)


torch.library.define(
    "mylib::cvmm_triton",
    "(Tensor x, Tensor sel_index, Tensor sel, Tensor keys, ScalarType out_dtype,"
    " Tensor out_index) -> Tensor",
)


    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride] # noqa: ANN205, D102
        ctx,  # pyright: ignore[reportMissingParameterType]
        grad_output,  # pyright: ignore[reportMissingParameterType]
    ):
        x, keys, sel, sel_index, out_index, reduction_weight = ctx.saved_tensors
        keys_dt = keys

        # Backward for weight
        if reduction_weight is not None:
            # Project back the grads with he reduction weight, so the grad for the
            # weight matrix is ok
            grad_output_w = reduction_weight.unsqueeze(-1).type_as(
                grad_output,
            ) @ grad_output.unsqueeze(-2)
        else:
            grad_output_w = grad_output

        out_index_is_none = False
        if out_index is None:
            out_index_is_none = True
            out_index = torch.tensor(-1).cuda()

        grad_w = cvmm_triton_backward(
            x,
            sel_index,
            sel,
            grad_output_w,
            keys_dt.shape[0],
            ctx.keys_type,
            ctx.is_autocast,
            out_index=out_index,
        )

        # Backward for input and reduction weight
        grad_w_off = None

        bw_index = sel_index if out_index_is_none else out_index
        bw_index_out = torch.tensor(-1).cuda()
        if reduction_weight is not None:
            # Manipulate the output indices to emulate repeats
            bw_index_out = bw_index
            bw_index = bw_index // reduction_weight.shape[-1]  # noqa: PLR6104

        grad_x_full = torch.ops.mylib.cvmm_triton(
            grad_output,
            bw_index,
            sel,
            keys_dt.transpose(1, 2),
            ctx.op_type,
            bw_index_out,
        )  # pyright: ignore[reportCallIssue]

        grad_x_full = grad_x_full.view(*x.shape[:-1], -1, x.shape[-1])
        if reduction_weight is not None:
            # The grad_x_full is the unscaled grad. For the input, we have to scale it,
            # for the reduction weight, we have to compute dot products with the input.
            grad_x = (
                reduction_weight.view(*grad_x_full.shape[:-1])
                .unsqueeze(-2)
                .type_as(grad_x_full)
                @ grad_x_full
            ).squeeze(-2)
            grad_w_off = (
                (
                    grad_x_full.type_as(reduction_weight)
                    @ x.unsqueeze(-1).type_as(reduction_weight)
                )
                .squeeze(-1)
                .view_as(reduction_weight)
            )
        elif grad_x_full.shape[-2] != 1:
            grad_x = grad_x_full.sum(-2)
        else:
            grad_x = grad_x_full

        grad_x = grad_x.view_as(x)

        return grad_x, None, None, grad_w, None, grad_w_off


def cvmm(
    x: torch.Tensor,
    sel: torch.Tensor | CVMMSel,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Compute the CVMM operation.

    This function calls the CVMM kernel with the prepared selection tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor of shape (batch_size, seq_length, d_model).
    sel : torch.Tensor | CVMMSel
        Prepared selection structure or raw selection tensor.
    keys : torch.Tensor
        Weight tensor of shape (O, K, N). The shape can be:
        - (n_experts, d_model, expert_size) for the up projection, or
        - (n_experts, expert_size, d_model) for the down projection.

    Returns
    -------
    torch.Tensor
        The output tensor.
    """
    if not isinstance(sel, CVMMSel):
        sel = cvmm_prepare_sel(sel)

    return cast(
        "torch.Tensor",
        CVMM.apply(
            x,
            # Token position in original selection tensor of the sorted by expert ids
            # selection tensor, (batch_size * seq_length * n_active_experts).
            sel.sel_index,
            # Sorted by expert ids selection tensor,
            # (batch_size, seq_length, n_active_experts).
            sel.sel,
            # Up/Down projection parameters
            keys,
            # Indices of the sorted selection tensor in the flatten original selection
            # tensor, (batch_size * seq_length * n_active_experts).
            sel.out_index,
            sel.reduction_weight,  # None for the up projection
        ),
    )
