"""Module for CVMMSel and CVMM Triton kernels.

This module is taken as it is from: https://github.com/RobertCsordas/moeut
All credits to their authors.

"""

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import triton
import triton.language as tl

# Based on https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py
# torch.compile() fixes by Julian Büchel <jub@zurich.ibm.com>, based on https://github.com/pytorch/pytorch/issues/115344

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_dtype() -> torch.dtype:
    """Get the appropriate dtype based on autocast settings.

    Returns
    -------
    torch.dtype
        The appropriate dtype for the current context.

    """
    if not torch.is_autocast_enabled():
        return torch.float32
    return torch.get_autocast_gpu_dtype()


def dtype_to_type_id(dtype: torch.dtype) -> Literal[0, 1, 2]:
    """Map torch dtype to triton type id.

    Parameters
    ----------
    dtype : torch.dtype
        The input torch dtype.

    Returns
    -------
    Literal[0, 1, 2]
        The corresponding triton type id.

    Raises
    ------
    ValueError
        If the input dtype is not supported.

    """
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2

    msg = "Unknown dtype"
    raise ValueError(msg)



def cvmm_kernel(  # noqa: PLR0913, PLR0914, PLR0917
    # Pointers to matrices
    a_ptr: torch.Tensor,
    b_ptr: torch.Tensor,
    c_ptr: torch.Tensor,
    index_ptr: torch.Tensor,
    sel_ptr: torch.Tensor,
    out_index_ptr: torch.Tensor,
    # Matrix dimensions
    M: int,  # noqa: N803
    N: int,  # noqa: N803
    K: int,  # noqa: N803
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
    dtype_id: tl.constexpr,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_N: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_K: tl.constexpr,  # noqa: N803
    GROUP_SIZE_M: tl.constexpr,  # noqa: N803
) -> None:
    """Kernel for computing the matmul C = A x B.

    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.


    

    sel_first = tl.load(sel_ptr + pid_m * BLOCK_SIZE_M * stride_sel)
    sel_last = tl.load(sel_ptr + (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) * stride_sel)
    sel_all = tl.load(
        sel_ptr
        + stride_sel * ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M),
    )

    for matrix_id in range(sel_first, sel_last + 1):
        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetics` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        remap_offs_am = tl.load(index_ptr + stride_index * offs_am)

        # Create offset pointers
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (
            remap_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + matrix_id * stride_bo
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K
            # dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.

            # Triton was unhappy with passing dtypes as vars.
            if dtype_id == 1:
                a = a.to(tl.float16)
                b = b.to(tl.float16)
            elif dtype_id == 2:  # noqa: PLR2004
                a = a.to(tl.bfloat16)
                b = b.to(tl.bfloat16)

            accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if dtype_id == 1:
            c = accumulator.to(tl.float16)
        elif dtype_id == 2:  # noqa: PLR2004
            c = accumulator.to(tl.bfloat16)
        else:
            c = accumulator

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        if out_index_is_none:
            remap_offs_cm = remap_offs_am
        else:
            remap_offs_cm = tl.load(out_index_ptr + stride_out_index * offs_am)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = ((offs_cm[:, None] < M) & (sel_all[:, None] == matrix_id)) & (
            offs_cn[None, :] < N
        )
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def cvmm_backward_kernel(  # noqa: C901, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917
    # Pointers to matrices
    a_ptr: torch.Tensor,
    b_ptr: torch.Tensor,
    c_ptr: torch.Tensor,
    index_ptr: torch.Tensor,
    sel_ptr: torch.Tensor,
    out_index_ptr: torch.Tensor,
    # Matrix dimensions
    M: int,  # noqa: N803
    N: int,  # noqa: N803
    K: int,  # noqa: N803
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
    out_dtype_id: tl.constexpr,
    allow_tf32: tl.constexpr,
    dtype_id: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_N: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_K: tl.constexpr,  # noqa: N803
    GROUP_SIZE_M: tl.constexpr,  # noqa: N803
    K_BLOCKS: tl.constexpr,  # noqa: N803
) -> None:

    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.

    a_ptrs_this = a_ptr + offs_am[:, None] * stride_am
    b_ptrs_this = b_ptr + offs_bn[None, :] * stride_bn

    block_start_index = k_block_id * BLOCK_SIZE_K * K_BLOCKS
    block_end_index = min(block_start_index + BLOCK_SIZE_K * K_BLOCKS, K) - 1

    first_mat = tl.load(sel_ptr + stride_sel * block_start_index)
    last_mat = tl.load(sel_ptr + stride_sel * block_end_index)

    for matrix_index in range(first_mat, last_mat + 1):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        start_i = block_start_index
        end_i = block_end_index + 1
        while start_i < end_i:
            middle = (start_i + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix < matrix_index:
                start_i = middle + 1
            else:
                end_i = middle

        # # Continue binary search: find the first matrix that is > matrix_index
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

        block_mem_indices_f_base = start_i + tl.arange(0, BLOCK_SIZE_K)

        if count > 0:
            for k in range((count + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
                block_mem_indices_f = block_mem_indices_f_base + k * BLOCK_SIZE_K
                block_mem_indices = block_mem_indices_f % K
                a_index = tl.load(index_ptr + stride_index * block_mem_indices)
                if out_index_is_none:
                    b_index = a_index
                else:
                    b_index = tl.load(
                        out_index_ptr + stride_out_index * block_mem_indices,
                    )
                sel_ok = block_mem_indices_f < end_i

                a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk

                # Load the next block of A and B, generate a mask by checking the K
                # dimension.
                # If it is out of bounds, set it to 0.
                a = tl.load(a_ptrs, mask=sel_ok[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=sel_ok[:, None], other=0.0)

                if dtype_id == 1:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)
                elif dtype_id == 2:  # noqa: PLR2004
                    a = a.to(tl.bfloat16)
                    b = b.to(tl.bfloat16)

                # We accumulate along the K dimension.
                accumulator += tl.dot(a, b, allow_tf32=allow_tf32)

            if out_dtype_id == 1:
                c = accumulator.to(tl.float16)
            elif out_dtype_id == 2:  # noqa: PLR2004
                c = accumulator.to(tl.bfloat16)
            else:
                c = accumulator

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = (
                c_ptr
                + stride_co * matrix_index
                + stride_cm * offs_cm[:, None]
                + stride_cn * offs_cn[None, :]
            )
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.atomic_add(c_ptrs, c, mask=c_mask)






    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride] # noqa: ANN205, D102
        ctx,  # pyright: ignore[reportMissingParameterType]  # noqa: ANN001
        grad_output,  # pyright: ignore[reportMissingParameterType]  # noqa: ANN001
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
            ctx.dtype,
            out_index=out_index,
        )

        # Backward for input and reduction weight
        grad_w_off = None

        bw_index = sel_index if out_index_is_none else out_index
        bw_index_out = torch.tensor(-1).cuda()
        if reduction_weight is not None:
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
            # grad_x_full is the unscaled grad. For the input, we have to scale it, for
            # the reduction weight, we have to compute dot products with the input.
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


def cvmm_prepare_sel(sel: torch.Tensor) -> CVMMSel:
    """Prepare the selection tensor for the CVMM kernel.

    This function takes the selection tensor and prepares it for the CVMM kernel
    by flattening it and computing the selection scores and indices.

    Parameters
    ----------
    sel : torch.Tensor
        The selection tensor to prepare.

    Returns
    -------
    CVMMSel
        The prepared CVMMSel object.

    """
    flatten_sel = sel.flatten()
    sel_scores, sel_index = flatten_sel.sort()
    return CVMMSel(sel, sel_scores.view_as(sel), sel_index, None)


def cvmm_prepare_sel2(sel: torch.Tensor, w: torch.Tensor | None = None) -> CVMMSel:
    """Prepare the selection tensor for CVMM.

    This function takes a selection tensor and prepares it for use in the
    CVMM operation by computing the necessary indices and weights.

    Parameters
    ----------
    sel : torch.Tensor
        The selection tensor (indices), expected shape is
        (batch_size, seq_length, n_active_experts).
    w : torch.Tensor | None
        The weights tensor, expected shape is
        (batch_size, seq_length, n_active_experts).

    Returns
    -------
    CVMMSel
        The prepared CVMM selection.

    """
    # Get the number of active experts (selected indices) for this batch
    n_selected_indices_per_batch = sel.shape[-1]
    # Flatten the sel tensor to have size (batch_size * seq_length * n_active_experts)
    flatten_sel = sel.flatten()
    # Sort by expert index, 0-th expert first, and return the flattened sel tensor
    # (sorted_original_sel_index)  and, also, return the indices of the original
    # positions in the `flatten_sel` (sorted_per_expert_sel_index)
    # NOTE: The values in `sorted_per_expert_sel_index` are original experts indices and
    # thus can be in the range [0, n_total_experts - 1].
    # NOTE: the values in `sorted_per_expert_sel_index` are the original indices of the
    # `flattened_sel` and thus can be in the range
    # [0, batch_size * seq_length * n_active_experts - 1].
    sorted_flatten_sel, indices_of_flatten_sel = flatten_sel.sort()
    # Indices referring to the token position in the unflattened `sel` as if the samples
    # in a batch are concatenated sequentially
    in_index = indices_of_flatten_sel // n_selected_indices_per_batch

    return CVMMSel(
        # Original selection tensor, (batch_size, seq_length, n_active_experts)
        raw_sel=sel,
        # Sorted by expert ids selection tensor,
        # (batch_size, seq_length, n_active_experts).
        sel=sorted_flatten_sel.view_as(sel),
        # Token position in original selection tensor of the sorted by expert ids
        # selection tensor, (batch_size * seq_length * n_active_experts).
        sel_index=in_index,
        # Indices of the sorted selection tensor in the flatten original selection
        # tensor, (batch_size * seq_length * n_active_experts).
        out_index=indices_of_flatten_sel,
        # None for the up projection
        reduction_weight=w,
    )