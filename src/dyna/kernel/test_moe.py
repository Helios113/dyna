@lib_decorator
def cvmm_triton(
    x: torch.Tensor,
    selction_index: torch.Tensor,
    affinity: torch.Tensor,
    keys: torch.Tensor,
    out_dtype: torch.dtype,
    out_index: torch.Tensor,
):
    x = x.flatten(end_dim=-2)
    assert x.shape[-1] == keys.shape[1]

    sel_shape = sel.shape
    sel = sel.flatten()

    M = sel.shape[0]
    O, K, N = keys.shape
    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)
    # out = torch.zeros((M, N), device=x.device, dtype=out_dtype)
    # 1D launch kernel where each block gets its own program.

    # expected_m_per_matrix = int(math.ceil(M / O * 1.5))
    # expected_m_per_matrix = M

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    out_index_is_none = False
    if out_index.numel() == 1 and out_index == -1:
        out_index_is_none = True

    cvmm_kernel[grid](
        x,
        keys,
        out,
        sel_index,
        sel,
        out_index,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        out.stride(0),
        out.stride(1),
        sel_index.stride(0),
        sel.stride(0),
        0 if out_index_is_none else out_index.stride(0),
        out_index_is_none=out_index_is_none,
        dtype_id=dtype_to_type_id(out.dtype),
        allow_tf32=False,  # torch.backends.cuda.matmul.allow_tf32
    )

    return out.view(*sel_shape, N)
