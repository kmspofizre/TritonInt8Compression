import torch
import triton
import triton.language as tl


@triton.jit
def get_pid(pid, GROUP_SIZE_M, num_pid_m, num_pid_n):
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def filter_invalid_configs(configs, named_args, **kwargs):
    valid_configs = []
    
    K = kwargs["K"]
    M = kwargs["M"]
    N = kwargs["N"]
    COMPRESS_FACTOR = kwargs["COMPRESS_FACTOR"]
    BLOCK_SIZE_K = kwargs.get("BLOCK_SIZE_K")

    for config in configs:
        BLOCK_SIZE_M = config.kwargs.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.kwargs.get("BLOCK_SIZE_N")
        SPLIT_K = config.kwargs.get("SPLIT_K")

        if K >= BLOCK_SIZE_K * COMPRESS_FACTOR * SPLIT_K \
            and K % (BLOCK_SIZE_K * COMPRESS_FACTOR * SPLIT_K) == 0 \
            and N >= BLOCK_SIZE_N \
            and N % BLOCK_SIZE_N == 0 \
            and 2 * M >= BLOCK_SIZE_M:
            valid_configs.append(config)

    if len(valid_configs) == 0:
        return [configs[0]]
        
    return valid_configs

@triton.autotune(
    configs= [
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 256,'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16,'GROUP_SIZE_M': 8}, num_stages=1, num_warps=2),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_N': 256,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_N': 128,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 1,  'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 64,'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1,'GROUP_SIZE_M': 8}, num_stages=1, num_warps=2),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_N': 256,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_N': 128,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 1,  'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 8,  'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 64,'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1,'GROUP_SIZE_M': 8}, num_stages=1, num_warps=2),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_N': 256,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 128,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 4,  'BLOCK_SIZE_N': 128,'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 1,  'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 8,  'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 64,'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
	    triton.Config({'SPLIT_K': 4, 'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 32,'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
    ],
    key=["M", "N", "K", "BLOCK_SIZE_K", "COMPRESS_FACTOR"],
    reset_to_zero=['output_ptr'],
    prune_configs_by={'early_config_prune': filter_invalid_configs},
    cache_results=True,
)
@triton.jit
def _decompress_int4_matmul_float16_fused_kernel(
    weights_ptr,
    scale_ptr,
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    K_compressed: tl.constexpr,
    K_scale: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    COMPRESS_FACTOR: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = get_pid(pid, GROUP_SIZE_M, num_pid_m, num_pid_n)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    rk = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)  
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    mask_m = rm < M
    mask_n = rn < N

    tl.static_assert(K_scale % SPLIT_K == 0, "K must be divisible by SPLIT_K")

    stride_xm, stride_xk = K, 1
    stride_wk, stride_wn = 1, K_compressed
    stride_sk, stride_sn = 1, K_scale

    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk) + pid_k * (K // SPLIT_K) * stride_xk
    weights_ptrs = weights_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn) + pid_k * (K_compressed // SPLIT_K) * stride_wk
    scale_ptrs = scale_ptr + (rn[None, :] * stride_sn) + pid_k * (K_scale // SPLIT_K) * stride_sk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    DTYPE = tl.uint8 if COMPRESS_FACTOR == 2 else tl.uint32
    weights = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=DTYPE)

    for i in tl.static_range(0, K_scale // SPLIT_K):
        local_step = i % COMPRESS_FACTOR
        if local_step == 0:
            weights = tl.load(weights_ptrs, mask=mask_n[None, :], other=0)
            weights_ptrs += BLOCK_SIZE_K * stride_wk
            
        weights_local = ((weights >> (local_step * 4)) & 0xF).to(tl.float16)
        scale = tl.load(scale_ptrs, mask=mask_n[None, :], other=0.0)

        weights_recon = (weights_local - 8.0) * scale
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        
        accumulator = tl.dot(x, weights_recon, accumulator, input_precision="tf32")

        x_ptrs += BLOCK_SIZE_K * stride_xk
        scale_ptrs += stride_sk

    accum_casted = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    stride_om, stride_on = N, 1
    output_ptrs = output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    output_mask = mask_m[:, None] & mask_n[None, :]

    if SPLIT_K == 1:
        tl.store(output_ptrs, accum_casted, mask=output_mask)
    else:
        tl.atomic_add(output_ptrs, accum_casted, mask=output_mask)


def decompress_int4_matmul_float16_fused(
    weights_compressed: torch.Tensor, scale: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    assert weights_compressed.is_contiguous(), "Weights compressed matrix have to be contiguous"
    assert scale.is_contiguous(), "Scales matrix have to be contiguous"
    assert x.is_contiguous(), "Input matrix have to be contiguous"
    assert len(x.shape) == 2, f"Input matrix must have only 2 dims but have {len(x.shape)}"

    assert weights_compressed.dtype in [torch.uint8, torch.uint32], f"Compressed weights must have type uint8 or uint32, but have {weights_compressed.dtype}"
    assert scale.dtype == torch.float16,  f"Scales must have float16 dtype, but have {scale.dtype}"
    assert x.dtype == torch.float16, f"Input x must have float16 dtype, but have {scale.dtype}"

    M, K = x.shape
    N, K_compressed = weights_compressed.shape

    N_scale, K_scale = scale.shape

    assert N == N_scale, "Compress weights and scale must have the same number of rows"

    compress_factor = 2 if weights_compressed.dtype == torch.uint8 else 8

    assert K == K_compressed * compress_factor, f"Mismatch in shape K != K_compressed * compress_factor {K} != {K_compressed} * {compress_factor}"
    quant_block_size = K // K_scale
    assert quant_block_size >= 16, (
        "quant_block_size must be >= 16 for tl.dot compatibility, "
        f"got quant_block_size={quant_block_size}"
    )

    output = torch.zeros(
        (M, N), device=weights_compressed.device, dtype=torch.float16
    )

    grid = lambda meta: (
        (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), meta['SPLIT_K'])
    )

    _decompress_int4_matmul_float16_fused_kernel[grid](
        M=M,
        N=N,
        K=K,
        K_compressed=K_compressed,
        K_scale=K_scale,
        weights_ptr=weights_compressed,
        scale_ptr=scale,
        x_ptr=x,
        output_ptr=output,
        BLOCK_SIZE_K=quant_block_size,
        COMPRESS_FACTOR=compress_factor,
    )
    return output
