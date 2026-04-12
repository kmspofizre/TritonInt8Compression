import torch
import triton
import triton.language as tl


@triton.jit
def _quant_compress_kernel_to_int4(
    rows,
    cols,
    x_ptr,
    output_ptr,
    scale_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_FACTOR: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    input_start = (
        x_ptr
        + (pid_n * cols + pid_m * BLOCK_SIZE * COMPRESS_FACTOR)
        + tl.arange(0, BLOCK_SIZE)
    )
    output_start = (
        output_ptr
        + (pid_n * cols // COMPRESS_FACTOR + pid_m * BLOCK_SIZE)
        + tl.arange(0, BLOCK_SIZE)
    )
    scale_start = scale_ptr + (pid_n * cols // BLOCK_SIZE + pid_m * COMPRESS_FACTOR)

    out = tl.zeros((BLOCK_SIZE,), dtype=OUTPUT_DTYPE)

    for i in tl.static_range(COMPRESS_FACTOR):
        x = tl.load(input_start)

        scale = tl.max(tl.abs(x)) / 7.0
        scale = tl.where(scale > 0, scale, 1.0)

        tl.store(scale_start, scale)

        x_scaled = x / scale
        x_scaled = tl.maximum(tl.minimum(x_scaled, 7), -8) + 8

        x_uint8 = x_scaled.cast(OUTPUT_DTYPE)
        out += x_uint8 << (i * 4)

        input_start += BLOCK_SIZE
        scale_start += 1

    tl.store(output_start, out)


def quant_compress_to_int4(
    weights: torch.Tensor, compress_factor: int = 2, quant_block_size: int = 128
) -> torch.Tensor:
    rows, cols = weights.shape
    assert weights.is_contiguous(), "Weights matrix have to be contiguous"
    assert cols % quant_block_size == 0, (
        f"Number of cols {cols} have to be divisible by {quant_block_size=}"
    )

    if compress_factor == 2:
        assert cols % 2 == 0, (
            f"Number of cols ({cols}) have to be divisible by 2, to pack into int8"
        )
        output_dtype = torch.uint8
        output_tl_dtype = tl.uint8
    elif compress_factor == 8:
        assert cols % 8 == 0, (
            f"Number of cols ({cols}) have to be divisible by 8, to pack into int32"
        )
        output_dtype = torch.uint32
        output_tl_dtype = tl.uint32
    else:
        raise ValueError(
            f"Invalid compress factor: {compress_factor} doesn't supported, use [2, 8] to compress into [uint8, uint32]"
        )

    output_cols = cols // compress_factor
    n_blocks_per_row = cols // quant_block_size
    output = torch.empty((rows, output_cols), device=weights.device, dtype=output_dtype)
    scale = torch.empty(
        (rows, n_blocks_per_row), device=weights.device, dtype=torch.float16
    )

    assert output.is_contiguous()
    assert scale.is_contiguous()

    grid = lambda meta: (
        # triton.cdiv(rows, meta["BLOCK_M"]),
        # triton.cdiv(IN, meta["BLOCK_N"]),
        rows,
        n_blocks_per_row // compress_factor,
    )

    _quant_compress_kernel_to_int4[grid](
        rows=rows,
        cols=cols,
        x_ptr=weights,
        output_ptr=output,
        scale_ptr=scale,
        BLOCK_SIZE=quant_block_size,
        COMPRESS_FACTOR=compress_factor,
        OUTPUT_DTYPE=output_tl_dtype,
    )
    return output, scale


@triton.jit
def _dequant_decompress_kernel_from_int4_to_float16(
    rows,
    cols_compressed,
    cols_original,
    cols_scale,
    x_ptr,
    scale_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_FACTOR: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    input_start = (
        x_ptr
        + (pid_n * cols_compressed + pid_m * BLOCK_SIZE)
        + tl.arange(0, BLOCK_SIZE)
    )
    output_start = (
        output_ptr
        + (pid_n * cols_original + pid_m * BLOCK_SIZE * COMPRESS_FACTOR)
        + tl.arange(0, BLOCK_SIZE)
    )
    scale_start = scale_ptr + (pid_n * cols_scale + pid_m * COMPRESS_FACTOR)
    x = tl.load(input_start)

    for i in tl.static_range(COMPRESS_FACTOR):
        x_local = (x & 0xF).cast(tl.float16)

        scale = tl.load(scale_start)

        out = (x_local - 8.0) * scale
        tl.store(output_start, out)

        x >>= 4
        output_start += BLOCK_SIZE
        scale_start += 1


def dequant_decompress_from_int4_to_float16(
    weights_compressed: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    assert weights_compressed.is_contiguous(), (
        "Weights compressed matrix have to be contiguous"
    )
    assert scale.is_contiguous(), "Scales matrix have to be contiguous"
    assert weights_compressed.dtype in [torch.uint8, torch.uint32], (
        f"Compressed weights must have type uint8 or uint32, but have {weights_compressed.dtype}"
    )
    assert scale.dtype == torch.float16, (
        f"Scales must have float16 dtype, but have {scale.dtype}"
    )
    rows, cols_compressed = weights_compressed.shape
    rows_scale, cols_scale = scale.shape
    assert rows == rows_scale, (
        "Compress weights and scale must have the same number of rows"
    )

    compress_factor = 2 if weights_compressed.dtype == torch.uint8 else 8

    cols_original = cols_compressed * compress_factor
    quant_block_size = cols_original // cols_scale

    output = torch.empty(
        (rows, cols_original), device=weights_compressed.device, dtype=torch.float16
    )

    grid = lambda meta: (
        # triton.cdiv(rows, meta["BLOCK_M"]),
        # triton.cdiv(IN, meta["BLOCK_N"]),
        rows,
        cols_original // (quant_block_size * compress_factor),
    )

    _dequant_decompress_kernel_from_int4_to_float16[grid](
        rows=rows,
        cols_compressed=cols_compressed,
        cols_original=cols_original,
        cols_scale=cols_scale,
        x_ptr=weights_compressed,
        scale_ptr=scale,
        output_ptr=output,
        BLOCK_SIZE=quant_block_size,
        COMPRESS_FACTOR=compress_factor,
    )
    return output
