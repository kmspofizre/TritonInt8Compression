import torch

from kernels.compression_int4 import quant_compress_to_int4
from kernels.decompress_int4_matmul_float16_fused import (
    decompress_int4_matmul_float16_fused,
)

_BLOCK_SIZE_CANDIDATES = (128, 64, 32, 16)


def _choose_quant_block_size(cols: int, requested: int, compress_factor: int) -> int:
    candidates = [requested] + [x for x in _BLOCK_SIZE_CANDIDATES if x != requested]
    for block_size in candidates:
        if block_size < 16:
            continue
        if cols % block_size != 0:
            continue
        n_blocks_per_row = cols // block_size
        if n_blocks_per_row % compress_factor != 0:
            continue
        return block_size

    raise ValueError(
        "Cannot choose valid quant_block_size for int4 packing: "
        f"cols={cols}, requested={requested}, compress_factor={compress_factor}. "
        "Require block_size >= 16, cols % block_size == 0, and (cols // block_size) % compress_factor == 0."
    )


def quantize_to_int4_no_pack(
    weights: torch.Tensor,
    quant_block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns unpacked int4 values in range [-8, 7] and per-block scales.
    """
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D, got shape {tuple(weights.shape)}")
    if weights.shape[1] % quant_block_size != 0:
        raise ValueError(
            f"weights.shape[1]={weights.shape[1]} must be divisible by quant_block_size={quant_block_size}"
        )

    rows, cols = weights.shape
    blocks = cols // quant_block_size
    reshaped = weights.contiguous().view(rows, blocks, quant_block_size).to(torch.float16)
    scales = reshaped.abs().amax(dim=-1) / 7.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quant = torch.round(reshaped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    return quant.view(rows, cols), scales


def pack_int4_to_int32(
    weights: torch.Tensor,
    quant_block_size: int = 128,
    compress_factor: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Packs 8 int4 values into uint32 and returns (packed, scales).
    """
    cols = weights.shape[1]
    effective_block_size = _choose_quant_block_size(
        cols=cols,
        requested=quant_block_size,
        compress_factor=compress_factor,
    )
    packed, scale = quant_compress_to_int4(
        weights=weights.to(torch.float16).contiguous(),
        compress_factor=compress_factor,
        quant_block_size=effective_block_size,
    )
    return packed, scale


def matmul_x16w4_dequant(
    x: torch.Tensor,
    packed_weight_int32: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused X16 @ W4 matmul with in-kernel dequantization.
    """
    if weight_zero_point is not None and weight_zero_point.numel() > 0:
        raise NotImplementedError("weight_zero_point is not used in symmetric int4 kernels")

    return decompress_int4_matmul_float16_fused(
        weights_compressed=packed_weight_int32.contiguous(),
        scale=weight_scale.contiguous(),
        x=x.to(torch.float16).contiguous(),
    )
