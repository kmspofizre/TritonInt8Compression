from __future__ import annotations

import torch

from kernels.compression_int4 import quant_compress_to_int4
from kernels.decompress_int4_matmul_float16_fused import (
    decompress_int4_matmul_float16_fused,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Packs 8 int4 values into uint32 and returns (packed, scales).
    """
    packed, scale = quant_compress_to_int4(
        weights=weights.to(torch.float16).contiguous(),
        compress_factor=8,
        quant_block_size=quant_block_size,
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
