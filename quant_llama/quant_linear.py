from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.int4_ops import matmul_x16w4_dequant, pack_int4_to_int32

QuantBackend = Literal["fp16_baseline", "triton_int4"]


class QuantLinear(nn.Module):
    """
    Quantized linear layer integration point.

    `fp16_baseline` backend is production-ready for integration tests.
    `triton_int4` backend uses fused int4 kernels from `kernels/`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        backend: QuantBackend = "fp16_baseline",
        quant_block_size: int = 128,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.quant_block_size = quant_block_size

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=weight_dtype, device=device)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=weight_dtype, device=device))
            if bias
            else None
        )

        # Prepared quantized representation for triton_int4 backend.
        self.register_buffer("packed_weight_int32", torch.empty(0, dtype=torch.uint32), persistent=False)
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float16), persistent=False)
        self.register_buffer("weight_zero_point", torch.empty(0, dtype=torch.float16), persistent=False)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        backend: QuantBackend = "fp16_baseline",
        quant_block_size: int = 128,
        compress_factor: int = 8,
    ) -> "QuantLinear":
        quant_layer = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            weight_dtype=layer.weight.dtype,
            device=layer.weight.device,
            backend=backend,
            quant_block_size=quant_block_size,
            compress_factor=compress_factor,
        )

        with torch.no_grad():
            quant_layer.weight.copy_(layer.weight)
            if layer.bias is not None and quant_layer.bias is not None:
                quant_layer.bias.copy_(layer.bias)
            if backend == "triton_int4":
                quant_layer.prepare_int4_weights()
        return quant_layer

    def prepare_int4_weights(self) -> None:
        if not self.weight.is_cuda:
            raise RuntimeError("triton_int4 backend requires CUDA weights")
        packed, scale = pack_int4_to_int32(
            self.weight.detach(),
            quant_block_size=self.quant_block_size,
            compress_factor=self.compress_factor,
        )
        self.packed_weight_int32 = packed
        self.weight_scale = scale
        # Symmetric quantization in current kernels does not use zero points.
        self.weight_zero_point = torch.empty(
            0,
            dtype=torch.float16,
            device=self.weight.device,
        )
        self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "fp16_baseline":
            return F.linear(x, self.weight, self.bias)

        if self.backend == "triton_int4":
            if self.packed_weight_int32.numel() == 0:
                self.prepare_int4_weights()

            original_shape = x.shape[:-1]
            x_2d = x.reshape(-1, self.in_features)
            result = matmul_x16w4_dequant(
                x_2d,
                self.packed_weight_int32,
                self.weight_scale,
                self.weight_zero_point,
            )
            result = result.reshape(*original_shape, self.out_features)
            if self.bias is not None:
                result = result + self.bias
            return result

        raise ValueError(f"Unsupported backend: {self.backend}")
