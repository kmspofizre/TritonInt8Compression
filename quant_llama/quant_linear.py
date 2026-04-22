from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.int4_ops import matmul_x16w4_dequant

QuantBackend = Literal["fp16_baseline", "triton_int4"]


class QuantLinear(nn.Module):
    """
    Quantized linear layer integration point.

    `fp16_baseline` backend is production-ready for integration tests.
    `triton_int4` backend depends on kernels owned by other participants.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        backend: QuantBackend = "fp16_baseline",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=weight_dtype, device=device)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=weight_dtype, device=device))
            if bias
            else None
        )

        # Placeholders for kernels owned by other participants.
        self.register_buffer("packed_weight_int32", torch.empty(0, dtype=torch.int32), persistent=False)
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("weight_zero_point", torch.empty(0, dtype=torch.float32), persistent=False)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        backend: QuantBackend = "fp16_baseline",
    ) -> "QuantLinear":
        quant_layer = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            weight_dtype=layer.weight.dtype,
            device=layer.weight.device,
            backend=backend,
        )

        with torch.no_grad():
            quant_layer.weight.copy_(layer.weight)
            if layer.bias is not None and quant_layer.bias is not None:
                quant_layer.bias.copy_(layer.bias)
        return quant_layer

    def prepare_int4_weights(self) -> None:
        """
        Zhenya/Kirill ownership:
        - quantize fp16/bf16 weights to int4
        - pack int4 into int32
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "fp16_baseline":
            return F.linear(x, self.weight, self.bias)

        if self.backend == "triton_int4":
            result = matmul_x16w4_dequant(
                x,
                self.packed_weight_int32,
                self.weight_scale,
                self.weight_zero_point,
            )
            if result is None:
                raise NotImplementedError(
                    "triton_int4 backend is blocked by pass-stubs in kernels/int4_ops.py"
                )
            if self.bias is not None:
                result = result + self.bias
            return result

        raise ValueError(f"Unsupported backend: {self.backend}")
