from collections.abc import Set
from typing import Optional

import torch
import torch.nn as nn

from quant_llama.quant_linear import QuantBackend, QuantLinear


def _should_replace(full_name: str, target_names: Optional[Set[str]]) -> bool:
    if target_names is None:
        return True
    short_name = full_name.rsplit(".", maxsplit=1)[-1]
    return full_name in target_names or short_name in target_names


@torch.no_grad()
def replace_llama_linears_with_triton_quant(
    model: nn.Module,
    target_names: Optional[Set[str]] = None,
    backend: QuantBackend = "fp16_baseline",
    quant_block_size: int = 128,
) -> int:
    """Recursively replace selected `nn.Linear` modules with `QuantLinear`."""
    replaced_count = 0

    def _replace_recursive(module: nn.Module, prefix: str = "") -> None:
        nonlocal replaced_count
        for child_name, child_module in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child_module, nn.Linear) and _should_replace(full_name, target_names):
                setattr(
                    module,
                    child_name,
                    QuantLinear.from_linear(
                        child_module,
                        backend=backend,
                        quant_block_size=quant_block_size,
                    ),
                )
                replaced_count += 1
                continue
            _replace_recursive(child_module, full_name)

    _replace_recursive(model)
    return replaced_count
