from typing import Iterable, Optional

import torch
import torch.nn as nn

@torch.no_grad()
def replace_llama_linears_with_triton_quant(
    model: nn.Module,
    target_names: Optional[set[str]] = None
) -> None:
    """
    Replace chosen nn.Linear layers by customize QuantLinearTriton recursively
    """
    return
    
