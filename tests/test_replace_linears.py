import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from quant_llama.quant_linear import QuantLinear
from quant_llama.utils import replace_llama_linears_with_triton_quant


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(128, 128)
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, 64)

    def forward(self, x):
        x = self.input(x)
        x = self.block(x)
        return self.head(x)


@pytest.fixture(scope="module", autouse=True)
def triton_cache_dir():
    cache_dir = Path.cwd() / ".triton_cache_tests"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


def _count_quant_linear(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, QuantLinear))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_replace_all_linears_baseline_backend():
    model = TinyModel().cuda().half()
    replaced = replace_llama_linears_with_triton_quant(
        model,
        backend="fp16_baseline",
        quant_block_size=128,
    )

    assert replaced == 3
    assert _count_quant_linear(model) == 3

    x = torch.randn((4, 128), device="cuda", dtype=torch.float16)
    y = model(x)
    assert y.shape == (4, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_replace_only_target_names():
    model = TinyModel().cuda().half()
    replaced = replace_llama_linears_with_triton_quant(
        model,
        target_names={"input", "head"},
        backend="fp16_baseline",
        quant_block_size=128,
    )

    assert replaced == 2
    assert isinstance(model.input, QuantLinear)
    assert isinstance(model.head, QuantLinear)
    assert isinstance(model.block[1], nn.Linear)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_e2e_replace_and_forward_triton_int4():
    model = TinyModel().cuda().half()
    replaced = replace_llama_linears_with_triton_quant(
        model,
        backend="triton_int4",
        quant_block_size=128,
    )
    assert replaced == 3
    assert _count_quant_linear(model) == 3

    x = torch.randn((2, 128), device="cuda", dtype=torch.float16)
    y = model(x)
    assert y.shape == (2, 64)
    assert y.dtype == torch.float16
