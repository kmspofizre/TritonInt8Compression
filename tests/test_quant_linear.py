import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from quant_llama.quant_linear import QuantLinear


@pytest.fixture(scope="module", autouse=True)
def triton_cache_dir():
    cache_dir = Path.cwd() / ".triton_cache_tests"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_from_linear_copies_weights_and_bias():
    layer = nn.Linear(128, 64, bias=True, device="cuda", dtype=torch.float16)
    q = QuantLinear.from_linear(layer, backend="fp16_baseline", quant_block_size=128)

    assert q.in_features == 128
    assert q.out_features == 64
    assert q.backend == "fp16_baseline"
    assert torch.equal(q.weight, layer.weight)
    assert torch.equal(q.bias, layer.bias)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp16_baseline_matches_linear():
    layer = nn.Linear(128, 64, bias=True, device="cuda", dtype=torch.float16)
    q = QuantLinear.from_linear(layer, backend="fp16_baseline", quant_block_size=128)
    x = torch.randn((16, 128), device="cuda", dtype=torch.float16)

    out_ref = layer(x)
    out_q = q(x)
    assert out_q.shape == out_ref.shape
    assert out_q.dtype == out_ref.dtype
    assert torch.allclose(out_q, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_int4_prepares_packed_buffers():
    layer = nn.Linear(128, 64, bias=True, device="cuda", dtype=torch.float16)
    q = QuantLinear.from_linear(layer, backend="triton_int4", quant_block_size=128)

    assert q.packed_weight_int32.numel() > 0
    assert q.weight_scale.numel() > 0
    assert q.packed_weight_int32.dtype == torch.uint32
    assert q.weight_scale.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_int4_forward_2d_and_3d():
    layer = nn.Linear(128, 96, bias=True, device="cuda", dtype=torch.float16)
    q = QuantLinear.from_linear(layer, backend="triton_int4", quant_block_size=128)

    x2d = torch.randn((8, 128), device="cuda", dtype=torch.float16)
    y2d = q(x2d)
    assert y2d.shape == (8, 96)
    assert y2d.dtype == torch.float16
    assert torch.isfinite(y2d).all()

    x3d = torch.randn((4, 3, 128), device="cuda", dtype=torch.float16)
    y3d = q(x3d)
    assert y3d.shape == (4, 3, 96)
    assert y3d.dtype == torch.float16
    assert torch.isfinite(y3d).all()

