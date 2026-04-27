import random

import numpy as np
import pytest
import torch

from kernels.decompress_int4_matmul_float16_fused import decompress_int4_matmul_float16_fused
from tests.utils import compare_tensors, get_correlation, run_and_check_conversion


@pytest.fixture(autouse=True)
def fix_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("quant_block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_factor", [2, 8])
@pytest.mark.parametrize("rows", [32, 64, 128])
@pytest.mark.parametrize("cols", [128, 256, 512, 1024])
def test_with_masked_ones_weights(batch_size, rows, cols, quant_block_size, compress_factor):
    if cols < quant_block_size * compress_factor:
        pytest.skip(f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}")

    weights = torch.ones((rows, cols), dtype=torch.float16, device="cuda")
    mask = torch.randn_like(weights) >= 0.5
    weights[mask] = -1

    weights_compressed, scale, weights_recon = run_and_check_conversion(weights, quant_block_size, compress_factor, zero_std=True)
    
    x = torch.ones((batch_size, cols), dtype=torch.float16).cuda()
    mask = torch.randn_like(x) >= 0.5
    x[mask] = -1

    expected_matmul = x @ weights_recon.T
    actual_matmul = decompress_int4_matmul_float16_fused(weights_compressed, scale, x)

    compare_tensors(expected_matmul, actual_matmul, check_match=True, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("rows", [32, 64, 128])
@pytest.mark.parametrize("cols", [128, 256, 512, 1024])
@pytest.mark.parametrize("quant_block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_factor", [2, 8])
def test_with_rand_weights(batch_size, rows, cols, quant_block_size, compress_factor):
    if cols < quant_block_size * compress_factor:
        pytest.skip(f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}")

    weights = torch.randn((rows, cols), dtype=torch.float16, device="cuda")

    weights_compressed, scale, weights_recon = run_and_check_conversion(weights, quant_block_size, compress_factor, check_match=False)
    
    x = torch.randn((batch_size, cols), dtype=torch.float16).cuda()

    expected_matmul = x @ weights_recon.T
    actual_matmul = decompress_int4_matmul_float16_fused(weights_compressed, scale, x)

    compare_tensors(expected_matmul, actual_matmul, check_match=True, atol=1e-2, rtol=1e-1)

    correlation = get_correlation(expected_matmul, actual_matmul)
    print(f"{correlation=}")
    assert correlation >= 0.999, f"Correlation must be more than 0.95, actual correlation is {correlation}"
