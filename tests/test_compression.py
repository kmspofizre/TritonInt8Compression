import random

import numpy as np
import pytest
import torch

from tests.utils import assert_reconstruction, get_correlation, run_and_check_conversion


@pytest.fixture(autouse=True)
def fix_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

@pytest.mark.parametrize("rows", [32, 64, 128])
@pytest.mark.parametrize("cols", [128, 256, 512, 1024])
@pytest.mark.parametrize("quant_block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_factor", [2, 8])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_with_ones_weights(rows, cols, quant_block_size, compress_factor, sign):
    if cols < quant_block_size * compress_factor:
        pytest.skip(f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}")

    weights = sign * torch.ones((rows, cols), dtype=torch.float16, device="cuda")

    run_and_check_conversion(weights, quant_block_size, compress_factor, zero_std=True)


@pytest.mark.parametrize("rows", [4, 32, 64, 128])
@pytest.mark.parametrize("cols", [256, 512, 1024])
@pytest.mark.parametrize("quant_block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_factor", [2, 8])
def test_with_masked_ones_weights(rows, cols, quant_block_size, compress_factor, request):
    if cols < quant_block_size * compress_factor:
        pytest.skip(f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}")

    mask = torch.randn((rows, cols)) >= 0.5
    weights = torch.ones((rows, cols), dtype=torch.float16, device="cuda")
    weights[mask] = -1

    _, _, weights_recon = run_and_check_conversion(weights, quant_block_size, compress_factor, zero_std=False, check_match=False)

    assert_reconstruction(
        weights,
        weights_recon,
        atol=1e-08,
        test_name=request.node.name,
        save_path=f"tests/images/{request.node.name}",
    )


@pytest.mark.parametrize("rows", [32, 64, 128, 256])
@pytest.mark.parametrize("cols", [128, 256, 512, 1024])
@pytest.mark.parametrize("quant_block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_factor", [2, 8])
def test_with_rand_weights(rows, cols, quant_block_size, compress_factor, request):
    if cols < quant_block_size * compress_factor:
        pytest.skip(f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}")

    weights = torch.randn((rows, cols), dtype=torch.float16, device="cuda")

    _, _, weights_recon = run_and_check_conversion(weights, quant_block_size, compress_factor, zero_std=False, check_match=False)

    assert_reconstruction(
        weights,
        weights_recon,
        atol=1e-1,
        test_name=request.node.name,
        save_path=f"tests/images/{request.node.name}",
        compare_mse=True,
    )

    correlation = get_correlation(weights, weights_recon)
    assert correlation >= 0.99, f"Correlation must be more than 0.95, actual correlation is {correlation}"
