import random

import numpy as np
import pytest
import torch

from kernels.compression_int4 import dequant_decompress_from_int4_to_float16, quant_compress_to_int4
from tests.utils import assert_reconstruction, get_correlation


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
        pytest.skip(
            f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}"
        )

    weights = sign * torch.ones((rows, cols), dtype=torch.float16, device="cuda")
    weights_compressed, scale = quant_compress_to_int4(
        weights, compress_factor=compress_factor, quant_block_size=quant_block_size
    )

    assert weights_compressed.shape == (rows, cols // compress_factor), (
        f"Weights compressed must have shape {(rows, cols // compress_factor)}, but have {weights_compressed.shape}"
    )
    assert scale.shape == (rows, cols // quant_block_size), (
        f"Scale must have shape {(rows, cols // quant_block_size)}, but have {scale.shape}"
    )
    assert weights_compressed.nbytes * 4 == weights.nbytes, (
        f"Weights compressed must reduce memory 4 times, but expected: {weights.nbytes // 4} actual: {weights_compressed.nbytes=}"
    )
    assert scale.is_contiguous(), "Scales matrix have to be contiguous"

    if compress_factor == 2:
        assert weights_compressed.dtype == torch.uint8, (
            "Compressed weights with compress_factor 2 must have type uint8"
        )
    elif compress_factor == 8:
        assert weights_compressed.dtype == torch.uint32, (
            "Compressed weights with compress_factor 2 must have type uint32"
        )

    assert scale.std().item() == 0, (
        f"For case of ones, scale.std() must be equal to zero, but {scale.std()=}"
    )

    weights_recon = dequant_decompress_from_int4_to_float16(weights_compressed, scale)

    assert weights_recon.dtype == weights.dtype, (
        f"Reconstructed weight dtype mismatch with original dtype, expected: {weights.dtype} actual: {weights_recon.dtype}"
    )
    assert weights_recon.shape == weights.shape, (
        f"Reconstructed weight shape mismatch with original shape, expected: {weights.shape} actual: {weights_recon.shape}"
    )
    assert torch.allclose(weights_recon, weights), (
        f"Reconstucted weights match with original expected: {weights=} actual: {weights_recon=}"
    )


@pytest.mark.parametrize("rows", [4, 32, 64, 128])
@pytest.mark.parametrize("cols", [256, 512, 1024])
@pytest.mark.parametrize("quant_block_size", [32, 64, 128])
@pytest.mark.parametrize("compress_factor", [2, 8])
def test_with_masked_ones_weights(rows, cols, quant_block_size, compress_factor, request):
    if cols < quant_block_size * compress_factor:
        pytest.skip(
            f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}"
        )

    mask = torch.randn((rows, cols)) >= 0.5
    weights = torch.ones((rows, cols), dtype=torch.float16, device="cuda")
    weights[mask] = -1
    weights_compressed, scale = quant_compress_to_int4(
        weights, compress_factor=compress_factor, quant_block_size=quant_block_size
    )

    assert weights_compressed.shape == (rows, cols // compress_factor), (
        f"Weights compressed must have shape {(rows, cols // compress_factor)}, but have {weights_compressed.shape}"
    )
    assert scale.shape == (rows, cols // quant_block_size), (
        f"Scale must have shape {(rows, cols // quant_block_size)}, but have {scale.shape}"
    )
    assert weights_compressed.nbytes * 4 == weights.nbytes, (
        f"Weights compressed must reduce memory 4 times, but expected: {weights.nbytes // 4} actual: {weights_compressed.nbytes=}"
    )

    if compress_factor == 2:
        assert weights_compressed.dtype == torch.uint8, (
            "Compressed weights with compress_factor 2 must have type uint8"
        )
    elif compress_factor == 8:
        assert weights_compressed.dtype == torch.uint32, (
            "Compressed weights with compress_factor 2 must have type uint32"
        )

    assert scale.std().item() == 0, (
        f"For case of ones, scale.std() must be equal to zero, but {scale.std()=}"
    )

    weights_recon = dequant_decompress_from_int4_to_float16(weights_compressed, scale)

    assert weights_recon.dtype == weights.dtype, (
        f"Reconstructed weight dtype mismatch with original dtype, expected: {weights.dtype} actual: {weights_recon.dtype}"
    )
    assert weights_recon.shape == weights.shape, (
        f"Reconstructed weight shape mismatch with original shape, expected: {weights.shape} actual: {weights_recon.shape}"
    )
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
        pytest.skip(
            f"Not enough cols. Cols should be not less quant_block_size * compress_factor, but {quant_block_size}*{compress_factor} > {cols}"
        )

    weights = torch.randn((rows, cols), dtype=torch.float16, device="cuda")
    weights_compressed, scale = quant_compress_to_int4(
        weights, compress_factor=compress_factor, quant_block_size=quant_block_size
    )

    assert torch.isfinite(scale).all(), "Scale values must be finite"
    assert weights_compressed.shape == (rows, cols // compress_factor), (
        f"Weights compressed must have shape {(rows, cols // compress_factor)}, but have {weights_compressed.shape}"
    )
    assert scale.shape == (rows, cols // quant_block_size), (
        f"Scale must have shape {(rows, cols // quant_block_size)}, but have {scale.shape}"
    )
    assert weights_compressed.nbytes * 4 == weights.nbytes, (
        f"Weights compressed must reduce memory 4 times, but expected: {weights.nbytes // 4} actual: {weights_compressed.nbytes=}"
    )

    if compress_factor == 2:
        assert weights_compressed.dtype == torch.uint8, (
            "Compressed weights with compress_factor 2 must have type uint8"
        )
    elif compress_factor == 8:
        assert weights_compressed.dtype == torch.uint32, (
            "Compressed weights with compress_factor 2 must have type uint32"
        )

    weights_recon = dequant_decompress_from_int4_to_float16(weights_compressed, scale)

    assert torch.isfinite(weights_recon).all(), "Reconstructed weights must be finite"
    assert weights_recon.dtype == weights.dtype, (
        f"Reconstructed weight dtype mismatch with original dtype, expected: {weights.dtype} actual: {weights_recon.dtype}"
    )
    assert weights_recon.shape == weights.shape, (
        f"Reconstructed weight shape mismatch with original shape, expected: {weights.shape} actual: {weights_recon.shape}"
    )

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
