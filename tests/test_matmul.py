import random

import numpy as np
import pytest
import torch

from kernels.matmul_bf16 import matmul


@pytest.fixture(autouse=True)
def fix_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("M", [128, 256, 512, 1024])
@pytest.mark.parametrize("K", [128, 256, 512, 1024])
def test_matmul(N, M, K):
    a = torch.randn((N, K), dtype=torch.float16).cuda()
    b = torch.randn((K, M), dtype=torch.float16).cuda()

    expected = a @ b
    actual = matmul(a, b)

    assert expected.shape == actual.shape, f"Matmul output shape mismath: expected {expected.shape} actual {actual.shape}"
    assert expected.dtype == actual.dtype, f"Matmul output dtype mismath: expected {expected.dtype} actual {actual.dtype}"

    assert torch.allclose(expected, actual, rtol=1e-1, atol=1e-1), f"Matmul output mismath: mse {torch.max((expected - actual).abs())}"
