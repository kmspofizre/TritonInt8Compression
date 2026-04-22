import os
from pathlib import Path

import pytest
import torch
import triton

from kernels.launcher import Autotuner, KernelConfig, Launcher, noop_kernel, run_noop_smoke_test


@pytest.fixture(scope="module", autouse=True)
def triton_cache_dir():
    cache_dir = Path.cwd() / ".triton_cache_tests"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_run_noop_smoke_test():
    out = run_noop_smoke_test(num_elements=257, block_size=128)
    expected = torch.arange(257, device="cuda", dtype=torch.float32)
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype
    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_launcher_launches_noop_kernel():
    n = 257
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(n, 128),)
    launcher = Launcher(noop_kernel, grid)
    launcher.launch(x, n, BLOCK_SIZE=128)
    assert torch.equal(x, torch.arange(n, device="cuda", dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_autotuner_selects_and_caches_best_config():
    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    launcher = Launcher(noop_kernel, (triton.cdiv(n, 256),))
    configs = [
        KernelConfig(meta={"BLOCK_SIZE": 256}, num_warps=2, num_stages=2),
        KernelConfig(meta={"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    ]
    autotuner = Autotuner(launcher=launcher, configs=configs, warmup=1, reps=2)

    best = autotuner.autotune(x, n)
    assert best in configs
    assert autotuner.best_config == best

    cached = autotuner.launch(x, n)
    assert cached == best

