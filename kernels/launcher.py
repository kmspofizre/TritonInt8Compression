from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Sequence

import torch
import triton
import triton.language as tl

GridSpec = tuple[int, ...] | Callable[[dict[str, Any]], tuple[int, ...]]


@triton.jit
def noop_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """No-op Triton kernel used to validate launch path from Python."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    value = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs, value, mask=mask)


class Launcher:
    """Thin Python wrapper over Triton kernel launch syntax."""

    def __init__(self, kernel: Any, grid: GridSpec):
        self.kernel = kernel
        self.grid = grid

    def launch(self, *kernel_args: Any, **meta: Any) -> None:
        grid = self.grid(meta) if callable(self.grid) else self.grid
        self.kernel[grid](*kernel_args, **meta)


@dataclass(frozen=True)
class KernelConfig:
    """Launch configuration candidate for autotuning."""

    meta: dict[str, Any] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 2

    def as_launch_meta(self) -> dict[str, Any]:
        launch_meta = dict(self.meta)
        launch_meta["num_warps"] = self.num_warps
        launch_meta["num_stages"] = self.num_stages
        return launch_meta


class Autotuner:
    """Minimal runtime autotuner selecting the fastest kernel config."""

    def __init__(
        self,
        launcher: Launcher,
        configs: Sequence[KernelConfig],
        warmup: int = 5,
        reps: int = 25,
    ) -> None:
        if not configs:
            raise ValueError("configs must not be empty")

        self.launcher = launcher
        self.configs = list(configs)
        self.warmup = warmup
        self.reps = reps
        self.best_config: KernelConfig | None = None

    def _time_ms(self, kernel_args: tuple[Any, ...], config: KernelConfig) -> float:
        launch_meta = config.as_launch_meta()
        for _ in range(self.warmup):
            self.launcher.launch(*kernel_args, **launch_meta)
        torch.cuda.synchronize()

        durations_ms = []
        for _ in range(self.reps):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.launcher.launch(*kernel_args, **launch_meta)
            end_event.record()
            torch.cuda.synchronize()
            durations_ms.append(start_event.elapsed_time(end_event))
        return float(median(durations_ms))

    def autotune(self, *kernel_args: Any) -> KernelConfig:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Triton autotuning")

        best_config = min(
            self.configs,
            key=lambda cfg: self._time_ms(tuple(kernel_args), cfg),
        )
        self.best_config = best_config
        return best_config

    def launch(self, *kernel_args: Any) -> KernelConfig:
        if self.best_config is None:
            self.autotune(*kernel_args)
        assert self.best_config is not None
        self.launcher.launch(*kernel_args, **self.best_config.as_launch_meta())
        return self.best_config


def run_noop_smoke_test(
    num_elements: int = 1024,
    block_size: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """Runs a no-op Triton kernel to validate Python wrapper integration."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run no-op smoke test")

    tensor = torch.arange(num_elements, device=device, dtype=torch.float32)
    grid = (triton.cdiv(num_elements, block_size),)
    launcher = Launcher(noop_kernel, grid)
    launcher.launch(tensor, num_elements, BLOCK_SIZE=block_size)
    return tensor
