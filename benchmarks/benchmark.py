import argparse
import gc
from dataclasses import asdict, dataclass
from statistics import median
from typing import Callable, Sequence

import torch
from tqdm import tqdm


@dataclass
class MatmulBenchmarkResult:
    m: int
    n: int
    k: int
    triton_ms: float
    torch_ms: float
    triton_tflops: float
    torch_tflops: float
    max_abs_error: float
    mean_abs_error: float


def build_metrics_dataset(results: list[MatmulBenchmarkResult]) -> list[dict[str, int | float | str]]:
    dataset: list[dict[str, int | float | str]] = []
    for result in results:
        torch_ms = result.torch_ms
        torch_tflops = result.torch_tflops
        speedup = (torch_ms / result.triton_ms) if result.triton_ms != 0.0 else float("inf")
        tflops_gain_pct = (
            ((result.triton_tflops - torch_tflops) / torch_tflops * 100.0)
            if torch_tflops != 0.0
            else float("inf")
        )

        row = {
            "shape": f"{result.m}x{result.n}x{result.k}",
            "m": result.m,
            "n": result.n,
            "k": result.k,
            "triton_ms": result.triton_ms,
            "torch_ms": result.torch_ms,
            "speedup_x": speedup,
            "triton_tflops": result.triton_tflops,
            "torch_tflops": result.torch_tflops,
            "tflops_gain_pct": tflops_gain_pct,
            "max_abs_error": result.max_abs_error,
            "mean_abs_error": result.mean_abs_error,
            "raw": asdict(result),
        }
        dataset.append(row)
    return dataset


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def format_metrics_dataset(dataset: list[dict[str, int | float | str]]) -> str:
    if not dataset:
        return "No benchmark rows."

    visible_columns = [
        "shape",
        "triton_ms",
        "torch_ms",
        "speedup_x",
        "triton_tflops",
        "torch_tflops",
        "tflops_gain_pct",
        "max_abs_error",
        "mean_abs_error",
    ]
    header_labels = {
        "shape": "Shape",
        "triton_ms": "Triton ms",
        "torch_ms": "Torch ms",
        "speedup_x": "Speedup x",
        "triton_tflops": "Triton TFLOPS",
        "torch_tflops": "Torch TFLOPS",
        "tflops_gain_pct": "TFLOPS delta %",
        "max_abs_error": "MaxAbsErr",
        "mean_abs_error": "MeanAbsErr",
    }
    rendered_rows: list[dict[str, str]] = []
    for row in dataset:
        rendered_rows.append({col: _format_cell(row[col]) for col in visible_columns})

    widths: dict[str, int] = {}
    for col in visible_columns:
        max_cell_width = max(len(r[col]) for r in rendered_rows)
        widths[col] = max(len(header_labels[col]), max_cell_width)

    def _render_row(cells: dict[str, str]) -> str:
        return " | ".join(cells[col].rjust(widths[col]) for col in visible_columns)

    header_row = _render_row(header_labels)
    separator = "-+-".join("-" * widths[col] for col in visible_columns)
    body = "\n".join(_render_row(r) for r in rendered_rows)
    return f"{header_row}\n{separator}\n{body}"


def benchmark_inference_loop(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    device: torch.device | str,
) -> dict[str, float]:
    total_loss = 0.0
    total_steps = len(dataset)
    inference_time = 0.0

    model.eval()
    with torch.no_grad():
        for item in tqdm(dataset):
            sample = tokenizer(item["text"], padding=False, truncation=True, return_tensors="pt")
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            labels = sample["input_ids"].to(device).clone()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            end_event.record()
            torch.cuda.synchronize()

            inference_time += start_event.elapsed_time(end_event)
            total_loss += loss.item()

    return {
        "loss": total_loss / total_steps,
        "inference_time": inference_time / total_steps,
    }


def _measure_cuda_ms(fn: Callable[[], torch.Tensor], warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(reps):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings_ms.append(start_event.elapsed_time(end_event))
    return float(median(timings_ms))


def _calc_tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    if latency_ms == 0.0:
        return float("inf")
    flops = 2.0 * m * n * k
    return flops / (latency_ms / 1000.0) / 1e12


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def benchmark_matmul_shapes(
    shapes: Sequence[tuple[int, int, int]],
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    reps: int = 50,
) -> list[MatmulBenchmarkResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for matrix benchmark")

    if kernel is None:
        from kernels.matmul_bf16 import matmul as triton_kernel
    else:
        triton_kernel = kernel

    results: list[MatmulBenchmarkResult] = []
    with torch.no_grad():
        for m, n, k in shapes:
            a = torch.randn((m, k), device=device, dtype=dtype)
            b = torch.randn((k, n), device=device, dtype=dtype)

            triton_out = triton_kernel(a, b)
            torch_out = torch.matmul(a, b)
            max_abs_error = (triton_out.float() - torch_out.float()).abs().max().item()
            mean_abs_error = (triton_out.float() - torch_out.float()).abs().mean().item()

            triton_ms = _measure_cuda_ms(lambda: triton_kernel(a, b), warmup=warmup, reps=reps)
            torch_ms = _measure_cuda_ms(lambda: torch.matmul(a, b), warmup=warmup, reps=reps)

            results.append(
                MatmulBenchmarkResult(
                    m=m,
                    n=n,
                    k=k,
                    triton_ms=triton_ms,
                    torch_ms=torch_ms,
                    triton_tflops=_calc_tflops(m, n, k, triton_ms),
                    torch_tflops=_calc_tflops(m, n, k, torch_ms),
                    max_abs_error=max_abs_error,
                    mean_abs_error=mean_abs_error,
                )
            )

            cleanup_cuda()
    return results


def _parse_shape(shape_spec: str) -> tuple[int, int, int]:
    dims = tuple(int(part) for part in shape_spec.lower().split("x"))
    if len(dims) != 3:
        raise ValueError(f"Shape must be MxNxK, got: {shape_spec}")
    m, n, k = dims
    return m, n, k


def _default_shapes() -> list[tuple[int, int, int]]:
    return [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16_384, 16_384, 16_384),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton matmul vs torch.matmul with TFLOPS and error metrics."
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Matrix shape in MxNxK format. Can be used multiple times.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw dataset rows as dictionaries after the table.",
    )
    args = parser.parse_args()

    shapes = [_parse_shape(shape) for shape in args.shape] if args.shape else _default_shapes()
    results = benchmark_matmul_shapes(shapes=shapes, warmup=args.warmup, reps=args.reps)
    metrics_dataset = build_metrics_dataset(results)
    print(format_metrics_dataset(metrics_dataset))
    if args.raw:
        print("\nRaw dataset:")
        for row in metrics_dataset:
            print(row)


if __name__ == "__main__":
    main()
