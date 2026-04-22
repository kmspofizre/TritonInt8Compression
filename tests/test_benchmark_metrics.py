import pytest
import torch

from benchmarks.benchmark import (
    MatmulBenchmarkResult,
    _parse_shape,
    benchmark_inference_loop,
    build_metrics_dataset,
    format_metrics_dataset,
)


def test_build_metrics_dataset_has_expected_fields_and_values():
    results = [
        MatmulBenchmarkResult(
            m=512,
            n=512,
            k=512,
            triton_ms=2.0,
            torch_ms=4.0,
            triton_tflops=10.0,
            torch_tflops=5.0,
            max_abs_error=0.1,
            mean_abs_error=0.01,
        )
    ]
    dataset = build_metrics_dataset(results)

    assert len(dataset) == 1
    row = dataset[0]
    assert row["shape"] == "512x512x512"
    assert row["m"] == 512
    assert row["n"] == 512
    assert row["k"] == 512
    assert row["speedup_x"] == pytest.approx(2.0)
    assert row["tflops_gain_pct"] == pytest.approx(100.0)
    assert "raw" in row


def test_format_metrics_dataset_renders_table():
    dataset = [
        {
            "shape": "128x256x512",
            "m": 128,
            "n": 256,
            "k": 512,
            "triton_ms": 0.5,
            "torch_ms": 0.7,
            "speedup_x": 1.4,
            "triton_tflops": 12.3,
            "torch_tflops": 8.1,
            "tflops_gain_pct": 51.8,
            "max_abs_error": 0.03,
            "mean_abs_error": 0.005,
            "raw": {},
        }
    ]

    table = format_metrics_dataset(dataset)
    assert "Shape" in table
    assert "Speedup x" in table
    assert "128x256x512" in table
    assert "1.4000" in table


def test_format_metrics_dataset_empty():
    assert format_metrics_dataset([]) == "No benchmark rows."


def test_parse_shape_valid():
    assert _parse_shape("512x1024x2048") == (512, 1024, 2048)


def test_parse_shape_invalid():
    with pytest.raises(ValueError):
        _parse_shape("512x1024")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_benchmark_inference_loop_smoke():
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask, labels=None):
            loss = (input_ids.float() - labels.float()).abs().mean()
            return type("Output", (), {"loss": loss})()

    def dummy_tokenizer(text, padding=False, truncation=True, return_tensors="pt"):
        _ = (text, padding, truncation, return_tensors)
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }

    dataset = [{"text": "hello"}, {"text": "world"}]
    result = benchmark_inference_loop(
        model=DummyModel().cuda(),
        tokenizer=dummy_tokenizer,
        dataset=dataset,
        device="cuda",
    )
    assert "loss" in result
    assert "inference_time" in result
    assert result["loss"] == pytest.approx(0.0)
    assert result["inference_time"] >= 0.0
