import gc
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_llama.utils import replace_llama_linears_with_triton_quant

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
CONTEXT_LENGTH = 512
NUM_TOKENS = 128
QUANT_BLOCK_SIZE_CANDIDATES = (128, 64, 32)
PLOT_DIR = Path("benchmarks") / "plots"
PLOT_PATH = PLOT_DIR / "speed_vs_quant_block_size.png"
# MLP-only quantization for Llama blocks (attention and lm_head stay FP16).
MLP_TARGET_NAMES = {"gate_proj", "up_proj", "down_proj"}


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_allocated_memory_mb() -> float:
    cleanup_cuda()
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_peak_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_memory_stats() -> None:
    torch.cuda.reset_peak_memory_stats()


def get_wikitext_inputs(tokenizer, context_length: int = 512) -> dict[str, torch.Tensor]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"][:200])
    inputs = tokenizer(text, return_tensors="pt")

    inputs["input_ids"] = inputs["input_ids"][:, :context_length]
    inputs["attention_mask"] = inputs["attention_mask"][:, :context_length]
    assert inputs["input_ids"].shape[1] == context_length, "Not enough text for requested context length"
    return inputs


def load_fp16_model(model_id: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
    )


def measure_generation_speed(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    num_tokens: int = 128,
) -> float:
    device_inputs = {k: v.to(model.device) for k, v in inputs.items()}
    assert device_inputs["input_ids"].shape[0] == 1, "Batch size must be 1"

    with torch.no_grad():
        _ = model.generate(**device_inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            **device_inputs,
            max_new_tokens=num_tokens,
            min_new_tokens=num_tokens,
            do_sample=False,
        )
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    return num_tokens / (end_time - start_time)


def quantize_mlp_only(model: torch.nn.Module, quant_block_size: int) -> int:
    return replace_llama_linears_with_triton_quant(
        model=model,
        target_names=MLP_TARGET_NAMES,
        backend="triton_int4",
        quant_block_size=quant_block_size,
    )


def benchmark_quant_block_sizes(
    model_id: str,
    inputs: dict[str, torch.Tensor],
) -> list[tuple[int, int, float, float, float]]:
    """
    Returns rows: (block_size, replaced_layers, tokens_per_sec, memory_static_mb, memory_peak_mb)
    """
    rows: list[tuple[int, int, float, float, float]] = []
    print("INT4 measurement by quant_block_size (MLP-only quantization)")

    for block_size in QUANT_BLOCK_SIZE_CANDIDATES:
        model = load_fp16_model(model_id)
        replaced = quantize_mlp_only(model, quant_block_size=block_size)
        quant_memory_static = get_allocated_memory_mb()
        reset_memory_stats()
        quant_speed = measure_generation_speed(model, inputs, num_tokens=NUM_TOKENS)
        quant_memory_peak = get_peak_memory_mb()

        print(
            f"  block={block_size:<3} replaced={replaced:<4} "
            f"speed={quant_speed:>7.1f} t/s static={quant_memory_static:>7.0f} MB peak={quant_memory_peak:>7.0f} MB"
        )
        rows.append((block_size, replaced, quant_speed, quant_memory_static, quant_memory_peak))

        del model
        cleanup_cuda()

    return rows


def save_speed_plot(
    quant_rows: list[tuple[int, int, float, float, float]],
    baseline_speed: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: matplotlib is not available ({exc})")
        return

    block_sizes = [row[0] for row in quant_rows]
    quant_speeds = [row[2] for row in quant_rows]

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(block_sizes, quant_speeds, marker="o", linewidth=2, label="INT4 (MLP-only)")
    plt.axhline(y=baseline_speed, color="red", linestyle="--", linewidth=2, label=f"FP16 baseline ({baseline_speed:.1f} t/s)")
    plt.gca().invert_xaxis()
    plt.xlabel("quant_block_size")
    plt.ylabel("generation speed (tokens/sec)")
    plt.title("Token Generation Speed vs quant_block_size")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    print("Tokenizer and input initialization")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = get_wikitext_inputs(tokenizer, context_length=CONTEXT_LENGTH)

    print("Baseline measurement (FP16)")
    model = load_fp16_model(MODEL_ID)
    base_memory_static = get_allocated_memory_mb()
    reset_memory_stats()
    base_speed = measure_generation_speed(model, inputs, num_tokens=NUM_TOKENS)
    base_memory_peak = get_peak_memory_mb()

    del model
    cleanup_cuda()

    quant_rows = benchmark_quant_block_sizes(MODEL_ID, inputs)
    quant_rows_sorted = sorted(quant_rows, key=lambda row: row[2], reverse=True)
    best_block_size, best_replaced, best_speed, best_static_mb, best_peak_mb = quant_rows_sorted[0]

    print(f"\nBest quant_block_size by speed: {best_block_size} ({best_speed:.1f} t/s)")
    print(f"{'Metric (Batch=1)':<28} | {'FP16 (Base)':<12} | {'INT4 best (MLP)':<16}")
    print(f"{'Quant block size':<28} | {'-':>12} | {best_block_size:>16}")
    print(f"{'Replaced MLP linears':<28} | {'-':>12} | {best_replaced:>16}")
    print(f"{'Memory (model weights)':<28} | {base_memory_static:>10.0f} MB | {best_static_mb:>14.0f} MB")
    print(f"{'Memory (peak during gen)':<28} | {base_memory_peak:>10.0f} MB | {best_peak_mb:>14.0f} MB")
    print(f"{'Generation speed':<28} | {base_speed:>10.1f} t/s | {best_speed:>14.1f} t/s")

    save_speed_plot(quant_rows, baseline_speed=base_speed)
