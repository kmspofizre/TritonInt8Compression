# Triton Quantization

## Структура проекта
*   **`kernels/`** — Ядра Triton: упаковка весов (`compression_int4.py`) и быстрый `fused` matmul (`decompress_int4_matmul_float16_fused.py`).
*   **`quant_llama/`** — Интеграция: слой `QuantLinear` и утилиты для замены слоев в моделях Llama.
*   **`benchmarks/benchmark.py`** — Сравнение производительности кастомного Matmul ядра с `torch.matmul` (TFLOPS, ошибки).
*   **`exp_speed_memory.py`** — Эксперимент по замеру скорости (tokens/s) и потребления памяти.
*   **`exp_perplexity.py`** — Замер точности (Perplexity) на датасете WikiText-2.

## Особенности экспериментов
*   **MLP-only:** Квантуются только FF-слои (`gate_proj`, `up_proj`, `down_proj`). Слои Attention и lm_head остаются в **FP16**.
*   **Результаты:** Скрипты автоматически сравнивают FP16 Baseline и Triton INT4.
*   **Графики:** Зависимость скорости от размера блока (`quant_block_size`) сохраняется в `benchmarks/plots/`.

## Запуск
1. **Тест ядра (Matmul):** 
   ```bash
   python benchmarks/benchmark.py --shape 4096x4096x4096
   ```

2. **Замер скорости и памяти (Llama):**
    ```bash
    python exp_speed_memory.py
    ```

3. **Замер точности (Perplexity):**
    ```bash
    python exp_perplexity.py
    ```