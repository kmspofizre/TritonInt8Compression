import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from exp_speed_memory import quantize_mlp_only
from quant_llama.quant_linear import QuantLinear


def evaluate_perplexity(model, tokenizer, max_length=512):
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(test_data["text"])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    nlls = []
    total_predicted_tokens = 0

    model.eval()
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, max_length), desc="Calculating PPL"):
            end_loc = min(begin_loc + max_length, seq_len)
            chunk_input_ids = input_ids[:, begin_loc:end_loc].to(model.device)

            outputs = model(chunk_input_ids, labels=chunk_input_ids)

            chunk_len = end_loc - begin_loc
            predicted_tokens_in_chunk = chunk_len - 1

            if predicted_tokens_in_chunk > 0:
                nll = outputs.loss * predicted_tokens_in_chunk
                nlls.append(nll)
                total_predicted_tokens += predicted_tokens_in_chunk

    total_nll = torch.stack(nlls).sum()
    mean_nll = total_nll / total_predicted_tokens
    return torch.exp(mean_nll).item()


def replace_linear_layers(module, backend="triton_int4", quant_block_size=128):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name != "lm_head":
            quant_layer = QuantLinear.from_linear(
                child,
                backend=backend,
                quant_block_size=quant_block_size
            )
            setattr(module, name, quant_layer)
        else:
            replace_linear_layers(child, backend, quant_block_size)


if __name__ == "__main__":
    model_id = "unsloth/Llama-3.2-1B-Instruct"

    print(f"1. Загрузка токенизатора и Baseline модели: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda"
    )

    print("\n2. Замер Baseline Perplexity (FP16)")
    baseline_ppl = evaluate_perplexity(model, tokenizer, max_length=512)
    print(f"-> Baseline PPL: {baseline_ppl:.4f}")

    print("\n3. Квантование модели в INT4")
    replaced = quantize_mlp_only(model, quant_block_size=128)
    torch.cuda.empty_cache()

    print("\n4. Замер INT4 Perplexity")
    quant_ppl = evaluate_perplexity(model, tokenizer, max_length=512)

    print("\n" + "=" * 40)
    print(f"Baseline (FP16) PPL: {baseline_ppl:.4f}")
    print(f"Triton (INT4)   PPL: {quant_ppl:.4f}")
    print(f"Degradation:        +{quant_ppl - baseline_ppl:.4f}")
    print("=" * 40)
