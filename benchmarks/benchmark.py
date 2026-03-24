from tqdm import tqdm
import torch

def benchmark_inference_loop(model, tokenizer, dataset, device):
    total_loss = 0.0
    total_steps = len(dataset)

    inference_time = 0.0

    model.eval()
    with torch.no_grad():
        for item in tqdm(dataset):
            sample = tokenizer(item["text"], padding=False, truncation=True, return_tensors="pt")
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            labels = sample['input_ids'].to(device).clone()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            ## TODO: split forward and loss computation
            torch.cuda.synchronize()
            start_event.record()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            end_event.record()
            torch.cuda.synchronize()

            inference_time += start_event.elapsed_time(end_event)
            total_loss += loss.item()
            # assert total_loss

    return {
        "loss" : total_loss / total_steps,
        "inference_time": inference_time / total_steps, 
    }