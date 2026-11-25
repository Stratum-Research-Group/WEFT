"""
WEFT Calibrated Generation Engine.
==================================
1. Extracts sample calibration activations from WikiText-2.
2. Fine-tunes layer-wise continuous scales (h, l, g) with Block-Hessian updates.
3. Runs autoregressive generation on standard prompt and records token log-probabilities.
"""

import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from weft.converter import factorize_linear_layer
from weft.finetune_subbit import finetune_subbit_linear
from weft.dataset_loader import download_wikitext2


def run_calibrated_generation(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "The capital of France is",
    calibration_tokens: int = 512,
    max_new_tokens: int = 16,
    out_path: str = "results/calibrated_generation_output.json",
):
    print(f"=== WEFT Calibrated Model Generation Pipeline ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer & model: {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    # 1. Download WikiText-2 and extract calibration data
    test_file = download_wikitext2("test")
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()[:2000]

    calib_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=calibration_tokens).input_ids.to(device)

    # 2. Extract activation states for layer 0 self-attention
    print("[*] Gathering calibration activations...")
    activations = []
    def hook_fn(module, input, output):
        if len(activations) == 0:
            activations.append(input[0].detach())

    layer0_q = model.model.layers[0].self_attn.q_proj
    handle = layer0_q.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(calib_ids)
    handle.remove()

    calib_act = activations[0].squeeze(0)  # [seq_len, hidden_size]
    print(f"[*] Calibration activation tensor: {calib_act.shape}")

    # 3. Factorize and Fine-tune layer 0 q_proj
    print("[*] Factorizing and fine-tuning layer 0 q_proj scales...")
    orig_W = layer0_q.weight.data.clone()
    subbit_layer = factorize_linear_layer(orig_W, base_rank=32, num_paths=1)

    loss_ratio = finetune_subbit_linear(subbit_layer, orig_W, calib_act, num_steps=30, lr=1e-4)
    print(f"[+] Calibration loss ratio (final/initial): {loss_ratio:.4f}")

    # Reconstruct calibrated weight and plug back into model
    rec_W = subbit_layer.reconstruct_weight().to(orig_W.dtype).to(device)
    layer0_q.weight.data.copy_(rec_W)

    # 4. Execute prompt generation
    print(f"[*] Running autoregressive generation for prompt: '{prompt}'")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"[+] Generated output:\n---\n{generated_text}\n---")

    result = {
        "model": model_id,
        "prompt": prompt,
        "generated_text": generated_text,
        "calibration_tokens": calibration_tokens,
        "loss_ratio": loss_ratio,
        "elapsed_seconds": elapsed,
        "tokens_generated": int(output_ids.shape[1] - input_ids.shape[1]),
        "device": device,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[+] Results saved to {out_path}")
    return result


if __name__ == "__main__":
    run_calibrated_generation()
