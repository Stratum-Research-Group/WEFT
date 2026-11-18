"""
Standalone Inference Engine for WEFT.
Demonstrates streaming inference with:
1. Bounded local KV window with persistent recurrent state cache.
2. Synchronized timing across CUDA and CPU backends.
3. Microsecond-level token generation metrics.
"""

import time
import torch
from weft import WeftConfig, WeftForCausalLM


class WeftInferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        target_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(target_device)
        print(f"Loading WEFT inference engine from {checkpoint_path} onto {self.device}...")
        self.model = WeftForCausalLM.from_pretrained(checkpoint_path).to(self.device)
        self.model.eval()
        self.config = self.model.config

    @torch.inference_mode()
    def generate_streaming(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 100):
        """
        Executes prefill followed by autoregressive decoding with accurate synchronization.
        """
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

        prompt_tokens = prompt_tokens.to(self.device)
        b, prompt_len = prompt_tokens.shape

        if max_new_tokens == 0:
            return {
                "tokens": torch.empty((b, 0), dtype=prompt_tokens.dtype, device=self.device),
                "prefill_ms": 0.0,
                "avg_decode_ms": 0.0,
                "tokens_per_sec": 0.0,
            }

        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize()

        # Prefill phase
        t0 = time.perf_counter()
        outputs = self.model(input_ids=prompt_tokens, use_cache=True)
        if is_cuda:
            torch.cuda.synchronize()
        prefill_time = (time.perf_counter() - t0) * 1000.0

        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token]
        decode_latencies = []

        # Decode loop
        for _ in range(max_new_tokens - 1):
            if is_cuda:
                torch.cuda.synchronize()
            t_step = time.perf_counter()

            step_out = self.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            if is_cuda:
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t_step) * 1000.0

            past_key_values = step_out.past_key_values
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            decode_latencies.append(step_ms)

        if len(decode_latencies) > 0:
            avg_decode_ms = sum(decode_latencies) / len(decode_latencies)
            tokens_per_sec = 1000.0 / avg_decode_ms if avg_decode_ms > 0 else 0.0
        else:
            avg_decode_ms = prefill_time
            tokens_per_sec = 1000.0 / prefill_time if prefill_time > 0 else 0.0

        return {
            "tokens": torch.cat(generated, dim=1),
            "prefill_ms": prefill_time,
            "avg_decode_ms": avg_decode_ms,
            "tokens_per_sec": tokens_per_sec,
        }


def benchmark_inference():
    print("=================================================================")
    print("  WEFT HIGH-THROUGHPUT STREAMING INFERENCE ENGINE BENCHMARK")
    print("=================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = WeftInferenceEngine("weft_shakespeare_ckpt", device=device)

    batch_sizes = [1, 4, 8, 16]
    prompt_len = 32
    gen_len = 128

    print(f"\nBenchmarking streaming decode for {gen_len} tokens (Prompt: {prompt_len} tokens):")
    print(f"{'Batch Size':>12} | {'Prefill (ms)':>14} | {'Decode (ms/tok)':>16} | {'Throughput (tok/s)':>20}")
    print("-" * 70)

    for b in batch_sizes:
        prompts = torch.randint(0, engine.config.vocab_size, (b, prompt_len), device=engine.device)
        # Warmup
        _ = engine.generate_streaming(prompts[:1], max_new_tokens=5)

        res = engine.generate_streaming(prompts, max_new_tokens=gen_len)
        throughput = b * res["tokens_per_sec"]
        print(f"{b:12d} | {res['prefill_ms']:12.2f} ms | {res['avg_decode_ms']:14.3f} ms | {throughput:18.1f} tok/s")

    print("\n=================================================================")
    print("  INFERENCE BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=================================================================")


if __name__ == "__main__":
    benchmark_inference()
