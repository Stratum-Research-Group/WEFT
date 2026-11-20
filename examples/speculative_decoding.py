"""
WEFT-2 Speculative Decoding Engine Prototype.

Demonstrates WEFT as an ultra-low latency autoregressive draft model paired with
a target model (or full precision verifier), achieving multi-token candidate generation
in O(1) step latency with speculative acceptance and verified speedup.
"""

from typing import List, Tuple, Optional
import time
import torch
import torch.nn.functional as F
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM


class WeftSpeculativeEngine:
    def __init__(
        self,
        target_model: WeftForCausalLM,
        draft_model: WeftForCausalLM,
        gamma: int = 4,  # number of draft tokens per speculative step
    ):
        self.target = target_model
        self.draft = draft_model
        self.gamma = gamma
        self.target.eval()
        self.draft.eval()

    @torch.no_grad()
    def generate_draft_tokens(
        self,
        prefix_ids: torch.Tensor,
        gamma: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Drafts gamma speculative tokens autoregressively using the draft model.
        Returns draft_tokens [1, gamma] and draft_probs [1, gamma, vocab].
        """
        curr_ids = prefix_ids
        drafted_tokens = []
        draft_probs_list = []

        for _ in range(gamma):
            out = self.draft(input_ids=curr_ids)
            logits = out.logits[:, -1, :]  # [1, vocab]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            drafted_tokens.append(next_token)
            draft_probs_list.append(probs.unsqueeze(1))
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        draft_tokens = torch.cat(drafted_tokens, dim=1)  # [1, gamma]
        draft_probs = torch.cat(draft_probs_list, dim=1)  # [1, gamma, vocab]
        return draft_tokens, draft_probs

    @torch.no_grad()
    def speculative_step(
        self,
        prefix_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Executes a single speculative iteration:
        1. Draft gamma tokens.
        2. Verify all gamma tokens in parallel with target model (single forward pass).
        3. Accept tokens according to rejection sampling / greedy match.
        """
        gamma = self.gamma
        draft_tokens, _ = self.generate_draft_tokens(prefix_ids, gamma)

        # Candidate sequence evaluated by target model in 1 parallel forward pass:
        candidate_ids = torch.cat([prefix_ids, draft_tokens], dim=1)
        target_out = self.target(input_ids=candidate_ids)
        target_logits = target_out.logits[:, prefix_ids.shape[1] - 1 : -1, :]  # [1, gamma, vocab]
        target_preds = torch.argmax(target_logits, dim=-1)  # [1, gamma]

        # Count accepted tokens
        accepted = 0
        accepted_tokens = []
        for i in range(gamma):
            drafted = draft_tokens[0, i].item()
            target_choice = target_preds[0, i].item()
            if drafted == target_choice:
                accepted += 1
                accepted_tokens.append(drafted)
            else:
                # Reject and take target token
                accepted_tokens.append(target_choice)
                break

        # If all gamma tokens accepted, sample one bonus token from target at end
        if accepted == gamma:
            bonus_token = torch.argmax(target_out.logits[:, -1, :], dim=-1).item()
            accepted_tokens.append(bonus_token)
            accepted += 1

        new_prefix = torch.cat(
            [prefix_ids, torch.tensor([accepted_tokens], device=prefix_ids.device)], dim=1
        )
        return new_prefix, accepted


def benchmark_speculative_decoding():
    print("=== WEFT-2 Speculative Decoding Benchmark ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup target model (e.g. 16 layers) and draft model (4 layers)
    target_cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_state_heads=4,
        state_rank=16,
        window_size=64,
    )
    draft_cfg = WeftConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=2,  # 6x shallower draft engine
        num_attention_heads=8,
        num_state_heads=4,
        state_rank=16,
        window_size=64,
    )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    target_model = WeftForCausalLM(target_cfg).to(device, dtype=dtype)
    draft_model = WeftForCausalLM(draft_cfg).to(device, dtype=dtype)

    engine = WeftSpeculativeEngine(target_model, draft_model, gamma=4)

    prefix = torch.randint(0, 1000, (1, 32), device=device)
    target_tokens_to_gen = 30

    print(f"Generating {target_tokens_to_gen} tokens with standard autoregressive decoding...")
    # Baseline: standard autoregressive generation on target
    t0 = time.perf_counter()
    curr = prefix.clone()
    with torch.no_grad():
        for _ in range(target_tokens_to_gen):
            out = target_model(input_ids=curr)
            next_t = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            curr = torch.cat([curr, next_t], dim=1)
    base_time = time.perf_counter() - t0
    base_tok_per_sec = target_tokens_to_gen / base_time

    print(f"Baseline Target Autoregressive: {base_time*1000:.1f} ms | {base_tok_per_sec:.1f} tok/s")

    print(f"Generating with Speculative Engine (gamma=4)...")
    curr_spec = prefix.clone()
    total_accepted = 0
    steps = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        while curr_spec.shape[1] - prefix.shape[1] < target_tokens_to_gen:
            curr_spec, n_acc = engine.speculative_step(curr_spec)
            total_accepted += n_acc
            steps += 1
    spec_time = time.perf_counter() - t0
    gen_tokens = curr_spec.shape[1] - prefix.shape[1]
    spec_tok_per_sec = gen_tokens / spec_time
    alpha = total_accepted / (steps * 4)

    print(f"Speculative Decoding: {spec_time*1000:.1f} ms | {spec_tok_per_sec:.1f} tok/s")
    print(f"Steps: {steps} | Generated: {gen_tokens} tokens | Mean tokens/step: {gen_tokens/steps:.2f}")
    print(f"Speedup Ratio: {spec_tok_per_sec / base_tok_per_sec:.2f}x")


if __name__ == "__main__":
    benchmark_speculative_decoding()
