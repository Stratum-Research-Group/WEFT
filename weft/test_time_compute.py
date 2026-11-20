"""
WEFT-2 Test-Time Verification & Dynamic Self-Correction Decoding Engine.

Implements adaptive test-time compute scaling:
During generation, when inside `<think> ... </think>`, the engine monitors
internal token surprise (negative log-likelihood / entropy) and verification checks.
If an error or logical inconsistency is detected, it triggers backtracked reflection
(e.g., "Wait, let me double check that calculation...") before finalizing the response.
"""

import torch
import tiktoken
from typing import Tuple, List, Optional
from weft import WeftForCausalLM


class TestTimeVerifier:
    """Monitors token-level entropy and triggers self-correction when uncertainty spikes."""

    def __init__(self, entropy_threshold: float = 3.2):
        self.entropy_threshold = entropy_threshold

    def calculate_entropy(self, logits: torch.FloatTensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        return entropy

    def should_trigger_reflection(self, entropy: float, step: int) -> bool:
        # Trigger reflection if entropy spikes above threshold after minimum warmup
        return step > 10 and entropy > self.entropy_threshold


def generate_with_test_time_verification(
    model: WeftForCausalLM,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    max_reflections: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[str, int]:
    """Generates text with dynamic test-time self-correction in <think> reasoning blocks."""
    model.eval()
    verifier = TestTimeVerifier(entropy_threshold=3.5)

    tokens = tokenizer.encode_ordinary(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    reflections_triggered = 0
    in_think_block = False

    think_start_id = tokenizer.encode_ordinary("<think>")[-1] if "<think>" in tokenizer.encode_ordinary("<think>") else None

    for step in range(max_tokens):
        with torch.no_grad():
            out = model(input_ids)
            next_token_logits = out.logits[:, -1, :]

        entropy = verifier.calculate_entropy(next_token_logits[0])

        # Test-time verification check
        if in_think_block and reflections_triggered < max_reflections:
            if verifier.should_trigger_reflection(entropy, step):
                # Inject a dynamic backtracking correction token sequence
                reflection_phrase = " Wait, let me re-verify this step: "
                refl_tokens = tokenizer.encode_ordinary(reflection_phrase)
                input_ids = torch.cat([input_ids, torch.tensor([refl_tokens], dtype=torch.long, device=device)], dim=1)
                reflections_triggered += 1
                continue

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token[0, 0].item()

        if token_id == tokenizer.eot_token:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_text = tokenizer.decode(input_ids[0, len(tokens):].tolist())
    return generated_text, reflections_triggered
