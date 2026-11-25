"""
Measured speculative-decoding acceptance from real draft/target logits.

Unlike weft/speculative_cascade.py (an analytical simulator with an assumed
acceptance probability), every number returned here is computed from logits that
the caller supplies. Nothing is drawn from an RNG.

Greedy verification rule: draft token d_i is accepted iff it equals
argmax(target_logits[i]) and all earlier draft tokens were accepted.
"""

from typing import Dict, List, Sequence
import torch


def accepted_prefix_len(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> int:
    """draft_tokens: [gamma] ints. target_logits: [gamma, vocab].
    Returns the number of leading draft tokens the target would accept."""
    if draft_tokens.ndim != 1 or target_logits.ndim != 2:
        raise ValueError("expected draft_tokens [gamma] and target_logits [gamma, vocab]")
    if target_logits.shape[0] != draft_tokens.shape[0]:
        raise ValueError("draft/target length mismatch")
    target_pred = target_logits.argmax(dim=-1)
    n = 0
    for d, t in zip(draft_tokens.tolist(), target_pred.tolist()):
        if d != t:
            break
        n += 1
    return n


def measure_acceptance(
    rounds: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, object]:
    """rounds: list of {"draft_tokens": [gamma], "target_logits": [gamma, vocab]}.
    Returns the empirical per-round distribution and mean acceptance rate."""
    if not rounds:
        raise ValueError("no rounds supplied")
    lens: List[int] = []
    gammas: List[int] = []
    for r in rounds:
        lens.append(accepted_prefix_len(r["draft_tokens"], r["target_logits"]))
        gammas.append(int(r["draft_tokens"].shape[0]))
    total_drafted = sum(gammas)
    total_accepted = sum(lens)
    hist: Dict[int, int] = {}
    for n in lens:
        hist[n] = hist.get(n, 0) + 1
    return {
        "rounds": len(rounds),
        "tokens_drafted": total_drafted,
        "tokens_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_drafted,
        "accepted_prefix_histogram": dict(sorted(hist.items())),
        "source": "measured_from_supplied_logits",
    }
