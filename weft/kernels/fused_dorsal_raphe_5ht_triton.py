"""Triton 3.9 GPU kernel for fused Dorsal Raphe Serotonin (5-HT) temporal discounting.

Computes batched hyperbolic temporal discounting, subjective delayed reward value,
accumulated waiting cost, and perseverance probability across simulated decision agents.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dorsal_raphe_5ht_kernel(
    serotonin_tone_ptr,
    delay_duration_ptr,
    nominal_reward_ptr,
    opportunity_cost_ptr,
    out_k_eff_ptr,
    out_value_ptr,
    out_net_value_ptr,
    out_prob_ptr,
    k_base,
    gamma,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load vectors
    tone = tl.load(serotonin_tone_ptr + offsets, mask=mask, other=0.0)
    delay = tl.load(delay_duration_ptr + offsets, mask=mask, other=0.0)
    reward = tl.load(nominal_reward_ptr + offsets, mask=mask, other=0.0)
    cost = tl.load(opportunity_cost_ptr + offsets, mask=mask, other=0.0)

    # Effective temporal discount factor k_eff = k_base / (1.0 + gamma * 5HT_tone)
    k_denom = 1.0 + gamma * tone
    k_eff = k_base / k_denom

    # Hyperbolic discounted value: V(t) = reward / (1.0 + k_eff * delay)
    v_denom = 1.0 + k_eff * delay
    value = reward / v_denom

    # Net subjective value: Net_V = value - cost * delay
    cost_accum = cost * delay
    net_value = value - cost_accum

    # Approximate perseverance probability: sigmoid(0.25 * net_value)
    # Using clamped algebraic approximation: clamp(0.5 + 0.02 * net_value, 0.0, 1.0)
    prob_raw = 0.5 + 0.02 * net_value
    prob = tl.maximum(0.0, tl.minimum(1.0, prob_raw))

    # Store results
    tl.store(out_k_eff_ptr + offsets, k_eff, mask=mask)
    tl.store(out_value_ptr + offsets, value, mask=mask)
    tl.store(out_net_value_ptr + offsets, net_value, mask=mask)
    tl.store(out_prob_ptr + offsets, prob, mask=mask)


def launch_fused_dorsal_raphe_5ht(
    serotonin_tone: torch.Tensor,
    delay_duration: torch.Tensor,
    nominal_reward: torch.Tensor,
    opportunity_cost: torch.Tensor,
    k_base: float = 0.85,
    gamma: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch Triton fused dorsal raphe 5-HT patience and discounting kernel."""
    assert serotonin_tone.is_contiguous()
    assert delay_duration.is_contiguous()
    assert nominal_reward.is_contiguous()
    assert opportunity_cost.is_contiguous()

    n_elements = serotonin_tone.numel()
    out_k_eff = torch.empty_like(serotonin_tone)
    out_value = torch.empty_like(serotonin_tone)
    out_net_value = torch.empty_like(serotonin_tone)
    out_prob = torch.empty_like(serotonin_tone)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_dorsal_raphe_5ht_kernel[grid](
        serotonin_tone,
        delay_duration,
        nominal_reward,
        opportunity_cost,
        out_k_eff,
        out_value,
        out_net_value,
        out_prob,
        k_base,
        gamma,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out_k_eff, out_value, out_net_value, out_prob
