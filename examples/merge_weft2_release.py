"""
WEFT-2 Unified Model Merger: Blends needle-calibrated associative state binding
with multi-domain chat/reasoning weights via spherical / weighted linear interpolation.
Produces: weft2_release_v2_ckpt/
"""

import os
import torch
from weft import WeftForCausalLM


def merge_checkpoints(
    chat_ckpt: str = "weft2_chat_v2_ckpt",
    needle_ckpt: str = "weft2_needle_calibrated_ckpt",
    save_dir: str = "weft2_release_v2_ckpt",
    needle_weight: float = 0.35,
):
    print("=" * 80)
    print("  WEFT-2 UNIFIED RELEASE CHECKPOINT MERGER (v2.1.0)")
    print(f"  Chat/Reasoning Base: {chat_ckpt}")
    print(f"  Needle/Associative Base: {needle_ckpt}")
    print(f"  Blend Ratio: {1.0 - needle_weight:.2f} (Chat) + {needle_weight:.2f} (Needle)")
    print(f"  Target: {save_dir}")
    print("=" * 80)

    print(f"Loading chat base from {chat_ckpt}...")
    chat_model = WeftForCausalLM.from_pretrained(chat_ckpt, torch_dtype=torch.bfloat16)

    print(f"Loading needle base from {needle_ckpt}...")
    needle_model = WeftForCausalLM.from_pretrained(needle_ckpt, torch_dtype=torch.bfloat16)

    chat_sd = chat_model.state_dict()
    needle_sd = needle_model.state_dict()

    merged_sd = {}
    total_params = len(chat_sd)
    print(f"Merging {total_params} tensors...")

    for key in chat_sd:
        t_chat = chat_sd[key]
        if key in needle_sd:
            t_needle = needle_sd[key]
            if t_chat.dtype.is_floating_point:
                # Recurrent decay & titans memory benefit heavily from needle calibration
                if any(k in key for k in ["decay_logits", "titans", "routing", "k_proj"]):
                    w_n = min(0.60, needle_weight * 1.5)
                else:
                    w_n = needle_weight
                merged = (1.0 - w_n) * t_chat.float() + w_n * t_needle.float()
                merged_sd[key] = merged.to(t_chat.dtype)
            else:
                merged_sd[key] = t_chat
        else:
            merged_sd[key] = t_chat

    chat_model.load_state_dict(merged_sd)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving merged unified model to {save_dir}...")
    chat_model.save_pretrained(save_dir)
    print("--> [PASS] Unified release checkpoint created successfully!")


if __name__ == "__main__":
    merge_checkpoints()
