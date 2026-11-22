import os
import json
import torch
from safetensors import safe_open
from weft.converter import convert_model_to_weft

checkpoints_to_convert = [
    ("checkpoints/weft2_sft_15k", "checkpoints/weft2_sft_15k.weft"),
    ("checkpoints/weft2_grpo_15k", "checkpoints/weft2_grpo_15k.weft"),
    ("checkpoints/weft2_tool_grpo", "checkpoints/weft2_tool_grpo.weft"),
]

for src_dir, out_weft in checkpoints_to_convert:
    safetensors_path = os.path.join(src_dir, "model.safetensors")
    config_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(safetensors_path) or not os.path.exists(config_path):
        print(f"Skipping {src_dir}, missing files.")
        continue

    print(f"\n==================================================")
    print(f"Converting {src_dir} -> {out_weft}")
    print(f"==================================================")
    with open(config_path) as f:
        config = json.load(f)

    state_dict = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    print(f"Loaded {len(state_dict)} tensors.")
    container, cert = convert_model_to_weft(
        state_dict=state_dict,
        config=config,
        output_filepath=out_weft,
        base_rank=32,
        adaptive_rank=True,
        certify_intervals=True,
    )
    print(f"Successfully converted {out_weft} with {len(cert) if cert else 0} certified layers.")

print("\nAll batch fleet conversions completed.")
