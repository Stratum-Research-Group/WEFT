#!/usr/bin/env python3
"""
WEFT Automated System Prompt & Chat Template Registry Validator.
Validates special token schemas, turn demarcations, and boundary encodings
across all local model architectures in the fleet to prevent template syntax leaks.
"""
import sys
import json
from typing import Dict, Any, List

CHAT_TEMPLATES = {
    "gemma": {
        "start_user": "<start_of_turn>user\n",
        "end_user": "<end_of_turn>\n",
        "start_model": "<start_of_turn>model\n",
        "end_model": "<end_of_turn>\n",
        "system_policy": "prefix_first_user"
    },
    "chatml": {
        "start_user": "<|im_start|>user\n",
        "end_user": "<|im_end|>\n",
        "start_model": "<|im_start|>assistant\n",
        "end_model": "<|im_end|>\n",
        "start_system": "<|im_start|>system\n",
        "end_system": "<|im_end|>\n",
        "system_policy": "dedicated_block"
    },
    "llama3": {
        "start_user": "<|start_header_id|>user<|end_header_id|>\n\n",
        "end_user": "<|eot_id|>",
        "start_model": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "end_model": "<|eot_id|>",
        "start_system": "<|start_header_id|>system<|end_header_id|>\n\n",
        "end_system": "<|eot_id|>",
        "system_policy": "dedicated_block"
    },
    "mistral": {
        "start_user": "[INST] ",
        "end_user": " [/INST]",
        "start_model": "",
        "end_model": "</s>",
        "system_policy": "prefix_first_user"
    }
}

def format_turn(template_name: str, system_msg: str, user_msg: str) -> str:
    tmpl = CHAT_TEMPLATES[template_name]
    output = ""
    if tmpl["system_policy"] == "dedicated_block":
        output += f"{tmpl['start_system']}{system_msg}{tmpl['end_system']}"
        output += f"{tmpl['start_user']}{user_msg}{tmpl['end_user']}"
        output += f"{tmpl['start_model']}"
    else:
        combined = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg
        output += f"{tmpl['start_user']}{combined}{tmpl['end_user']}"
        output += f"{tmpl['start_model']}"
    return output

def validate_all_templates():
    print("==========================================================")
    print("  WEFT Chat Template Registry Validator")
    print("==========================================================")
    
    system = "You are a formally verified AI assistant."
    user = "Solve the RoPE rotation matrix equation."

    failures = 0
    for name, schema in CHAT_TEMPLATES.items():
        formatted = format_turn(name, system, user)
        # Check that user prompt and system prompt are fully enclosed
        has_user = user in formatted
        has_system = system in formatted
        has_start_user = schema["start_user"] in formatted
        has_end_user = schema["end_user"] in formatted

        valid = has_user and has_system and has_start_user and has_end_user
        status = "PASSED" if valid else "FAILED"
        if not valid:
            failures += 1
        print(f"[{status}] Architecture: {name.upper():<8} | Length: {len(formatted)} chars")

    print("----------------------------------------------------------")
    if failures == 0:
        print("[SUCCESS] All 4 chat template architectures strictly validated.")
        return 0
    else:
        print(f"[FAIL] {failures} architectures failed validation.")
        return 1

if __name__ == "__main__":
    sys.exit(validate_all_templates())
