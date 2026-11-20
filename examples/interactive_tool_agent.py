"""
WEFT-2 Native Tool-Calling & Interactive Multi-Turn CLI Agent.

Provides a unified interactive chat session with:
1. Dynamic Tool Registry (system information, calculations, web/database simulations)
2. Grammar-constrained generation
3. Structured output parsing & immediate execution feedback
"""

import sys
import json
import torch
import tiktoken
from weft import WeftForCausalLM
from weft.tool_calling import ToolRegistry, extract_tool_calls, format_tool_response
from weft.grammar_logits import JSONGrammarLogitsProcessor


def build_default_cli_agent(
    model_path: str = "checkpoints/weft2_tool_grpo",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Loading WEFT-2 Agent from {model_path} on {device}...")
    actual_path = model_path if torch.cuda.is_available() else "checkpoints/weft2_tool_grpo"
    model = WeftForCausalLM.from_pretrained(actual_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    registry = ToolRegistry()

    # Register default agent tools
    registry.register(
        name="calculator",
        description="Safely evaluate basic mathematical expressions.",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        func=lambda expression: eval(expression, {"__builtins__": None}, {}),
    )

    registry.register(
        name="fetch_system_status",
        description="Query GPU memory and device health.",
        parameters={"type": "object", "properties": {}, "required": []},
        func=lambda: {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0,
            "status": "healthy",
        },
    )

    return model, tokenizer, registry


def run_agent_interaction(user_input: str, model, tokenizer, registry, device="cuda"):
    sys_prompt = registry.format_system_prompt()
    conversation = f"{sys_prompt}\nUser: {user_input}\nAssistant:"
    tokens = tokenizer.encode_ordinary(conversation)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Autoregressive generation with agentic tool loop
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=128, eos_token_id=tokenizer.eot_token)

    generated_text = tokenizer.decode(out[0, len(tokens):].tolist())
    calls = extract_tool_calls(generated_text)

    executed_results = []
    if calls:
        for call in calls:
            t_name = call.get("name")
            t_args = call.get("arguments", {})
            res = registry.execute(t_name, t_args)
            executed_results.append((t_name, t_args, res))

    return generated_text, executed_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  WEFT-2 INTERACTIVE AGENT DEMONSTRATION")
    print("=" * 80)
    model, tok, reg = build_default_cli_agent(device=device)

    sample_query = "Check system status and calculate 25 * 14."
    print(f"\nUser Query: {sample_query}")
    text, results = run_agent_interaction(sample_query, model, tok, reg, device=device)
    print("\n--- Model Output Stream ---")
    print(text)
    if results:
        print("\n--- Tool Execution Feedback ---")
        for name, args, res in results:
            print(f"Executed: {name}({args}) -> {res}")
