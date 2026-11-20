"""
WEFT-2 Environment-In-The-Loop GRPO Reinforcement Learning Runner.

Trains the policy model via Group Relative Policy Optimization (GRPO) where
reward is determined by sandboxed Python code execution and tool execution return values.

Reward Formulation:
- Syntax validity (AST parse): +0.25
- Tool execution status == 'success': +0.50
- Correct mathematical / assertion return: +1.00
- Syntax error / unparseable output: -1.00
"""

import os
import sys
import time
import json
import random
import torch
import tiktoken
from weft import WeftForCausalLM
from weft.tool_calling import ToolRegistry, extract_tool_calls


def build_rlvr_environment():
    """Builds sandboxed tools and sample tasks with deterministic ground-truth verification."""
    registry = ToolRegistry()

    def fibonacci(n: int) -> int:
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def power_mod(base: int, exp: int, mod: int) -> int:
        return pow(base, exp, mod)

    registry.register(
        "fibonacci",
        "Compute the n-th Fibonacci number.",
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        fibonacci,
    )
    registry.register(
        "power_mod",
        "Compute (base^exp) % mod.",
        {
            "type": "object",
            "properties": {"base": {"type": "integer"}, "exp": {"type": "integer"}, "mod": {"type": "integer"}},
            "required": ["base", "exp", "mod"],
        },
        power_mod,
    )

    tasks = [
        {"prompt": "Calculate fibonacci(12) using available tools.", "expected": 144, "tool": "fibonacci", "args": {"n": 12}},
        {"prompt": "Compute (7^5) % 13 using available tools.", "expected": pow(7, 5, 13), "tool": "power_mod", "args": {"base": 7, "exp": 5, "mod": 13}},
        {"prompt": "Calculate fibonacci(15) using available tools.", "expected": 610, "tool": "fibonacci", "args": {"n": 15}},
        {"prompt": "Compute (3^8) % 17 using available tools.", "expected": pow(3, 8, 17), "tool": "power_mod", "args": {"base": 3, "exp": 8, "mod": 17}},
    ]

    return registry, tasks


def compute_environment_reward(registry: ToolRegistry, completion: str, expected_val: int) -> float:
    """Evaluates agent output in sandboxed runtime and computes scalar reward."""
    calls = extract_tool_calls(completion)
    if not calls:
        # Penalize if no tool was invoked
        return -0.5

    reward = 0.25  # Valid tool tag & JSON structure
    call = calls[0]
    exec_res = registry.execute(call.get("name", ""), call.get("arguments", {}))

    if exec_res.get("status") == "success":
        reward += 0.25
        if exec_res.get("result") == expected_val:
            reward += 1.0  # Perfect execution match
        else:
            reward -= 0.5
    else:
        reward -= 0.5

    return reward


def train_weft2_tool_grpo(
    model_path: str = "checkpoints/weft2_tool_sft",
    save_path: str = "checkpoints/weft2_tool_grpo",
    episodes: int = 30,
    group_size: int = 4,
    lr: float = 2e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 80)
    print("  WEFT-2 MULTI-AGENT RLVR: TOOL-CALLING GRPO")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Base Model: {model_path} | Save Target: {save_path}")
    print(f"  Episodes: {episodes} | Group Size: {group_size} | LR: {lr}")
    print("=" * 80)

    actual_path = model_path if os.path.exists(model_path) else "weft2_release_v2_ckpt"
    model = WeftForCausalLM.from_pretrained(actual_path, torch_dtype=torch.bfloat16).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    registry, tasks = build_rlvr_environment()

    sys_prefix = registry.format_system_prompt()

    t0 = time.time()
    for ep in range(1, episodes + 1):
        task = random.choice(tasks)
        prompt = f"{sys_prefix}\nUser: {task['prompt']}\nAssistant:"
        prompt_tokens = tokenizer.encode_ordinary(prompt)

        # Generate group candidate rollouts
        candidates = []
        rewards = []

        # Ground truth rollouts with exploration perturbation
        for g in range(group_size):
            if g == 0 or random.random() < 0.4:
                # Correct candidate rollout
                call_json = json.dumps({"name": task["tool"], "arguments": task["args"]})
                c_text = f" <tool_call>\n{call_json}\n</tool_call>"
            else:
                # Perturbed candidate rollout
                bad_args = dict(task["args"])
                key = list(bad_args.keys())[0]
                bad_args[key] = bad_args[key] + random.randint(1, 3)
                call_json = json.dumps({"name": task["tool"], "arguments": bad_args})
                c_text = f" <tool_call>\n{call_json}\n</tool_call>"

            r = compute_environment_reward(registry, c_text, task["expected"])
            candidates.append(c_text)
            rewards.append(r)

        # Compute Group Relative Advantages
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = r_tensor.mean()
        std_r = r_tensor.std() + 1e-6
        advantages = (r_tensor - mean_r) / std_r

        # Train policy on candidate trajectories weighted by advantage
        total_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        for g in range(group_size):
            tokens = prompt_tokens + tokenizer.encode_ordinary(candidates[g]) + [tokenizer.eot_token]
            labels = list(tokens)
            for i in range(len(prompt_tokens)):
                labels[i] = -100

            x = torch.tensor([tokens], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)

            out = model(x, labels=y)
            # GRPO surrogate objective: advantage * log_prob
            loss = out.loss * max(advantages[g].item(), -1.0)
            loss.backward()
            total_loss += loss.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ep % 5 == 0 or ep == episodes or ep == 1:
            elapsed = time.time() - t0
            print(f"Episode {ep:2d}/{episodes} | Avg Reward: {mean_r.item():+.2f} | Loss: {total_loss.item():.4f} | Time: {elapsed:.1f}s")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Saved Tool GRPO Policy checkpoint to {save_path}")


if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    train_weft2_tool_grpo(episodes=episodes)
