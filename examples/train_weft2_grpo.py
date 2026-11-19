"""
WEFT-2 Group Relative Policy Optimization (GRPO) Reinforcement Learning Pipeline.

Fine-tunes the 589M WEFT-2 foundation model on arithmetic and algorithmic reasoning
using deterministic rule verifiers (MathRuleVerifier & CodeRuleVerifier).

Demonstrates:
1. Critic-free reward baseline across K rollout completions:
   A_i = (R_i - mean(R)) / (std(R) + 1e-4)
2. Group relative policy gradient objective with length-normalized sequence log-probs:
   L = -E[ A_i * (1/|y_i|) * sum_t log pi_theta(y_{i,t} | x, y_{i,<t}) ]
3. Sub-linear generation footprint using WEFT-2 constant recurrent cache + Titans memory.
"""

import os
import re
import time
import random
import torch
import torch.nn.functional as F
import tiktoken

from weft import WeftConfig, WeftForCausalLM
from weft.verifier import MathRuleVerifier, CodeRuleVerifier


class ArithmeticReasoningTask:
    verifier = MathRuleVerifier()

    @staticmethod
    def generate_problem():
        a = random.randint(3, 25)
        b = random.randint(3, 25)
        c = random.randint(2, 9)
        d = random.randint(1, 30)
        true_answer = ((a + b) * c) - d

        prompt = (
            f"Question: Compute (({a} + {b}) * {c}) - {d}.\n"
            f"Think step by step inside <think>...</think> and write the exact final integer answer as #### [number].\n"
            f"Answer:"
        )
        return prompt, true_answer


def train_weft2_grpo(
    model_path: str = "weft2_sft_ckpt",
    save_path: str = "weft2_grpo_aligned_ckpt",
    num_episodes: int = 300,
    group_size: int = 4,
    max_gen_tokens: int = 96,
    lr: float = 1e-5,
    device: str = "cuda",
):
    print("=" * 80)
    print("  WEFT-2 GRPO REINFORCEMENT LEARNING WITH RULE VERIFIERS")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Group Size K: {group_size} completions/prompt | Max Gen: {max_gen_tokens} tokens")
    print("=" * 80)

    # Load pretrained WEFT-2 model
    print(f"Loading pretrained WEFT-2 foundation checkpoint from {model_path}...")
    model = WeftForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.train()
    model.gradient_checkpointing_enable()

    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    task = ArithmeticReasoningTask()

    print("\nBeginning GRPO Policy Optimization...\n")
    t0 = time.time()

    for episode in range(1, num_episodes + 1):
        prompt, true_ans = task.generate_problem()
        prompt_tokens = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_tokens], device=device).expand(group_size, -1)

        # 1. Rollout K completions using model.generate
        with torch.no_grad():
            gen_out = model.generate(
                prompt_tensor,
                max_new_tokens=max_gen_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eot_token,
            )

        gen_tokens = gen_out[:, prompt_tensor.shape[1] :]
        completions = [tokenizer.decode(t.tolist()) for t in gen_tokens]

        # 2. Evaluate deterministic rewards via MathRuleVerifier
        rewards = []
        accurate_completions = 0
        for comp in completions:
            res = task.verifier.verify_solution(comp, float(true_ans))
            rewards.append(res.total_reward)
            if res.accuracy_reward >= 1.0:
                accurate_completions += 1

        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

        # 3. Compute group relative advantage
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std() + 1e-4
        advantages = (rewards_tensor - mean_r) / std_r

        # 4. Surrogate Policy Gradient step
        outputs = model(input_ids=gen_out, labels=gen_out)
        logits = outputs.logits[:, prompt_tensor.shape[1] - 1 : -1, :]
        log_probs = F.log_softmax(logits, dim=-1)

        # Token-level log probabilities
        targets = gen_tokens.unsqueeze(-1)
        selected_log_probs = log_probs.gather(dim=-1, index=targets).squeeze(-1)

        # Mask positions strictly after the first terminal EOT while including the terminal EOT itself
        is_eot = (gen_tokens == tokenizer.eot_token)
        # Cumulative sum of EOT: 0 before EOT, 1 at first EOT, >1 for subsequent padding
        eot_cumsum = torch.cumsum(is_eot.long(), dim=-1)
        valid_mask = (eot_cumsum <= 1)

        masked_log_probs = selected_log_probs * valid_mask.float()
        seq_log_probs = masked_log_probs.sum(dim=-1)
        comp_lens = valid_mask.sum(dim=-1).clamp(min=1)
        norm_seq_log_probs = seq_log_probs / comp_lens.float()

        # REINFORCE with Group Relative baseline
        loss = -(advantages * norm_seq_log_probs).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if episode % 10 == 0 or episode == num_episodes:
            elapsed = time.time() - t0
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            acc = accurate_completions / group_size
            print(
                f"Episode {episode:4d}/{num_episodes} | "
                f"Mean Reward: {mean_r.item():5.2f} | "
                f"Accuracy: {acc * 100:4.1f}% | "
                f"Loss: {loss.item():6.3f} | "
                f"VRAM: {vram_mb:6.1f} MB | Elapsed: {elapsed:5.1f}s"
            )

        # Checkpoint saving every 50 episodes or on completion
        if episode % 50 == 0 or episode == num_episodes:
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            print(f"  [Checkpoint] Saved GRPO policy model to {save_path}/ at episode {episode}")

    print(f"\nGRPO training episode complete. Final aligned model saved to {save_path}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="weft2_grpo_aligned_ckpt")
    args = parser.parse_args()

    train_weft2_grpo(num_episodes=args.episodes, group_size=args.group_size, save_path=args.save_path)
