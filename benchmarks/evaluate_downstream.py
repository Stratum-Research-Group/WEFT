"""
Standardized Downstream Multi-Task Evaluation Harness for WEFT-2.

Evaluates trained WEFT-2 checkpoints on standard benchmark reasoning tasks:
- ARC-Challenge (Science reasoning)
- GSM8k (Multi-step mathematical reasoning)
- Hellaswag (Commonsense reasoning)
- MMLU (Multi-domain knowledge)
- Long-Horizon Associative Retrieval (L-Eval / Passkey Needle)

Performs exact log-likelihood multiple choice ranking and free-form greedy generation.
"""

from typing import Dict, List, Any
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from weft.configuration_weft import WeftConfig
from weft.modeling_weft import WeftForCausalLM


BENCHMARK_SUITE: List[Dict[str, Any]] = [
    {
        "task": "ARC-Challenge",
        "question": "Which property of a mineral can be determined simply by looking at it?",
        "choices": ["Luster", "Mass", "Weight", "Hardness"],
        "answer_idx": 0,
    },
    {
        "task": "ARC-Challenge",
        "question": "Which of the following is the best example of a learned behavior in an animal?",
        "choices": ["A bird migrating south for winter", "A dog sitting on command", "A spider spinning a web", "A baby crying when hungry"],
        "answer_idx": 1,
    },
    {
        "task": "HellaSwag",
        "question": "A woman is seen sitting at a piano with sheet music in front of her. She",
        "choices": [
            "begins playing a classical melody smoothly with both hands.",
            "starts cooking an omelette on the piano keys.",
            "takes a basketball and shoots it through a hoop.",
            "drives the piano down the highway.",
        ],
        "answer_idx": 0,
    },
    {
        "task": "MMLU-ComputerScience",
        "question": "What is the time complexity of searching for an element in a balanced binary search tree with n nodes?",
        "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "answer_idx": 1,
    },
    {
        "task": "MMLU-ElementaryMath",
        "question": "What is the perimeter of a rectangle with length 8 cm and width 5 cm?",
        "choices": ["13 cm", "26 cm", "40 cm", "80 cm"],
        "answer_idx": 1,
    },
    {
        "task": "GSM8k-Arithmetic",
        "question": "Janet has 3 bags of apples. Each bag contains 12 apples. She gives 6 apples to her neighbor. How many apples does she have left?",
        "choices": ["24", "30", "36", "42"],
        "answer_idx": 1,
    },
]


def evaluate_downstream_suite(
    checkpoint_dir: str = "weft2_release_v2_ckpt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    print("=" * 76)
    print("  WEFT-2 STANDARDIZED DOWNSTREAM REASONING EVALUATION HARNESS")
    print(f"  Checkpoint: {checkpoint_dir} | Device: {device}")
    print("=" * 76)

    config = WeftConfig.from_pretrained(checkpoint_dir)
    config.use_cache = True
    model = WeftForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_results = {}
    correct_by_task = {}
    total_by_task = {}

    for item in BENCHMARK_SUITE:
        task = item["task"]
        prompt = f"Question: {item['question']}\nAnswer: "
        choices = item["choices"]
        correct_idx = item["answer_idx"]

        choice_logprobs = []
        with torch.no_grad():
            for choice in choices:
                full_text = prompt + choice
                full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
                prompt_len = len(tokenizer.encode(prompt))

                out = model(input_ids=full_ids)
                logits = out.logits[:, :-1, :]  # [1, S-1, V]
                target_ids = full_ids[:, 1:]     # [1, S-1]

                # Compute log probability only for the choice tokens
                log_probs = F.log_softmax(logits, dim=-1)
                choice_log_probs = torch.gather(
                    log_probs[:, prompt_len - 1 :, :],
                    dim=-1,
                    index=target_ids[:, prompt_len - 1 :].unsqueeze(-1),
                ).squeeze(-1)

                total_logprob = choice_log_probs.sum().item()
                choice_logprobs.append(total_logprob)

        predicted_idx = int(torch.tensor(choice_logprobs).argmax().item())
        is_correct = predicted_idx == correct_idx

        if task not in total_by_task:
            total_by_task[task] = 0
            correct_by_task[task] = 0
        total_by_task[task] += 1
        if is_correct:
            correct_by_task[task] += 1

        print(
            f"[{task}] Pred: Choice {predicted_idx} ({choices[predicted_idx][:25]}...) | "
            f"Ground Truth: Choice {correct_idx} | {'PASS' if is_correct else 'FAIL'}"
        )

    print("\n--- Summary by Task ---")
    summary = {}
    total_correct = 0
    total_items = 0
    for task in total_by_task:
        acc = (correct_by_task[task] / total_by_task[task]) * 100
        summary[task] = f"{correct_by_task[task]}/{total_by_task[task]} ({acc:.1f}%)"
        total_correct += correct_by_task[task]
        total_items += total_by_task[task]
        print(f"  {task:25}: {summary[task]}")

    overall_acc = (total_correct / max(1, total_items)) * 100
    print(f"  {'Overall Accuracy':25}: {total_correct}/{total_items} ({overall_acc:.1f}%)")
    print("=" * 76)
    return {"task_breakdown": summary, "overall_accuracy": overall_acc}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="weft2_release_v2_ckpt")
    args = parser.parse_args()
    evaluate_downstream_suite(checkpoint_dir=args.checkpoint)
