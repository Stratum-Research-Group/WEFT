"""
Item 9: Fixed 50-Prompt Side-by-Side Generation Evaluation
Evaluates 50 deterministic prompts with fixed seed on Qwen2.5-0.5B-Instruct,
comparing FP16 baseline completions against WEFT Sub-Bit completions,
and saves side-by-side completions to results/eval_50prompt_generation.json.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from weft.converter import factorize_linear_layer

PROMPTS = [
    "The capital of France is",
    "In mathematics, Euler's identity states that",
    "The primary function of mitochondria is",
    "Python is an interpreted, high-level, general-purpose programming language that",
    "Photosynthesis is the biological process by which",
    "The theory of general relativity was developed by",
    "To reverse a linked list in computer science,",
    "The speed of light in a vacuum is approximately",
    "An electric current flowing through a conductor produces",
    "DNA is composed of four nucleotide bases:",
    "Machine learning algorithms build a mathematical model based on",
    "The Roman Empire was founded by",
    "Water has a chemical formula of",
    "Newton's second law of motion states that force is equal to",
    "A prime number is defined as a natural number that",
    "The solar system consists of the Sun and",
    "In economics, supply and demand is a model of",
    "The process of cellular mitosis results in",
    "Quantum mechanics is a fundamental theory in physics that provides",
    "The human heart is composed of four chambers:",
    "A binary search algorithm has a time complexity of",
    "The boiling point of pure water at sea level is",
    "The United Nations was established following",
    "Thermodynamics is the branch of physics that deals with",
    "The Pythagorean theorem applies to right-angled triangles and states that",
    "Artificial neural networks are computing systems inspired by",
    "The law of conservation of energy states that energy cannot be",
    "A database transaction must satisfy ACID properties, which stand for",
    "The Earth rotates on its axis once every",
    "Photosynthetic organisms convert carbon dioxide and water into",
    "The Renaissance was a cultural movement that began in",
    "In object-oriented programming, encapsulation is the concept of",
    "The atomic number of an element corresponds to the number of",
    "A Turing machine is a mathematical model of computation that",
    "The Pacific Ocean is the largest and",
    "Graph theory is the study of mathematical structures used to model",
    "The Industrial Revolution began in Great Britain during the",
    "Enzymes act as biological catalysts by",
    "TCP/IP is a suite of communication protocols used to",
    "The square root of 144 is",
    "The Great Wall of China was primarily built to",
    "A hash table provides average-case time complexity of",
    "Electromagnetic radiation travels in the form of",
    "The human brain contains approximately eighty-six billion",
    "Plate tectonics is the scientific theory describing the large-scale motion of",
    "A group in abstract algebra is a set equipped with a binary operation that satisfies",
    "The Magna Carta was agreed to by King John of England in",
    "The fundamental theorem of calculus establishes a relationship between",
    "Chlorophyll is the pigment responsible for",
    "The Fibonacci sequence is defined such that each number is the sum of"
]

def run_evaluation(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", out_path: str = "results/eval_50prompt_generation.json"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[GEN EVAL] Running 50-prompt side-by-side comparison on {model_id} ({device})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 1. FP16 Baseline Generations
    print("[*] Generating completions with FP16 baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model_fp16.eval()

    fp16_completions = []
    torch.manual_seed(42)
    with torch.no_grad():
        for p in PROMPTS:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            out = model_fp16.generate(**inputs, max_new_tokens=16, do_sample=False)
            gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
            fp16_completions.append(gen_text)

    del model_fp16
    torch.cuda.empty_cache()

    # 2. WEFT Sub-Bit Model Generations (Rank 64, 1 Path)
    print("[*] Quantizing model with WEFT (r=64, p=1) and generating completions...")
    model_weft = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model_weft.eval()

    for layer in model_weft.model.layers:
        projs = [
            layer.self_attn.q_proj, layer.self_attn.k_proj,
            layer.self_attn.v_proj, layer.self_attn.o_proj,
            layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
        ]
        for module in projs:
            orig_W = module.weight.data
            subbit = factorize_linear_layer(orig_W, base_rank=64, adaptive_rank=False, num_paths=1, itq_iterations=3)
            rec_W = subbit.reconstruct_weight().to(orig_W.dtype).to(device)
            module.weight.data.copy_(rec_W)

    weft_completions = []
    torch.manual_seed(42)
    with torch.no_grad():
        for p in PROMPTS:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            out = model_weft.generate(**inputs, max_new_tokens=16, do_sample=False)
            gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
            weft_completions.append(gen_text)

    del model_weft
    torch.cuda.empty_cache()

    results = {
        "model": model_id,
        "num_prompts": len(PROMPTS),
        "seed": 42,
        "max_new_tokens": 16,
        "comparisons": []
    }

    for i in range(len(PROMPTS)):
        results["comparisons"].append({
            "prompt": PROMPTS[i],
            "fp16": fp16_completions[i],
            "weft_subbit": weft_completions[i]
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Saved 50 side-by-side prompt completions to {out_path}")

if __name__ == "__main__":
    run_evaluation()
