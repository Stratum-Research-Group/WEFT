"""Real speculative decoding benchmark (greedy). Draft=Qwen2.5-Coder-0.5B, Target=Qwen2.5-Coder-1.5B.
Measures: acceptance rate, wall-clock tokens/s with vs without speculation, and exact-match vs plain greedy."""
import os, sys, json, time, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
snap = lambda n: glob.glob(f"{HUB}/models--Qwen--{n}/snapshots/*")[0]
DT = torch.bfloat16
PROMPTS = [
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "import numpy as np\n\ndef softmax(x):\n",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, value):\n",
    "# Merge sort implementation in Python\ndef merge_sort(arr):\n",
    "#include <stdio.h>\n\nint main(void) {\n",
    "SELECT name, COUNT(*) AS n\nFROM orders\nGROUP BY",
    "async function fetchJson(url) {\n",
    "def is_prime(n):\n",
]
NEW = 128


@torch.no_grad()
def greedy(model, ids, n):
    out = model(ids, use_cache=True); cache = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True); toks = [nxt.item()]
    for _ in range(n - 1):
        out = model(nxt, past_key_values=cache, use_cache=True); cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True); toks.append(nxt.item())
    return toks


@torch.no_grad()
def speculative(target, draft, ids, n, k):
    """Cache invariant: caches cover seq[:-1]; seq[-1] is the next input."""
    seq = ids[0].tolist()
    t_cache = target(ids[:, :-1], use_cache=True).past_key_values
    d_cache = draft(ids[:, :-1], use_cache=True).past_key_values
    gen, proposed, accepted, rounds = [], 0, 0, 0
    dev = ids.device
    while len(gen) < n:
        L = len(seq)
        inp = torch.tensor([[seq[-1]]], device=dev)
        drafts = []
        for _ in range(k):
            o = draft(inp, past_key_values=d_cache, use_cache=True); d_cache = o.past_key_values
            inp = o.logits[:, -1].argmax(-1, keepdim=True); drafts.append(inp.item())
        v_in = torch.tensor([[seq[-1]] + drafts], device=dev)
        o = target(v_in, past_key_values=t_cache, use_cache=True); t_cache = o.past_key_values
        pred = o.logits[0].argmax(-1).tolist()          # k+1 predictions
        m = 0
        while m < k and drafts[m] == pred[m]: m += 1
        new = drafts[:m] + [pred[m]]
        seq += new; gen += new
        proposed += k; accepted += m; rounds += 1
        crop_to(t_cache, L + m)                          # cover new_seq[:-1]
        if m == k:                                       # draft is missing d_{k-1}
            o = draft(torch.tensor([[drafts[-1]]], device=dev), past_key_values=d_cache, use_cache=True); d_cache = o.past_key_values
        else:
            crop_to(d_cache, L + m)
    return gen[:n], proposed, accepted, rounds


def sync(): torch.cuda.synchronize()


def crop_to(cache, n):
    """transformers>=5: crop() takes a negative count of tokens to remove."""
    cur = cache.get_seq_length()
    assert cur >= n, (cur, n)
    if cur > n:
        cache.crop(-(cur - n))


def main():
    tok = AutoTokenizer.from_pretrained(snap("Qwen2.5-Coder-1.5B"))
    target = AutoModelForCausalLM.from_pretrained(snap("Qwen2.5-Coder-1.5B"), torch_dtype=DT).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(snap("Qwen2.5-Coder-0.5B"), torch_dtype=DT).cuda().eval()
    enc = [tok(p, return_tensors="pt").input_ids.cuda() for p in PROMPTS]
    greedy(target, enc[0], 8); speculative(target, draft, enc[0], 8, 4)   # warmup
    res = {"draft": "Qwen2.5-Coder-0.5B", "target": "Qwen2.5-Coder-1.5B", "dtype": str(DT), "new_tokens": NEW,
           "num_prompts": len(PROMPTS), "gpu": torch.cuda.get_device_name(0), "runs": []}
    sync(); t0 = time.perf_counter()
    base = [greedy(target, e, NEW) for e in enc]
    sync(); base_t = time.perf_counter() - t0
    base_tps = len(PROMPTS) * NEW / base_t
    res["baseline_tokens_per_s"] = base_tps
    print(f"baseline greedy target: {base_tps:.1f} tok/s")
    for k in [2, 3, 4, 6, 8]:
        sync(); t0 = time.perf_counter()
        outs = [speculative(target, draft, e, NEW, k) for e in enc]
        sync(); dt = time.perf_counter() - t0
        P = sum(o[1] for o in outs); A = sum(o[2] for o in outs); R = sum(o[3] for o in outs)
        match = sum(o[0] == b for o, b in zip(outs, base))
        # token-level agreement up to first divergence
        pref = [next((i for i, (x, y) in enumerate(zip(o[0], b)) if x != y), NEW) for o, b in zip(outs, base)]
        r = {"k": k, "tokens_per_s": len(PROMPTS) * NEW / dt, "speedup_vs_baseline": (len(PROMPTS) * NEW / dt) / base_tps,
             "acceptance_rate": A / P, "mean_tokens_per_target_pass": (A + R) / R,
             "prompts_exact_match": f"{match}/{len(PROMPTS)}", "mean_matching_prefix_tokens": sum(pref) / len(pref)}
        res["runs"].append(r); print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/speculative_real_qwen_coder.json", "w"), indent=2)


if __name__ == "__main__":
    main()
