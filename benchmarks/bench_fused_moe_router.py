"""Real CUDA-event benchmark: flat vs cluster router, Triton-fused vs PyTorch. D=8192, E=128, C=16."""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from weft.kernels.fused_cluster_moe_triton import (
    fused_topk_router, fused_cluster_router, ref_topk_router, ref_cluster_router)


def timeit(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def main():
    D, E, C = 8192, 128, 16
    torch.manual_seed(0)
    g = torch.randn(E, D, device="cuda"); cg = torch.randn(C, D, device="cuda")
    out = {"device": torch.cuda.get_device_name(0), "D": D, "E": E, "C": C,
           "method": "cuda events, 200 iters, 30 warmup, fp32", "rows": []}
    for B in [1, 4, 16, 64, 256]:
        x = torch.randn(B, D, device="cuda")
        r = {"B": B,
             "torch_flat_us": timeit(lambda: ref_topk_router(x, g)),
             "torch_cluster_us": timeit(lambda: ref_cluster_router(x, cg, g)),
             "triton_flat_us": timeit(lambda: fused_topk_router(x, g)),
             "triton_cluster_us": timeit(lambda: fused_cluster_router(x, cg, g))}
        r["cluster_vs_flat_triton_speedup"] = r["triton_flat_us"] / r["triton_cluster_us"]
        r["triton_flat_vs_torch_flat_speedup"] = r["torch_flat_us"] / r["triton_flat_us"]
        out["rows"].append(r)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/fused_moe_router_rtx5060ti.json", "w"), indent=2)


if __name__ == "__main__":
    main()
