"""
WEFT Command Line Interface (CLI): Tooling for inspecting, converting,
benchmarking, and validating .weft sub-bit binary model containers.
"""

import os
import sys
import argparse
import time
import json
import torch
from weft.format import WeftModelContainer
from weft.converter import convert_model_to_weft
from safetensors import safe_open


def inspect_model(weft_path: str):
    if not os.path.exists(weft_path):
        print(f"Error: File not found: {weft_path}")
        sys.exit(1)

    container = WeftModelContainer.load(weft_path)
    file_size_mb = os.path.getsize(weft_path) / 1e6

    print("=" * 60)
    print(f"  WEFT MODEL INSPECTION: {os.path.basename(weft_path)}")
    print("=" * 60)
    print(f"File Size:            {file_size_mb:.2f} MB")
    print(f"Architecture:         {container.metadata.get('architecture', container.metadata.get('model_type', 'unknown'))}")
    print(f"Hidden Size:          {container.metadata.get('hidden_size', 'N/A')}")
    print(f"Vocab Size:           {container.metadata.get('vocab_size', 'N/A')}")
    print(f"Total Tensors:        {len(container.tensors)}")
    print("-" * 60)
    print(f"{'Tensor Name':<40} {'Shape (d_out x d_in)':<20} {'Rank':<6} {'BPW':<6}")
    print("-" * 60)

    bpw_list = []
    for name, tdata in container.tensors.items():
        layer = container.get_subbit_linear(name)
        bpw = layer.bits_per_weight()
        bpw_list.append(bpw)
        shape_str = f"{tdata['out_features']} x {tdata['in_features']}"
        print(f"{name:<40} {shape_str:<20} {tdata['rank']:<6} {bpw:.3f}")

    if bpw_list:
        mean_bpw = sum(bpw_list) / len(bpw_list)
        print("-" * 60)
        print(f"Effective Mean Bitrate: {mean_bpw:.4f} bits/weight")
    print("=" * 60)


def convert_checkpoint(input_path: str, output_path: str, base_rank: int, adaptive: bool):
    if not os.path.exists(input_path):
        print(f"Error: Input checkpoint not found: {input_path}")
        sys.exit(1)

    print(f"Converting {input_path} to {output_path} (base_rank={base_rank}, adaptive={adaptive})...")
    state_dict = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight"):
                t = f.get_tensor(k)
                if t.ndim == 2:
                    state_dict[k] = t

    config = {"source_checkpoint": os.path.basename(input_path)}
    t0 = time.time()
    convert_model_to_weft(
        state_dict=state_dict,
        config=config,
        output_filepath=output_path,
        base_rank=base_rank,
        adaptive_rank=adaptive,
    )
    duration = time.time() - t0
    print(f"Conversion complete in {duration:.2f}s. Saved to {output_path}.")


def benchmark_model(weft_path: str):
    if not os.path.exists(weft_path):
        print(f"Error: Model not found: {weft_path}")
        sys.exit(1)

    container = WeftModelContainer.load(weft_path)
    print(f"Benchmarking inference latency on {len(container.tensors)} layers from {weft_path}...")

    times_cpu = []
    for name in list(container.tensors.keys())[:4]:
        layer = container.get_subbit_linear(name)
        x = torch.randn(1, 16, layer.in_features)
        
        # Warmup
        _ = layer(x)
        
        t0 = time.perf_counter()
        for _ in range(10):
            _ = layer(x)
        times_cpu.append((time.perf_counter() - t0) / 10.0 * 1000.0)

    print(f"CPU Forward Latency (mean across sample layers): {sum(times_cpu) / len(times_cpu):.3f} ms / projection")


def main():
    parser = argparse.ArgumentParser(prog="weft", description="WEFT Sub-Bit Model Container CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a .weft model container")
    inspect_parser.add_argument("model_path", type=str, help="Path to .weft file")

    # Convert
    convert_parser = subparsers.add_parser("convert", help="Convert dense safetensors to .weft")
    convert_parser.add_argument("--input", "-i", type=str, required=True, help="Input safetensors file")
    convert_parser.add_argument("--output", "-o", type=str, required=True, help="Output .weft file")
    convert_parser.add_argument("--rank", "-r", type=int, default=64, help="Base rank")
    convert_parser.add_argument("--adaptive", action="store_true", default=True, help="Adaptive rank allocation")

    # Bench
    bench_parser = subparsers.add_parser("bench", help="Benchmark .weft model inference")
    bench_parser.add_argument("model_path", type=str, help="Path to .weft file")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_model(args.model_path)
    elif args.command == "convert":
        convert_checkpoint(args.input, args.output, args.rank, args.adaptive)
    elif args.command == "bench":
        benchmark_model(args.model_path)


if __name__ == "__main__":
    main()
