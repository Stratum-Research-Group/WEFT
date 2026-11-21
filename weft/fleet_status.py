"""
Fleet Status Monitor: Real-time telemetry dashboard for multi-architecture
sub-bit quantization streams and consolidated foundation deployments.
"""

import os
import glob
import time
import argparse
from typing import Dict, Any, List


def inspect_fleet_status(models_root: str = "/run/media/akarta/AI/models") -> Dict[str, Any]:
    models = {
        "llama-3.3-70b-subbit": {
            "name": "Llama-3.3-70B-Instruct",
            "total_shards": 30,
            "raw_fp16_gb": 140.0,
        },
        "qwen-2.5-72b-subbit": {
            "name": "Qwen-2.5-72B-Instruct",
            "total_shards": 37,
            "raw_fp16_gb": 145.0,
        },
        "goedel-prover-32b-subbit": {
            "name": "Goedel-Prover-V2-32B",
            "total_shards": 8,
            "raw_fp16_gb": 64.0,
        },
        "lfm3-vl-2b-subbit": {
            "name": "LiquidAI/LFM3-VL-2B",
            "total_shards": 1,
            "raw_fp16_gb": 3.84,
        },
    }

    fleet_report = {}
    for dir_name, meta in models.items():
        model_path = os.path.join(models_root, dir_name)
        if not os.path.exists(model_path):
            continue

        weft_files = sorted(glob.glob(os.path.join(model_path, "*.weft")))
        gguf_files = sorted(glob.glob(os.path.join(model_path, "*.gguf")))

        total_weft_bytes = sum(os.path.getsize(p) for p in weft_files)
        total_weft_mb = total_weft_bytes / 1e6
        gguf_mb = sum(os.path.getsize(p) for p in gguf_files) / 1e6

        completed_shards = len(weft_files)
        total_shards = meta["total_shards"]
        pct_complete = (completed_shards / total_shards) * 100

        # Estimated parameters
        ratio = completed_shards / total_shards
        processed_gb = meta["raw_fp16_gb"] * ratio
        compression_ratio = (processed_gb * 1000) / max(total_weft_mb, 1e-3)

        fleet_report[dir_name] = {
            "display_name": meta["name"],
            "completed_shards": completed_shards,
            "total_shards": total_shards,
            "percent_complete": pct_complete,
            "weft_storage_mb": total_weft_mb,
            "consolidated_gguf_mb": gguf_mb,
            "compression_factor": compression_ratio,
        }

    return fleet_report


def print_fleet_dashboard(fleet_report: Dict[str, Any]):
    print("=" * 80)
    print("        WEFT SUB-BIT FOUNDATION FLEET: REAL-TIME TELEMETRY")
    print("=" * 80)
    header = f"{'Model':<26} | {'Shards':<10} | {'Progress':<9} | {'SubBit MB':<10} | {'Compression':<12}"
    print(header)
    print("-" * 80)
    for model_id, data in fleet_report.items():
        shards_str = f"{data['completed_shards']}/{data['total_shards']}"
        pct_str = f"{data['percent_complete']:.1f}%"
        mb_str = f"{data['weft_storage_mb']:.1f} MB"
        comp_str = f"{data['compression_factor']:.1f}x"
        print(f"{data['display_name']:<26} | {shards_str:<10} | {pct_str:<9} | {mb_str:<10} | {comp_str:<12}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Fleet status dashboard")
    parser.add_argument("--models-root", type=str, default="/run/media/akarta/AI/models")
    args = parser.parse_args()

    report = inspect_fleet_status(args.models_root)
    print_fleet_dashboard(report)


if __name__ == "__main__":
    main()
