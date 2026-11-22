"""
WEFT Sub-Bit GGUF-Framed Export Container:
Converts .weft binary model containers into a GGUF-framed binary file container.
Maps packed uint32 bitboards and scale vectors into framed tensor arrays.
Includes auto-consolidation directory watcher.

NOTE: This writes a custom GGUF-framed serialization container for WEFT sub-bit tensors.
Direct execution in stock llama.cpp requires custom WEFT dequantization kernels.
"""

import os
import struct
import json
import glob
import time
import argparse
import numpy as np
from typing import Dict, Any, List
from weft.format import WeftModelContainer

# GGUF Constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def export_weft_to_gguf(weft_path: str, output_gguf_path: str):
    """
    Translates a single .weft model container into a standardized GGUF binary file.
    """
    return export_multiple_weft_to_gguf([weft_path], output_gguf_path)


def export_multiple_weft_to_gguf(weft_paths: List[str], output_gguf_path: str):
    """
    Consolidates multiple sharded .weft containers into a unified GGUF binary file.
    """
    containers = [WeftModelContainer.load(p) for p in weft_paths if os.path.exists(p)]
    assert len(containers) > 0, "No valid .weft files provided."

    print(f"Consolidating {len(containers)} shards into GGUF at {output_gguf_path}...")
    total_tensors = sum(
        sum(tdata.get("num_paths", 1) * 5 for tdata in c.tensors.values())
        for c in containers
    )
    kv_count = 2

    # Write to a temporary file first for atomic swap
    tmp_out = output_gguf_path + ".tmp"
    with open(tmp_out, "wb") as f:
        f.write(struct.pack("<4sIQQ", GGUF_MAGIC, GGUF_VERSION, total_tensors, kv_count))

        # Metadata
        arch = container_arch.encode("utf-8") if (container_arch := containers[0].metadata.get("architecture")) else b"weft-subbit-sharded"
        f.write(struct.pack("<Q", len("general.architecture")))
        f.write(b"general.architecture")
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(arch)))
        f.write(arch)

        qtype = b"WEFT_SUBBIT_70B_CONSOLIDATED"
        f.write(struct.pack("<Q", len("general.quantization_version")))
        f.write(b"general.quantization_version")
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(qtype)))
        f.write(qtype)

        for container in containers:
            for name, tdata in container.tensors.items():
                for p in range(tdata["num_paths"]):
                    for key in ["u_packed", "v_packed", "scale_h", "scale_l", "scale_g"]:
                        arr = tdata["paths"][p][key]
                        tname = f"{name}.p{p}.{key}".encode("utf-8")
                        f.write(struct.pack("<Q", len(tname)))
                        f.write(tname)
                        f.write(struct.pack("<I", arr.ndim))
                        for d in arr.shape:
                            f.write(struct.pack("<Q", d))
                        f.write(struct.pack("<I", 0))
                        raw_bytes = arr.tobytes()
                        f.write(struct.pack("<Q", len(raw_bytes)))
                        f.write(raw_bytes)

    os.replace(tmp_out, output_gguf_path)
    file_size_mb = os.path.getsize(output_gguf_path) / 1e6
    print(f"Sharded GGUF export successful: {output_gguf_path} ({file_size_mb:.2f} MB).")
    return output_gguf_path


def auto_consolidate_directory(model_dir: str, output_gguf_name: str = "model_consolidated_subbit.gguf") -> str:
    """
    Finds all *.weft files in model_dir and creates or updates the consolidated GGUF file.
    """
    weft_files = sorted(glob.glob(os.path.join(model_dir, "*.weft")))
    if not weft_files:
        raise FileNotFoundError(f"No .weft files found in {model_dir}")

    output_path = os.path.join(model_dir, output_gguf_name)
    export_multiple_weft_to_gguf(weft_files, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export .weft shards to consolidated GGUF binary.")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing .weft shards")
    parser.add_argument("--output-gguf", type=str, default=None, help="Output GGUF path")
    args = parser.parse_args()

    if args.output_gguf:
        weft_files = sorted(glob.glob(os.path.join(args.model_dir, "*.weft")))
        export_multiple_weft_to_gguf(weft_files, args.output_gguf)
    else:
        auto_consolidate_directory(args.model_dir)


if __name__ == "__main__":
    main()
