"""
WEFT to GGUF Export Bridge:
Converts .weft binary model containers into GGUF files compatible with llama.cpp.
Maps packed uint32 bitboards and scale vectors into custom GGUF tensor arrays.
"""

import os
import struct
import json
import numpy as np
from typing import Dict, Any
from weft.format import WeftModelContainer

# GGUF Constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def export_weft_to_gguf(weft_path: str, output_gguf_path: str):
    """
    Translates a single .weft model container into a standardized GGUF binary file.
    """
    return export_multiple_weft_to_gguf([weft_path], output_gguf_path)


def export_multiple_weft_to_gguf(weft_paths: list[str], output_gguf_path: str):
    """
    Consolidates multiple sharded .weft containers into a unified GGUF binary file.
    """
    containers = [WeftModelContainer.load(p) for p in weft_paths if os.path.exists(p)]
    assert len(containers) > 0, "No valid .weft files provided."

    print(f"Consolidating {len(containers)} shards into GGUF at {output_gguf_path}...")
    total_tensors = sum(len(c.tensors) for c in containers) * 5
    kv_count = 2

    with open(output_gguf_path, "wb") as f:
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

    print(f"Sharded GGUF export successful: {output_gguf_path} ({os.path.getsize(output_gguf_path) / 1e6:.2f} MB).")
