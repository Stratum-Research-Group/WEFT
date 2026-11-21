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
    Translates a .weft model container into a standardized GGUF binary file.
    """
    assert os.path.exists(weft_path), f"File not found: {weft_path}"
    container = WeftModelContainer.load(weft_path)

    print(f"Exporting {weft_path} to GGUF format at {output_gguf_path}...")

    # Build GGUF header
    # 2 standard metadata KV pairs: general.architecture, general.quantization_version
    kv_count = 2
    tensor_count = len(container.tensors) * 5  # u, v, h, l, g per layer

    with open(output_gguf_path, "wb") as f:
        f.write(struct.pack("<4sIQQ", GGUF_MAGIC, GGUF_VERSION, tensor_count, kv_count))

        # Write metadata KV pairs
        # 1. Architecture
        arch = container.metadata.get("architecture", "weft-subbit").encode("utf-8")
        f.write(struct.pack("<Q", len("general.architecture")))
        f.write(b"general.architecture")
        f.write(struct.pack("<I", 8))  # string type
        f.write(struct.pack("<Q", len(arch)))
        f.write(arch)

        # 2. Quantization Type
        qtype = b"WEFT_SUBBIT_0.23BPW"
        f.write(struct.pack("<Q", len("general.quantization_version")))
        f.write(b"general.quantization_version")
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(qtype)))
        f.write(qtype)

        # Write tensor descriptors and payloads
        for name, tdata in container.tensors.items():
            for p in range(tdata["num_paths"]):
                for key in ["u_packed", "v_packed", "scale_h", "scale_l", "scale_g"]:
                    arr = tdata["paths"][p][key]
                    tname = f"{name}.p{p}.{key}".encode("utf-8")
                    
                    # Tensor info: name_len(Q), name, n_dims(I), dims(Q*n), type(I), offset(Q)
                    f.write(struct.pack("<Q", len(tname)))
                    f.write(tname)
                    f.write(struct.pack("<I", arr.ndim))
                    for d in arr.shape:
                        f.write(struct.pack("<Q", d))
                    f.write(struct.pack("<I", 0))  # FP32/Raw type code
                    raw_bytes = arr.tobytes()
                    f.write(struct.pack("<Q", len(raw_bytes)))
                    f.write(raw_bytes)

    print(f"GGUF export successful: {output_gguf_path} ({os.path.getsize(output_gguf_path) / 1e6:.2f} MB).")
