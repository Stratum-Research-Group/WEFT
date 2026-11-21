"""
Unit tests for WEFT to GGUF Export Bridge.
"""

import os
import tempfile
import struct
import pytest
from weft.export_subbit_gguf import export_weft_to_gguf, GGUF_MAGIC
from weft.format import WeftSubBitLinear, WeftModelContainer


def test_weft_to_gguf_export():
    container = WeftModelContainer(metadata={"architecture": "test-gguf"})
    layer = WeftSubBitLinear(in_features=64, out_features=128, rank=32)
    container.add_subbit_linear("layers.0.mlp.up_proj", layer)

    with tempfile.NamedTemporaryFile(suffix=".weft", delete=False) as tmp_weft:
        weft_path = tmp_weft.name
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp_gguf:
        gguf_path = tmp_gguf.name

    try:
        container.save(weft_path)
        export_weft_to_gguf(weft_path, gguf_path)

        assert os.path.exists(gguf_path)
        assert os.path.getsize(gguf_path) > 0

        # Validate GGUF magic bytes
        with open(gguf_path, "rb") as f:
            magic, version = struct.unpack("<4sI", f.read(8))
            assert magic == GGUF_MAGIC
            assert version == 3
    finally:
        for p in [weft_path, gguf_path]:
            if os.path.exists(p):
                os.remove(p)
