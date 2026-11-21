"""
Unit tests for GGUF Sub-Bit Export and Third-Party Binary Loader Compatibility.
"""

import os
import struct
import numpy as np
import torch
import pytest

from weft.kernels.subbit_cpu import cpu_unpack_bitboards
from weft.format import WeftModelContainer, WeftSubBitLinear
from weft.export_subbit_gguf import export_weft_to_gguf, GGUF_MAGIC, GGUF_VERSION


def test_gguf_export_and_decode(tmp_path):
    # Create mock weft container
    container = WeftModelContainer(metadata={"architecture": "test-subbit", "quantization_version": "v1"})
    
    m, n, rank = 128, 128, 32
    layer = WeftSubBitLinear(in_features=n, out_features=m, rank=rank, num_paths=1, bias=False)
    
    layer.u_packed[0].data.copy_(torch.ones((m, rank // 32), dtype=torch.int32) * 0x55555555)
    layer.v_packed[0].data.copy_(torch.ones((n, rank // 32), dtype=torch.int32) * 0x33333333)
    layer.scale_h[0].data.copy_(torch.ones(m, dtype=torch.float16) * 0.1)
    layer.scale_l[0].data.copy_(torch.ones(rank, dtype=torch.float16) * 1.5)
    layer.scale_g[0].data.copy_(torch.ones(n, dtype=torch.float16) * 0.2)

    container.add_subbit_linear("layer.0.attn.proj", layer)
    
    weft_file = str(tmp_path / "model.weft")
    gguf_file = str(tmp_path / "model.gguf")
    container.save(weft_file)
    
    # Export to GGUF
    export_weft_to_gguf(weft_file, gguf_file)
    assert os.path.exists(gguf_file)
    assert os.path.getsize(gguf_file) > 0

    # Read binary header
    with open(gguf_file, "rb") as f:
        magic, version, tensor_count, kv_count = struct.unpack("<4sIQQ", f.read(24))
        assert magic == GGUF_MAGIC
        assert version == GGUF_VERSION
        assert tensor_count == 5
        assert kv_count == 2

        # Read KVs
        kvs = {}
        for _ in range(kv_count):
            klen = struct.unpack("<Q", f.read(8))[0]
            k = f.read(klen).decode("utf-8")
            vtype = struct.unpack("<I", f.read(4))[0]
            vlen = struct.unpack("<Q", f.read(8))[0]
            v = f.read(vlen).decode("utf-8")
            kvs[k] = v

        assert kvs["general.architecture"] == "test-subbit"

        # Read tensors
        tensors = {}
        for _ in range(tensor_count):
            tlen = struct.unpack("<Q", f.read(8))[0]
            tname = f.read(tlen).decode("utf-8")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
            ttype = struct.unpack("<I", f.read(4))[0]
            payload_len = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(payload_len)
            key = tname.split(".")[-1]
            if "packed" in key:
                tensors[key] = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
            else:
                tensors[key] = np.frombuffer(raw, dtype=np.float16).reshape(shape)

    # Dequantize and verify reconstruction
    u_tensor = torch.from_numpy(tensors["u_packed"].astype(np.int32))
    v_tensor = torch.from_numpy(tensors["v_packed"].astype(np.int32))
    u_unpacked = cpu_unpack_bitboards(u_tensor, rank).numpy()
    v_unpacked = cpu_unpack_bitboards(v_tensor, rank).numpy()

    assert u_unpacked.shape == (m, rank)
    assert v_unpacked.shape == (n, rank)
    W_rec = (tensors["scale_h"][:, None] * u_unpacked) @ (np.diag(tensors["scale_l"]) @ (v_unpacked.T * tensors["scale_g"][None, :]))
    assert W_rec.shape == (m, n)
    assert np.all(np.isfinite(W_rec))
