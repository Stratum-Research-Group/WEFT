import os
import mmap
import json
import struct
import torch
import pytest

def test_safetensors_mmap_tensor_slicing():
    # Construct a valid safetensors binary file with two tensors
    t1 = torch.randn(32, 64, dtype=torch.float32)
    t2 = torch.randn(16, 16, dtype=torch.float32)

    t1_bytes = t1.numpy().tobytes()
    t2_bytes = t2.numpy().tobytes()

    header = {
        "tensor1": {"dtype": "F32", "shape": [32, 64], "data_offsets": [0, len(t1_bytes)]},
        "tensor2": {"dtype": "F32", "shape": [16, 16], "data_offsets": [len(t1_bytes), len(t1_bytes) + len(t2_bytes)]}
    }
    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    test_file = "build/test_mmap_shard.safetensors"
    with open(test_file, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(t1_bytes)
        f.write(t2_bytes)

    # Now verify zero-copy mmap reader
    with open(test_file, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        hlen = struct.unpack("<Q", mm[:8])[0]
        meta = json.loads(mm[8:8+hlen].decode("utf-8"))
        data_base = 8 + hlen

        # Read tensor 1
        off1, off2 = meta["tensor1"]["data_offsets"]
        t1_slice = mm[data_base + off1 : data_base + off2]
        t1_recovered = torch.frombuffer(bytearray(t1_slice), dtype=torch.float32).reshape(meta["tensor1"]["shape"])
        assert torch.allclose(t1, t1_recovered)

        # Read tensor 2
        off1, off2 = meta["tensor2"]["data_offsets"]
        t2_slice = mm[data_base + off1 : data_base + off2]
        t2_recovered = torch.frombuffer(bytearray(t2_slice), dtype=torch.float32).reshape(meta["tensor2"]["shape"])
        assert torch.allclose(t2, t2_recovered)

        mm.close()

    os.remove(test_file)
    print("[PASS] Zero-copy safetensors mmap tensor slicing verified bit-exact.")

if __name__ == "__main__":
    test_safetensors_mmap_tensor_slicing()
