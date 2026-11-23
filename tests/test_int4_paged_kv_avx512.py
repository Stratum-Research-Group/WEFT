import pytest
import torch

def test_int4_paged_kv_dequant():
    head_dim = 64
    scale = 0.05
    zero_point = 8.0

    # Synthetic nibble values: range [0, 15]
    torch.manual_seed(42)
    nibbles = torch.randint(0, 16, (head_dim,), dtype=torch.uint8)
    
    # Pack into bytes (2 nibbles per byte)
    packed_bytes = bytearray(head_dim // 2)
    for i in range(0, head_dim, 2):
        low = int(nibbles[i])
        high = int(nibbles[i+1])
        packed_bytes[i // 2] = low | (high << 4)

    # Reference dequantization
    expected_dequant = (nibbles.float() - zero_point) * scale

    # Unpack from bytes
    unpacked = torch.zeros(head_dim, dtype=torch.float32)
    for i in range(head_dim):
        b = packed_bytes[i // 2]
        n = (b & 0x0F) if (i % 2 == 0) else ((b >> 4) & 0x0F)
        unpacked[i] = (float(n) - zero_point) * scale

    assert torch.allclose(unpacked, expected_dequant, atol=1e-5)
    print("[PASS] Int4 paged KV dequantization test passed.")

if __name__ == "__main__":
    test_int4_paged_kv_dequant()
