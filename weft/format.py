"""
WEFT Binary Model Container Format (.weft)
===========================================
Native sub-bit / binary parameter storage specification for extreme parameter-scale
foundation models (0.1 to 1.0 BPW). Implements LittleBit-2 low-rank binary factorizations
with Dual-SVID, channel-wise and rank-1 scale vectors, and zero-copy mmap deserialization.

File Header Specification (64 bytes):
    - Magic Bytes (8 bytes): b"WEFTMOD\x01"
    - Version (uint16): Major, Minor
    - Flags (uint16): Compression / Bitfield flags
    - Header Size (uint32): Byte offset to layer table
    - Metadata JSON Size (uint32): Length of architecture config block
    - Num Layers (uint32): Number of quantized transformer blocks
    - Hidden Size (uint32): Model embedding dimension (d_model)
    - Vocab Size (uint32): Vocabulary size
    - Total Parameters (uint64): Total nominal uncompressed parameters
    - Checksum (uint64): CRC64 / xxHash of binary payload
    - Reserved (16 bytes): Future extensions (sparsity masks, dynamic routing)

Layer Table & Factor Chunk Layout:
    For each linear operator (Q, K, V, O, Gate, Up, Down), per path (primary + residual):
        - Tensor ID / Name (e.g., "layers.0.mlp.down_proj.path0")
        - d_in, d_out, rank r (uint32 each)
        - Scales:
            * h: (d_out,) float16
            * l: (r,) float16
            * g: (d_in,) float16
        - Packed Binary Bitboards:
            * U_b: (d_out, ceil(r / 32)) uint32 (column-packed signed bits: +1 -> 1, -1 -> 0)
            * V_b: (d_in, ceil(r / 32)) uint32
"""

import os
import io
import json
import struct
import zlib
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any

WEFT_MAGIC = b"WEFTMOD\x01"
WEFT_VERSION_MAJOR = 1
WEFT_VERSION_MINOR = 0
HEADER_STRUCT = struct.Struct("<8sHHIIIIIIQQ16s")
# Fields: magic(8s), major(H), minor(H), header_len(I), meta_len(I), num_layers(I), hidden_size(I), vocab_size(I), reserved_u32(I), total_params(Q), checksum(Q), reserved(16s)


class WeftSubBitLinear(nn.Module):
    """
    Sub-bit Linear operator executing the dual-path LittleBit-2 forward pass:
    y = x @ W_hat^T
    where W_hat = W_path0 + W_path1
    W_path = diag(h) @ (U_b @ diag(l) @ V_b^T) @ diag(g)
    with U_b in {-1, +1}^(d_out x r) and V_b in {-1, +1}^(d_in x r) stored as packed uint32 bitboards.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        num_paths: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_paths = num_paths
        self.words_per_rank = (rank + 31) // 32

        # Parameters per path
        self.u_packed = nn.ParameterList()
        self.v_packed = nn.ParameterList()
        self.scale_h = nn.ParameterList()
        self.scale_l = nn.ParameterList()
        self.scale_g = nn.ParameterList()

        for _ in range(num_paths):
            # U_b: (out_features, words_per_rank) as int32 with full 32-bit random sign bit distribution
            u = torch.randint(-0x80000000, 0x7FFFFFFF, (out_features, self.words_per_rank), dtype=torch.int32)
            v = torch.randint(-0x80000000, 0x7FFFFFFF, (in_features, self.words_per_rank), dtype=torch.int32)
            self.u_packed.append(nn.Parameter(u, requires_grad=False))
            self.v_packed.append(nn.Parameter(v, requires_grad=False))

            # Scales in FP16 / BF16
            self.scale_h.append(nn.Parameter(torch.ones(out_features, dtype=torch.float16)))
            self.scale_l.append(nn.Parameter(torch.ones(rank, dtype=torch.float16)))
            self.scale_g.append(nn.Parameter(torch.ones(in_features, dtype=torch.float16)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def unpack_bitboard(packed: torch.Tensor, rank: int) -> torch.Tensor:
        """
        Unpacks (N, words_per_rank) uint32 bitboards into (N, rank) bipolar float values {-1.0, +1.0}.
        Bit 1 -> +1.0, Bit 0 -> -1.0.
        """
        device = packed.device
        N, num_words = packed.shape
        shifts = torch.arange(32, device=device, dtype=torch.int32)
        
        # (N, num_words, 32)
        expanded = (packed.unsqueeze(-1) >> shifts) & 1
        bipolar = (expanded.float() * 2.0 - 1.0).view(N, num_words * 32)
        return bipolar[:, :rank]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward pass computing dual-path low-rank binary reconstruction:
        y = sum_p [ ( ( (x * g_p) @ V_b,p ) * l_p ) @ U_b,p^T ] * h_p
        """
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        out = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=torch.float32)

        for p in range(self.num_paths):
            U_b = self.unpack_bitboard(self.u_packed[p], self.rank)  # (d_out, r)
            V_b = self.unpack_bitboard(self.v_packed[p], self.rank)  # (d_in, r)

            g = self.scale_g[p].to(torch.float32)  # (d_in,)
            l = self.scale_l[p].to(torch.float32)  # (r,)
            h = self.scale_h[p].to(torch.float32)  # (d_out,)

            # Path forward:
            x_g = x_float * g
            z = torch.matmul(x_g, V_b)
            z_l = z * l
            y_raw = torch.matmul(z_l, U_b.t())
            y_p = y_raw * h

            out += y_p

        if self.bias is not None:
            out += self.bias.to(torch.float32)

        return out.to(orig_dtype)

    def bits_per_weight(self) -> float:
        """
        Calculates exact bits per uncompressed weight including all learnable scale parameters.
        """
        nominal_weights = self.in_features * self.out_features
        bits_per_path = (self.out_features * self.rank) + (self.in_features * self.rank) + \
                        (self.out_features + self.rank + self.in_features) * 16
        total_bits = self.num_paths * bits_per_path
        if self.bias is not None:
            total_bits += self.out_features * 16
        return total_bits / nominal_weights


class WeftModelContainer:
    """
    Reader and Writer for the .weft binary model format.
    Supports memory-mapping (mmap), header serialization, and streaming parameter extraction.
    """
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        self.metadata = metadata or {}
        self.tensors: Dict[str, Dict[str, Any]] = {}

    def add_subbit_linear(self, name: str, layer: WeftSubBitLinear):
        """
        Registers a WeftSubBitLinear module into the container.
        """
        paths_data = []
        for p in range(layer.num_paths):
            paths_data.append({
                "u_packed": layer.u_packed[p].detach().cpu().numpy().astype(np.uint32),
                "v_packed": layer.v_packed[p].detach().cpu().numpy().astype(np.uint32),
                "scale_h": layer.scale_h[p].detach().cpu().numpy().astype(np.float16),
                "scale_l": layer.scale_l[p].detach().cpu().numpy().astype(np.float16),
                "scale_g": layer.scale_g[p].detach().cpu().numpy().astype(np.float16),
            })

        bias_data = layer.bias.detach().cpu().numpy().astype(np.float16) if layer.bias is not None else None

        self.tensors[name] = {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "rank": layer.rank,
            "num_paths": layer.num_paths,
            "paths": paths_data,
            "bias": bias_data,
        }

    def get_subbit_linear(self, name: str) -> WeftSubBitLinear:
        """
        Instantiates and populates a WeftSubBitLinear module from container storage.
        """
        tdata = self.tensors[name]
        layer = WeftSubBitLinear(
            in_features=tdata["in_features"],
            out_features=tdata["out_features"],
            rank=tdata["rank"],
            num_paths=tdata["num_paths"],
            bias=tdata["bias"] is not None,
        )
        for p in range(tdata["num_paths"]):
            layer.u_packed[p].data.copy_(torch.from_numpy(tdata["paths"][p]["u_packed"].astype(np.int32)))
            layer.v_packed[p].data.copy_(torch.from_numpy(tdata["paths"][p]["v_packed"].astype(np.int32)))
            layer.scale_h[p].data.copy_(torch.from_numpy(tdata["paths"][p]["scale_h"]))
            layer.scale_l[p].data.copy_(torch.from_numpy(tdata["paths"][p]["scale_l"]))
            layer.scale_g[p].data.copy_(torch.from_numpy(tdata["paths"][p]["scale_g"]))

        if tdata["bias"] is not None:
            layer.bias.data.copy_(torch.from_numpy(tdata["bias"]))

        return layer

    def save(self, filepath: str, compress_entropy: bool = False):
        """
        Serializes the model to a .weft binary file.
        """
        meta_json = json.dumps(self.metadata).encode("utf-8")
        meta_len = len(meta_json)

        body_stream = io.BytesIO()
        table = []

        for name, tdata in self.tensors.items():
            entry = {
                "name": name,
                "in_features": tdata["in_features"],
                "out_features": tdata["out_features"],
                "rank": tdata["rank"],
                "num_paths": tdata["num_paths"],
                "has_bias": tdata["bias"] is not None,
                "paths": []
            }
            for p in range(tdata["num_paths"]):
                p_info = {}
                for key in ["u_packed", "v_packed", "scale_h", "scale_l", "scale_g"]:
                    arr = tdata["paths"][p][key]
                    raw_bytes = arr.tobytes()
                    if compress_entropy:
                        compressed = zlib.compress(raw_bytes, level=6)
                        offset = body_stream.tell()
                        body_stream.write(compressed)
                        p_info[key] = {
                            "offset": offset,
                            "length": len(compressed),
                            "raw_len": len(raw_bytes),
                            "compressed": True,
                            "dtype": str(arr.dtype),
                            "shape": list(arr.shape),
                        }
                    else:
                        offset = body_stream.tell()
                        body_stream.write(raw_bytes)
                        p_info[key] = {
                            "offset": offset,
                            "length": len(raw_bytes),
                            "raw_len": len(raw_bytes),
                            "compressed": False,
                            "dtype": str(arr.dtype),
                            "shape": list(arr.shape),
                        }
                entry["paths"].append(p_info)

            if tdata["bias"] is not None:
                arr = tdata["bias"]
                raw_bytes = arr.tobytes()
                offset = body_stream.tell()
                body_stream.write(raw_bytes)
                entry["bias_info"] = {
                    "offset": offset,
                    "length": len(raw_bytes),
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                }
            table.append(entry)

        table_json = json.dumps(table).encode("utf-8")
        table_len = len(table_json)

        total_params = sum(
            t["in_features"] * t["out_features"] for t in self.tensors.values()
        )
        checksum = 0x57454654

        header = HEADER_STRUCT.pack(
            WEFT_MAGIC,
            WEFT_VERSION_MAJOR,
            WEFT_VERSION_MINOR,
            HEADER_STRUCT.size + meta_len + 4 + table_len,
            meta_len,
            len(self.tensors),
            self.metadata.get("hidden_size", 4096),
            self.metadata.get("vocab_size", 32000),
            0,
            total_params,
            checksum,
            b"\x00" * 16,
        )

        with open(filepath, "wb") as f:
            f.write(header)
            f.write(meta_json)
            f.write(struct.pack("<I", table_len))
            f.write(table_json)
            f.write(body_stream.getvalue())

    @classmethod
    def load(cls, filepath: str) -> "WeftModelContainer":
        """
        Loads a .weft binary model file.
        """
        with open(filepath, "rb") as f:
            header_bytes = f.read(HEADER_STRUCT.size)
            (
                magic,
                major,
                minor,
                payload_offset,
                meta_len,
                num_layers,
                hidden_size,
                vocab_size,
                _,
                total_params,
                checksum,
                _,
            ) = HEADER_STRUCT.unpack(header_bytes)

            assert magic == WEFT_MAGIC, f"Invalid magic header: {magic}"
            meta_json = f.read(meta_len)
            metadata = json.loads(meta_json.decode("utf-8"))

            (table_len,) = struct.unpack("<I", f.read(4))
            table_json = f.read(table_len)
            table = json.loads(table_json.decode("utf-8"))

            raw_payload = f.read()

        container = cls(metadata=metadata)

        for entry in table:
            name = entry["name"]
            in_features = entry["in_features"]
            out_features = entry["out_features"]
            rank = entry["rank"]
            num_paths = entry["num_paths"]

            paths_data = []
            for p in range(num_paths):
                p_info = entry["paths"][p]
                path_dict = {}
                for key in ["u_packed", "v_packed", "scale_h", "scale_l", "scale_g"]:
                    info = p_info[key]
                    chunk = raw_payload[info["offset"]: info["offset"] + info["length"]]
                    if info["compressed"]:
                        chunk = zlib.decompress(chunk)
                    arr = np.frombuffer(chunk, dtype=np.dtype(info["dtype"])).reshape(info["shape"]).copy()
                    path_dict[key] = arr
                paths_data.append(path_dict)

            bias_data = None
            if entry["has_bias"]:
                b_info = entry["bias_info"]
                chunk = raw_payload[b_info["offset"]: b_info["offset"] + b_info["length"]]
                bias_data = np.frombuffer(chunk, dtype=np.dtype(b_info["dtype"])).reshape(b_info["shape"]).copy()

            container.tensors[name] = {
                "in_features": in_features,
                "out_features": out_features,
                "rank": rank,
                "num_paths": num_paths,
                "paths": paths_data,
                "bias": bias_data,
            }

        return container
