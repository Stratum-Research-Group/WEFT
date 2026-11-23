"""
Low-Rank Latent Key-Value Cache Compression (MLA-KV):
Implements low-rank projection of Multi-Head Attention Key-Value tensors:
Compresses KV cache footprint by 8x (from 16-bit float to rank-compressed manifold),
expanding context window from 8k to 64k+ tokens on a 16 GB edge host without memory overflow.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class MLAKVConfig:
    d_model: int = 4096
    n_heads: int = 32
    d_head: int = 128
    latent_dim: int = 512  # Low-rank compressed latent projection
    max_context: int = 65536

class MLALowRankKVCache:
    def __init__(self, config: MLAKVConfig = MLAKVConfig()):
        self.cfg = config
        # Compression projection matrices: W_compress: d_model -> latent_dim
        # Decompression projection: W_decompress: latent_dim -> (n_heads * d_head)
        rng = np.random.default_rng(42)
        self.W_comp_k = rng.normal(0, 0.02, (self.cfg.d_model, self.cfg.latent_dim)).astype(np.float16)
        self.W_decomp_k = rng.normal(0, 0.02, (self.cfg.latent_dim, self.cfg.d_model)).astype(np.float16)
        
        self.W_comp_v = rng.normal(0, 0.02, (self.cfg.d_model, self.cfg.latent_dim)).astype(np.float16)
        self.W_decomp_v = rng.normal(0, 0.02, (self.cfg.latent_dim, self.cfg.d_model)).astype(np.float16)
        
        # Compressed storage
        self.k_latent_cache = []
        self.v_latent_cache = []
        
    def append_and_compress(self, k: np.ndarray, v: np.ndarray) -> Dict[str, Any]:
        """Compresses incoming token KV representations into low-rank latent manifold."""
        t0 = time.perf_counter()
        
        # k, v shape: (seq_len, d_model)
        # Latent projection: (seq_len, latent_dim)
        k_latent = (k.astype(np.float32) @ self.W_comp_k.astype(np.float32)).astype(np.float16)
        v_latent = (v.astype(np.float32) @ self.W_comp_v.astype(np.float32)).astype(np.float16)
        
        self.k_latent_cache.append(k_latent)
        self.v_latent_cache.append(v_latent)
        
        raw_bytes = (k.nbytes + v.nbytes)
        compressed_bytes = (k_latent.nbytes + v_latent.nbytes)
        comp_ratio = raw_bytes / compressed_bytes
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        return {
            "tokens": k.shape[0],
            "raw_kb": raw_bytes / 1024.0,
            "compressed_kb": compressed_bytes / 1024.0,
            "compression_ratio": round(comp_ratio, 2),
            "compression_latency_us": round(elapsed_us, 2)
        }
        
    def decompress_at_step(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """On-the-fly registers decompression for cross-attention dot product."""
        k_lat = self.k_latent_cache[idx].astype(np.float32)
        v_lat = self.v_latent_cache[idx].astype(np.float32)
        
        k_rec = (k_lat @ self.W_decomp_k.astype(np.float32)).astype(np.float16)
        v_rec = (v_lat @ self.W_decomp_v.astype(np.float32)).astype(np.float16)
        return k_rec, v_rec
        
    def total_memory_footprint(self) -> Dict[str, float]:
        """Calculates memory footprint across 64k tokens."""
        raw_64k_mb = (65536 * self.cfg.d_model * 2 * 2) / (1024 * 1024)  # 2 for K and V, 2 bytes/float16
        comp_64k_mb = (65536 * self.cfg.latent_dim * 2 * 2) / (1024 * 1024)
        return {
            "uncompressed_64k_mb": round(raw_64k_mb, 2),
            "compressed_64k_mb": round(comp_64k_mb, 2),
            "saved_mb": round(raw_64k_mb - comp_64k_mb, 2),
            "memory_reduction_factor": round(raw_64k_mb / comp_64k_mb, 2)
        }
