"""
WEFT Multi-Tenant Memory Quota & GPU OOM Circuit-Breaker Guard.
Monitors VRAM headroom on CUDA devices, detects rapid memory inflation,
and triggers preemptive KV-cache evictions or request shedding before CUDA OOM.
"""
import torch
import logging
import os
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("weft.gpu_circuit_breaker")

class GPUCircuitBreaker:
    def __init__(
        self,
        device_index: int = 0,
        min_free_vram_mb: float = 500.0,
        max_allocated_ratio: float = 0.92,
    ):
        self.device_index = device_index
        self.min_free_vram_mb = min_free_vram_mb
        self.max_allocated_ratio = max_allocated_ratio
        self.tripped = False
        self.eviction_counter = 0

    def query_vram_state(self) -> Dict[str, float]:
        """Queries current physical VRAM allocated and free on the target device."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 8192.0, "total_mb": 8192.0}

        allocated = torch.cuda.memory_allocated(self.device_index) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self.device_index) / (1024 * 1024)
        total = torch.cuda.get_device_properties(self.device_index).total_memory / (1024 * 1024)
        free = total - reserved

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free,
            "total_mb": total,
            "usage_ratio": reserved / total if total > 0 else 0.0
        }

    def check_or_trip(self) -> bool:
        """
        Evaluates memory pressure. Returns True if execution may proceed,
        or False if circuit breaker is tripped to prevent OOM crash.
        """
        state = self.query_vram_state()
        if state["free_mb"] < self.min_free_vram_mb or state["usage_ratio"] >= self.max_allocated_ratio:
            self.tripped = True
            logger.warning(
                f"[CIRCUIT_BREAKER_TRIPPED] Free VRAM ({state['free_mb']:.1f} MB) below floor "
                f"({self.min_free_vram_mb} MB) or usage ({state['usage_ratio']*100:.1f}%) >= {self.max_allocated_ratio*100}%"
            )
            return False
        self.tripped = False
        return True

    def trigger_emergency_eviction(self) -> int:
        """Forces CUDA memory cache clearance and returns reclaimed estimate."""
        if not torch.cuda.is_available():
            return 0
        before = torch.cuda.memory_reserved(self.device_index)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved(self.device_index)
        reclaimed_mb = int((before - after) / (1024 * 1024))
        self.eviction_counter += 1
        logger.info(f"[EMERGENCY_EVICTION] Reclaimed {reclaimed_mb} MB cache memory.")
        return reclaimed_mb

# Singleton instance for live server engine
breaker = GPUCircuitBreaker()
