"""
Unit and integration tests for vLLM Sub-Bit Worker Adapter.
"""

import os
import tempfile
import torch
import pytest
from weft.vllm_subbit_worker import VLLMSubBitWorkerAdapter
from weft.format import WeftModelContainer, WeftSubBitLinear


def test_vllm_subbit_worker_lifecycle():
    container = WeftModelContainer(metadata={"architecture": "vllm-test"})
    l1 = WeftSubBitLinear(in_features=64, out_features=128, rank=32)
    l2 = WeftSubBitLinear(in_features=128, out_features=64, rank=32)
    container.add_subbit_linear("block.0.q_proj", l1)
    container.add_subbit_linear("block.0.out_proj", l2)

    with tempfile.NamedTemporaryFile(suffix=".weft", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        container.save(tmp_path)
        
        # Initialize worker adapter
        worker = VLLMSubBitWorkerAdapter(model_path=tmp_path, device="cpu")
        stats = worker.get_memory_stats()
        
        assert stats["num_layers_active"] == 2
        assert stats["compression_ratio"] > 1.0
        assert stats["mean_bpw"] < 4.0
        
        # Test projection execution
        x1 = torch.randn(2, 4, 64)
        out1 = worker.execute_projection("block.0.q_proj", x1)
        assert out1.shape == (2, 4, 128)
        
        # Test batch execution
        x2 = torch.randn(2, 4, 128)
        batch_requests = [("block.0.q_proj", x1), ("block.0.out_proj", x2)]
        batch_outs = worker.batch_execute_projections(batch_requests)
        
        assert len(batch_outs) == 2
        assert batch_outs[0].shape == (2, 4, 128)
        assert batch_outs[1].shape == (2, 4, 64)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
