"""
Unit tests for the WEFT CLI commands: inspect, convert, and bench.
"""

import os
import tempfile
import pytest
from weft.cli import inspect_model, benchmark_model
from weft.format import WeftSubBitLinear, WeftModelContainer


def test_cli_inspect_and_bench():
    container = WeftModelContainer(metadata={"architecture": "test-model", "hidden_size": 128})
    layer = WeftSubBitLinear(in_features=64, out_features=128, rank=32)
    container.add_subbit_linear("test.linear", layer)

    with tempfile.NamedTemporaryFile(suffix=".weft", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        container.save(tmp_path)
        
        # Test inspect without raising
        inspect_model(tmp_path)
        
        # Test bench without raising
        benchmark_model(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
