"""Unit tests for AsyncTokenizerPipeline."""
import os
from pathlib import Path
import pytest
import asyncio
from weft.async_tokenizer_pipeline import AsyncTokenizerPipeline

@pytest.mark.anyio
async def test_async_tokenizer_pipeline():
    models_root = Path(os.environ.get("WEFT_MODELS_ROOT", "/run/media/akarta/AI/models"))
    tokenizer_dir = models_root / "gemma-4-E2B-it"
    if not tokenizer_dir.exists():
        pytest.skip(f"Tokenizer directory {tokenizer_dir} does not exist.")

    pipeline = AsyncTokenizerPipeline(str(tokenizer_dir))
    sample_text = "def hello_world(): return 'WEFT-2'"

    toks = await pipeline.encode_async(sample_text)
    assert isinstance(toks, list)
    assert len(toks) > 0

    decoded = await pipeline.decode_async(toks)
    assert "hello_world" in decoded
    pipeline.shutdown()
