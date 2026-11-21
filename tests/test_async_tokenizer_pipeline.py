"""Unit tests for AsyncTokenizerPipeline."""
import pytest
import asyncio
from weft.async_tokenizer_pipeline import AsyncTokenizerPipeline

@pytest.mark.anyio
async def test_async_tokenizer_pipeline():
    pipeline = AsyncTokenizerPipeline()
    sample_text = "def hello_world(): return 'WEFT-2'"

    toks = await pipeline.encode_async(sample_text)
    assert isinstance(toks, list)
    assert len(toks) > 0

    decoded = await pipeline.decode_async(toks)
    assert "hello_world" in decoded
    pipeline.shutdown()
