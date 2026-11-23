import pytest
import os
import shutil
from weft.multilingual_tokenizer import WeftTokenizer, SPECIAL_TOKENS


def test_weft_tokenizer_special_tokens():
    tokenizer = WeftTokenizer()
    text = "<|im_start|>system\n<think>testing</think>\n<tool_call>{}</tool_call><|im_end|>"
    tokens = tokenizer.encode(text)

    # Decode and check roundtrip
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_weft_tokenizer_multilingual_fallback():
    tokenizer = WeftTokenizer()
    # Test complex multilingual text (Greek, Arabic, CJK, Emojis, mathematical symbols)
    text = "Hello! Bonjour, こんにちは, مرحباً, ∑_{i=1}^N x_i 🚀"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_weft_tokenizer_save_load(tmp_path):
    tokenizer = WeftTokenizer()
    save_dir = str(tmp_path / "tokenizer_save")
    tokenizer.save_pretrained(save_dir)

    loaded = WeftTokenizer.from_pretrained(save_dir)
    sample = "<think>2+2=4</think>"
    assert loaded.decode(loaded.encode(sample)) == sample
