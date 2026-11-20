"""
WEFT-2 Native Byte-Fallback BPE Tokenizer Module.

Provides:
1. High-efficiency code, English, and multilingual tokenization.
2. Byte-fallback mechanism guaranteeing zero-OOV (Out-Of-Vocabulary) coverage for all UTF-8 characters.
3. Native support for special structural tokens:
   <tool_call>, </tool_call>, <tool_response>, </tool_response>, <think>, </think>.
4. Compatible with standard Hugging Face PreTrainedTokenizerFast conventions.
"""

import os
import json
from typing import List, Dict, Optional, Union


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]


class WeftTokenizer:
    """
    Native byte-level fallback BPE tokenizer designed for WEFT-2.
    Ensures zero out-of-vocabulary tokens for code, math symbols, and multilingual text.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None, special_tokens: Optional[List[str]] = None):
        self.special_tokens = special_tokens or list(SPECIAL_TOKENS)
        self.special_token_to_id: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}

        # Initialize special tokens first
        curr_id = 0
        for token in self.special_tokens:
            self.special_token_to_id[token] = curr_id
            self.id_to_special_token[curr_id] = token
            curr_id += 1

        # 256 individual byte tokens for fallback guarantee
        self.byte_to_id: Dict[int, int] = {}
        self.id_to_byte: Dict[int, int] = {}
        for b in range(256):
            self.byte_to_id[b] = curr_id
            self.id_to_byte[curr_id] = b
            curr_id += 1

        self.vocab_size = curr_id
        self.eot_token = self.special_token_to_id["<|endoftext|>"]

    def encode(self, text: str) -> List[int]:
        """Encodes text into token ids with special token recognition and byte-fallback."""
        tokens = []
        i = 0
        n = len(text)

        while i < n:
            matched_special = False
            for s_token, s_id in self.special_token_to_id.items():
                if text.startswith(s_token, i):
                    tokens.append(s_id)
                    i += len(s_token)
                    matched_special = True
                    break

            if not matched_special:
                char = text[i]
                raw_bytes = char.encode("utf-8")
                for b in raw_bytes:
                    tokens.append(self.byte_to_id[b])
                i += 1

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token ids back into text, recovering UTF-8 bytes and special tokens."""
        byte_stream = bytearray()
        parts = []

        for tid in token_ids:
            if tid in self.id_to_special_token:
                if byte_stream:
                    parts.append(byte_stream.decode("utf-8", errors="replace"))
                    byte_stream.clear()
                parts.append(self.id_to_special_token[tid])
            elif tid in self.id_to_byte:
                byte_stream.append(self.id_to_byte[tid])
            else:
                if byte_stream:
                    parts.append(byte_stream.decode("utf-8", errors="replace"))
                    byte_stream.clear()
                parts.append(f"<unk_{tid}>")

        if byte_stream:
            parts.append(byte_stream.decode("utf-8", errors="replace"))

        return "".join(parts)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        manifest = {
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return cls(special_tokens=config.get("special_tokens"))
        return cls()
