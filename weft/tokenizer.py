"""
Simple Character-level Tokenizer for WEFT demo checkpoint.
"""

from typing import List, Union, Any, Optional
import json
import logging
import os

logger = logging.getLogger(__name__)


class WeftCharTokenizer:
    def __init__(self, vocab_file: str):
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        self.vocab_size = len(self.char_to_id)

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        vocab_path = os.path.join(pretrained_path, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.json not found in {pretrained_path}")
        return cls(vocab_path)

    def encode(self, text: str) -> List[int]:
        ids = []
        unknown_chars = []
        for c in text:
            if c in self.char_to_id:
                ids.append(self.char_to_id[c])
            else:
                unknown_chars.append(c)
        if unknown_chars:
            logger.warning(
                "WeftCharTokenizer: Dropped unknown character(s) not in vocabulary: %s",
                repr("".join(unknown_chars[:10]))
            )
        return ids

    def decode(self, token_ids: Union[List[int], Any]) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return "".join([self.id_to_char[i] for i in token_ids if i in self.id_to_char])
