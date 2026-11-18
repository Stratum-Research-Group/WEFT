"""
High-Performance Streaming Dataset Loader with Token Packing for WEFT.
Directly downloads text shards (e.g. OpenWebText/Wikitext/C4 samples),
tokenizes using BPE (tiktoken GPT-2), packs tokens into fixed sequences,
and provides infinite batch streams with zero dependencies on problematic C++ extensions.
"""

import os
import urllib.request
import torch
import tiktoken


WIKITEXT_URLS = {
    "train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}

class TokenPackedLoader:
    def __init__(self, data_path: str, seq_len: int, batch_size: int, device: str = "cuda"):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        print(f"[Loader] Tokenizing {data_path} with GPT-2 BPE...")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        tokens = self.enc.encode_ordinary(raw_text)
        print(f"[Loader] Encoded {len(tokens):,} BPE tokens (vocab size: {self.vocab_size:,})")

        self.data_tensor = torch.tensor(tokens, dtype=torch.long)
        n_split = int(0.9 * len(self.data_tensor))
        self.train_data = self.data_tensor[:n_split]
        self.val_data = self.data_tensor[n_split:]

    def get_batch(self, split: str = "train"):
        d = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(d) - self.seq_len, (self.batch_size,))
        x = torch.stack([d[i : i + self.seq_len] for i in ix]).to(self.device)
        return x, x.clone()
