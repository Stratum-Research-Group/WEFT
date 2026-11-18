"""
FineWeb-Edu Dataset Loader for WEFT Pretraining.
Supports fast cached multi-megabyte training directly from local data/fineweb_edu_cache.json
with zero network timeouts, full GPT-2 BPE token packing, and dynamic fallback.
"""

import os
import json
import urllib.request
import torch
import tiktoken


class FineWebEduStreamer:
    def __init__(self, seq_len: int, batch_size: int, device: str = "cuda", max_rows: int = 5000):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.max_rows = max_rows
        self.current_offset = 0
        self.token_queue = []
        self._local_docs = []
        self._doc_idx = 0
        self._load_local_cache()

    def _load_local_cache(self):
        cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fineweb_edu_cache.json")
        if not os.path.exists(cache_file):
            cache_file = "data/fineweb_edu_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self._local_docs = json.load(f)

    def _fetch_next_rows(self, count: int = 25):
        if self._local_docs:
            rows = self._local_docs[self._doc_idx : self._doc_idx + count]
            self._doc_idx += len(rows)
            if self._doc_idx >= len(self._local_docs):
                self._doc_idx = 0  # wrap around local corpus
            return rows

        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train&"
            f"offset={self.current_offset}&length={count}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = [item["row"]["text"] for item in data.get("rows", [])]
                self.current_offset += len(rows)
                return rows
        except Exception:
            self.current_offset += count
            return []

    def get_batch(self):
        target_tokens = self.batch_size * self.seq_len
        while len(self.token_queue) < target_tokens:
            rows = self._fetch_next_rows(count=25)
            if not rows:
                import time
                time.sleep(0.1)
                continue
            for text in rows:
                if not text.strip():
                    continue
                toks = self.enc.encode_ordinary(text)
                toks.append(self.enc.eot_token)
                self.token_queue.extend(toks)

        # Extract fixed-size batch
        batch_chunks = []
        for _ in range(self.batch_size):
            chunk = self.token_queue[: self.seq_len]
            self.token_queue = self.token_queue[self.seq_len :]
            batch_chunks.append(torch.tensor(chunk, dtype=torch.long))

        x = torch.stack(batch_chunks).to(self.device)
        return x, x.clone()
