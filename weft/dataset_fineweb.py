"""
Streaming FineWeb-Edu Dataset Loader for WEFT Pretraining.
Streams real educational web content directly via the HuggingFace datasets server API
with zero local multi-gigabyte disk footprint and full GPT-2 BPE token packing.
"""

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
        self.buffer = []
        self.token_queue = []

    def _fetch_next_rows(self, count: int = 100):
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train&"
            f"offset={self.current_offset}&length={count}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = [item["row"]["text"] for item in data.get("rows", [])]
                self.current_offset += len(rows)
                return rows
        except Exception as e:
            # Wrap around on connection timeout or EOF
            self.current_offset = 0
            return []

    def get_batch(self):
        while len(self.token_queue) < self.batch_size * self.seq_len:
            rows = self._fetch_next_rows(count=100)
            if not rows:
                break
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
