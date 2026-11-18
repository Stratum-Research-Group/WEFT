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
        self._consecutive_failures = 0

    def _fetch_next_rows(self, count: int = 25):
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
                if not rows:
                    # Genuine end of rows reached: wrap around
                    self.current_offset = 0
                else:
                    self.current_offset += len(rows)
                self._consecutive_failures = 0
                return rows
        except urllib.error.HTTPError as e:
            if e.code == 404 or e.code == 400:
                # Genuine end-of-split bounds reached
                self.current_offset = 0
                self._consecutive_failures = 0
                return []
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                # If specific server row index times out or returns 502/504 gateway error, advance offset past corrupted slice
                self.current_offset += count
                self._consecutive_failures = 0
            return []
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                # Advance past unresponsive remote row chunk
                self.current_offset += count
                self._consecutive_failures = 0
            return []

    def get_batch(self):
        target_tokens = self.batch_size * self.seq_len
        retries = 0
        max_retries = 20
        while len(self.token_queue) < target_tokens:
            rows = self._fetch_next_rows(count=25)
            if not rows:
                retries += 1
                if retries >= max_retries:
                    raise RuntimeError(
                        f"FineWebEduStreamer failed after {max_retries} consecutive fetch attempts at offset {self.current_offset}"
                    )
                import time
                time.sleep(1.0)
                continue
            retries = 0
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
