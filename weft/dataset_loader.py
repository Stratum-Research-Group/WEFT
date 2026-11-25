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


TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

WIKITEXT2_REPO = "Salesforce/wikitext"
WIKITEXT2_FILES = {
    "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "validation": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
    "test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
}


def download_wikitext2(split: str = "test", out_dir: str = "data/wikitext2") -> str:
    """Download the real WikiText-2 (raw) split and write it as one text file.

    Joins rows exactly as the standard perplexity protocol does ("\n\n".join).
    Returns the path to the written .txt file.
    """
    if split not in WIKITEXT2_FILES:
        raise ValueError(f"split must be one of {sorted(WIKITEXT2_FILES)}")
    import pandas as pd
    from huggingface_hub import hf_hub_download

    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"wikitext-2-raw-{split}.txt")
    if not os.path.exists(txt_path):
        pq = hf_hub_download(WIKITEXT2_REPO, WIKITEXT2_FILES[split], repo_type="dataset")
        text = "\n\n".join(pd.read_parquet(pq)["text"].tolist())
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    return txt_path

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
        # Contract: targets are an unshifted copy of the inputs. WeftForCausalLM
        # (weft/modeling_weft.py, labels[..., 1:] vs logits[..., :-1, :]) performs the
        # next-token shift internally, following the HuggingFace convention.
        # Do NOT pre-shift here or the objective would be shifted twice.
        return x, x.clone()
