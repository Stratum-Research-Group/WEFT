#!/usr/bin/env python3
"""Downloads raw WikiText-2 test set if absent."""
import os
import urllib.request
import hashlib

WIKITEXT_DIR = "data/wikitext2"
WIKITEXT_FILE = os.path.join(WIKITEXT_DIR, "wikitext-2-raw-test.txt")
URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"

def ensure_wikitext2():
    os.makedirs(WIKITEXT_DIR, exist_ok=True)
    if not os.path.exists(WIKITEXT_FILE) or os.path.getsize(WIKITEXT_FILE) == 0:
        print(f"Fetching WikiText-2 test set from {URL}...")
        urllib.request.urlretrieve(URL, WIKITEXT_FILE)
    
    with open(WIKITEXT_FILE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"WikiText-2 ready: {WIKITEXT_FILE} (SHA256: {digest})")
    return digest

if __name__ == "__main__":
    ensure_wikitext2()
