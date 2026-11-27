"""
WEFT Stratum Papers API Python Client SDK.
Provides type-safe programmatic access to the 95-paper monograph and Lean 4 proofs.
"""
import json
import urllib.request
import urllib.parse
from typing import Dict, List, Optional

class StratumPapersClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8095"):
        self.base_url = base_url.rstrip("/")

    def list_papers(self, search: Optional[str] = None) -> List[Dict]:
        url = f"{self.base_url}/v1/papers"
        if search:
            url += f"?search={urllib.parse.quote(search)}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_paper(self, paper_id: int) -> Dict:
        url = f"{self.base_url}/v1/papers/{paper_id}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_proof(self, paper_id: int) -> Dict:
        url = f"{self.base_url}/v1/papers/{paper_id}/proof"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
