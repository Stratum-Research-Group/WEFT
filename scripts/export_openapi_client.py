#!/usr/bin/env python3
"""
Exports OpenAPI 3.1.0 specification for WEFT Stratum Papers REST API
and generates client bindings (docs/papers_api_schema.json & weft/client/papers_client.py).
"""
import json
import os

def export_openapi():
    # Construct exact OpenAPI schema
    schema = {
        "openapi": "3.1.0",
        "info": {
            "title": "WEFT Stratum Papers REST API",
            "version": "1.0.0",
            "description": "Programmatic interface to the 95-Paper Motor Systems & Theoretical Neuroscience Monograph Volume IV co-authored by Nicholas Charles Mynhardt & Kevin W. McCairn, Ph.D."
        },
        "servers": [{"url": "http://127.0.0.1:8095"}],
        "paths": {
            "/v1/papers": {
                "get": {
                    "summary": "List all 95 scientific papers",
                    "parameters": [{
                        "name": "search",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string"},
                        "description": "Substring search query across title and abstract"
                    }],
                    "responses": {
                        "200": {
                            "description": "List of paper metadata records",
                            "content": {"application/json": {"schema": {"type": "array"}}}
                        }
                    }
                }
            },
            "/v1/papers/{paper_id}": {
                "get": {
                    "summary": "Retrieve complete paper details, parameters, and figure paths",
                    "parameters": [{
                        "name": "paper_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"}
                    }],
                    "responses": {
                        "200": {"description": "Full manuscript metadata and abstract"},
                        "404": {"description": "Paper not found"}
                    }
                }
            },
            "/v1/papers/{paper_id}/proof": {
                "get": {
                    "summary": "Retrieve machine-checked Lean 4 formal proof for paper",
                    "parameters": [{
                        "name": "paper_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"}
                    }],
                    "responses": {
                        "200": {"description": "Lean 4 theorem code"},
                        "404": {"description": "Paper not found"}
                    }
                }
            }
        }
    }

    out_schema = "docs/papers_api_schema.json"
    with open(out_schema, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Generate Python Client SDK
    client_code = '''"""
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
'''
    os.makedirs("weft/client", exist_ok=True)
    out_client = "weft/client/papers_client.py"
    with open(out_client, "w", encoding="utf-8") as f:
        f.write(client_code)

    print(f"Generated {out_schema} and {out_client}.")

if __name__ == '__main__':
    export_openapi()
