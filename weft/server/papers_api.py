"""
FastAPI REST API server for the 95 Stratum Research Group & Synaptek Labs
technical reports. Exposes endpoints for papers, formal invariants, and benchmarks.
"""
from fastapi import FastAPI, HTTPException
import json
import os
from typing import List, Optional

app = FastAPI(
    title="WEFT Stratum Papers REST API",
    description="Programmatic access to the 95-Paper Phase IV Monograph Series",
    version="1.0.0"
)

MANIFEST_PATH = "docs/stratum_papers_manifest.json"

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/")
def root():
    return {
        "service": "WEFT Stratum Papers Microservice",
        "total_manuscripts": 95,
        "series": "Motor Systems Neurobiology & Theoretical Neuroscience",
        "authors": ["Nicholas Charles Mynhardt", "Kevin W. McCairn, Ph.D."],
        "affiliation": "Stratum Research Group & Synaptek Labs",
        "documentation": "/docs"
    }

@app.get("/v1/papers")
def list_papers(search: Optional[str] = None):
    papers = load_manifest()
    if search:
        s = search.lower()
        papers = [p for p in papers if s in p["title"].lower() or s in p["abstract"].lower()]
    return [{"paper_number": p["paper_number"], "target_number": p["target_number"], "title": p["title"]} for p in papers]

@app.get("/v1/papers/{paper_id}")
def get_paper(paper_id: int):
    papers = load_manifest()
    for p in papers:
        if p["paper_number"] == paper_id:
            return p
    raise HTTPException(status_code=404, detail=f"Paper #{paper_id} not found.")

@app.get("/v1/papers/{paper_id}/proof")
def get_paper_proof(paper_id: int):
    papers = load_manifest()
    for p in papers:
        if p["paper_number"] == paper_id:
            return {
                "paper_number": paper_id,
                "target_number": p["target_number"],
                "verification_level": p["verification_level"],
                "lean_proof": p["formal_proof"]
            }
    raise HTTPException(status_code=404, detail=f"Paper #{paper_id} not found.")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8095)
