"""Real behaviour tests for the speculative verification endpoints.
Expected values come from the Python reference (weft.speculative_tree), not from the server."""
import json, os, subprocess, time, signal, urllib.request, urllib.error
import pytest
import torch
from weft.speculative_tree import SpeculativeTreeEngine

MODEL = "/run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft"
PORT = 8131
_ref = SpeculativeTreeEngine(None, None)


def ref(draft, target, parents=None):
    parents = parents if parents is not None else [-1] + list(range(len(draft) - 1))
    acc = _ref.verify_tree(torch.zeros(1, 1, dtype=torch.long), draft, target, parents)
    div = -1 if len(acc) == len(draft) else len(acc)
    return acc, div


@pytest.fixture(scope="module")
def server():
    if not os.path.exists(MODEL):
        pytest.skip("model not found")
    p = subprocess.Popen(["./build/weft-server", MODEL, str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    yield p
    os.kill(p.pid, signal.SIGTERM); p.wait(timeout=3)


def post(path, obj, raw=False):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, (r.read().decode() if raw else json.loads(r.read().decode()))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


CASES = [
    ([5, 6, 7, 8], [5, 6, 7, 8]),          # all accepted
    ([5, 6, 9, 8], [5, 6, 7, 8]),          # diverges at 2
    ([1, 6, 7, 8], [5, 6, 7, 8]),          # rejected at root
    ([5], [5]),
    ([5, 6, 7, 8, 9, 10], [5, 6, 7, 0, 9, 10]),
]


@pytest.mark.parametrize("draft,target", CASES)
def test_verify_tree_matches_reference(server, draft, target):
    st, d = post("/v1/speculative/verify_tree", {"draft_tokens": draft, "target_tokens": target})
    acc, div = ref(draft, target)
    assert st == 200
    assert d["accepted_tokens"] == acc and d["accepted_count"] == len(acc) and d["diverged_index"] == div


def test_verify_tree_with_parents(server):
    draft, target, parents = [5, 6, 7, 8], [5, 6, 7, 8], [-1, 0, 0, 2]  # node 2's parent is 0, not last-accepted(1)
    st, d = post("/v1/speculative/verify_tree", {"draft_tokens": draft, "target_tokens": target, "parents": parents})
    acc, div = ref(draft, target, parents)
    assert (d["accepted_tokens"], d["diverged_index"]) == (acc, div) == ([5, 6], 2)


def test_verify_tree_rejects_malformed(server):
    assert post("/v1/speculative/verify_tree", {"draft_tokens": [1, 2]})[0] == 400
    assert post("/v1/speculative/verify_tree", {"draft_tokens": [1, 2, 3], "target_tokens": [1]})[0] == 400
    assert post("/v1/speculative/verify_tree", {"draft_tokens": [], "target_tokens": []})[0] == 400


def test_batch_independent_streams(server):
    streams = [{"stream_id": 10 + i, "draft_tokens": d, "target_tokens": t} for i, (d, t) in enumerate(CASES)]
    st, d = post("/v1/speculative/verify_tree_batch", {"streams": streams})
    assert st == 200 and d["batch_size"] == len(CASES)
    for i, (dr, tg) in enumerate(CASES):
        acc, div = ref(dr, tg)
        r = d["stream_results"][i]
        assert r["stream_id"] == 10 + i and r["accepted_tokens"] == acc and r["diverged_index"] == div


def test_batch_reports_per_stream_error(server):
    st, d = post("/v1/speculative/verify_tree_batch",
                 {"streams": [{"stream_id": 1, "draft_tokens": [1], "target_tokens": [1]}, {"stream_id": 2, "draft_tokens": [1]}]})
    assert st == 200 and "error" in d["stream_results"][1] and d["stream_results"][0]["accepted_count"] == 1


def test_batch_empty_is_400(server):
    assert post("/v1/speculative/verify_tree_batch", {"streams": []})[0] == 400


@pytest.mark.parametrize("draft,target", CASES)
def test_sse_stream_matches_reference(server, draft, target):
    st, raw = post("/v1/speculative/verify_tree_stream", {"draft_tokens": draft, "target_tokens": target}, raw=True)
    assert st == 200
    ev = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    acc, div = ref(draft, target)
    assert [e["token_id"] for e in ev if e["accepted"]] == acc
    rej = [e for e in ev if not e["accepted"]]
    assert (len(rej) == 0) == (div == -1)
    if rej:
        assert rej[0]["index"] == div and rej[0]["target_token"] == target[div]
    assert raw.strip().endswith("data: [DONE]")
