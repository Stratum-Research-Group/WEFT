import pytest
import os
import torch
from weft.self_improving_agent import SelfImprovingCodeAgent
from weft.lifelong_memory import LifelongMemoryStore
from weft.multimodal_streaming import StreamingMultimodalProjector
from weft.asymmetric_speculator import AsymmetricSpeculator


def test_self_improving_agent_execution(tmp_path):
    agent = SelfImprovingCodeAgent(workspace_dir=str(tmp_path))

    # Test file write and read
    test_file = "buggy.py"
    agent.apply_patch(test_file, "def fn(): return 1 + 1")
    assert "return 1 + 1" in agent.read_file(test_file)

    # Test autonomous solve loop simulation
    def mock_proposer(code, error, attempt):
        return "def test_answer(): assert True\n"

    res = agent.solve_issue_loop(
        target_file="test_sample.py",
        test_command="python3 -c 'exit(0)'",
        propose_fix_fn=mock_proposer,
        max_attempts=2,
    )
    assert res["success"] is True


def test_lifelong_memory_store(tmp_path):
    store = LifelongMemoryStore(memory_dir=str(tmp_path / "memory"))
    session = "user_491"

    dummy_state = torch.randn(8, 4, 64, 16)
    store.save_state(session, dummy_state)

    loaded = store.load_state(session, device="cpu")
    assert loaded is not None
    assert torch.allclose(dummy_state, loaded)

    # Test surprisal consolidation
    new_state = dummy_state + 0.1
    consolidated = store.consolidate_state(dummy_state, new_state, surprisal=2.0)
    assert consolidated.shape == dummy_state.shape


def test_streaming_multimodal_projector():
    projector = StreamingMultimodalProjector(
        input_dim=768,
        hidden_dim=512,
        state_dim=64,
        state_rank=16,
    )
    batch_size = 2
    seq_sensory = 16
    mock_features = torch.randn(batch_size, seq_sensory, 768)

    proj, k_s, v_s = projector(mock_features)
    assert proj.shape == (batch_size, seq_sensory, 512)
    assert k_s.shape == (batch_size, seq_sensory, 64)
    assert v_s.shape == (batch_size, seq_sensory, 16)


def test_asymmetric_speculator():
    # Mock draft model
    class MockDraftModel:
        def eval(self): pass
        def __call__(self, x):
            class Out:
                logits = torch.randn(1, x.shape[1], 100)
            return Out()

    spec = AsymmetricSpeculator(MockDraftModel(), None, gamma=3)
    input_ids = torch.tensor([[10, 20]], dtype=torch.long)

    candidate_ids, drafted = spec.draft_tokens(input_ids)
    assert len(drafted) == 3
    assert candidate_ids.shape == (1, 5)

    # Target verifier mock
    def mock_verify_fn(x):
        # Target echoes candidate sequence
        return torch.cat([x[:, 1:], x[:, :1]], dim=1)

    accepted_ids, count = spec.verify_tokens(candidate_ids, 3, mock_verify_fn)
    assert accepted_ids.shape[0] == 1
