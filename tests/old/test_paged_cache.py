import pytest
import torch
from weft.paged_cache import PagedWeftBlockManager, PagedWeftServingEngine


def test_paged_block_allocation_and_free():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mgr = PagedWeftBlockManager(num_layers=4, block_size=16, max_blocks=32, latent_dim=64, device=device)

    assert len(mgr.free_blocks) == 32
    b1 = mgr.allocate(seq_id=1, num_blocks=2)
    assert len(b1) == 2
    assert len(mgr.free_blocks) == 30

    b2 = mgr.allocate(seq_id=2, num_blocks=4)
    assert len(b2) == 4
    assert len(mgr.free_blocks) == 26

    mgr.free(seq_id=1)
    assert len(mgr.free_blocks) == 28

    mgr.free(seq_id=2)
    assert len(mgr.free_blocks) == 32


def test_paged_latent_read_write():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mgr = PagedWeftBlockManager(num_layers=2, block_size=4, max_blocks=16, latent_dim=32, device=device)

    mgr.allocate(seq_id=10, num_blocks=3)  # Capacity: 12 tokens
    v0 = torch.randn(32, dtype=torch.bfloat16, device=device)
    v5 = torch.randn(32, dtype=torch.bfloat16, device=device)

    mgr.write_latent(layer_idx=0, seq_id=10, logical_token_pos=0, latent_vec=v0)
    mgr.write_latent(layer_idx=0, seq_id=10, logical_token_pos=5, latent_vec=v5)

    window = mgr.read_window_latents(layer_idx=0, seq_id=10, start_token=0, end_token=6)
    assert window.shape == (6, 32)
    assert torch.allclose(window[0], v0)
    assert torch.allclose(window[5], v5)


def test_paged_serving_engine_lifecycle():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = PagedWeftServingEngine(num_layers=8, block_size=16, max_blocks=64, latent_dim=64, device=device)

    s1 = engine.add_sequence()
    s2 = engine.add_sequence()
    stats = engine.stats()
    assert stats["active_sequences"] == 2
    assert stats["free_blocks"] == 62

    engine.remove_sequence(s1)
    engine.remove_sequence(s2)
    stats2 = engine.stats()
    assert stats2["active_sequences"] == 0
    assert stats2["free_blocks"] == 64
