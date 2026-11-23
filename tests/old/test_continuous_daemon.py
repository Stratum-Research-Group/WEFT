"""Unit test for continuous self-distillation daemon."""
from weft.continuous_daemon import ContinuousDistillationDaemon

def test_continuous_daemon_cycle(tmp_path):
    daemon = ContinuousDistillationDaemon(checkpoint_dir=str(tmp_path / "checkpoints"))
    stats = daemon.run_cycle(max_iterations=2)
    assert stats["cycles_completed"] == 2
    assert stats["theorems_verified"] == 2
