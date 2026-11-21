"""
Autonomous Continuous Self-Improvement Daemon Runner:
Runs an autonomous distillation & Lean 4 verification cycle,
saving progress metrics to /run/media/akarta/AI/checkpoints/aligned_head/daemon_stats.json.
"""
import os
import sys
import json
import time
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft.continuous_daemon import ContinuousDistillationDaemon

def run_daemon_step(num_cycles: int = 10):
    print("=" * 80)
    print("  LAUNCHING AUTONOMOUS CONTINUOUS SELF-IMPROVEMENT DAEMON")
    print("=" * 80)
    
    daemon = ContinuousDistillationDaemon()
    stats = daemon.run_cycle(max_iterations=num_cycles)
    
    stats_file = "/run/media/akarta/AI/checkpoints/aligned_head/daemon_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Cycle finished successfully: {stats}")
    print(f"Metrics saved to {stats_file}")
    print("=" * 80)

if __name__ == "__main__":
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_daemon_step(num_cycles=cycles)
