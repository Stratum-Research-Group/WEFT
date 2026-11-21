"""
Background Continuous Distillation & Lean 4 RSI Daemon runner.
Runs in an asynchronous loop, validating theorem proofs and updating checkpoints.
"""
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from weft.continuous_daemon import ContinuousDistillationDaemon

def main():
    print("Initializing Continuous Distillation Daemon...")
    daemon = ContinuousDistillationDaemon()
    print("Daemon initialized. Running 10 continuous cycles on GPU...")
    stats = daemon.run_cycle(max_iterations=10)
    print(f"Cycle completed successfully: {stats}")

if __name__ == "__main__":
    main()
