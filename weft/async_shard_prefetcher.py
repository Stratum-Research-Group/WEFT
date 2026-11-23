"""
Asynchronous Double-Buffered Shard Prefetcher:
Pipelines sequential shard streaming under strict storage invariants (>= 700 GB floor).
Overlaps download of shard k+1 during factorization of shard k.
"""

import os
import time
import threading
import queue
from typing import Optional, Callable
from huggingface_hub import hf_hub_download

class AsyncDoubleBufferedShardPrefetcher:
    def __init__(self, repo_id: str, shard_filenames: list, work_dir: str, min_free_gb: float = 700.0):
        self.repo_id = repo_id
        self.shard_filenames = shard_filenames
        self.work_dir = work_dir
        self.min_free_gb = min_free_gb
        self.prefetch_queue = queue.Queue(maxsize=1)
        self.stop_signal = threading.Event()
        self.worker_thread = None

    def _get_free_space_gb(self) -> float:
        st = os.statvfs(self.work_dir)
        return (st.f_bavail * st.f_frsize) / (1024**3)

    def _prefetch_loop(self):
        for idx, filename in enumerate(self.shard_filenames):
            if self.stop_signal.is_set():
                break

            free_gb = self._get_free_space_gb()
            if free_gb < self.min_free_gb:
                time.sleep(5)
                continue

            target_path = os.path.join(self.work_dir, filename)
            if not os.path.exists(target_path):
                # Prefetch next shard asynchronously
                downloaded_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    local_dir=self.work_dir,
                    cache_dir=self.work_dir
                )
            else:
                downloaded_path = target_path

            self.prefetch_queue.put((idx, filename, downloaded_path))

    def start(self):
        self.worker_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.worker_thread.start()

    def get_next_shard(self, timeout: float = 300.0) -> Optional[tuple]:
        try:
            return self.prefetch_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def release_shard(self, local_path: str):
        if os.path.exists(local_path):
            os.remove(local_path)

    def stop(self):
        self.stop_signal.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
