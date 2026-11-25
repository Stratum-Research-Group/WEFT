"""
Tree KV Cache Rollback & /dev/shm Zero-Copy IPC Ring Polling Daemon.
Milestone Paper #126 / Phase 5 Autonomous Runtime.
Connects Python execution swarms directly to the shared memory ring buffer (/dev/shm/weft_spec_ring)
for zero-copy candidate token retrieval and speculative rollback synchronization.
"""

import os
import mmap
import struct
import time
from typing import Optional, List, Tuple

SHM_FILE_PATH = "/dev/shm/weft_spec_ring"
RING_CAPACITY = 1024
NODE_RECORD_FORMAT = "iiiif" # node_id (int), parent_id (int), token_id (int), accepted (int), confidence (float)
NODE_RECORD_SIZE = struct.calcsize(NODE_RECORD_FORMAT)

# Header: head (uint64), tail (uint64)
HEADER_FORMAT = "QQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
TOTAL_SHM_SIZE = HEADER_SIZE + (RING_CAPACITY * NODE_RECORD_SIZE)

class WeftShmTreeRingPoller:
    def __init__(self, shm_path: str = SHM_FILE_PATH):
        self.shm_path = shm_path
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        self._ensure_shm_mapped()

    def _ensure_shm_mapped(self):
        if not os.path.exists(self.shm_path):
            with open(self.shm_path, "wb") as f:
                f.write(b"\x00" * TOTAL_SHM_SIZE)
        
        self._fd = os.open(self.shm_path, os.O_RDWR)
        self._mmap = mmap.mmap(self._fd, TOTAL_SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def read_pointers(self) -> Tuple[int, int]:
        if not self._mmap:
            return 0, 0
        head, tail = struct.unpack_from(HEADER_FORMAT, self._mmap, 0)
        return head, tail

    def write_pointers(self, head: int, tail: int):
        if self._mmap:
            struct.pack_into(HEADER_FORMAT, self._mmap, 0, head, tail)

    def poll_nodes(self, max_batch: int = 16) -> List[dict]:
        """Poll newly arrived tree nodes from the zero-copy ring buffer."""
        if not self._mmap:
            return []
        
        head, tail = self.read_pointers()
        nodes = []
        
        while head < tail and len(nodes) < max_batch:
            idx = head % RING_CAPACITY
            offset = HEADER_SIZE + (idx * NODE_RECORD_SIZE)
            node_id, parent_id, token_id, accepted, conf = struct.unpack_from(NODE_RECORD_FORMAT, self._mmap, offset)
            nodes.append({
                "node_id": node_id,
                "parent_id": parent_id,
                "token_id": token_id,
                "accepted": bool(accepted),
                "confidence": conf
            })
            head += 1
            
        self.write_pointers(head, tail)
        return nodes

    def push_node(self, node_id: int, parent_id: int, token_id: int, accepted: bool, confidence: float):
        """Producer method to push speculative node directly into SHM ring."""
        if not self._mmap:
            return
        head, tail = self.read_pointers()
        idx = tail % RING_CAPACITY
        offset = HEADER_SIZE + (idx * NODE_RECORD_SIZE)
        struct.pack_into(NODE_RECORD_FORMAT, self._mmap, offset, node_id, parent_id, token_id, int(accepted), confidence)
        self.write_pointers(head, tail + 1)

    def close(self):
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

if __name__ == "__main__":
    poller = WeftShmTreeRingPoller()
    print(f"[*] Initialized WeftShmTreeRingPoller on {SHM_FILE_PATH}")
    poller.push_node(1, -1, 1042, True, 0.99)
    poller.push_node(2, 1, 2084, True, 0.95)
    nodes = poller.poll_nodes()
    print(f"[+] Polled {len(nodes)} nodes from SHM ring: {nodes}")
    poller.close()
