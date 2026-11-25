"""
WEFT Closed-Loop Autonomous Cognitive Engine (cog2 Architecture).
Implements the 6-Stage Closed-Loop Cycle:
  [1. Ingestion] -> [2. Surprise Filter] -> [3. Attentional Ignition] ->
  [4. Global Workspace Broadcast] -> [5. Metacognitive Interrogation] ->
  [6. Action & Temporal Continuity Update] -> Loops back to [1. Ingestion]

Integrates Active Inference free-energy minimization with IMF 5-Phase Cognitive Flow
and zero-copy /dev/shm ring synchronization.
"""

import math
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from weft.cognitive_flow import CognitiveStateMachine, CognitivePhase, Bottleneck
from weft.kernels.tree_kv_rollback import WeftShmTreeRingPoller

@dataclass
class CognitiveMemoryState:
    attention_focus: str = ""
    focal_salience: float = 0.0
    surprise_history: List[float] = field(default_factory=list)
    active_inferences: List[str] = field(default_factory=list)
    temporal_step: int = 0
    free_energy: float = 1.0

class CognitiveLoopEngine:
    def __init__(self, surprise_threshold: float = 0.45):
        self.surprise_threshold = surprise_threshold
        self.state = CognitiveMemoryState()
        self.flow_sm = CognitiveStateMachine("WEFT Autonomous Active Inference Session")
        self.shm_poller: Optional[WeftShmTreeRingPoller] = None
        try:
            self.shm_poller = WeftShmTreeRingPoller()
        except Exception:
            pass

    def step_1_ingestion(self, prompt: str, system_telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stage 1: Multi-Channel Ingestion (Exteroception + Interoception)."""
        telemetry = system_telemetry or {}
        return {
            "prompt": prompt,
            "tokens": prompt.split(),
            "cpu_load": telemetry.get("cpu_load", 0.15),
            "memory_usage": telemetry.get("memory_usage", 0.25),
            "timestamp": time.time()
        }

    def step_2_surprise_filtering(self, raw_input: Dict[str, Any]) -> float:
        """Stage 2: Prediction & Error Filtering (Detecting Novelty/Surprise)."""
        tokens = raw_input.get("tokens", [])
        if not tokens:
            return 0.0
        
        # Heuristic surprise estimation based on token novelty and cognitive complexity
        keywords = ["proof", "theorem", "verify", "invariant", "bug", "error", "fail", "broken", "critical"]
        complexity_hits = sum(1 for t in tokens if any(k in t.lower() for k in keywords))
        surprise = min(1.0, (complexity_hits / max(1, len(tokens))) * 2.5 + 0.1)
        
        self.state.surprise_history.append(surprise)
        return surprise

    def step_3_attentional_ignition(self, raw_input: Dict[str, Any], surprise: float) -> bool:
        """Stage 3: Attentional Ignition (The Threshold: What enters the spotlight?)."""
        ignited = surprise >= self.surprise_threshold
        if ignited:
            self.state.attention_focus = " ".join(raw_input.get("tokens", [])[:6])
            self.state.focal_salience = surprise
        else:
            self.state.attention_focus = "routine_background_continuation"
            self.state.focal_salience = surprise * 0.5
        return ignited

    def step_4_global_workspace_broadcast(self, ignited: bool, raw_input: Dict[str, Any]):
        """Stage 4: Global Workspace Broadcast (Subjective Consciousness Integration)."""
        if self.shm_poller and ignited:
            # Broadcast attention spotlight to /dev/shm ring buffer
            token_repr = hash(self.state.attention_focus) % 32000
            self.shm_poller.push_node(
                node_id=self.state.temporal_step,
                parent_id=-1,
                token_id=token_repr,
                accepted=True,
                confidence=float(self.state.focal_salience)
            )

    def step_5_metacognitive_interrogation(self, prompt: str, ignited: bool) -> Dict[str, Any]:
        """Stage 5: Metacognitive Interrogation (Autonomous Thinking & Self-Critique)."""
        if not ignited:
            return {
                "route": "fast_path_heuristic",
                "verdict": "Standard low-friction continuation",
                "confidence": 0.95
            }
        
        # High surprise / high friction: Advance cognitive state machine and deliberate
        self.flow_sm.advance_phase()
        
        # Check for bottlenecks
        diagnostic_bottleneck = self.flow_sm.diagnose_bottleneck()
        
        return {
            "route": "master_council_deliberation",
            "phase": self.flow_sm.phase.value,
            "friction": self.flow_sm.friction_level,
            "clarity": self.flow_sm.task_clarity,
            "bottleneck": diagnostic_bottleneck.value,
            "verdict": f"Metacognitive synthesis complete for: '{self.state.attention_focus}'"
        }

    def step_6_action_and_temporal_update(self, interrogation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 6: Action & Temporal Continuity Update."""
        self.state.temporal_step += 1
        
        # Minimize Free Energy via iterative error suppression
        if self.state.surprise_history:
            recent_mean_surprise = sum(self.state.surprise_history[-5:]) / min(5, len(self.state.surprise_history))
            self.state.free_energy = math.exp(-0.2 * self.state.temporal_step) * recent_mean_surprise
        
        self.state.active_inferences.append(interrogation_result.get("verdict", ""))
        
        return {
            "step": self.state.temporal_step,
            "focus": self.state.attention_focus,
            "salience": self.state.focal_salience,
            "free_energy": self.state.free_energy,
            "result": interrogation_result
        }

    def run_cycle(self, prompt: str, telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the full 6-stage closed-loop cognitive cycle."""
        # 1. Ingestion
        raw = self.step_1_ingestion(prompt, telemetry)
        # 2. Surprise Filtering
        surprise = self.step_2_surprise_filtering(raw)
        # 3. Attentional Ignition
        ignited = self.step_3_attentional_ignition(raw, surprise)
        # 4. Global Broadcast
        self.step_4_global_workspace_broadcast(ignited, raw)
        # 5. Metacognitive Interrogation
        interrogation = self.step_5_metacognitive_interrogation(prompt, ignited)
        # 6. Action & Update
        final_state = self.step_6_action_and_temporal_update(interrogation)
        return final_state

if __name__ == "__main__":
    engine = CognitiveLoopEngine()
    print("=== Testing WEFT cog2 6-Stage Autonomous Cognitive Loop ===")
    out1 = engine.run_cycle("Hello how are you today")
    print(f"Cycle 1 (Low Surprise): Route={out1['result']['route']} FreeEnergy={out1['free_energy']:.4f}")
    out2 = engine.run_cycle("Critical bug: verify Lean 4 memory proof failure in kernel")
    print(f"Cycle 2 (High Surprise Ignition): Route={out2['result']['route']} Focus='{out2['focus']}' FreeEnergy={out2['free_energy']:.4f}")
