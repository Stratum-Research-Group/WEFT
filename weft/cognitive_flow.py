"""
Cognitive Flow State Machine & Troubleshooting Engine for WEFT Sub-Bit Swarms.
Ported from IMF Cognitive Workflows & State Machine Architecture Specification.
Enforces the 5-Phase Cognitive Flow Protocol across Council & Swarm Agent Collaborations:
1. Setup: Context boundary conditioning, task narrowing, working memory protection.
2. Struggle: Latent schema loading, multi-agent friction tolerance, task monogamy.
3. Release: Decoupling conscious strain, state vector reset, entropy relaxation.
4. Flow: Uninterrupted high-throughput execution (> 280 tok/s), speculative drafting.
5. Recovery: State compaction, KV eviction, checkpointing, and clean handoff.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import time

class CognitivePhase(str, Enum):
    SETUP = "setup"         # Phase 1: Boundary conditioning & context narrowing
    STRUGGLE = "struggle"   # Phase 2: Schema loading & adversarial friction
    RELEASE = "release"     # Phase 3: Hidden state reset & tension decoupling
    FLOW = "flow"           # Phase 4: High-throughput speculative execution
    RECOVERY = "recovery"   # Phase 5: Consolidation & zero-copy state eviction

class Bottleneck(str, Enum):
    ANXIOUS_FROZEN = "anxious_frozen" # Task too ambiguous or high-stakes
    BORED_RESTLESS = "bored_restless" # Task too trivial, lacks tight feedback
    FOGGY_SLUGGISH = "foggy_sluggish" # Cold start resistance or context fatigue
    DISTRACTED = "distracted"         # High entropy, intrusive token drift
    NONE = "none"

class CognitiveStateMachine:
    def __init__(self, task_description: str):
        self.task_description = task_description
        self.phase = CognitivePhase.SETUP
        self.task_clarity = 5 # 0-10
        self.friction_level = 5 # 0-10
        self.distraction_parking_lot: List[str] = []
        self.phase_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self._record_transition(CognitivePhase.SETUP, "Initiating session setup & objective lockdown")

    def _record_transition(self, phase: CognitivePhase, note: str):
        self.phase = phase
        self.phase_history.append({
            "phase": phase.value,
            "note": note,
            "timestamp": round(time.time() - self.start_time, 3),
            "clarity": self.task_clarity,
            "friction": self.friction_level
        })

    def advance_phase(self) -> CognitivePhase:
        """Transitions forward strictly along the canonical cognitive flow loop."""
        transitions = {
            CognitivePhase.SETUP: CognitivePhase.STRUGGLE,
            CognitivePhase.STRUGGLE: CognitivePhase.RELEASE,
            CognitivePhase.RELEASE: CognitivePhase.FLOW,
            CognitivePhase.FLOW: CognitivePhase.RECOVERY,
            CognitivePhase.RECOVERY: CognitivePhase.SETUP,
        }
        next_phase = transitions[self.phase]
        self._record_transition(next_phase, f"Advancing to {next_phase.value} phase")
        return self.phase

    def park_distraction(self, transient_thought: str):
        """Zero-latency interrupt buffer preserving working memory."""
        self.distraction_parking_lot.append(transient_thought)

    def diagnose_bottleneck(self) -> Bottleneck:
        if self.task_clarity < 3:
            return Bottleneck.ANXIOUS_FROZEN
        elif self.friction_level == 0:
            return Bottleneck.BORED_RESTLESS
        elif self.friction_level > 8:
            return Bottleneck.FOGGY_SLUGGISH
        return Bottleneck.NONE

    def apply_algorithmic_fix(self, bottleneck: Bottleneck) -> Dict[str, Any]:
        """Applies IMF-certified deterministic resolution actions."""
        if bottleneck == Bottleneck.ANXIOUS_FROZEN:
            self.task_clarity = 10
            self.friction_level = 2
            return {
                "action": "Bisect task scope by half and set 10-minute micro-sprint",
                "resolved_clarity": self.task_clarity
            }
        elif bottleneck == Bottleneck.BORED_RESTLESS:
            self.friction_level = 5
            return {
                "action": "Elevate formal constraints or double token throughput limit",
                "resolved_friction": self.friction_level
            }
        elif bottleneck == Bottleneck.FOGGY_SLUGGISH:
            self.friction_level = 3
            return {
                "action": "Execute Lie SSM state reset & decouple stale KV caches",
                "resolved_friction": self.friction_level
            }
        elif bottleneck == Bottleneck.DISTRACTED:
            self.task_clarity = 8
            self.friction_level = 2
            return {
                "action": "Flush distraction buffer and re-prime short-term context",
                "resolved_clarity": self.task_clarity
            }
        return {"action": "No intervention required"}
