import pytest
from weft.autonomous_collective import AutonomousCollectiveEngine, GoalContract

def test_autonomous_collective_lifecycle():
    engine = AutonomousCollectiveEngine()
    contract = GoalContract(
        goal_id="TEST-001",
        description="Verify quadratic formula and test code",
        modality_requirements=["math", "code"]
    )
    state = engine.execute_goal(contract)
    assert state.current_stage == "COMPLETED"
    assert state.completed_subgoals == 4
    assert state.collective_ram_mb < 1400.0
    assert "mathematical_proof" in state.verified_outputs
    assert "code_kernel" in state.verified_outputs
