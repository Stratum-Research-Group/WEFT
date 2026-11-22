import pytest
import os
from weft.multi_file_agent import MultiFileCodebaseAgent


def test_multi_file_agent_workflow(tmp_path):
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    agent = MultiFileCodebaseAgent(str(repo))

    lib_file = "core_math.py"
    test_file = "test_flow.py"

    agent.write_file(lib_file, "def multiply(a, b): return a + b\n")
    agent.write_file(
        test_file,
        "from core_math import multiply\nassert multiply(3, 4) == 12\n",
    )

    # Initial test fails
    res = agent.execute_test_suite(f"python3 {test_file}")
    assert res["passed"] is False

    # Proposer returns fix
    def multi_file_proposer(err, files, turn):
        return {
            lib_file: "def multiply(a, b): return a * b\n",
        }

    solve_res = agent.solve_multi_file_issue(
        test_cmd=f"python3 {test_file}",
        proposer_fn=multi_file_proposer,
        max_turns=2,
    )

    assert solve_res["success"] is True
    assert solve_res["turns_used"] == 1
    assert "return a * b" in agent.read_file(lib_file)


def test_multi_file_agent_path_traversal_guard(tmp_path):
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret_data")

    agent = MultiFileCodebaseAgent(str(repo))

    # Attempting to read outside repo must return None
    assert agent.read_file("../secret.txt") is None
    assert agent.read_file("/etc/passwd") is None

    # Attempting to write outside repo must return False and not write
    assert agent.write_file("../pwned.txt", "exploit") is False
    assert not (tmp_path / "pwned.txt").exists()

