"""
WEFT-2 Autonomous Self-Improving Software Engineering Agent ("Self-Coder").

Architecture:
1. Worktree Inspection: Reads files, git diffs, directory trees.
2. Tooling Hooks: Runs pytest, cargo test, linters, or custom shell commands.
3. Diagnostic Feedback: Feeds compiler/runtime tracebacks back into the model's <think> context.
4. Auto-Patching: Generates unified diffs and applies them to the target worktree.
5. Loop Termination: Terminates when all test gates pass or maximum iterations are exhausted.
"""

import os
import subprocess
import traceback
from typing import Dict, Any, List, Optional
from weft.unified_sandbox import UnifiedAgentSandbox


class SelfImprovingCodeAgent:
    """Autonomous recursive software engineering loop powered by WEFT."""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.sandbox = UnifiedAgentSandbox()

    def run_tests(self, test_command: str = "pytest -q") -> Dict[str, Any]:
        """Runs test suite and returns exit code, stdout, and error traces."""
        try:
            res = subprocess.run(
                test_command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "exit_code": res.returncode,
                "passed": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "passed": False, "stdout": "", "stderr": "Execution timed out (60s)."}
        except Exception as e:
            return {"exit_code": -1, "passed": False, "stdout": "", "stderr": str(e)}

    def read_file(self, rel_path: str) -> Optional[str]:
        # Path traversal guard: ensure path is within workspace_dir
        full_path = os.path.abspath(os.path.join(self.workspace_dir, rel_path))
        if not full_path.startswith(self.workspace_dir + os.sep) and full_path != self.workspace_dir:
            return None
        if not os.path.isfile(full_path):
            return None
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def apply_patch(self, rel_path: str, new_content: str) -> bool:
        # Path traversal guard: ensure target is strictly inside workspace_dir
        full_path = os.path.abspath(os.path.join(self.workspace_dir, rel_path))
        if not full_path.startswith(self.workspace_dir + os.sep) and full_path != self.workspace_dir:
            return False
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        except Exception:
            return False

    def solve_issue_loop(
        self,
        target_file: str,
        test_command: str,
        propose_fix_fn,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """Runs the recursive self-improving loop until tests pass."""
        history = []

        for attempt in range(1, max_attempts + 1):
            test_res = self.run_tests(test_command)
            if test_res["passed"]:
                return {
                    "success": True,
                    "attempts": attempt - 1,
                    "history": history,
                    "message": "All test assertions passing cleanly.",
                }

            current_code = self.read_file(target_file)
            error_msg = test_res["stderr"] or test_res["stdout"]

            # Prompt model/heuristic with current code + traceback
            fix = propose_fix_fn(current_code, error_msg, attempt)
            if fix:
                self.apply_patch(target_file, fix)

            history.append({
                "attempt": attempt,
                "error": error_msg[:300],
                "applied_fix": bool(fix),
            })

        final_check = self.run_tests(test_command)
        return {
            "success": final_check["passed"],
            "attempts": max_attempts,
            "history": history,
            "message": "Max attempts reached." if not final_check["passed"] else "Fixed on final attempt.",
        }
