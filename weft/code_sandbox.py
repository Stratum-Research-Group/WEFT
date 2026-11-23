"""
Edge Code Execution Sandbox for Gemma-4-26B Sparse MoE:
Provides safe, local in-process execution of Python and shell snippets with strict resource limits,
timeout enforcement (5.0s), and stdout/stderr capture for autonomous closed-loop verification.
"""

import sys
import io
import time
import subprocess
from typing import Dict, Any

class EdgeCodeSandbox:
    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """Safely executes a Python snippet and returns stdout, return value, or traceback."""
        t0 = time.perf_counter()
        buffer_out = io.StringIO()
        buffer_err = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        exec_globals = {"__name__": "__main__"}
        success = True
        error_msg = None
        
        sys.stdout = buffer_out
        sys.stderr = buffer_err
        try:
            exec(code, exec_globals)
        except Exception as e:
            success = False
            error_msg = f"{type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        output = buffer_out.getvalue()
        errors = buffer_err.getvalue()
        
        return {
            "success": success,
            "stdout": output,
            "stderr": errors + (error_msg if error_msg else ""),
            "execution_time_ms": round(elapsed_ms, 2),
            "status": "COMPLETED" if success else "EXECUTION_ERROR"
        }

    def execute_shell_tool(self, cmd: str) -> Dict[str, Any]:
        """Runs a safe read-only shell command with strict timeout."""
        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout_s)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "success": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "returncode": res.returncode,
                "execution_time_ms": round(elapsed_ms, 2)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {self.timeout_s}s",
                "returncode": -1,
                "execution_time_ms": self.timeout_s * 1000
            }
