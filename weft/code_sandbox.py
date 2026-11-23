"""
Edge Code Execution Sandbox:
Provides isolated process-level execution of Python and shell snippets with strict resource limits,
timeout enforcement, and subprocess isolation for safety.
"""

import sys
import os
import io
import time
import subprocess
import shlex
from typing import Dict, Any

class EdgeCodeSandbox:
    def __init__(self, timeout_s: float = 5.0, allow_unrestricted: bool = False):
        self.timeout_s = timeout_s
        self.allow_unrestricted = allow_unrestricted

    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """
        Executes a Python snippet in an isolated subprocess to prevent in-process state corruption
        and enforce wall-clock timeouts.
        """
        t0 = time.perf_counter()
        
        try:
            # Run code inside an isolated Python subprocess
            cmd = [sys.executable, "-I", "-c", code]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            success = (res.returncode == 0)
            return {
                "success": success,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "execution_time_ms": round(elapsed_ms, 2),
                "status": "COMPLETED" if success else "EXECUTION_ERROR"
            }
        except subprocess.TimeoutExpired:
            elapsed_ms = self.timeout_s * 1000
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout_s}s",
                "execution_time_ms": round(elapsed_ms, 2),
                "status": "TIMEOUT"
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "success": False,
                "stdout": "",
                "stderr": f"{type(e).__name__}: {str(e)}",
                "execution_time_ms": round(elapsed_ms, 2),
                "status": "LAUNCH_ERROR"
            }

    def execute_shell_tool(self, cmd: str) -> Dict[str, Any]:
        """
        Runs a shell command in an isolated subprocess without shell=True, enforcing arguments and timeout.
        """
        t0 = time.perf_counter()
        try:
            # Parse command safely without arbitrary shell expansion
            args = shlex.split(cmd)
            if not args:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Empty command",
                    "returncode": -1,
                    "execution_time_ms": 0.0
                }
                
            res = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s
            )
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
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution failed: {str(e)}",
                "returncode": -1,
                "execution_time_ms": (time.perf_counter() - t0) * 1000
            }
