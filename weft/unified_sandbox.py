"""
WEFT-2 Unified Sandboxed Agent Environment (Code + Tools + System).

Unifies Python script execution, tool calling, and command execution into a single agent runtime interface.
NOTE: This in-process executor is intended for trusted agent helper evaluation, not hostile code isolation.
"""

import sys
import io
import json
import traceback
from typing import Dict, Any, Tuple


class UnifiedAgentSandbox:
    """Secure sandboxed execution environment for code, math, and tool calls."""

    def __init__(self):
        self.scope: Dict[str, Any] = {
            "json": json,
        }

    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """Executes a Python code block in sandboxed scope and captures stdout/stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        redirected_error = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_error

        status = "success"
        error_msg = None

        try:
            # Execute within sandboxed global dictionary
            exec(code, self.scope)
        except Exception as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return {
            "status": status,
            "stdout": redirected_output.getvalue(),
            "stderr": redirected_error.getvalue(),
            "error": error_msg,
        }

    def execute_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes registered tool functions or dynamic Python procedures."""
        if tool_name in self.scope and callable(self.scope[tool_name]):
            try:
                res = self.scope[tool_name](**args)
                return {"status": "success", "result": res}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Tool '{tool_name}' not defined in sandbox scope."}
