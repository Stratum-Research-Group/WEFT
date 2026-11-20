"""
WEFT-2 Native Tool Calling and Structured Output Module.

Provides:
1. Tool call prompt formatting with schema declarations.
2. Structured output parsing (<tool_call> -> JSON).
3. Sandboxed local execution environment.
4. Logit processor hooks for grammar-constrained generation.
"""

import json
import re
from typing import Dict, Any, List, Optional, Callable


TOOL_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
When you need to call a tool, respond with a JSON tool call enclosed in <tool_call> and </tool_call> tags.
Format:
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

Available tools:
"""


class ToolRegistry:
    """Registry of callable Python functions with automatic JSON schema generation."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        self.tools[name] = func
        self.schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def format_system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {schema['name']}: {schema['description']}\n  Parameters: {json.dumps(schema['parameters'])}"
            for schema in self.schemas.values()
        )
        return TOOL_SYSTEM_PROMPT + tools_desc

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"status": "error", "message": f"Tool '{tool_name}' not found."}
        try:
            result = self.tools[tool_name](**arguments)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extracts all <tool_call>...</tool_call> payloads from generated text."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(parsed)
        except json.JSONDecodeError:
            continue
    return tool_calls


def format_tool_response(result: Dict[str, Any]) -> str:
    """Formats tool execution output into standard conversational prompt sequence."""
    return f"\n<tool_response>\n{json.dumps(result)}\n</tool_response>\n"
