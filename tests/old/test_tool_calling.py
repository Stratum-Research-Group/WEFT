import pytest
import json
from weft.tool_calling import ToolRegistry, extract_tool_calls, format_tool_response


def test_tool_registry_registration_and_execution():
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    registry.register(
        name="calculator_add",
        description="Add two numbers together.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        func=add,
    )

    sys_prompt = registry.format_system_prompt()
    assert "calculator_add" in sys_prompt
    assert "Add two numbers together." in sys_prompt

    # Test successful execution
    res = registry.execute("calculator_add", {"a": 12, "b": 30})
    assert res["status"] == "success"
    assert res["result"] == 42

    # Test unknown tool
    err_res = registry.execute("unknown_tool", {})
    assert err_res["status"] == "error"
    assert "not found" in err_res["message"]


def test_extract_tool_calls():
    sample_text = """
    Let me check the weather for Paris.
    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}
    </tool_call>
    I am waiting for the tool output.
    """
    calls = extract_tool_calls(sample_text)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"]["city"] == "Paris"


def test_format_tool_response():
    resp = {"temperature": 18, "condition": "sunny"}
    formatted = format_tool_response(resp)
    assert "<tool_response>" in formatted
    assert "</tool_response>" in formatted
    assert '"temperature": 18' in formatted
