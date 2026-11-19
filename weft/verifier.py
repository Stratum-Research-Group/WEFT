"""
Subprocess-Isolated Deterministic Rule & Code Execution Verifiers for WEFT RLVR.
Adapted from AETHER-1 / RLVR engine (Specification v1.1.0).

Performs objective, deterministic verification of student completions without teacher bias:
1. Python Code Execution: Runs code snippets against hidden unit test assertions in a subprocess.
2. Mathematical Proof / Numerical Extraction: Resolves \\boxed{} and #### delimiters, verifies exact parity.
3. Attractor / Repetition Loop Penalties: Detects repetitive word cycles or degenerate digit loops and penalizes them.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class RewardBreakdown:
    total_reward: float
    accuracy_reward: float
    format_reward: float
    loop_penalty: float
    details: Dict[str, Any]


class CodeRuleVerifier:
    """Executes generated Python functions against unit test assertions in an isolated subprocess."""

    def __init__(self, timeout_seconds: float = 2.0):
        self.timeout = timeout_seconds

    def clean_code(self, raw_text: str) -> str:
        # Extract markdown python code block if present
        if "```python" in raw_text:
            return raw_text.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_text:
            return raw_text.split("```")[1].split("```")[0].strip()
        return raw_text.strip()

    def verify_solution(self, generated_code: str, test_assertions: str) -> RewardBreakdown:
        code = self.clean_code(generated_code)
        
        # 1. Syntax check via AST
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError:
            return RewardBreakdown(
                total_reward=0.0,
                accuracy_reward=0.0,
                format_reward=0.0,
                loop_penalty=0.0,
                details={"status": "syntax_error"}
            )

        # 2. Subprocess execution with explicit test assertion sentinel
        # Prevent premature exit bypass (e.g. SystemExit(0) or sys.exit(0))
        # by wrapping execution in a test harness that reports success via stdout sentinel
        harness = (
            f"import sys\n\n"
            f"# Generated candidate code executed in isolated namespace\n"
            f"user_scope = {{}}\n"
            f"code_str = {repr(code)}\n"
            f"test_str = {repr(test_assertions)}\n\n"
            f"try:\n"
            f"    exec(code_str, user_scope)\n"
            f"    exec(test_str, user_scope)\n"
            f"    print('__WEFT_TEST_PASSED_SENTINEL__')\n"
            f"except SystemExit as e:\n"
            f"    sys.exit(1 if e.code != 0 else 2)\n"
            f"except Exception:\n"
            f"    sys.exit(1)\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(harness)
            tmp_path = f.name

        try:
            res = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            passed = (res.returncode == 0) and ("__WEFT_TEST_PASSED_SENTINEL__" in res.stdout)
        except subprocess.TimeoutExpired:
            passed = False
            status = "timeout"
        except Exception:
            passed = False
            status = "subprocess_error"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if passed:
            return RewardBreakdown(
                total_reward=1.0,
                accuracy_reward=0.8,
                format_reward=0.2,
                loop_penalty=0.0,
                details={"status": "passed_all_tests"}
            )
        else:
            return RewardBreakdown(
                total_reward=0.2 if syntax_valid else 0.0,
                accuracy_reward=0.0,
                format_reward=0.2 if syntax_valid else 0.0,
                loop_penalty=0.0,
                details={"status": "failed_assertions"}
            )


class MathRuleVerifier:
    """Deterministic rule-based verifier for mathematical step-by-step reasoning."""

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        if not text:
            return None
        target = text
        if "</think>" in text:
            post = text.split("</think>")[-1].strip()
            if post:
                target = post

        # Check for #### <num>
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", target)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Check for \\boxed{<num>}
        match = re.search(r"\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}", target)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Standalone trailing number
        cleaned = re.sub(r"(?:^|\n)\s*\d+[\.\)]\s+", " ", target)
        numbers = re.findall(r"(?<![a-zA-Z])-?[\d,]+(?:\.\d+)?(?![a-zA-Z])", cleaned)
        if numbers:
            try:
                val = numbers[-1].replace(",", "").rstrip(".")
                if val:
                    return float(val)
            except ValueError:
                pass
        return None

    @staticmethod
    def detect_loop(text: str) -> bool:
        # Digit attractor: 8+ repeating digits
        if re.search(r"(\d)\1{7,}", text):
            return True
        # Phrase repetition
        words = text.split()
        if len(words) >= 12:
            for n in range(3, 7):
                for i in range(len(words) - 2 * n):
                    if words[i : i + n] == words[i + n : i + 2 * n]:
                        if i + 3 * n <= len(words) and words[i + 2 * n : i + 3 * n] == words[i : i + n]:
                            return True
        return False

    def verify_solution(
        self,
        completion_text: str,
        ground_truth_numeric: float,
        tol: float = 1e-4,
    ) -> RewardBreakdown:
        format_reward = 0.0
        accuracy_reward = 0.0
        loop_penalty = 0.0

        # Check format tags
        has_think_open = "<think>" in completion_text
        has_think_close = "</think>" in completion_text
        has_box_or_hash = ("####" in completion_text) or ("\\boxed" in completion_text)

        if has_think_open and has_think_close:
            format_reward += 0.3
        elif has_think_open:
            format_reward += 0.1

        if has_box_or_hash:
            format_reward += 0.2

        if self.detect_loop(completion_text):
            loop_penalty = -0.5

        # Check numerical accuracy
        extracted = self.extract_number(completion_text)
        if extracted is not None:
            if abs(extracted - ground_truth_numeric) <= tol:
                accuracy_reward = 1.0

        total_reward = max(0.0, format_reward + accuracy_reward + loop_penalty)
        return RewardBreakdown(
            total_reward=total_reward,
            accuracy_reward=accuracy_reward,
            format_reward=format_reward,
            loop_penalty=loop_penalty,
            details={"extracted": extracted, "ground_truth": ground_truth_numeric},
        )

