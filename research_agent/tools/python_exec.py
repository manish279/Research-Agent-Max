from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from research_agent.tools.base import ToolResult


BLOCKED_IMPORTS = {
    "ctypes",
    "http",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
BLOCKED_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


class PythonSafetyError(ValueError):
    pass


class SafePythonExecutor:
    """Run small analysis snippets in a subprocess after AST safety checks."""

    def __init__(self, timeout_seconds: int = 12) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, code: str, context: dict[str, Any] | None = None) -> ToolResult:
        try:
            validate_python_safety(code)
        except PythonSafetyError as exc:
            return ToolResult(ok=False, error=str(exc), data={"stdout": "", "stderr": ""})

        runner = _build_runner(code, context or {})
        with tempfile.TemporaryDirectory(prefix="odc_agent_py_") as tmp:
            runner_path = Path(tmp) / "runner.py"
            runner_path.write_text(runner, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, str(runner_path)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(ok=False, error=f"Python execution timed out after {self.timeout_seconds}s")

        return ToolResult(
            ok=completed.returncode == 0,
            data={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            },
            error=None if completed.returncode == 0 else "Python snippet failed",
        )


def validate_python_safety(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PythonSafetyError(f"Syntax error in generated Python: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_IMPORTS:
                    raise PythonSafetyError(f"Blocked import in generated Python: {root}")
        elif isinstance(node, ast.ImportFrom):
            # Check the source module, e.g. `from os import path` has module="os"
            if node.module:
                root = node.module.split(".")[0]
                if root in BLOCKED_IMPORTS:
                    raise PythonSafetyError(f"Blocked import in generated Python: {root}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                raise PythonSafetyError(f"Blocked Python call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                raise PythonSafetyError("Blocked dunder attribute call in generated Python")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PythonSafetyError("Blocked dunder attribute access in generated Python")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise PythonSafetyError("Blocked dunder name access in generated Python")


def _build_runner(code: str, context: dict[str, Any]) -> str:
    context_json = json.dumps(context, default=str)
    return f"""
import json
import math
import statistics

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

context = json.loads({context_json!r})

{code}
"""
