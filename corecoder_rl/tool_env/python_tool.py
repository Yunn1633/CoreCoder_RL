"""Restricted Python tool used by tool-use RL smoke runs.

This is intentionally small: it is meant for arithmetic/data calculation tasks,
not for general code execution.
"""

from __future__ import annotations

import ast
import json
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any

PYTHON_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run a short, safe Python calculation. Use this for arithmetic, algebraic checking, or small data calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code for a small calculation. Print the final result, or leave a final expression as the last line.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

_DENIED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "breakpoint",
    "globals", "locals", "vars", "dir", "help", "getattr", "setattr", "delattr",
    "memoryview", "classmethod", "staticmethod", "super", "type", "object",
}
_DENIED_ATTR_PREFIX = "__"
_ALLOWED_IMPORTS = set()


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str = ""
    timeout: bool = False

    def to_message_content(self) -> str:
        payload = {"ok": self.ok, "output": self.output}
        if self.error:
            payload["error"] = self.error
        if self.timeout:
            payload["timeout"] = True
        return json.dumps(payload, ensure_ascii=False)


def _validate_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".", 1)[0] for alias in getattr(node, "names", [])]
            if any(name not in _ALLOWED_IMPORTS for name in names):
                raise ValueError("imports are disabled in the python tool")
        if isinstance(node, ast.Name) and node.id in _DENIED_NAMES:
            raise ValueError(f"name '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith(_DENIED_ATTR_PREFIX):
            raise ValueError("dunder attributes are not allowed")
        if isinstance(node, (ast.With, ast.AsyncWith, ast.Lambda, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"{type(node).__name__} is not allowed in the python tool")


def _wrap_code(code: str) -> str:
    tree = ast.parse(code, mode="exec")
    body = list(tree.body)
    if body and isinstance(body[-1], ast.Expr):
        expr = ast.Expression(body[-1].value)
        ast.fix_missing_locations(expr)
        prefix = ast.Module(body=body[:-1], type_ignores=[])
        ast.fix_missing_locations(prefix)
        prefix_code = ast.unparse(prefix).strip()
        expr_code = ast.unparse(expr.body).strip()
        code = (prefix_code + "\n" if prefix_code else "") + f"print({expr_code})"
    indented = textwrap.indent(code, "    ")
    return textwrap.dedent(f"""
    import math, statistics, fractions, decimal
    from fractions import Fraction
    from decimal import Decimal
    SAFE_BUILTINS = {{
        'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
        'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
        'int': int, 'float': float, 'str': str, 'repr': repr, 'print': print,
        'list': list, 'tuple': tuple, 'dict': dict, 'set': set,
        'sorted': sorted, 'pow': pow,
    }}
    globals()['__builtins__'] = SAFE_BUILTINS
{indented}
    """)


def execute_python_tool(code: str, timeout: float = 3.0) -> ToolResult:
    code = str(code or "").strip()
    if not code:
        return ToolResult(ok=False, output="", error="empty code")
    try:
        _validate_ast(code)
        wrapped = _wrap_code(code)
    except Exception as exc:
        return ToolResult(ok=False, output="", error=str(exc))
    try:
        proc = subprocess.run(
            ["/root/miniconda3/envs/openclaw-rl/bin/python", "-I", "-S", "-c", wrapped],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output="", error=f"timeout after {timeout}s", timeout=True)
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        return ToolResult(ok=False, output=stdout[-2000:], error=stderr[-2000:] or f"returncode={proc.returncode}")
    return ToolResult(ok=True, output=stdout[-4000:])


def parse_python_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return str(arguments.get("code", ""))
    if isinstance(arguments, str):
        try:
            data = json.loads(arguments)
            if isinstance(data, dict):
                return str(data.get("code", ""))
        except json.JSONDecodeError:
            return arguments
    return ""
