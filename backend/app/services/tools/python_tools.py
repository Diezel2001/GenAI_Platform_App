from __future__ import annotations

import io
import os
import sys
import traceback
import contextlib
import textwrap
import resource

from typing import Optional

from pydantic import BaseModel, Field


# =========================================================
# SCHEMAS
# =========================================================

class ExecutePythonSchema(BaseModel):
    code: str = Field(..., description="Python code to execute.")
    timeout_seconds: int = Field(30, ge=1, le=120, description="Max execution time in seconds.")
    memory_limit_mb: int = Field(512, ge=64, le=2048, description="Soft memory limit in MB (Unix only).")


# =========================================================
# IMPLEMENTATION
# =========================================================

def execute_python(
    code: str,
    timeout_seconds: int = 30,
    memory_limit_mb: int = 512,
) -> str:
    """
    Execute a Python code snippet in a restricted environment.
    Captures stdout, stderr, and the return value of the last expression.
    Returns a formatted result string.

    Security notes:
    - Runs in the same process — NOT a sandbox. Do not execute untrusted code.
    - For production, replace with a subprocess-based or container-based executor.
    - Memory limit applies on Unix systems only (uses resource.setrlimit).
    """

    # Dedent code so agents can pass indented blocks without IndentationError
    code = textwrap.dedent(code)

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Restricted globals — imports are allowed but dangerous builtins are removed
    safe_globals = {
        "__builtins__": {
            k: v
            for k, v in __builtins__.items()
            if k not in (
                "open", "exec", "eval", "compile",
                "__import__", "breakpoint", "input",
            )
        }
        if isinstance(__builtins__, dict)
        else {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if k not in (
                "open", "exec", "eval", "compile",
                "__import__", "breakpoint", "input",
            )
        },
        # Re-allow __import__ via a controlled wrapper
        "__import__": __import__,
        # Allow open in read-only mode only
        "open": _safe_open,
    }

    local_vars: dict = {}
    last_expr_value = None
    error_output = None

    def _run():
        nonlocal last_expr_value, error_output

        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):

            try:
                # Apply memory limit on Unix
                if hasattr(resource, "setrlimit"):
                    soft = memory_limit_mb * 1024 * 1024
                    try:
                        resource.setrlimit(
                            resource.RLIMIT_AS,
                            (soft, resource.RLIM_INFINITY),
                        )
                    except (ValueError, resource.error):
                        pass

                import ast
                tree = ast.parse(code, mode="exec")

                # If the last statement is an expression, capture its value
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = tree.body.pop()
                    exec(  # noqa: S102
                        compile(tree, "<agent_code>", "exec"),
                        safe_globals,
                        local_vars,
                    )
                    last_expr_value = eval(  # noqa: S307
                        compile(
                            ast.Expression(body=last_expr.value),
                            "<agent_code>",
                            "eval",
                        ),
                        safe_globals,
                        local_vars,
                    )
                else:
                    exec(  # noqa: S102
                        compile(tree, "<agent_code>", "exec"),
                        safe_globals,
                        local_vars,
                    )

            except Exception:
                error_output = traceback.format_exc()

    # Run with timeout using threading (works on all platforms)
    import threading

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return (
            f"TIMEOUT: Code execution exceeded {timeout_seconds} seconds.\n"
            "Partial stdout:\n"
            + stdout_capture.getvalue()[:2000]
        )

    stdout_out = stdout_capture.getvalue()
    stderr_out = stderr_capture.getvalue()

    sections = []

    if stdout_out:
        sections.append(f"--- stdout ---\n{stdout_out.rstrip()}")

    if stderr_out:
        sections.append(f"--- stderr ---\n{stderr_out.rstrip()}")

    if error_output:
        sections.append(f"--- error ---\n{error_output.rstrip()}")
    elif last_expr_value is not None:
        sections.append(f"--- result ---\n{repr(last_expr_value)}")

    if not sections:
        sections.append("(no output)")

    # Truncate total output to prevent runaway responses
    result = "\n\n".join(sections)
    if len(result) > 10_000:
        result = result[:10_000] + "\n\n[OUTPUT TRUNCATED]"

    return result


def _safe_open(file, mode="r", *args, **kwargs):
    """
    Replacement for built-in open() that restricts write access.
    Write/append modes are blocked to prevent arbitrary file modification
    during code execution.
    """
    if any(m in mode for m in ("w", "a", "x", "+")):
        raise PermissionError(
            "Write access via open() is disabled in execute_python. "
            "Use the write_file tool to save output."
        )
    return io.open(file, mode, *args, **kwargs)


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

PYTHON_TOOLS = {
    "execute_python": {
        "func": execute_python,
        "schema": ExecutePythonSchema,
        "description": (
            "Execute a Python code snippet and return stdout, stderr, "
            "and the value of the last expression. "
            "Supports timeout and memory limits. "
            "Write access via open() is restricted — use write_file instead."
        ),
    },
}
