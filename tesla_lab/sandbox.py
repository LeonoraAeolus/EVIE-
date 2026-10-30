"""Run a small LLM-written calculation safely.

Two layers:
1. Static check (AST): only whitelisted imports (math, numpy, scipy, sympy, ...), no
   dunder access, no file / network / process / eval builtins.
2. Isolated subprocess: `python -I`, empty environment, temp working dir, CPU + memory
   limits (POSIX), wall-clock timeout. The code must assign a dict to `result`.

SANDBOX_BACKEND=nebius sends the same code to a Nebius Sandbox instead (optional, see
run_in_nebius_sandbox; written from the Sandboxes quickstart, NOT tested).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap

ALLOWED_IMPORTS = {"math", "cmath", "fractions", "decimal", "statistics", "numpy", "scipy", "sympy"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "help", "exit", "quit", "memoryview",
}
TIMEOUT_S = 10
MEM_BYTES = 768 * 1024 * 1024


class UnsafeCode(ValueError):
    pass


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise UnsafeCode(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                raise UnsafeCode(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise UnsafeCode(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise UnsafeCode(f"private attribute not allowed: {node.attr}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCode(f"dunder name not allowed: {node.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCode(f"construct not allowed: {type(node).__name__}")
    if "result" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}:
        raise UnsafeCode("code must assign a dict to `result`")


_RUNNER = """
import json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
except Exception:
    pass
_src = sys.stdin.read()
_ns = {{}}
exec(compile(_src, "<calc>", "exec"), _ns)
_r = _ns.get("result")
if not isinstance(_r, dict):
    raise SystemExit("result must be a dict")
def _clean(v):
    try:
        import sympy
        if isinstance(v, sympy.Basic):
            return float(v) if v.is_number else str(v)
    except Exception:
        pass
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {{str(k): _clean(x) for k, x in v.items()}}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)
print("__RESULT__" + json.dumps(_clean(_r)))
"""


def run_local(code: str, timeout_s: int = TIMEOUT_S) -> dict:
    check_code(code)
    runner = _RUNNER.format(cpu=timeout_s, mem=MEM_BYTES)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                input=textwrap.dedent(code),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": tmp},
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {timeout_s}s"}
    out = proc.stdout
    if proc.returncode != 0 or "__RESULT__" not in out:
        err = (proc.stderr or out).strip().splitlines()[-1:] or ["unknown error"]
        return {"ok": False, "error": err[0][:500]}
    return {"ok": True, "result": json.loads(out.split("__RESULT__", 1)[1])}


def run_in_nebius_sandbox(code: str, timeout_s: int = 60) -> dict:  # pragma: no cover
    """UNTESTED sketch from https://docs.tokenfactory.nebius.com/sandboxes/start/python-quickstart

    Needs `pip install contree-sdk`, NEBIUS_API_KEY and NEBIUS_PROJECT_ID. The quickstart
    uses `alpine:3.19`; pick an image with python3 (+ sympy if you use it).
    """
    import asyncio

    from contree_sdk import Contree  # type: ignore

    check_code(code)
    image_name = os.getenv("NEBIUS_SANDBOX_IMAGE", "python:3.12-slim")
    payload = _RUNNER.format(cpu=timeout_s, mem=MEM_BYTES).replace("sys.stdin.read()", repr(code))
    shell = "python3 - <<'PYEOF'\n" + payload + "\nPYEOF"

    async def go() -> dict:
        client = Contree()
        image = await client.images.use(image_name, strict=True)
        res = await image.run(shell=shell, disposable=True)
        out = res.stdout or ""
        if res.exit_code != 0 or "__RESULT__" not in out:
            return {"ok": False, "error": (res.stderr or out)[-500:]}
        return {"ok": True, "result": json.loads(out.split("__RESULT__", 1)[1])}

    return asyncio.run(go())


def run_code(code: str, backend: str = "local") -> dict:
    try:
        if backend == "nebius":
            return run_in_nebius_sandbox(code)
        return run_local(code)
    except UnsafeCode as e:
        return {"ok": False, "error": f"rejected by sandbox: {e}"}
