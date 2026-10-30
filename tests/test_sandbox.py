import pytest

from tesla_lab.sandbox import UnsafeCode, check_code, run_code


def test_runs_math():
    r = run_code("import math\nresult = {'metrics': {'x': math.sqrt(16)}}")
    assert r["ok"] and r["result"]["metrics"]["x"] == 4.0


def test_runs_sympy():
    pytest.importorskip("sympy")
    r = run_code("import sympy as sp\nx = sp.symbols('x')\nresult = {'roots': sp.solve(x**2 - 4, x)}")
    assert r["ok"] and sorted(r["result"]["roots"]) == [-2.0, 2.0]


@pytest.mark.parametrize("code", [
    "import os\nresult = {}",
    "import subprocess\nresult = {}",
    "result = open('/etc/passwd').read()",
    "result = ().__class__.__bases__",
    "result = __import__('os')",
    "x = 1",  # no result
])
def test_rejects_unsafe(code):
    with pytest.raises(UnsafeCode):
        check_code(code)
    assert not run_code(code)["ok"]


def test_timeout():
    r = run_code("while True:\n    pass\nresult = {}")
    assert not r["ok"] and "timed out" in r["error"] or "error" in r
