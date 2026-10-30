"""Check your OpenMath-Nemotron endpoint (the v3 math checker).

    python scripts/check_openmath.py

Reads OPENMATH_BASE_URL / OPENMATH_MODEL / OPENMATH_API_KEY from .env, lists the served models,
solves one small physics problem, and runs the full cross-check against the Python calculator.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tesla_lab import mathcheck as MC  # noqa: E402
from tesla_lab.calc_runner import run_calc  # noqa: E402
from tesla_lab.config import get_settings  # noqa: E402
from tesla_lab.openmath import OpenMathClient  # noqa: E402
from tesla_lab.schema import CalcSpec, Claim, ClaimVerdict  # noqa: E402


def main() -> int:
    s = get_settings()
    if not s.openmath_base_url:
        print("OPENMATH_BASE_URL is not set. Start the GPU VM (deploy/openmath/README.md), open the SSH tunnel and set\n"
              "  OPENMATH_BASE_URL=http://localhost:8000/v1 in .env")
        return 1
    print(f"Endpoint: {s.openmath_base_url}\nModel:    {s.openmath_model}")
    client = OpenMathClient(s)
    try:
        ids = client.ping()
    except Exception as e:  # noqa: BLE001
        print(f"Could not reach the server: {e}\n(Is the VM running, the tunnel open, and vLLM finished loading?)")
        return 1
    print("Served models:", ids)
    if s.openmath_model not in ids:
        print(f"WARNING: {s.openmath_model} is not served here; set OPENMATH_MODEL to one of {ids}")
        return 1

    spec = CalcSpec(template="earth_resonator", params={"f0_hz": 7.83, "fractional_bandwidth": 0.2})
    v = ClaimVerdict(claim=Claim(id="C1", text="Earth resonator ring-down", calc=spec), calc=run_calc(spec),
                     verdict="MISSING PIECE", explanation="")
    prob = MC.problem_for_claim(v)
    print("\nProblem:", prob["problem"], "\nSolving (OpenMath thinks before answering; this can take a minute)...")
    comp = client.solve(prob["problem"])
    chk = MC.build_check("C1", prob, comp.text, reasoning=comp.reasoning, python_value=v.calc.metrics["energy_ring_down_s"],
                         tol=s.math_check_tolerance, source="openmath", model=comp.model, origin="template",
                         latency_s=comp.latency_s, completion_tokens=comp.completion_tokens)
    for st in chk.steps:
        print(f"  {st.n}. {st.text[:300]}")
    print(f"\nBoxed answer: {chk.final_answer_latex!r}")
    print(f"{chk.headline}\nlatency {comp.latency_s}s · {comp.completion_tokens} tokens"
          + (f" · {comp.completion_tokens / comp.latency_s:.0f} tokens/s" if comp.completion_tokens and comp.latency_s else ""))
    return 0 if chk.status == "agreed" else 2


if __name__ == "__main__":
    sys.exit(main())
