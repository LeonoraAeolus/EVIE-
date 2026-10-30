"""Run a CalcSpec: a vetted template, or sandboxed custom code."""
from __future__ import annotations

import inspect

from .calcs import TEMPLATES
from .sandbox import run_code
from .schema import CalcResult, CalcSpec


def calc_source(spec: CalcSpec) -> str:
    """The exact code that produced the numbers, shown in the UI ("verified math")."""
    if spec.template and spec.template in TEMPLATES:
        return inspect.getsource(TEMPLATES[spec.template])
    return spec.custom_code or ""


def run_calc(spec: CalcSpec, backend: str = "local") -> CalcResult:
    if spec.template:
        fn = TEMPLATES.get(spec.template)
        if fn is None:
            return CalcResult(ok=False, template=spec.template, error=f"unknown template {spec.template}")
        try:
            out = fn(dict(spec.params))
        except Exception as e:  # bad params from the model
            return CalcResult(ok=False, template=spec.template, error=f"{type(e).__name__}: {e}")
        out = {k: v for k, v in out.items() if not k.startswith("_")}
        return CalcResult(**out)
    if spec.custom_code:
        res = run_code(spec.custom_code, backend=backend)
        if not res["ok"]:
            return CalcResult(ok=False, template="custom", error=res["error"])
        r = res["result"]
        flags = {k: bool(r[k]) for k in ("violates_energy_conservation", "violates_second_law") if k in r}
        return CalcResult(
            ok=True, template="custom", metrics=r.get("metrics", {}), series=r.get("series"),
            series_labels=r.get("series_labels"), flags=flags, summary=r.get("summary", ""),
            suggested_verdict=r.get("suggested_verdict"),
        )
    return CalcResult(ok=False, error="no calculation specified")
