r"""v3 math checker: OpenMath-Nemotron derives, Python computes, we compare.

For every claim (and for the product spec) we write a self-contained maths problem with the
same inputs the Python calculator used:
  * vetted templates and design calculators -> deterministic problem builders below
  * custom sandboxed code -> Nemotron 3 Super (the lead) writes the problem (live mode), or a
    canned problem (mock mode)
OpenMath-Nemotron solves it step by step and boxes a final answer. We parse the boxed answer
(number or formula, via latexmath.py), evaluate it at the same inputs, and compare it with the
Python number:
  agreed        |checker - python| <= tolerance (relative, default 2%) or within the python
                value's own rounding; a x100 / x1000 unit slip (fraction vs percent, m vs mm)
                is accepted and stated
  disagreed     outside tolerance -> flagged on the verdict, the verdict itself is NOT changed
                (Python is the calculator of record; a disagreement means "a human should look")
  inconclusive  no boxed answer, unparseable, or no numeric python value to compare
  skipped       checker not configured / turned off / nothing to check
  error         the call to the model failed
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import latexmath as LM
from .config import Settings
from .schema import ClaimVerdict, DerivationStep, MathCheck, MathCheckerInfo, ProductSpec

MU0 = 4e-7 * math.pi
SCALES = [(1.0, ""), (100.0, "x100 (fraction vs percent)"), (0.01, "/100 (percent vs fraction)"),
          (1000.0, "x1000 (unit prefix)"), (0.001, "/1000 (unit prefix)")]


# ----------------------------------------------------------------------------- problems
def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _p(params: dict, key: str, default: float) -> float:
    v = params.get(key, default)
    return float(default if v is None else v)


def problem_for_claim(c: ClaimVerdict) -> dict | None:
    r"""Deterministic problem for a vetted-template claim, or None (custom code)."""
    t, p, m = c.claim.calc.template, c.claim.calc.params or {}, c.calc.metrics or {}
    if not c.calc.ok or not t:
        return None
    if t == "resonant_coupling":
        r1, r2, a = _p(p, "r_tx_m", 0.3), _p(p, "r_rx_m", 0.3), _p(p, "wire_radius_m", 0.003)
        q1, q2, d = _p(p, "q_tx", 950), _p(p, "q_rx", 950), _p(p, "claimed_distance_m", 2.0)
        mm = MU0 * math.pi * r1**2 * r2**2 / (2 * (d**2 + r1**2) ** 1.5)
        l1, l2 = MU0 * r1 * (math.log(8 * r1 / a) - 2), MU0 * r2 * (math.log(8 * r2 / a) - 2)
        k = mm / math.sqrt(l1 * l2)
        return dict(
            quantity="maximum link efficiency at the claimed distance", target_metric="efficiency_at_claimed_distance_pct",
            unit="%", variables={"k": k, "Q_1": q1, "Q_2": q2, "Q1": q1, "Q2": q2, "r_1": r1, "r_2": r2, "a": a, "d": d},
            reference="100*k**2*Q_1*Q_2/(1+sqrt(1+k**2*Q_1*Q_2))**2",
            problem=(
                f"Two coaxial single-turn circular loops are tuned to the same resonant frequency and exchange power by "
                f"magnetic coupling. Transmitter loop radius r_1 = {r1:g} m, receiver loop radius r_2 = {r2:g} m, wire radius "
                f"a = {a:g} m, separation d = {d:g} m, unloaded quality factors Q_1 = {q1:g} and Q_2 = {q2:g}. "
                "Use these models: loop self-inductance L = mu_0 r (ln(8r/a) - 2); mutual inductance (dipole approximation) "
                "M = mu_0 pi r_1^2 r_2^2 / (2 (d^2 + r_1^2)^(3/2)); coupling coefficient k = M / sqrt(L_1 L_2); "
                "mu_0 = 4 pi x 10^-7 H/m. First derive the maximum power-transfer efficiency of the link, with the receiver "
                "load chosen optimally, as a formula in k, Q_1 and Q_2. Then evaluate it. Give the efficiency in percent."),
        )
    if t == "earth_resonator":
        f0 = _p(p, "f0_hz", 7.83)
        if "q_factor" in p:
            qtxt, q = f"The quality factor is Q = {float(p['q_factor']):g}.", float(p["q_factor"])
        else:
            fb = _p(p, "fractional_bandwidth", 0.2)
            qtxt, q = f"Its bandwidth is {fb:g} f_0 (full width at half maximum), so Q = f_0 / bandwidth.", 1 / fb
        return dict(
            quantity="energy ring-down time", target_metric="energy_ring_down_s", unit="s",
            variables={"Q": q, "f_0": f0, "f0": f0, "omega_0": 2 * math.pi * f0}, reference="Q/(2*pi*f_0)",
            problem=(f"A resonant cavity rings at f_0 = {f0:g} Hz. {qtxt} With Q defined as omega_0 x (stored energy) / "
                     "(power lost), the stored energy decays as exp(-t/tau). Derive tau in terms of Q and f_0, then compute "
                     "tau in seconds."),
        )
    if t == "energy_balance":
        if _num(m.get("out_over_in")) is None:
            return None
        ins, outs = p.get("inputs") or {}, p.get("outputs") or {}
        unit = p.get("unit", "W")
        fmt = lambda d: "; ".join(f"{k} = {float(v):g} {unit}" for k, v in d.items())  # noqa: E731
        return dict(
            quantity="energy out / energy in", target_metric="out_over_in", unit="", variables={},
            reference="",
            problem=(f"A process takes in: {fmt(ins)}. It delivers: {fmt(outs)}. Using conservation of energy, write "
                     "the overall efficiency as total energy out divided by total energy in, and evaluate it as a decimal."),
        )
    if t == "electrolysis_loop":
        ee, en, eg = _p(p, "electrolyser_eff", 0.7), _p(p, "engine_eff", 0.25), _p(p, "generator_eff", 0.9)
        return dict(
            quantity="loop gain (energy back / energy in)", target_metric="loop_gain_realistic", unit="",
            variables={"eta_e": ee, "eta_engine": en, "eta_g": eg, "HHV": 285.8, "LHV": 241.8},
            reference="LHV*eta_engine*eta_g*eta_e/HHV",
            problem=(f"A car splits 1 mol of liquid water with an electrolyser of efficiency eta_e = {ee:g}; splitting needs "
                     "the higher heating value HHV = 285.8 kJ/mol, so the electricity used is HHV / eta_e. It burns the "
                     f"hydrogen in an engine of efficiency eta_engine = {en:g}, which releases the lower heating value "
                     f"LHV = 241.8 kJ/mol as heat, and a generator of efficiency eta_g = {eg:g} turns the shaft work back "
                     "into electricity. Derive the loop gain G = (electricity returned) / (electricity used) and evaluate "
                     "it as a decimal."),
        )
    if t == "disc_turbine":
        rpm, nu = _p(p, "rpm", 10000), _p(p, "kinematic_viscosity_m2_s", 1.5e-5)
        omega = rpm * 2 * math.pi / 60
        return dict(
            quantity="boundary-layer thickness on the disc", target_metric="boundary_layer_mm", unit="mm",
            variables={"nu": nu, "omega": omega}, reference="1000*5*sqrt(nu/omega)",
            problem=(f"A smooth disc spins at {rpm:g} revolutions per minute in a fluid of kinematic viscosity "
                     f"nu = {nu:g} m^2/s. For a rotating disc (von Karman flow) the boundary-layer thickness is "
                     "delta = 5 sqrt(nu / omega). Convert the speed to omega in rad/s, then compute delta. "
                     "Give delta in millimetres."),
        )
    if t == "carnot_limit":
        th, tc = _p(p, "t_hot_k", 600), _p(p, "t_cold_k", 300)
        return dict(
            quantity="Carnot efficiency limit", target_metric="carnot_limit_pct", unit="%",
            variables={"T_h": th, "T_c": tc}, reference="100*(1-T_c/T_h)",
            problem=(f"A heat engine runs between a hot reservoir at T_h = {th:g} K and a cold reservoir at T_c = {tc:g} K. "
                     "Derive the maximum (Carnot) efficiency from the second law and evaluate it in percent."),
        )
    if t == "broadcast_spreading":
        ds = [float(x) for x in p.get("distances_km", [1, 10, 100, 1000, 5000])]
        key = f"received_at_{ds[-1]:g}km_w"
        if key not in m:
            return None
        ptx, arx = _p(p, "tx_power_w", 200e3), _p(p, "rx_area_m2", 1.0)
        return dict(
            quantity=f"power received at {ds[-1]:g} km", target_metric=key, unit="W",
            variables={"P": ptx, "A": arx, "d": ds[-1] * 1000}, reference="P*A/(2*pi*d**2)",
            problem=(f"A transmitter radiates P = {ptx:g} W spread evenly over a hemisphere. A receiver of area "
                     f"A = {arx:g} m^2 faces it at distance d = {ds[-1]:g} km. Derive the received power and evaluate it "
                     "in watts."),
        )
    return None


def _spec_value(spec: ProductSpec, name: str) -> float | None:
    for it in spec.specs:
        if it.name == name:
            return _num(it.value)
    return None


def problem_for_spec(spec: ProductSpec) -> dict | None:
    r"""The design calculator's key governing equation, re-derived."""
    if spec.error:
        return None
    r = spec.requirements or {}
    if spec.design == "wireless_power":
        u = _spec_value(spec, "Figure of merit k·√(Q1·Q2)")
        eff = _spec_value(spec, "Coil-to-coil efficiency (optimal load)")
        if u is None or eff is None:
            return None
        return dict(
            quantity="coil-to-coil efficiency at the design distance (optimal load)",
            target_metric="Coil-to-coil efficiency (optimal load)", unit="%", python_value=eff,
            variables={"U": u}, reference="100*U**2/(1+sqrt(1+U**2))**2",
            problem=("Two magnetically coupled resonators (series RLC transmitter driving a resonant receiver) have "
                     f"figure of merit U = k sqrt(Q_1 Q_2) = {u:g}. The receiver load resistance can be chosen freely. "
                     "Derive the load that maximises the power-transfer efficiency and the resulting maximum efficiency as "
                     "a function of U. Evaluate it in percent."),
        )
    if spec.design == "h2_backup":
        kg = _spec_value(spec, "Hydrogen needed")
        load, hours = _num(r.get("load_w")), _num(r.get("backup_hours"))
        fc = _num(r.get("fuel_cell_eff")) or 0.5
        if kg is None or load is None or hours is None:
            return None
        return dict(
            quantity="hydrogen needed per outage", target_metric="Hydrogen needed", unit="kg", python_value=kg,
            variables={"P": load, "t": hours, "eta": fc, "LHV": 33.33}, reference="P*t/1000/(eta*LHV)",
            problem=(f"A home backup system must supply an average load of P = {load:g} W for t = {hours:g} hours from "
                     f"hydrogen. The fuel cell converts hydrogen to electricity with efficiency eta = {fc:g} relative to the "
                     "lower heating value, LHV = 33.33 kWh per kg of hydrogen. Derive the mass of hydrogen needed and "
                     "evaluate it in kilograms."),
        )
    if spec.design == "disc_pump":
        ph = _spec_value(spec, "Hydraulic power ρgQH")
        q, h = _num(r.get("flow_lpm")), _num(r.get("head_m"))
        if ph is None or q is None or h is None:
            return None
        return dict(
            quantity="hydraulic power", target_metric="Hydraulic power ρgQH", unit="W", python_value=ph,
            variables={"rho": 998.0, "g": 9.81, "Q": q / 60000, "H": h}, reference="rho*g*Q*H",
            problem=(f"A pump moves water at {q:g} litres per minute against a head of H = {h:g} m. Take the water density "
                     "rho = 998 kg/m^3 and g = 9.81 m/s^2. Convert the flow to m^3/s, derive the hydraulic power the pump "
                     "delivers to the water, and evaluate it in watts."),
        )
    return None


# ----------------------------------------------------------------------------- compare
def _rounding_slack(v: float) -> float:
    r"""Half a unit in the last decimal place the python value shows (0.102 -> 0.0005)."""
    txt = repr(float(v))
    if "e" in txt or "." not in txt:
        return 0.0
    decimals = len(txt.split(".")[1].rstrip("0"))
    return 0.5 * 10 ** (-decimals) if 0 < decimals <= 4 else 0.0


def cross_check(checker: float | None, python: float | None, tol: float = 0.02) -> dict:
    r"""Compare the checker's number with Python's. Returns status, rel_diff, scale_note, checker_scaled."""
    if checker is None or python is None:
        return {"status": "inconclusive", "rel_diff": None, "scale_note": "", "checker_scaled": checker}
    best = None
    for factor, note in SCALES:
        c = checker * factor
        denom = max(abs(python), 1e-12)
        rel = abs(c - python) / denom
        ok = rel <= tol or abs(c - python) <= _rounding_slack(python)
        cand = (not ok, rel, factor != 1.0, c, note)
        if best is None or cand < best:
            best = cand
        if ok and factor == 1.0:
            break
    not_ok, rel, _, c, note = best
    if not_ok:  # report the unscaled difference when nothing matches
        rel = abs(checker - python) / max(abs(python), 1e-12)
        return {"status": "disagreed", "rel_diff": rel, "scale_note": "", "checker_scaled": checker}
    return {"status": "agreed", "rel_diff": rel, "scale_note": note, "checker_scaled": c}


def sym(expr: str):
    r"""sympify with every bare name as a Symbol (plain sympify turns 'Q' into SymPy's assumptions module)."""
    import re

    import sympy as sp
    funcs = {"sqrt": sp.sqrt, "pi": sp.pi, "log": sp.log, "exp": sp.exp, "E": sp.E, "sin": sp.sin, "cos": sp.cos}
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expr)) - set(funcs)
    local = {n: sp.Symbol(n) for n in names}
    local.update(funcs)
    return sp.parse_expr(expr, local_dict=local)


def symbolic_match(answer_expr: str | None, reference: str, variables: dict[str, float]) -> bool | None:
    r"""Is the model's boxed formula the same function as the calculator's? None if not comparable."""
    if not answer_expr or not reference:
        return None
    import sympy as sp
    try:
        a = sym(answer_expr)
        ref = sym(reference)
    except Exception:  # noqa: BLE001
        return None
    if not a.free_symbols:
        return None
    # align names loosely (Q1 ~ Q_1)
    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())  # noqa: E731
    ref_syms = {norm(s): s for s in ref.free_symbols}
    a = a.subs({s: ref_syms[norm(s)] for s in a.free_symbols if norm(s) in ref_syms})
    for scale in (1, 100, sp.Rational(1, 100), 1000, sp.Rational(1, 1000)):
        try:
            if sp.simplify(a * scale - ref) == 0:
                return True
        except Exception:  # noqa: BLE001
            return None
    return False


def _fmt(v: float | None) -> str:
    if v is None:
        return "?"
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.3g}"
    return f"{v:.4g}"


def headline(chk: MathCheck) -> str:
    u = f" {chk.unit}" if chk.unit and chk.unit != "%" else chk.unit
    if chk.status == "agreed":
        extra = f" ({chk.scale_note})" if chk.scale_note else ""
        return (f"Math checker agreed: derived {_fmt(chk.checker_value)}{u} vs Python {_fmt(chk.python_value)}{u}"
                f"{extra}.")
    if chk.status == "disagreed":
        pct = f"{100 * chk.rel_diff:.0f}% apart" if chk.rel_diff is not None else "apart"
        return (f"Math checker DISAGREED: derived {_fmt(chk.checker_value)}{u} vs Python {_fmt(chk.python_value)}{u} "
                f"({pct}). Review the derivation before trusting this number.")
    if chk.status == "inconclusive":
        return f"Math checker inconclusive: {chk.note or 'no comparable answer'}."
    if chk.status == "error":
        return f"Math checker error: {chk.note}."
    return f"Math checker skipped: {chk.note}."


def build_check(target: str, prob: dict, completion_text: str, *, reasoning: str = "", python_value: float | None,
                tol: float, source: str, model: str, origin: str, latency_s: float | None = None,
                completion_tokens: int | None = None) -> MathCheck:
    r"""Parse a model reply and compare it with python_value. Pure: no I/O (unit-tested)."""
    think, solution = LM.split_reasoning(completion_text)
    reasoning = reasoning or think
    body = solution or ""
    if not body and reasoning:  # ran out of tokens mid-thought
        body = reasoning[-4000:]
    boxed = LM.last_boxed(solution) or LM.last_boxed(completion_text)
    variables = prob.get("variables") or {}
    parsed = LM.parse_answer(boxed, variables)
    formula = _boxed_formula(boxed, variables)
    cmp = cross_check(parsed["value"], python_value, tol)
    note = ""
    if cmp["status"] == "inconclusive":
        note = parsed["error"] or ("no numeric Python value to compare" if python_value is None else "")
        if not solution and reasoning:
            note = "the model ran out of tokens before finishing (raise OPENMATH_MAX_TOKENS)"
    chunks = [c for c in LM.split_steps(_drop_trailing_boxed(body)) if c.strip()]
    steps = [DerivationStep(n=i + 1, text=LM.to_markdown_math(t)) for i, t in enumerate(chunks)]
    chk = MathCheck(
        target=target, status=cmp["status"], headline="", source=source, model=model, problem=prob["problem"],
        problem_origin=origin, quantity=prob.get("quantity", ""), target_metric=prob.get("target_metric", ""),
        unit=prob.get("unit", ""), steps=steps, key_equations=LM.display_equations(body),
        final_answer_latex=boxed or "", answer_expr=parsed["expr"] or "", answer_symbolic=parsed["symbolic"],
        checker_value=cmp["checker_scaled"], python_value=python_value, scale_note=cmp["scale_note"],
        rel_diff=None if cmp["rel_diff"] is None else round(cmp["rel_diff"], 5), tolerance=tol,
        symbolic_match=symbolic_match(formula, prob.get("reference", ""), variables) if formula else None,
        reasoning_chars=len(reasoning), latency_s=latency_s, completion_tokens=completion_tokens, note=note,
    )
    chk.headline = headline(chk)
    return chk


def _drop_trailing_boxed(body: str) -> str:
    r"""Remove a final display line that only holds the boxed answer (the UI shows it as 'Final answer')."""
    i = body.rfind("\\boxed")
    if i < 0:
        return body
    head = body[:i].rstrip()
    for opener, closer in (("\\[", "\\]"), ("$$", "$$")):
        if head.endswith(opener):
            brace = body.find("{", i)
            end = LM._match_brace(body, brace) if brace >= 0 else -1
            if end > 0 and body[end + 1:].strip() == closer:
                return head[: -len(opener)].rstrip()
    return body


def _boxed_formula(boxed: str | None, variables: dict) -> str | None:
    r"""A formula inside the boxed answer ('\tau = \frac{Q}{2\pi f_0} \approx 0.102' -> 'Q/(2*pi*f_0)')."""
    import re

    if not boxed:
        return None
    for part in re.split(r"\\approx|\\simeq|=", boxed):
        try:
            e = LM.latex_to_sympy(part, variables)
        except ValueError:
            continue
        if getattr(e, "free_symbols", None):
            # the left side of 'tau = ...' is just the name of the answer; skip lone symbols
            if len(e.free_symbols) == 1 and e.is_Symbol:
                continue
            return str(e)
    return None


def skipped(target: str, why: str, status: str = "skipped", source: str = "none") -> MathCheck:
    chk = MathCheck(target=target, status=status, headline="", note=why, source=source)
    chk.headline = headline(chk)
    return chk


# ----------------------------------------------------------------------------- canned (mock)
_CANNED_PATH = Path(__file__).parent / "mock" / "derivations.json"


def load_canned() -> dict:
    return json.loads(_CANNED_PATH.read_text(encoding="utf-8")) if _CANNED_PATH.exists() else {}


# ----------------------------------------------------------------------------- orchestration
ProblemWriter = Callable[[ClaimVerdict], "dict | None"]


def nemotron_problem_writer(llm) -> ProblemWriter:
    r"""Live mode: the lead model (Nemotron 3 Super) turns a custom calculation into a maths problem."""
    from . import prompts

    def write(c: ClaimVerdict) -> dict | None:
        numeric = {k: v for k, v in (c.calc.metrics or {}).items() if _num(v) is not None and not isinstance(v, bool)}
        if not numeric:
            return None
        out = llm.json_call(prompts.MATHCHECK_PROBLEM_SYSTEM,
                            f"Claim: {c.claim.text}\nEquation: {c.claim.equation}\n"
                            f"Python code that computed it:\n{c.claim.calc.custom_code or ''}\n"
                            f"Numeric results available: {json.dumps(list(numeric))}")
        key = out.get("target_metric")
        if key not in numeric or not out.get("problem"):
            return None
        return dict(problem=str(out["problem"]), target_metric=key, quantity=out.get("quantity", key),
                    unit=out.get("unit", ""), variables={}, reference="")

    return write


def info_for(s: Settings, mock: bool) -> MathCheckerInfo:
    if not s.math_check:
        return MathCheckerInfo(enabled=False, source="none", reason="MATH_CHECK=0")
    if s.openmath_configured:
        return MathCheckerInfo(enabled=True, source="openmath", model=s.openmath_model, base_url=s.openmath_base_url or "")
    if mock:
        return MathCheckerInfo(enabled=True, source="canned", model=f"{s.openmath_model} (canned)",
                               reason="OPENMATH_BASE_URL not set: showing CANNED derivations written in OpenMath's "
                                      "format for the demo claims. The Python comparison runs for real.")
    return MathCheckerInfo(enabled=False, source="none",
                           reason="OPENMATH_BASE_URL not set: math checker skipped (see deploy/openmath/README.md)")


def run_checks(verdicts: list[ClaimVerdict], spec: ProductSpec | None, s: Settings, *, demo_id: str | None = None,
               model: "object | None" = None, problem_writer: ProblemWriter | None = None,
               step: Callable[[str, str], None] = lambda a, b: None, max_workers: int = 4) -> MathCheckerInfo:
    r"""Attach a MathCheck to every verdict (and the spec). Never raises."""
    info = info_for(s, mock=demo_id is not None)
    targets: list[tuple[str, object, dict | None, str]] = []  # (target id, owner, problem, origin)
    canned = load_canned().get(demo_id or "", {}) if info.source == "canned" else {}
    for v in verdicts:
        prob, origin = problem_for_claim(v), "template"
        if prob is None and info.source == "canned" and v.claim.id in canned:
            prob, origin = canned[v.claim.id].get("problem_spec"), "canned"
        if prob is None and info.source == "openmath" and demo_id is not None:
            cp = load_canned().get(demo_id, {}).get(v.claim.id, {}).get("problem_spec")
            prob, origin = (cp, "canned") if cp else (None, "")
        if prob is None and info.source == "openmath" and problem_writer is not None:
            try:
                prob, origin = problem_writer(v), "nemotron"
            except Exception as exc:  # noqa: BLE001
                v.math_check = skipped(v.claim.id, f"lead model could not write a check problem ({exc})", "inconclusive")
                continue
        targets.append((v.claim.id, v, prob, origin))
    if spec is not None:
        targets.append((f"spec:{spec.design}", spec, problem_for_spec(spec), "design"))

    if not info.enabled:
        for tid, owner, _, _ in targets:
            owner.math_check = skipped(tid, info.reason)
        info.counts = {"skipped": len(targets)}
        info.line = "Math checker skipped (OpenMath endpoint not configured)." if s.math_check else "Math checker off."
        return info

    def python_value_for(owner, prob) -> float | None:
        if "python_value" in prob:
            return _num(prob["python_value"])
        return _num((owner.calc.metrics or {}).get(prob["target_metric"]))

    client = model
    if info.source == "openmath" and client is None:
        try:
            from .openmath import OpenMathClient
            client = OpenMathClient(s)
        except Exception as exc:  # noqa: BLE001
            for tid, owner, _, _ in targets:
                owner.math_check = skipped(tid, str(exc), "error")
            info.counts = {"error": len(targets)}
            info.line = f"Math checker error: {exc}"
            return info

    def one(item):
        tid, owner, prob, origin = item
        if prob is None:
            return owner, skipped(tid, "no check problem for this calculation", "inconclusive", info.source)
        pv = python_value_for(owner, prob)
        if info.source == "canned":
            entry = canned.get(tid.split(":")[0] if not tid.startswith("spec:") else tid)
            if not entry:
                return owner, skipped(tid, "no canned derivation for this claim", "inconclusive", "canned")
            return owner, build_check(tid, prob, entry["response"], python_value=pv, tol=s.math_check_tolerance,
                                      source="canned", model=info.model, origin=origin)
        try:
            comp = client.solve(prob["problem"])
        except Exception as exc:  # noqa: BLE001
            return owner, skipped(tid, f"OpenMath call failed: {type(exc).__name__}: {exc}"[:300], "error", "openmath")
        return owner, build_check(tid, prob, comp.text, reasoning=comp.reasoning, python_value=pv,
                                  tol=s.math_check_tolerance, source="openmath", model=comp.model or s.openmath_model,
                                  origin=origin, latency_s=comp.latency_s, completion_tokens=comp.completion_tokens)

    label = "canned derivations" if info.source == "canned" else info.model
    step("mathcheck", f"Math checker ({label}) is re-deriving {len(targets)} results step by step")
    workers = 1 if info.source == "canned" else max(1, min(max_workers, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for owner, chk in ex.map(one, targets):
            owner.math_check = chk
    counts = Counter(getattr(o, "math_check").status for _, o, _, _ in targets)
    info.counts = dict(counts)
    order = ["agreed", "disagreed", "inconclusive", "error", "skipped"]
    parts = [f"{counts[k]} {k}" for k in order if counts.get(k)]
    tag = " (canned derivations)" if info.source == "canned" else ""
    info.line = "Math checker" + tag + ": " + ", ".join(parts) + "."
    return info
