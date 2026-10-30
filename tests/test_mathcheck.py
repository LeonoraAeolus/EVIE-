"""v3 math checker: parsing, cross-check logic, mocked OpenMath client, skip path, mock-mode canned run."""
from types import SimpleNamespace

import pytest

from tesla_lab import latexmath as LM
from tesla_lab import mathcheck as MC
from tesla_lab.calc_runner import run_calc
from tesla_lab.config import get_settings
from tesla_lab.openmath import PROMPT, Completion, OpenMathClient
from tesla_lab.schema import CalcSpec, Claim, ClaimVerdict


# ----------------------------------------------------------------------------- helpers
def _settings(monkeypatch, url="http://gpu.example:8000/v1", mode="mock", **env):
    monkeypatch.setenv("OPENMATH_BASE_URL", url or "")
    monkeypatch.setenv("MATH_CHECK", env.pop("MATH_CHECK", "1"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return get_settings(mode)


def _verdict(cid="C1", template="earth_resonator", params=None, custom_code=None, verdict="MISSING PIECE"):
    spec = CalcSpec(template=template, params=params if params is not None else {"f0_hz": 7.83, "fractional_bandwidth": 0.2},
                    custom_code=custom_code)
    claim = Claim(id=cid, text="test claim", calc=spec)
    return ClaimVerdict(claim=claim, calc=run_calc(spec), verdict=verdict, explanation="x")


class FakeModel:
    """Stands in for OpenMathClient: answers from a dict keyed by a word in the problem."""

    model = "nvidia/OpenMath-Nemotron-7B"

    def __init__(self, answers: dict[str, str], fail_on: str | None = None):
        self.answers, self.fail_on, self.calls = answers, fail_on, []

    def solve(self, problem: str) -> Completion:
        self.calls.append(problem)
        if self.fail_on and self.fail_on in problem:
            raise TimeoutError("server did not answer")
        for key, text in self.answers.items():
            if key in problem:
                return Completion(text=text, latency_s=1.5, completion_tokens=321, model=self.model)
        return Completion(text="I am not sure.", model=self.model)


TAU_OK = ("<think>Q = 5, omega = 2 pi f0 ... so tau is about 0.1016</think>\n"
          "**Step 1:** \\(Q = 5\\).\n\n**Step 2:** \\[ \\tau = \\frac{Q}{2\\pi f_0} \\]\n\n"
          "**Step 3:** \\[ \\tau = \\frac{5}{2\\pi \\cdot 7.83} \\approx 0.1016 \\]\n\n\\[ \\boxed{0.1016} \\]")


# ----------------------------------------------------------------------------- latexmath
def test_boxed_extraction_handles_nested_braces():
    text = r"so \boxed{\frac{Q}{2\pi f_{0}}} and finally \boxed{0.102\text{ s}}"
    assert LM.boxed_answers(text) == [r"\frac{Q}{2\pi f_{0}}", r"0.102\text{ s}"]
    assert LM.last_boxed(text) == r"0.102\text{ s}"
    assert LM.last_boxed("no answer here") is None


@pytest.mark.parametrize("latex,value", [
    ("0.102", 0.102),
    (r"\tau = \frac{Q}{2\pi f_0} \approx 0.102 \text{ s}", 0.102),
    (r"18.1\%", 18.1),
    (r"1.35 \times 10^{-1}", 0.135),
    (r"2 \times 10^{9}\text{ W}", 2e9),
    (r"1,219.5", 1219.5),
    (r"\eta_{\text{max}} = 62.9\%", 62.9),
])
def test_parse_numeric_answers(latex, value):
    out = LM.parse_answer(latex)
    assert out["value"] == pytest.approx(value)
    assert out["symbolic"] is False


def test_parse_formula_answer_is_evaluated_at_the_inputs():
    out = LM.parse_answer(r"\dfrac{k^2 Q_1 Q_2}{\left(1+\sqrt{1+k^2 Q_1 Q_2}\right)^2}", {"k": 0.0066, "Q_1": 717, "Q_2": 590})
    assert out["symbolic"] is True
    u = 0.0066**2 * 717 * 590
    assert out["value"] == pytest.approx(u / (1 + (1 + u) ** 0.5) ** 2)
    assert LM.parse_answer(r"\frac{a}{b}")["value"] is None  # undefined symbols -> no number, not a guess


def test_split_reasoning_and_steps():
    think, sol = LM.split_reasoning(TAU_OK)
    assert "tau is about" in think and "<think>" not in sol
    steps = LM.split_steps(sol)
    assert len(steps) >= 3 and steps[0].startswith("**Step 1")
    assert LM.split_reasoning("<think>cut off mid")[1] == ""
    md = LM.to_markdown_math(r"\( x \) and \[ y = 1 \]")
    assert "$x$" in md and "$$y = 1$$" in md


# ----------------------------------------------------------------------------- cross-check
def test_cross_check_agrees_within_tolerance_and_rounding():
    assert MC.cross_check(0.1016, 0.102, 0.02)["status"] == "agreed"
    assert MC.cross_check(18.1, 18.12, 0.02)["status"] == "agreed"
    assert MC.cross_check(0.3305, 0.33, 0.0)["status"] == "agreed"  # inside 0.33's own rounding


def test_cross_check_flags_disagreement():
    out = MC.cross_check(0.64, 0.102, 0.02)
    assert out["status"] == "disagreed" and out["rel_diff"] > 1


def test_cross_check_accepts_and_states_unit_scaling():
    out = MC.cross_check(0.1812, 18.12, 0.02)  # fraction vs percent
    assert out["status"] == "agreed" and "x100" in out["scale_note"]
    assert out["checker_scaled"] == pytest.approx(18.12)


def test_cross_check_inconclusive_without_numbers():
    assert MC.cross_check(None, 1.0)["status"] == "inconclusive"
    assert MC.cross_check(1.0, None)["status"] == "inconclusive"


def test_symbolic_match():
    assert MC.symbolic_match("Q/(2*pi*f_0)", "Q/(2*pi*f_0)", {}) is True
    assert MC.symbolic_match("Q1*Q2*k**2/(sqrt(Q1*Q2*k**2+1)+1)**2", "100*k**2*Q_1*Q_2/(1+sqrt(1+k**2*Q_1*Q_2))**2", {}) is True
    assert MC.symbolic_match("Q/(pi*f_0)", "Q/(2*pi*f_0)", {}) is False
    assert MC.symbolic_match(None, "Q", {}) is None


def test_build_check_agreed_from_model_text():
    v = _verdict()
    prob = MC.problem_for_claim(v)
    assert prob and prob["target_metric"] == "energy_ring_down_s" and "7.83" in prob["problem"]
    chk = MC.build_check("C1", prob, TAU_OK, python_value=v.calc.metrics["energy_ring_down_s"], tol=0.02,
                         source="openmath", model="m", origin="template")
    assert chk.status == "agreed" and chk.checker_value == pytest.approx(0.1016)
    assert chk.steps and chk.key_equations and chk.reasoning_chars > 0
    assert chk.headline.startswith("Math checker agreed")


def test_build_check_disagreed_and_inconclusive():
    v = _verdict()
    prob = MC.problem_for_claim(v)
    bad = TAU_OK.replace(r"\boxed{0.1016}", r"\boxed{0.6385}")  # e.g. forgot the 2*pi
    chk = MC.build_check("C1", prob, bad, python_value=0.102, tol=0.02, source="openmath", model="m", origin="template")
    assert chk.status == "disagreed" and "DISAGREED" in chk.headline
    none = MC.build_check("C1", prob, "Step 1: hmm.", python_value=0.102, tol=0.02, source="openmath", model="m",
                          origin="template")
    assert none.status == "inconclusive"
    cut = MC.build_check("C1", prob, "<think>long thinking that never ends", python_value=0.102, tol=0.02,
                         source="openmath", model="m", origin="template")
    assert cut.status == "inconclusive" and "ran out of tokens" in cut.note


def test_problem_builders_cover_every_vetted_template():
    from tesla_lab.calcs import TEMPLATES
    params = {
        "resonant_coupling": {"claimed_distance_m": 2.0}, "earth_resonator": {"f0_hz": 7.83, "fractional_bandwidth": 0.2},
        "energy_balance": {"inputs": {"a": 10}, "outputs": {"b": 4}}, "electrolysis_loop": {},
        "disc_turbine": {"claimed_efficiency": 0.4}, "carnot_limit": {"t_hot_k": 600, "t_cold_k": 300, "claimed_efficiency": 0.3},
        "broadcast_spreading": {},
    }
    for name in TEMPLATES:
        v = _verdict(template=name, params=params[name])
        prob = MC.problem_for_claim(v)
        assert prob, name
        assert MC._num(v.calc.metrics[prob["target_metric"]]) is not None, name
        # the reference formula, evaluated at the problem's variables, reproduces the Python number
        if prob["reference"]:
            ref = MC.sym(prob["reference"])
            val = MC.LM.evaluate(ref, prob["variables"])
            assert MC.cross_check(val, float(v.calc.metrics[prob["target_metric"]]), 0.02)["status"] == "agreed", name


# ----------------------------------------------------------------------------- client (mocked)
def test_openmath_client_uses_model_card_prompt(monkeypatch):
    s = _settings(monkeypatch, OPENMATH_MODEL="nvidia/OpenMath-Nemotron-14B", OPENMATH_TEMPERATURE="0.6")
    seen = {}

    def create(**kw):
        seen.update(kw)
        msg = SimpleNamespace(content=r"Step 1. \boxed{4}", reasoning_content="thinking")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(completion_tokens=12))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    out = OpenMathClient(s, client=fake).solve("What is 2+2?")
    assert seen["model"] == "nvidia/OpenMath-Nemotron-14B" and seen["temperature"] == 0.6
    assert seen["messages"] == [{"role": "user", "content": PROMPT.format(problem="What is 2+2?")}]
    assert seen["messages"][0]["content"].startswith("Solve the following math problem. Make sure to put the answer "
                                                     "(and only answer) inside \\boxed{}.")
    assert out.text.endswith(r"\boxed{4}") and out.reasoning == "thinking" and out.completion_tokens == 12


def test_run_checks_live_with_mocked_model(monkeypatch):
    s = _settings(monkeypatch, mode="live")
    good, bad = _verdict("C1"), _verdict("C2", template="electrolysis_loop", params={}, verdict="BREAKS PHYSICS")
    model = FakeModel({"resonant cavity": TAU_OK,
                       "electrolyser": r"**Step 1.** \[G = \frac{241.8 \cdot 0.25 \cdot 0.9}{285.8/0.7}\] \boxed{0.5}"})
    info = MC.run_checks([good, bad], None, s, model=model)
    assert info.source == "openmath" and info.enabled
    assert good.math_check.status == "agreed" and good.math_check.latency_s == 1.5
    assert bad.math_check.status == "disagreed"
    assert bad.verdict == "BREAKS PHYSICS"  # the checker flags; it never rewrites the verdict
    assert info.counts == {"agreed": 1, "disagreed": 1} and "1 disagreed" in info.line


def test_run_checks_model_failure_is_an_error_not_a_crash(monkeypatch):
    s = _settings(monkeypatch, mode="live")
    v = _verdict()
    info = MC.run_checks([v], None, s, model=FakeModel({}, fail_on="cavity"))
    assert v.math_check.status == "error" and "TimeoutError" in v.math_check.note
    assert info.counts == {"error": 1}


def test_custom_code_claims_get_a_problem_from_the_lead_model(monkeypatch):
    s = _settings(monkeypatch, mode="live")
    code = "result = {'metrics': {'tower_power_needed_gw': 2.0}, 'summary': 'x', 'suggested_verdict': 'BREAKS PHYSICS'}"
    v = _verdict("C3", template=None, params={}, custom_code=code, verdict="BREAKS PHYSICS")
    assert v.calc.ok and MC.problem_for_claim(v) is None

    class FakeLead:
        def json_call(self, system, user, **kw):
            assert "tower_power_needed_gw" in user
            return {"target_metric": "tower_power_needed_gw", "quantity": "tower power", "unit": "GW",
                    "problem": "One million homes use 1000 W each through a 50% link. How many GW must the tower send?"}

    model = FakeModel({"million homes": r"\[P = \frac{Np}{\eta}\] \boxed{2}"})
    MC.run_checks([v], None, s, model=model, problem_writer=MC.nemotron_problem_writer(FakeLead()))
    assert v.math_check.status == "agreed" and v.math_check.problem_origin == "nemotron"

    class BadLead:
        def json_call(self, *a, **k):
            return {"target_metric": "not_a_metric", "problem": "x"}

    v2 = _verdict("C3", template=None, params={}, custom_code=code)
    MC.run_checks([v2], None, s, model=model, problem_writer=MC.nemotron_problem_writer(BadLead()))
    assert v2.math_check.status == "inconclusive"


# ----------------------------------------------------------------------------- skip path
def test_skip_path_live_without_endpoint(monkeypatch):
    s = _settings(monkeypatch, url="", mode="live")
    assert not s.openmath_configured
    v = _verdict()
    info = MC.run_checks([v], None, s, model=FakeModel({"cavity": TAU_OK}))
    assert info.enabled is False and "OPENMATH_BASE_URL not set" in info.reason
    assert v.math_check.status == "skipped" and v.math_check.steps == []


def test_math_check_can_be_turned_off(monkeypatch):
    s = _settings(monkeypatch, mode="mock", MATH_CHECK="0")
    v = _verdict()
    info = MC.run_checks([v], None, s, demo_id="wardenclyffe")
    assert info.enabled is False and v.math_check.status == "skipped"


def test_pipeline_live_skips_checker_gracefully(monkeypatch):
    """No endpoint: the live pipeline's checker pass marks everything skipped and adds one note."""
    from tesla_lab.pipeline import math_check_pass
    s = _settings(monkeypatch, url="", mode="live")
    v, notes = _verdict(), []
    info = math_check_pass([v], None, s, lambda a, b: None, notes)
    assert info.enabled is False and v.math_check.status == "skipped"
    assert any("OPENMATH_BASE_URL not set" in n for n in notes)


# ----------------------------------------------------------------------------- mock mode (canned)
@pytest.mark.parametrize("demo", ["Wardenclyffe", "water car", "Tesla turbine"])
def test_mock_mode_shows_canned_derivations_clearly_labelled(monkeypatch, demo):
    from tesla_lab.pipeline import run
    s = _settings(monkeypatch, url="", mode="mock")
    r = run(demo, s, image_dir=None)
    assert r.math_checker.source == "canned" and "CANNED" in r.math_checker.reason
    assert any("canned derivations" in n.lower() for n in r.notes)
    for v in r.verdicts:
        assert v.math_check.source == "canned" and v.math_check.steps, v.claim.id
        assert v.math_check.status == "agreed", (v.claim.id, v.math_check.headline)
    assert r.spec.math_check.status == "agreed"
    assert "Math checker (canned derivations): 4 agreed." in r.overall
    js = r.model_dump()
    assert js["verdicts"][0]["math_check"]["steps"][0]["text"]
    assert js["spec"]["math_check"]["final_answer_latex"]


def test_mock_mode_with_endpoint_uses_the_real_model_on_canned_problems(monkeypatch):
    """Research stays canned, but OpenMath is called for real (here: a fake) when OPENMATH_BASE_URL is set."""
    from tesla_lab.mock import pick_demo
    s = _settings(monkeypatch, mode="mock")
    demo, _ = pick_demo("Wardenclyffe")
    verdicts = []
    for c in demo["claims"]:
        claim = Claim(**c)
        verdicts.append(ClaimVerdict(claim=claim, calc=run_calc(claim.calc), verdict="WORKS", explanation=""))
    model = FakeModel({"resonant cavity": TAU_OK, "1,000,000 homes": r"\boxed{2}"})
    info = MC.run_checks(verdicts, None, s, demo_id="wardenclyffe", model=model)
    assert info.source == "openmath" and len(model.calls) == 3
    by_id = {v.claim.id: v.math_check for v in verdicts}
    assert by_id["C2"].status == "agreed" and by_id["C3"].status == "agreed"
    assert by_id["C3"].problem_origin == "canned" and by_id["C3"].source == "openmath"
    assert by_id["C1"].status == "inconclusive"  # the fake had no answer for it: reported, never guessed


def test_real_model_output_cut_off_mid_thought_is_inconclusive():
    """Real output: OpenMath-Nemotron-1.5B (CPU, transformers, temperature 0.6) stopped at a 2,500-token cap while
    still inside <think>. The checker must say so, not guess an answer."""
    from pathlib import Path
    text = (Path(__file__).parent / "fixtures" / "openmath_1p5b_truncated.txt").read_text()
    assert text.lstrip().startswith("<think>")
    v = _verdict()
    chk = MC.build_check("C2", MC.problem_for_claim(v), text, python_value=0.102, tol=0.02, source="openmath",
                         model="nvidia/OpenMath-Nemotron-1.5B", origin="template")
    assert chk.status == "inconclusive" and "ran out of tokens" in chk.note and chk.reasoning_chars > 5000
