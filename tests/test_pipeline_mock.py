import json

import pytest

from tesla_lab.config import get_settings
from tesla_lab.mock import DEMOS
from tesla_lab.pipeline import apply_guardrails, run
from tesla_lab.schema import CalcResult, Report


@pytest.mark.parametrize("demo", DEMOS, ids=[d["id"] for d in DEMOS])
def test_mock_pipeline_end_to_end(demo):
    steps = []
    report = run(demo["input"], get_settings("mock"), on_step=lambda s, m: steps.append(s))
    assert report.mode == "mock"
    assert {"research", "extract", "math"} <= set(steps)
    assert report.verdicts and all(v.calc.ok for v in report.verdicts), [v.calc.error for v in report.verdicts]
    assert all("{" not in v.explanation for v in report.verdicts), "unfilled placeholder"
    ids = {s.id for s in report.sources}
    for v in report.verdicts:
        assert set(v.source_ids) <= ids
    Report.model_validate_json(json.dumps(report.model_dump()))  # JSON round-trip


def test_expected_verdicts():
    s = get_settings("mock")
    wc = run(DEMOS[0]["input"], s)
    assert [v.verdict for v in wc.verdicts] == ["WORKS", "MISSING PIECE", "BREAKS PHYSICS"]
    assert wc.concept is not None
    water = run(DEMOS[1]["input"], s)
    assert water.verdicts[0].verdict == "BREAKS PHYSICS"


def test_unknown_claim_falls_back_with_note():
    r = run("A perpetual motion machine made of magnets", get_settings("mock"))
    assert any("closest" in n for n in r.notes)


def test_guardrail_overrules_model():
    calc = CalcResult(ok=True, flags={"violates_energy_conservation": True}, suggested_verdict="BREAKS PHYSICS")
    v, note = apply_guardrails("WORKS", calc)
    assert v == "BREAKS PHYSICS" and note
