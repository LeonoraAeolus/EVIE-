import json

import pytest

from tesla_lab.config import get_settings
from tesla_lab.mock import DEMOS
from tesla_lab.pipeline import run
from tesla_lab.schema import Report


@pytest.mark.parametrize("demo", DEMOS, ids=[d["id"] for d in DEMOS])
def test_mock_demo_designs_a_product(demo):
    steps = []
    r = run(demo["input"], get_settings("mock"), on_step=lambda s, m: steps.append(s))
    assert {"archive", "design"} <= set(steps)
    assert r.spec is not None and r.spec.error is None and r.spec.headline and r.spec.bom
    assert r.concept.image_prompt == r.spec.render_prompt  # render follows the computed numbers
    assert any(v.archive_refs for v in r.verdicts)
    Report.model_validate_json(json.dumps(r.model_dump()))


def test_gap_search_runs_for_missing_pieces():
    s = get_settings("mock")
    wc = run(DEMOS[0]["input"], s)
    assert [g.claim_id for g in wc.gaps] == ["C2"]
    reqs = wc.gaps[0].requirements
    assert [r.rank for r in reqs] == list(range(1, len(reqs) + 1))
    assert reqs == sorted(reqs, key=lambda r: -r.score)
    assert all(l.url.startswith("http") for r in reqs for l in r.leads)
    turbine = run(DEMOS[2]["input"], s)
    assert {g.claim_id for g in turbine.gaps} == {"C2", "C3"}
    water = run(DEMOS[1]["input"], s)
    assert water.gaps == []  # no MISSING PIECE, nothing to search


def test_archive_citation_in_wardenclyffe():
    r = run(DEMOS[0]["input"], get_settings("mock"))
    earth = next(v for v in r.verdicts if v.claim.id == "C2")
    assert earth.archive_refs[0].doc_id == "US787412A" and earth.archive_refs[0].page >= 1
    assert "page" in earth.archive_refs[0].cite


def test_v1_reports_still_load():
    r = run(DEMOS[1]["input"], get_settings("mock")).model_dump()
    for k in ("spec", "gaps", "archive_hits"):
        r.pop(k)
    for v in r["verdicts"]:
        v.pop("archive_refs")
    assert Report.model_validate(r).spec is None
