import datetime as dt

from tesla_lab.gaps import (clean_leads, fallback_requirements, gap_search_live, parse_arxiv_atom,
                            score_requirements)
from tesla_lab.schema import CalcResult, Claim, ClaimVerdict, Lead, Requirement

ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry><id>http://arxiv.org/abs/2501.00001v1</id><published>2025-01-02T00:00:00Z</published>
  <title>Zenneck surface wave power transfer</title><summary>We launch a Zenneck wave for power transfer.</summary>
  <author><name>A. Author</name></author>
  <link href="https://arxiv.org/abs/2501.00001v1" rel="alternate" type="text/html"/></entry>
</feed>"""


def _verdict(v="MISSING PIECE"):
    return ClaimVerdict(claim=Claim(id="C2", text="The Earth carries power anywhere"), calc=CalcResult(ok=True),
                        verdict=v, explanation="leaky")


def test_parse_arxiv_atom():
    leads = parse_arxiv_atom(ATOM, '"Zenneck" wave power transfer')
    assert len(leads) == 1 and leads[0].kind == "arxiv" and leads[0].published == "2025-01-02"
    assert leads[0].authors == ["A. Author"] and leads[0].relevance > 0.5


def test_clean_leads_drops_social_and_weak():
    leads = [Lead(title="a", url="https://facebook.com/x", kind="web", relevance=0.9),
             Lead(title="b &amp; c", url="https://ok.org", kind="web", relevance=0.9),
             Lead(title="weak", url="https://weak.org", kind="web", relevance=0.1)]
    out = clean_leads(leads)
    assert [l.url for l in out] == ["https://ok.org"] and out[0].title == "b & c"


def test_scoring_prefers_feasible_and_recent():
    today = dt.date(2026, 10, 13)
    new = Lead(title="n", url="https://n.org", kind="arxiv", published="2026-09-01", relevance=1.0)
    old = Lead(title="o", url="https://o.org", kind="arxiv", published="2010-01-01", relevance=1.0)
    reqs = score_requirements([Requirement(statement="old", feasibility=0.4, leads=[old]),
                               Requirement(statement="new", feasibility=0.4, leads=[new])], today)
    assert [r.statement for r in reqs] == ["new", "old"] and reqs[0].rank == 1
    assert reqs[0].score == round(0.6 * 0.4 + 0.4 * reqs[0].evidence / 3, 3)


def test_gap_search_live_with_fakes(monkeypatch):
    import tesla_lab.gaps as g

    monkeypatch.setattr(g, "web_search", lambda q, s, n: [Lead(title="web " + q, url="https://w.org/" + q, kind="web", relevance=0.8)])
    monkeypatch.setattr(g, "arxiv_search", lambda q, n: [Lead(title="paper", url="https://arxiv.org/abs/1" + q, kind="arxiv",
                                                               published="2026-01-01", relevance=1.0)])

    class FakeLLM:
        def json_call(self, *a, **k):
            return {"missing_piece": "no low-loss path", "requirements": [
                {"statement": "Surface waves carry kW", "target": ">50% at 10 km", "feasibility": 0.1, "web_query": "zenneck", "arxiv_query": "zenneck"},
                {"statement": "Beamed power", "target": ">30%", "feasibility": 0.5, "web_query": "beaming", "arxiv_query": "beaming"}]}

    class S:
        fast_model = "m"

    out = gap_search_live([_verdict(), _verdict("WORKS")], S(), llm=FakeLLM())
    assert len(out) == 1 and out[0].missing_piece == "no low-loss path"
    assert out[0].requirements[0].statement == "Beamed power" and len(out[0].requirements[0].leads) == 2


def test_fallback_requirement_without_model():
    r = fallback_requirements(_verdict())[0]
    assert r.web_query and r.arxiv_query and 0 <= r.feasibility <= 1
