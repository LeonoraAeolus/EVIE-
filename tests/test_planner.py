"""v4 Patent Search Planner + Wildcard tests. Every network call is mocked; nothing here touches the internet."""
import datetime as dt
import gzip
import json
from pathlib import Path

import pytest

from tesla_lab.config import get_settings
from tesla_lab.patents import cpc as CPC
from tesla_lab.patents import gpatents as G
from tesla_lab.patents import planner as P
from tesla_lab.patents import sizing as S
from tesla_lab.patents import wildcard as W
from tesla_lab.patents.brain import CannedBrain, NemotronBrain, load_example, parse_constraint
from tesla_lab.patents.sources import BigQueryPatents, BigQueryUnavailable, TavilyPatents
from tesla_lab.schema import SearchPlan

FIX = Path(__file__).parent / "fixtures" / "US5365025A.html.gz"
HTML = gzip.decompress(FIX.read_bytes()).decode("utf-8")


# ----------------------------------------------------------------------------- robots + fetcher
RULES = G.parse_robots(G.ROBOTS_SNAPSHOT)


@pytest.mark.parametrize("path,ok", [
    ("/patent/US5365025A/en", True),
    ("/patent/US5365025A", True),
    ("/", True),
    ("/?q=muffler", False),
    ("/?q=(muffler)&cpc=F01N1/02", False),
    ("/xhr/query?url=q%3Dmuffler", False),
    ("/scholar/123", False),
])
def test_robots_evaluator(path, ok):
    assert G.robots_allows(path, RULES) is ok


@pytest.mark.parametrize("url", [
    "https://patents.google.com/?q=muffler",
    "https://patents.google.com/?q=muffler&before=priority:20000101",
    "https://patents.google.com/xhr/query?url=q%3Dpipe",
    "https://example.com/patent/US5365025A/en",
    "http://patents.google.com/patent/US5365025A/en",
])
def test_check_url_refuses_search_and_foreign_urls(url):
    with pytest.raises(G.DisallowedURL):
        G.check_url(url, RULES)


class FakeHTTP:
    """Records every URL; serves robots.txt + the captured US5365025A page; 404 for everything else."""

    def __init__(self, pages=None, fail=()):
        self.urls = []
        self.pages = {"US5365025A": HTML, **(pages or {})}
        self.fail = set(fail)

    def __call__(self, url, headers, timeout):
        self.urls.append(url)
        assert "?q=" not in url and "/xhr/" not in url, f"search URL requested: {url}"
        if url.endswith("/robots.txt"):
            return 200, G.ROBOTS_SNAPSHOT
        pub = G.pub_from_url(url)
        if pub in self.fail:
            raise ConnectionError("simulated network failure")
        if pub in self.pages:
            return 200, self.pages[pub]
        return 404, ""


def _fetcher(tmp_path, http=None):
    return G.PatentFetcher(cache_dir=tmp_path / "cache", http_get=http or FakeHTTP(), min_interval_s=0)


def test_fetcher_reads_patent_page_and_caches(tmp_path):
    http = FakeHTTP()
    f = _fetcher(tmp_path, http)
    page = f.get("US5365025A")
    assert page and page.pub == "US5365025A"
    assert "muffler" in page.title.lower()
    assert page.claims_text and "tubular member" in page.claims_text.lower()
    assert any(c.startswith("F01N") for c in page.cpc_codes)
    assert page.backward and page.forward  # backward + forward citations parsed
    n = len(http.urls)
    assert f.get("https://patents.google.com/patent/US5365025A/en").pub == "US5365025A"
    assert len(http.urls) == n, "second read must come from the cache"
    assert all("/patent/" in u or u.endswith("/robots.txt") for u in http.urls)


def test_fetcher_refuses_search_url_before_any_request(tmp_path):
    http = FakeHTTP()
    f = _fetcher(tmp_path, http)
    with pytest.raises((G.DisallowedURL, ValueError)):
        f.get("https://patents.google.com/?q=muffler")
    assert not any("?q=" in u for u in http.urls)


def test_fetcher_and_planner_skip_missing_failing_and_disallowed(tmp_path):
    f = _fetcher(tmp_path, FakeHTTP(fail={"US1111111A"}))
    assert f.get("US9999999B2") is None  # 404 → None, no exception
    planner = P.Planner(get_settings(), mode_data="live", brain=None, fetcher=f)
    skipped = []
    assert planner._read("US9999999B2", skipped) is None
    assert planner._read("US1111111A", skipped) is None  # network error → skipped, run continues
    assert planner._read("https://patents.google.com/?q=muffler", skipped) is None  # disallowed → skipped
    assert len(skipped) == 3
    assert "not available" in skipped[0] and "fetch failed" in skipped[1]


def test_parse_page_legal_status_note():
    page = G.parse_patent_html(HTML, "US5365025A")
    flag, why = G.likely_expired_flag(page, dt.date(2026, 10, 13))
    assert flag == "expired"
    assert "legal advice" in G.LEGAL_NOTE.lower()


def test_search_links_are_built_not_fetched():
    url = G.search_link('pipe "fluid conduit"', cpc=["F01N1/02"], before="2000-01-01")
    assert url.startswith("https://patents.google.com/?q=")
    assert "cpc=F01N1/02" in url and "before=priority:20000101" in url
    with pytest.raises(G.DisallowedURL):
        G.check_url(url, RULES)


def test_pub_normalisation():
    assert G.normalize_pub("US 5,365,025 A") == "US5365025A"
    assert G.pub_from_url("https://patents.google.com/patent/US5365025A/en") == "US5365025A"
    assert G.pub_from_url("https://patents.google.com/?q=US5365025A") is None


# ----------------------------------------------------------------------------- ranking + pure helpers
def test_rank_formula_and_weights():
    assert P.rank_score(1, 1, 999, 1) == pytest.approx(1.0, abs=1e-3)
    assert P.rank_score(0, 0, 0, 0) == 0
    # fit dominates: a perfect-fit patent beats a heavily cited off-topic one
    assert P.rank_score(0.9, 0.5, 0, 0.2) > P.rank_score(0.2, 0.5, 500, 0.2)


def test_quote_must_be_on_page():
    claims = "1. A muffler comprising a tubular member having imperforate inlet and outlet end portions."
    assert P.quote_on_page("tubular member having imperforate inlet", claims)
    assert not P.quote_on_page("a magic noise-cancelling crystal lattice", claims)
    assert not P.quote_on_page("tubular member", claims)  # too short to count


def test_sort_hits_mode_emphasis():
    from tesla_lab.schema import Hit
    a = Hit(pub="A", rank_score=0.9, prior_art=False, is_new=False)
    b = Hit(pub="B", rank_score=0.5, prior_art=True, is_new=True)
    assert [h.pub for h in P.sort_hits([a, b], "find")] == ["A", "B"]
    assert [h.pub for h in P.sort_hits([a, b], "check")] == ["B", "A"]  # prior art first
    assert [h.pub for h in P.sort_hits([a, b], "improve")] == ["B", "A"]  # new first


def test_cpc_distance_and_validation():
    assert CPC.normalize("F01N 1/023") == "F01N1/023"
    assert CPC.distance("G10D7/10", ["F01N1/02"]) == 1.0  # other section
    assert CPC.distance("F01N1/06", ["F01N1/02"]) < 0.5
    assert CPC.cpc_match(["F01N1/02"], ["F01N1/02"]) == 1.0


def test_constraint_parsing():
    c = parse_constraint("max tube length 1.0 m")
    assert c.name and c.value == pytest.approx(1.0) and c.op in ("<=", "<")
    c2 = parse_constraint("max volume 8 L")
    assert c2.value == pytest.approx(8.0)


# ----------------------------------------------------------------------------- Wildcard math (spec worked example)
def test_exhaust_sizing_numbers_reproducible():
    c = S.speed_of_sound(400, 1.35)
    assert c == pytest.approx(510, abs=2)
    assert S.firing_frequency(3000, 4) == pytest.approx(100)
    qw = S.run_sizing("quarter_wave_tube", {"freq_hz": 100, "c_m_s": 510})
    assert qw["value"] == pytest.approx(1.275, abs=0.005) and qw["unit"] == "m"
    assert qw["check"]["python_value"] == pytest.approx(qw["value"])  # what the math checker must re-derive
    hh = S.run_sizing("helmholtz_resonator", {"freq_hz": 100, "gas_temp_c": 400})
    assert hh["value"] == pytest.approx(8.8, abs=0.1) and hh["unit"] == "L"


def test_math_fit_and_surprise():
    ok, _ = W.math_fit(True, [{"verdict": "ok", "value": 1.2, "unit": "m", "op": "<=", "limit": 2, "constraint": "x"}])
    assert ok == 1.0
    no, why = W.math_fit(False, [], "acts at 64 kHz")
    assert no == 0.0 and "64 kHz" in why
    s_far, _ = W.surprise("G10D7/10", ["F01N1/02"], ["quarter-wave"], ["perforated tube muffler"])
    s_used, _ = W.surprise("G10K11/172", ["F01N1/02"], ["helmholtz"], ["a helmholtz resonator chamber"])
    assert s_far > s_used


# ----------------------------------------------------------------------------- full runs, mock mode
@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setenv("TESLA_LAB_MODE", "mock")
    return get_settings()


def test_find_exhaust_mock_end_to_end(mock_settings):
    inp = P.example_input("exhaust", "find")
    plan = P.run_plan("find", inp, mock_settings)
    assert plan.data_mode == "mock" and "CANNED" in plan.canned_note
    steps = [t.step for t in plan.trail]
    for s in ("abstract", "expand", "classify", "chain", "claims"):
        assert any(s in x.lower() for x in steps), steps
    assert plan.abstraction.conflict.without
    assert any(t.patent_speak for t in plan.terms)
    roles = {c.role for c in plan.cpc}
    assert {"home", "cross"} <= roles  # Find = cross-industry
    assert plan.hits and plan.hits[0].rank == 1
    assert all(h.url.startswith("https://patents.google.com/patent/") for h in plan.hits)
    assert any("backward" in h.found_via or "forward" in h.found_via for h in plan.hits)  # chain step
    assert "legal advice" in plan.legal_note.lower()
    links = [q for q in plan.queries if q.engine == "gpatents_link"]
    assert links and all(q.url.startswith("https://patents.google.com/?q=") for q in links)
    # Wildcard: quarter-wave tube ≈1.28 m and Helmholtz ≈8.8 L, ranked by surprise × math_fit
    assert plan.wildcard
    vals = {}
    for w in plan.wildcard:
        assert w.score == pytest.approx(w.surprise * w.math_fit, abs=1e-3)
        vals[w.niche] = w.sizing.value
    assert any(abs(v - 1.277) < 0.01 for v in vals.values())
    assert any(abs(v - 8.83) < 0.05 for v in vals.values())
    top = plan.wildcard[0]
    assert top.bucket == "try" and top.niche.lower().startswith("music")
    assert plan.wildcard[-1].bucket == "fun_but_no"  # owl serrations: physics doesn't apply
    json.loads(plan.model_dump_json())  # JSON output round-trips


def test_check_mode_is_narrow_and_dated(mock_settings):
    inp = P.example_input("exhaust", "check")
    inp.before = "2000-01-01"
    plan = P.run_plan("check", inp, mock_settings, wildcard=False)
    assert {c.role for c in plan.cpc} == {"home"}
    assert not plan.wildcard
    assert any(h.prior_art for h in plan.hits)
    pa = [bool(h.prior_art) for h in plan.hits]
    assert pa == sorted(pa, reverse=True)  # prior art listed first
    assert all("before=priority:20000101" in q.url for q in plan.queries if q.engine == "gpatents_link")


def test_improve_reruns_with_new_constraints(mock_settings, tmp_path):
    first = P.run_plan("find", P.example_input("exhaust", "find"), mock_settings)
    P.save_plan(first, tmp_path / "p.json")
    last = P.load_plan(tmp_path / "p.json")
    inp = P.plan_input("improve", test_results=["1.28 m tube cut the 100 Hz drone by 9 dB but is too long"],
                       constraints=["max tube length 1.0 m"], example="exhaust")
    plan = P.run_plan("improve", inp, mock_settings, last=last)
    assert plan.diff is not None and (plan.diff.new or plan.diff.reranked or plan.diff.dropped)
    assert plan.watch and "after=" in plan.watch[0].url  # new-filings watch link
    qw = next(w for w in plan.wildcard if w.niche.lower().startswith("music"))
    assert qw.math_fit < 1.0  # 1.28 m now breaks the 1.0 m limit
    slide = next(w for w in plan.wildcard if "slide" in w.niche.lower())
    assert slide.bucket == "fun_but_no"


def test_tesla_example_mock(mock_settings):
    plan = P.run_plan("find", P.example_input("tesla_wireless", "find"), mock_settings)
    assert plan.hits and plan.wildcard
    assert any(c.code.startswith("H02J50") for c in plan.cpc if c.role == "home")


def test_mock_run_never_hits_network(mock_settings, monkeypatch):
    import requests

    def boom(*a, **k):
        raise AssertionError("network call in mock mode")
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests.Session, "request", boom)
    plan = P.run_plan("find", P.example_input("exhaust", "find"), mock_settings)
    assert plan.hits


# ----------------------------------------------------------------------------- live path with mocked clients
class DelegatingBrain:
    """Not a CannedBrain instance, so the planner takes its LIVE code paths (Tavily discovery, live niches)."""
    source = "model"

    def __init__(self):
        self._b = CannedBrain(load_example("exhaust"))

    def __getattr__(self, name):
        return getattr(self._b, name)

    def transfer(self, principle, niche, snippets, calculators):
        # live niche names differ from the canned ones: hand back the canned idea for the matching family
        fam = niche.split()[0].lower()
        return [i for i in self._b.ex["wildcard"]["ideas"] if i["niche"].split()[0].lower() == fam]


class FakeTavilyClient:
    def __init__(self):
        self.calls = []

    def search(self, query, **kw):
        self.calls.append((query, kw))
        return {"results": [
            {"url": "https://patents.google.com/patent/US5365025A/en", "title": "Low backpressure muffler",
             "content": "straight-through"},
            {"url": "https://patents.google.com/?q=muffler", "title": "search page", "content": ""},
            {"url": "https://patents.google.com/patent/US9999999B2/en", "title": "gone", "content": ""},
            {"url": "https://example.com/blog", "title": "blog", "content": ""},
        ]}


def test_tavily_filters_to_patent_pages():
    client = FakeTavilyClient()
    t = TavilyPatents(client=client)
    seeds = t.search("muffler low backpressure")
    assert [s.pub for s in seeds] == ["US5365025A", "US9999999B2"]
    assert client.calls and "patents.google.com" in json.dumps(client.calls[0][1])


def test_live_path_with_mocked_clients(tmp_path, monkeypatch):
    import tesla_lab.gaps as gaps
    monkeypatch.setattr(gaps, "arxiv_search", lambda *a, **k: [])
    settings = get_settings()
    http = FakeHTTP()
    client = FakeTavilyClient()
    planner = P.Planner(settings, mode_data="live", brain=DelegatingBrain(), fetcher=_fetcher(tmp_path, http),
                        discovery=TavilyPatents(client=client), bq=None, max_read=6)
    plan = planner.run("find", P.example_input("exhaust", "find"))
    assert client.calls, "Tavily discovery should run in live mode"
    assert [h.pub for h in plan.hits][:1] == ["US5365025A"]
    assert not any("?q=" in u for u in http.urls), http.urls
    assert all(("/patent/" in u) or u.endswith("/robots.txt") for u in http.urls)
    assert plan.wildcard  # sized ideas even though niches came from the live path


def test_bigquery_skips_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with pytest.raises(BigQueryUnavailable):
        BigQueryPatents(project=None)


def test_bigquery_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("PATENTS_BIGQUERY", raising=False)
    assert get_settings().patents_bigquery is False


def test_nemotron_brain_parses_json_from_llm():
    class FakeLLM:
        def __init__(self):
            self.systems = []

        def json_call(self, system, user, model=None, max_tokens=0, **kw):
            self.systems.append(system)
            if "FUNCTION plus a CONFLICT" in system:
                return {"function": "attenuate pressure pulses", "conflict": {"improve": "lower backpressure",
                        "without": "more noise"}, "principle": "pulses in gas", "quantities": {"freq_hz": 100}}
            if "claims" in system.lower():
                return {"fit": 0.8, "quote": "a tubular member", "why": "straight-through"}
            return {}
    from tesla_lab.schema import PlanInput
    b = NemotronBrain(FakeLLM(), model="nemotron-3-super")
    ab = b.abstract("find", PlanInput(goal="quiet exhaust"))
    assert ab.conflict.without == "more noise" and ab.quantities["freq_hz"] == 100
    sc = b.score_claims(ab, "US5365025A", "Muffler", "1. A muffler ...", "find")
    assert sc["fit"] == 0.8


def test_cli_plan_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TESLA_LAB_MODE", "mock")
    from tesla_lab.cli import main
    out = tmp_path / "plan.json"
    rc = main(["--plan", "find", "--example", "exhaust", "--plan-out", str(out)])
    assert rc in (0, None)
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["mode"] == "find" and d["hits"] and d["wildcard"]
    assert "CANNED" in capsys.readouterr().out
    plan = SearchPlan.model_validate(d)
    assert plan.hits[0].rank == 1
