"""Innovation loop: for every MISSING PIECE, search who is working on it right now.

  1. Nemotron turns the missing piece into 2-4 "what would need to be true" requirements, each
     with a measurable target, a feasibility judgment, a web query and an arXiv query.
  2. Tavily (web, last ~2 years) and the arXiv API (newest first) look for current work on each.
  3. Each requirement is scored transparently:
         evidence = sum over leads of relevance * recency,  recency = 0.5 ** (age_years / 2)
         score    = 0.6 * feasibility + 0.4 * min(1, evidence / 3)
     and the list is ranked by score (most promising path first).

arXiv API (no key): https://info.arxiv.org/help/api/user-manual.html  -- please keep to
about one request every 3 seconds (their terms of use).
"""
from __future__ import annotations

import datetime as dt
import html
import re
import time
import xml.etree.ElementTree as ET

import requests

from .schema import ClaimVerdict, GapAnalysis, Lead, Requirement

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
RANKING_RULE = ("score = 0.6 × feasibility + 0.4 × evidence (recency-weighted count of relevant "
                "papers and articles, capped at 3)")
_last_arxiv_call = [0.0]


# ----------------------------------------------------------------------------- arXiv
def parse_arxiv_atom(xml_text: str, query: str = "") -> list[Lead]:
    root = ET.fromstring(xml_text)
    qterms = {w for w in re.findall(r"[a-z]{4,}", query.lower()) if w not in {"with", "from", "that", "this"}}
    leads = []
    for e in root.findall("a:entry", ATOM):
        title = " ".join((e.findtext("a:title", "", ATOM) or "").split())
        summary = " ".join((e.findtext("a:summary", "", ATOM) or "").split())
        link = e.findtext("a:id", "", ATOM) or ""
        for l in e.findall("a:link", ATOM):
            if l.get("rel") == "alternate":
                link = l.get("href", link)
        published = (e.findtext("a:published", "", ATOM) or "")[:10] or None
        authors = [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)][:4]
        words = set(re.findall(r"[a-z]{4,}", (title + " " + summary).lower()))
        rel = len(qterms & words) / len(qterms) if qterms else 0.5
        if title:
            leads.append(Lead(title=title, url=link, kind="arxiv", published=published, authors=authors,
                              snippet=summary[:400], relevance=round(rel, 3)))
    return leads


def arxiv_get(params: dict, timeout: int = 30, tries: int = 4) -> requests.Response:
    """GET the arXiv API politely: >= 3 s between calls, back off on HTTP 429/503."""
    for attempt in range(tries):
        wait = 3.1 - (time.time() - _last_arxiv_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_arxiv_call[0] = time.time()
        r = requests.get(ARXIV_API, params=params, timeout=timeout)
        if r.status_code in (429, 503) and attempt < tries - 1:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("arXiv API unavailable")


def arxiv_query_string(query: str) -> str:
    return " AND ".join(f"all:{w}" if " " not in w else f'all:"{w}"' for w in _arxiv_terms(query))


def arxiv_search(query: str, max_results: int = 5, timeout: int = 30) -> list[Lead]:
    terms = _arxiv_terms(query)
    leads: list[Lead] = []
    while terms:
        q = " AND ".join(f"all:{w}" if " " not in w else f'all:"{w}"' for w in terms)
        r = arxiv_get({"search_query": q, "sortBy": "relevance", "sortOrder": "descending",
                       "max_results": max_results}, timeout=timeout)
        leads = parse_arxiv_atom(r.text, query)
        if len(leads) >= 2 or len(terms) <= 2:
            break
        terms = terms[:-1]  # too narrow: drop the last keyword and ask again
    return leads


def _arxiv_terms(query: str) -> list[str]:
    """'"Zenneck wave" power transmission' -> ['Zenneck wave', 'power', 'transmission']"""
    phrases = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]+"', " ", query).split()
    return phrases + [w for w in rest if len(w) > 2]


# ----------------------------------------------------------------------------- web (Tavily)
def web_search(query: str, settings, max_results: int = 5) -> list[Lead]:
    from .research import TavilyClient

    data = TavilyClient(settings).search(query, max_results=max_results, depth="basic", time_range="year")
    leads = []
    for r in data.get("results", []):
        leads.append(Lead(title=r.get("title") or r.get("url", ""), url=r.get("url", ""), kind="web",
                          published=(r.get("published_date") or None), snippet=(r.get("content") or "")[:400],
                          relevance=round(float(r.get("score") or 0.5), 3)))
    return leads


# ----------------------------------------------------------------------------- filtering
JUNK_DOMAINS = ("facebook.com", "instagram.com", "tiktok.com", "pinterest.", "quora.com", "youtube.com",
                "x.com/", "twitter.com", "reddit.com")


def clean_leads(leads: list[Lead], min_web: float = 0.3, min_arxiv: float = 0.5) -> list[Lead]:
    """Drop social-media posts and weak matches; tidy titles. Used for live and canned data alike."""
    out, seen = [], set()
    for l in leads:
        if l.url in seen or any(d in l.url for d in JUNK_DOMAINS):
            continue
        if l.relevance < (min_web if l.kind == "web" else min_arxiv):
            continue
        seen.add(l.url)
        l.title = html.unescape(l.title)
        l.snippet = html.unescape(l.snippet)
        out.append(l)
    return out


# ----------------------------------------------------------------------------- scoring
def _age_years(published: str | None, today: dt.date) -> float:
    if not published:
        return 1.0  # Tavily time_range=year: undated results are within ~a year
    try:
        d = dt.date.fromisoformat(published[:10])
    except ValueError:
        return 1.0
    return max(0.0, (today - d).days / 365.25)


def score_requirements(reqs: list[Requirement], today: dt.date | None = None) -> list[Requirement]:
    today = today or dt.date.today()
    for r in reqs:
        r.leads = clean_leads(r.leads)
        r.evidence = round(sum(l.relevance * 0.5 ** (_age_years(l.published, today) / 2) for l in r.leads), 3)
        r.score = round(0.6 * r.feasibility + 0.4 * min(1.0, r.evidence / 3), 3)
    reqs.sort(key=lambda r: -r.score)
    for i, r in enumerate(reqs, start=1):
        r.rank = i
    return reqs


# ----------------------------------------------------------------------------- live + mock
GAP_SYSTEM = """You are Tesla Lab's R&D scout. A physical claim was judged MISSING PIECE: not forbidden
by physics, but a crucial part is unproven or impractical. List 2-4 things that would NEED TO BE TRUE
for it to work, each a concrete, measurable engineering or science milestone (not hype). For each give:
a feasibility from 0 (essentially impossible) to 1 (already demonstrated at small scale) with a one-line
basis, a short web search query to find who is working on it now, and a short arXiv query (3-6 keywords,
use quotes for exact phrases). Never call anything that violates conservation of energy feasible.
Reply with JSON only: {"missing_piece": str, "requirements": [{"statement": str, "target": str,
 "feasibility": number, "feasibility_basis": str, "web_query": str, "arxiv_query": str}]}"""


def fallback_requirements(v: ClaimVerdict) -> list[Requirement]:
    """No model available: one generic requirement built from the claim text itself."""
    text = re.sub(r"[^\w\s-]", " ", v.claim.text)
    words = [w for w in text.split() if len(w) > 3][:6]
    q = " ".join(words)
    return [Requirement(statement=f"Someone demonstrates: {v.claim.text}", target="independent replication",
                        feasibility=0.3, feasibility_basis="default (no model judgment available)",
                        web_query=f"{q} latest research 2026", arxiv_query=q)]


def _collect(req: Requirement, settings, notes: list[str], per_source: int) -> None:
    if req.web_query:
        try:
            req.leads += web_search(req.web_query, settings, per_source)
        except Exception as e:
            notes.append(f"Tavily gap search failed for '{req.web_query}': {e}")
    if req.arxiv_query:
        try:
            req.leads += arxiv_search(req.arxiv_query, per_source)
        except Exception as e:
            notes.append(f"arXiv search failed for '{req.arxiv_query}': {e}")


def gap_search_live(verdicts: list[ClaimVerdict], settings, llm=None, notes: list[str] | None = None,
                    per_source: int = 4, step=None) -> list[GapAnalysis]:
    notes = notes if notes is not None else []
    out = []
    for v in verdicts:
        if v.verdict != "MISSING PIECE":
            continue
        if step:
            step("gaps", f"{v.claim.id}: what would need to be true? Searching Tavily + arXiv")
        missing, reqs = v.explanation, []
        if llm is not None:
            try:
                j = llm.json_call(GAP_SYSTEM, f"Claim: {v.claim.text}\nWhy it is a missing piece: {v.explanation}\n"
                                              f"Calculation: {v.calc.summary}", model=settings.fast_model)
                missing = j.get("missing_piece") or missing
                reqs = [Requirement(**{k: r.get(k) for k in ("statement", "target", "feasibility", "feasibility_basis",
                                                             "web_query", "arxiv_query") if r.get(k) is not None})
                        for r in j.get("requirements", [])][:4]
            except Exception as e:
                notes.append(f"Gap planner failed for {v.claim.id}, using a generic query: {e}")
        if not reqs:
            reqs = fallback_requirements(v)
        for r in reqs:
            r.feasibility = max(0.0, min(1.0, float(r.feasibility)))
            _collect(r, settings, notes, per_source)
        out.append(GapAnalysis(claim_id=v.claim.id, claim_text=v.claim.text, missing_piece=missing,
                               requirements=score_requirements(reqs), ranking_rule=RANKING_RULE,
                               captured=dt.date.today().isoformat()))
    return out


def gap_search_mock(verdicts: list[ClaimVerdict], demo: dict, today: dt.date | None = None) -> list[GapAnalysis]:
    """Canned requirements + evidence that was fetched for real from Tavily and arXiv (see
    scripts/capture_gap_mock.py); the scoring and ranking run for real."""
    from .mock import load_gaps

    canned = load_gaps().get(demo["id"], {})
    out = []
    for v in verdicts:
        if v.verdict != "MISSING PIECE":
            continue
        g = canned.get("claims", {}).get(v.claim.id)
        if not g:
            continue
        reqs = [Requirement(**r) for r in g["requirements"]]
        cap = canned.get("captured", "")
        ref_day = today or (dt.date.fromisoformat(cap) if cap else None)
        out.append(GapAnalysis(claim_id=v.claim.id, claim_text=v.claim.text, missing_piece=g["missing_piece"],
                               requirements=score_requirements(reqs, ref_day), ranking_rule=RANKING_RULE,
                               captured=cap))
    return out
