"""Patent Search Planner: Find / Check / Improve, five steps, one reasoning trail (spec v2).

    1 Abstract   -> function + conflict (+ principle, exclusions)              [model]
    2 Expand     -> synonyms + patent-speak; query strings                     [model + code]
    3 Classify   -> CPC codes: home + other industries' codes, validated        [model + local CPC table]
    4 Chain      -> seeds (Tavily / BigQuery / canned) + 1 hop of citations     [patent pages]
    5 Read claims-> fit to the conflict with a verbatim quote, legal-status flag [model + string match + page]
    + Wildcard   -> see wildcard.py

Ranking (spec v2 initial weights):
    rank = 0.5·fit + 0.2·cpc_match + 0.2·log-scaled forward citations + 0.1·recency
    fwd  = min(1, log10(1 + cited_by) / 3)          (1,000 citing documents -> 1.0)
    recency = 0.5 ** (age_years / 10)               (by priority date)
Check mode sorts prior art (older than the team's date) first; Improve sorts "new since last run" first.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

from ..schema import (Abstraction, ChainEdge, Constraint, CpcCode, Hit, PlanDiff, PlanInput, PlanQuery,
                      SearchPlan, Term, TrailStep)
from . import cpc as CPC
from .brain import CannedBrain, NemotronBrain, load_example, load_pages, pick_example
from .gpatents import (LEGAL_NOTE, DisallowedURL, PatentFetcher, PatentPage, likely_expired_flag, patent_url,
                       search_link)

StepCallback = Callable[[str, str], None]
W_FIT, W_CPC, W_FWD, W_REC = 0.5, 0.2, 0.2, 0.1


def _noop(step: str, msg: str) -> None:
    pass


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def quote_on_page(quote: str, claims_text: str, min_words: int = 4) -> bool:
    """True when the model's quote appears verbatim in the claims (case/punctuation/whitespace-insensitive)."""
    q = norm_text(quote)
    return bool(q) and len(q.split()) >= min_words and q in norm_text(claims_text)


def fwd_score(n: int) -> float:
    return min(1.0, math.log10(1 + max(0, n)) / 3.0)


def recency_score(date: str, today: dt.date) -> float:
    try:
        d = dt.date.fromisoformat(date[:10])
    except (ValueError, TypeError):
        return 0.0
    age = max(0.0, (today - d).days / 365.25)
    return 0.5 ** (age / 10.0)


def rank_score(fit: float, cpc_match: float, forward: int, recency: float) -> float:
    return round(W_FIT * fit + W_CPC * cpc_match + W_FWD * fwd_score(forward) + W_REC * recency, 4)


def sort_hits(hits: list[Hit], mode: str) -> list[Hit]:
    if mode == "check":
        key = lambda h: (0 if h.prior_art else 1, -h.rank_score)  # noqa: E731
    elif mode == "improve":
        key = lambda h: (0 if h.is_new else 1, -h.rank_score)  # noqa: E731
    else:
        key = lambda h: -h.rank_score  # noqa: E731
    out = sorted(hits, key=key)
    for i, h in enumerate(out, 1):
        h.rank = i
    return out


def diff_plans(last: SearchPlan, hits: list[Hit], top: int = 10) -> PlanDiff:
    old = {h.pub: h.rank for h in last.hits}
    new = {h.pub: h.rank for h in hits}
    d = PlanDiff(new=[p for p in new if p not in old], dropped=[p for p in old if p not in new])
    for p, r in new.items():
        if p in old and old[p] != r and (r <= top or old[p] <= top):
            d.reranked.append({"pub": p, "from": old[p], "to": r})
    return d


def _quote_ors(xs: list[str], limit: int = 4) -> str:
    out = []
    for x in xs[:limit]:
        x = x.strip()
        if x:
            out.append(f'"{x}"' if " " in x else x)
    return "(" + " OR ".join(out) + ")" if out else ""


def build_queries(ab: Abstraction, terms: list[Term], codes: list[CpcCode], mode: str, inp: PlanInput,
                  last_run: str | None = None) -> tuple[list[PlanQuery], list[PlanQuery]]:
    """(queries, watch). Google Patents links are for the user to click; Tavily queries are run by the app."""
    groups = [_quote_ors([t.term] + t.patent_speak[:2] + t.synonyms[:1]) for t in terms[:3]]
    q = " ".join(g for g in groups if g)
    home = [c.code for c in codes if c.role == "home"]
    cross = [c.code for c in codes if c.role == "cross"]
    before = inp.before if mode == "check" else None
    out: list[PlanQuery] = []
    if home:
        out.append(PlanQuery(engine="gpatents_link", text=f"{q} + CPC {', '.join(map(CPC.pretty, home))}",
                             url=search_link(q, cpc=home, before=before), purpose="home class"))
    if mode != "check":
        for c in cross[:6]:
            out.append(PlanQuery(engine="gpatents_link", text=f"{q} + CPC {CPC.pretty(c)}",
                                 url=search_link(q, cpc=[c]), purpose="same function, other industry"))
    out.append(PlanQuery(engine="gpatents_link", text=q + (" (keywords only)" if q else ""),
                         url=search_link(q, before=before), purpose="all classes"))
    # Tavily (run by the app): function + patent-speak, one per class
    speak = [p for t in terms for p in t.patent_speak[:1]]
    base = f"{ab.function} {' '.join(speak[:3])}".strip()
    tav = [base]
    if mode == "find":
        for c in codes:
            if c.role == "cross":
                tav.append(f"{ab.conflict.improve} {CPC.table().get(c.code, {}).get('title', '')} {speak[0] if speak else ''}".strip())
    elif mode == "check":
        tav.append(inp.idea or base)
    else:  # improve
        for con in inp.constraints[:3]:
            tav.append(f"{ab.function} {con.text}")
    for t in tav[:7]:
        out.append(PlanQuery(engine="tavily", text=re.sub(r"\s+", " ", t)[:380], purpose="patent page discovery"))
    watch = []
    if mode == "improve":
        since = last_run or dt.date.today().isoformat()
        watch.append(PlanQuery(engine="gpatents_link", text=f"{q} filed after {since}",
                               url=search_link(q, cpc=home + cross[:2], after=since), purpose="new filings watch"))
    return out, watch


# ----------------------------------------------------------------------------- the planner
class Planner:
    def __init__(self, settings, *, mode_data: str, brain, fetcher: PatentFetcher, discovery=None, bq=None,
                 on_step: StepCallback | None = None, today: dt.date | None = None, max_read: int = 14,
                 example: dict | None = None):
        self.s = settings
        self.data_mode = mode_data
        self.brain = brain
        self.fetcher = fetcher
        self.discovery = discovery
        self.bq = bq
        self.step = on_step or _noop
        self.today = today or dt.date.today()
        self.max_read = max_read
        self.example = example

    # --- step helpers
    def _trail(self, plan: SearchPlan, step: str, did: str, t0: float, kept=None, dropped=None, source=""):
        plan.trail.append(TrailStep(step=step, did=did, kept=kept or [], dropped=dropped or [], source=source,
                                    seconds=round(time.perf_counter() - t0, 2)))
        self.step(step, did)

    def classify(self, ab: Abstraction, terms: list[Term], mode: str) -> tuple[list[CpcCode], list[str], list[str]]:
        fn_words = [ab.function, ab.conflict.improve] + [p for t in terms for p in t.patent_speak]
        hints = CPC.match_titles(fn_words, top=10) if self.brain.source != "canned" else []
        proposals = self.brain.classify(ab, terms, hints, mode)
        kept: list[CpcCode] = []
        dropped: list[str] = []
        seen = set()
        for p in proposals:
            code, note = CPC.validate(p.get("code", ""))
            if not code:
                dropped.append(note)
                continue
            if code in seen:
                continue
            seen.add(code)
            kept.append(CpcCode(code=code, title=CPC.title(code), role=p.get("role", "cross") if p.get("role") in ("home", "cross") else "cross",
                                why=p.get("why", ""), origin="fixture" if self.brain.source == "canned" else "model", note=note))
        home_subs = {CPC.parts(c.code)["subclass"] for c in kept if c.role == "home"}
        notes = []
        if mode == "check":  # narrow and deep: drop cross-industry codes
            for c in [c for c in kept if c.role == "cross"]:
                dropped.append(f"{CPC.pretty(c.code)}: cross-industry code skipped in Check mode")
            kept = [c for c in kept if c.role == "home"]
        else:
            n_cross = sum(1 for c in kept if c.role == "cross")
            want = 6 if mode == "find" else 3
            if n_cross < want:
                for m in CPC.match_titles(fn_words, exclude_subclasses=home_subs, top=8):
                    if n_cross >= want:
                        break
                    if m["code"] in seen or any(CPC.similarity(m["code"], c.code) >= 0.75 for c in kept):
                        continue
                    seen.add(m["code"])
                    kept.append(CpcCode(code=m["code"], title=m["title"], role="cross", origin="title_match",
                                        why="CPC title shares: " + ", ".join(m["words"])))
                    n_cross += 1
                    notes.append(CPC.pretty(m["code"]))
        return kept, dropped, notes

    def _relevance_words(self, ab: Abstraction, terms: list[Term]) -> set[str]:
        txt = " ".join([ab.function, ab.conflict.improve, ab.conflict.without] +
                       [t.term for t in terms] + [p for t in terms for p in t.patent_speak + t.synonyms])
        return CPC.words(txt)

    def _hit_from_page(self, page: PatentPage, found_via: str, codes: list[CpcCode], inp: PlanInput) -> Hit:
        flag, why = likely_expired_flag(page, self.today)
        hit_codes = [c["code"] for c in page.cpc if c.get("leaf")] + [c["code"] for c in page.cpc if not c.get("leaf")]
        for c in page.cpc:
            CPC.learn(c["code"], c.get("title", ""))
        home = [c.code for c in codes if c.role == "home"]
        cross = [c.code for c in codes if c.role == "cross"]
        role = ""
        if any(CPC.parts(h).get("subclass") == CPC.parts(c).get("subclass") for h in hit_codes for c in home):
            role = "home"
        elif any(CPC.similarity(h, c) >= 0.75 for h in hit_codes for c in cross):
            role = "cross"
        h = Hit(pub=page.pub, title=page.title, date=page.priority_date, filing_date=page.filing_date,
                cpc=hit_codes[:12], found_via=found_via, status_text=page.status_text, status_flag=flag,
                status_why=why, legal_note=LEGAL_NOTE, forward_citations=page.forward_count or len(page.forward),
                backward_citations=page.backward_count or len(page.backward), url=patent_url(page.pub), role=role,
                cpc_match=round(CPC.cpc_match(hit_codes, [c.code for c in codes]), 3),
                recency=round(recency_score(page.priority_date, self.today), 3))
        if inp.before and page.priority_date:
            h.prior_art = page.priority_date[:10] < inp.before[:10]
        return h

    def run(self, mode: str, inp: PlanInput, last: SearchPlan | None = None, wildcard: bool = True) -> SearchPlan:
        t_start = time.perf_counter()
        pid = hashlib.sha1(f"{mode}{inp.model_dump_json()}{time.time()}".encode()).hexdigest()[:10]
        plan = SearchPlan(id=pid, mode=mode, data_mode=self.data_mode, input=inp,
                          canned_note=(self.example or {}).get("canned_note", "") if self.data_mode == "mock" else "",
                          created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        if mode == "improve" and last is not None:
            plan.input.last_plan_id = last.id

        # constraints: text -> numbers (regex first; the live brain fills the rest)
        if inp.constraints:
            parsed = self.brain.constraints([c.text for c in inp.constraints])
            plan.input.constraints = [p if p.name else c for p, c in zip(parsed, inp.constraints)]

        # 1 Abstract
        t0 = time.perf_counter()
        last_ab = last.abstraction.model_dump() if (last and last.abstraction) else None
        ab = self.brain.abstract(mode, plan.input, last_ab)
        plan.abstraction = ab
        self._trail(plan, "1 Abstract",
                    f"Function: {ab.function}. Conflict: {ab.conflict.improve} WITHOUT {ab.conflict.without}.",
                    t0, kept=[f"principle: {ab.principle}"] if ab.principle else [],
                    dropped=[f"exclude: {x}" for x in ab.exclude], source=self.brain.source)

        # 2 Expand
        t0 = time.perf_counter()
        plan.terms = self.brain.expand(ab)
        self._trail(plan, "2 Expand", f"{len(plan.terms)} terms → patent-speak: " +
                    "; ".join(f"{t.term} → {', '.join(t.patent_speak[:3])}" for t in plan.terms),
                    t0, kept=[p for t in plan.terms for p in t.patent_speak], source=self.brain.source)

        # 3 Classify
        t0 = time.perf_counter()
        codes, dropped, added = self.classify(ab, plan.terms, mode)
        if mode == "improve" and last is not None:
            for c in last.cpc:
                if c.code not in {k.code for k in codes} and c.role == "cross" and len(codes) < 8:
                    codes.append(c)
        plan.cpc = codes
        plan.queries, plan.watch = build_queries(ab, plan.terms, codes, mode, plan.input,
                                                 last_run=(last.created_at[:10] if last and last.created_at else None))
        home_txt = ", ".join(CPC.pretty(c.code) for c in codes if c.role == "home")
        cross_txt = ", ".join(CPC.pretty(c.code) for c in codes if c.role == "cross")
        self._trail(plan, "3 Classify", f"Home: {home_txt or '—'}. Other industries: {cross_txt or '— (Check mode stays in its own class)'}"
                    + (f". Added by CPC-title match: {', '.join(added)}" if added else ""),
                    t0, kept=[f"{CPC.pretty(c.code)} ({c.role}) {c.title[-70:]}" for c in codes], dropped=dropped,
                    source=f"{self.brain.source} + local CPC table")

        # 4 Discover + Chain
        t0 = time.perf_counter()
        seeds = self._discover(plan, mode)
        reads: dict[str, tuple[PatentPage, str]] = {}
        skipped: list[str] = []
        for pub, via in seeds:
            if len(reads) >= self.max_read:
                break
            page = self._read(pub, skipped)
            if page:
                reads.setdefault(pub, (page, via))
        rel = self._relevance_words(ab, plan.terms)
        chain_edges: list[ChainEdge] = []
        chained: list[tuple[str, str]] = []
        for pub, (page, _) in list(reads.items())[:5]:
            for kind, rows in (("backward", page.backward), ("forward", page.forward)):
                cands = []
                for r in rows:
                    ov = len(CPC.words(r.title) & rel)
                    if ov == 0:
                        continue
                    cands.append((ov + (0.5 if self.fetcher.available(r.pub) else 0), r.priority_date or "", r))
                if not cands:
                    continue
                # one MOST RELEVANT link + one OLDEST root (backward) / NEWEST improvement (forward)
                picks = [max(cands, key=lambda x: x[0])[2]]
                dated = [c for c in cands if c[1]]
                if dated:
                    edge = min(dated, key=lambda x: (x[1], -x[0])) if kind == "backward" else max(dated, key=lambda x: (x[1], x[0]))
                    if edge[2].pub != picks[0].pub:
                        picks.append(edge[2])
                for r in picks:
                    chain_edges.append(ChainEdge(src=pub, dst=r.pub, kind=kind, dst_title=r.title,
                                                 dst_date=r.priority_date))
                    via = (f"backward citation of {pub} (older root)" if kind == "backward"
                           else f"cites {pub} (newer improvement)")
                    chained.append((r.pub, via))
        n_chain_read = 0
        for pub, via in chained:
            if pub in reads or len(reads) >= self.max_read + 6:
                continue
            page = self._read(pub, skipped)
            if page:
                reads[pub] = (page, via)
                n_chain_read += 1
        plan.chain = chain_edges
        self._trail(plan, "4 Chain", f"{len(seeds)} seed patents; followed {len(chain_edges)} citation links "
                    f"(1 hop, backward = older roots, forward = newer improvements); read {len(reads)} patent pages "
                    f"({n_chain_read} reached only through the chain).", t0,
                    kept=[f"{p}: {v}" for p, (_, v) in reads.items()], dropped=skipped[:12],
                    source="patent pages (" + ("captured copies" if self.data_mode == "mock" else "patents.google.com/patent/…") + ")")

        # 5 Read claims
        t0 = time.perf_counter()
        hits: list[Hit] = []
        unconfirmed = []
        for pub, (page, via) in reads.items():
            h = self._hit_from_page(page, via, codes, plan.input)
            if not page.claims:
                h.fit_why = "no claims text on the page (old OCR'd patent): listed for the chain, not scored"
                h.read = False
            else:
                try:
                    sc = self.brain.score_claims(ab, pub, page.title, page.claims_text, mode)
                except Exception as exc:  # noqa: BLE001
                    sc = {"fit": 0.0, "quote": "", "why": f"model error: {exc}"}
                h.claim_quote = sc.get("quote", "")
                h.fit_why = sc.get("why", "")
                h.quote_verified = quote_on_page(h.claim_quote, page.claims_text)
                fit = max(0.0, min(1.0, float(sc.get("fit", 0) or 0)))
                if not h.quote_verified:
                    if h.claim_quote:
                        unconfirmed.append(pub)
                        h.fit_why = "UNCONFIRMED: the quoted line is not in this patent's claims, so fit = 0. " + h.fit_why
                    fit = 0.0
                h.fit = round(fit, 3)
                if page.claims_from_ocr:
                    h.fit_why += " (claims read from the OCR'd text)"
            h.rank_score = rank_score(h.fit, h.cpc_match, h.forward_citations, h.recency)
            if mode == "improve" and last is not None:
                h.is_new = pub not in {x.pub for x in last.hits}
            hits.append(h)
        plan.hits = sort_hits(hits, mode)
        verified = sum(1 for h in hits if h.quote_verified)
        self._trail(plan, "5 Read claims", f"Scored {len(hits)} patents against the conflict; {verified} quotes found "
                    f"verbatim in the claims" + (f", {len(unconfirmed)} unconfirmed (fit set to 0)" if unconfirmed else "")
                    + ". Legal status read from each page (not legal advice).", t0,
                    kept=[f"#{h.rank} {h.pub} fit {h.fit:.2f} · rank {h.rank_score:.2f} · {h.status_flag}" for h in plan.hits[:10]],
                    dropped=[f"{p}: quote not on the page" for p in unconfirmed], source=self.brain.source + " + string match")

        if mode == "improve" and last is not None:
            plan.diff = diff_plans(last, plan.hits)
        if mode == "check":
            plan.notes.append(f"Prior art = priority date before {plan.input.before or '(no date given)'}. "
                              "Check mode stays inside the home class and reads every hit's claims.")
            self._npl(plan)

        # Wildcard
        if wildcard:
            from .wildcard import run_wildcard

            t0 = time.perf_counter()
            run_wildcard(plan, self, last=last)
            if mode == "improve" and last is not None and plan.diff is not None:
                old = {w.niche: w for w in last.wildcard}
                for wi in plan.wildcard:
                    o = old.get(wi.niche)
                    if o and abs(o.score - wi.score) > 1e-9:
                        plan.diff.wildcard_changes.append({"niche": wi.niche, "from_score": o.score, "to_score": wi.score,
                                                           "why": wi.math_fit_why})
            self._trail(plan, "💡 Wildcard", f"Principle: {plan.wildcard_principle}. {len(plan.wildcard)} ideas from "
                        f"unrelated niches, ranked by surprise × math_fit (sizes computed in Python).", t0,
                        kept=[f"{w.niche}: score {w.score:.2f} (surprise {w.surprise:.2f} × math {w.math_fit:.1f})"
                              for w in plan.wildcard if w.bucket == "try"],
                        dropped=[f"{w.niche}: fun, but no — {w.math_fit_why}" for w in plan.wildcard if w.bucket != "try"],
                        source=self.brain.source + " + Python sizing")
        if mode == "improve":
            plan.notes.append("ROADMAP, not built: a closed loop where 3D-printed prototypes are tested and the "
                              "results flow back in automatically. Today you type the test results and constraints.")
        plan.timings_s["total"] = round(time.perf_counter() - t_start, 2)
        return plan

    # --- data access
    def _read(self, pub: str, skipped: list[str]) -> PatentPage | None:
        try:
            page = self.fetcher.get(pub)
        except DisallowedURL as exc:
            skipped.append(f"{pub}: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{pub}: fetch failed ({type(exc).__name__})")
            return None
        if page is None:
            skipped.append(f"{pub}: page not available" + (" in mock mode (not captured)" if self.data_mode == "mock" else ""))
        return page

    def _discover(self, plan: SearchPlan, mode: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        seen = set()
        if isinstance(self.brain, CannedBrain):
            for pub, q in self.brain.seeds(mode):
                if pub not in seen:
                    seen.add(pub)
                    out.append((pub, f"web search (captured 2026-10-13): {q}"))
        elif self.discovery is not None:
            for q in [x for x in plan.queries if x.engine == "tavily"]:
                try:
                    for sd in self.discovery.search(q.text, max_results=6):
                        if sd.pub not in seen:
                            seen.add(sd.pub)
                            out.append((sd.pub, sd.found_via))
                except Exception as exc:  # noqa: BLE001
                    plan.notes.append(f"Tavily search failed for “{q.text[:60]}”: {type(exc).__name__}")
        if self.bq is not None:
            for c in [c for c in plan.cpc][:3]:
                try:
                    words = [t.term for t in plan.terms][:4]
                    for sd in self.bq.search_cpc(c.code, words):
                        if sd.pub not in seen:
                            seen.add(sd.pub)
                            out.append((sd.pub, sd.found_via))
                    plan.queries.append(PlanQuery(engine="bigquery", text=f"cpc {c.code} title ~ {words}"))
                except Exception as exc:  # noqa: BLE001
                    plan.notes.append(f"BigQuery skipped: {exc}")
                    break
        elif getattr(self.s, "patents_bigquery", False):
            plan.notes.append("BigQuery path requested but unavailable (library, project or credentials missing): "
                              "using patent-page citations only.")
        return out

    def _npl(self, plan: SearchPlan) -> None:
        if self.data_mode == "mock":
            plan.notes.append("Non-patent literature (arXiv) is searched in live Check runs; skipped in mock mode.")
            return
        try:
            from ..gaps import arxiv_search

            leads = arxiv_search(plan.input.idea or plan.abstraction.function, max_results=4)
            for ld in leads:
                plan.npl.append({"title": ld.title, "url": ld.url, "published": ld.published, "source": "arXiv"})
        except Exception as exc:  # noqa: BLE001
            plan.notes.append(f"arXiv search failed: {type(exc).__name__}")


# ----------------------------------------------------------------------------- entry point
def make_planner(settings, example_id: str | None = None, on_step: StepCallback | None = None,
                 fetcher: PatentFetcher | None = None, llm=None, discovery=None, bq=None,
                 today: dt.date | None = None) -> Planner:
    if settings.is_mock:
        ex = load_example(example_id or "exhaust")
        fx = fetcher or PatentFetcher(cache_dir=settings.patent_cache_dir, offline=True, fixtures=load_pages())
        return Planner(settings, mode_data="mock", brain=CannedBrain(ex), fetcher=fx, on_step=on_step, today=today,
                       example=ex)
    if llm is None:
        from ..llm import NemotronClient

        llm = NemotronClient(settings)
    if discovery is None:
        from .sources import TavilyPatents

        discovery = TavilyPatents(settings)
    if bq is None and getattr(settings, "patents_bigquery", False):
        from .sources import BigQueryPatents, BigQueryUnavailable

        try:
            bq = BigQueryPatents(settings.bq_project, settings.bq_max_gb)
        except BigQueryUnavailable:
            bq = None
    fx = fetcher or PatentFetcher(cache_dir=settings.patent_cache_dir, min_interval_s=settings.patent_fetch_interval_s)
    return Planner(settings, mode_data="live", brain=NemotronBrain(llm, settings.model), fetcher=fx,
                   discovery=discovery, bq=bq, on_step=on_step, today=today, max_read=settings.planner_max_pages)


def plan_input(mode: str, *, goal: str = "", failed: list[str] | None = None, idea: str = "",
               before: str | None = None, constraints: list[str] | None = None,
               test_results: list[str] | None = None, example: str | None = None) -> PlanInput:
    return PlanInput(goal=goal, failed=failed or [], idea=idea, before=before,
                     constraints=[Constraint(text=c) for c in (constraints or []) if c.strip()],
                     test_results=test_results or [], example=example)


def run_plan(mode: str, inp: PlanInput, settings=None, *, last: SearchPlan | None = None, wildcard: bool = True,
             on_step: StepCallback | None = None, **kw) -> SearchPlan:
    """One planner run. Mock mode picks the closest canned example (exhaust or tesla_wireless)."""
    from ..config import get_settings

    s = settings or get_settings()
    if mode not in ("find", "check", "improve"):
        raise ValueError("mode must be find, check or improve")
    example = inp.example or (last.input.example if last is not None else None)
    if s.is_mock and not example:
        example, _ = pick_example(" ".join([inp.goal, inp.idea] + inp.failed + inp.test_results))
        inp.example = example
    if mode == "improve" and last is None:
        if s.is_mock:  # start from the canned Find run
            last = make_planner(s, example, on_step=None, **kw).run("find", _example_input(example, "find"), wildcard=wildcard)
        else:
            raise ValueError("Improve mode needs the last plan (pass last=..., or --last-plan FILE on the CLI)")
    if mode == "improve" and last is not None:
        # carry the last run's problem forward; the new results/constraints are added on top
        inp.goal = inp.goal or last.input.goal
        inp.failed = inp.failed or list(last.input.failed)
        inp.idea = inp.idea or last.input.idea
        known = {c.text for c in inp.constraints}
        inp.constraints = [c for c in last.input.constraints if c.text not in known and
                           not _superseded(c, inp.constraints)] + inp.constraints
    pl = make_planner(s, example, on_step=on_step, **kw)
    return pl.run(mode, inp, last=last, wildcard=wildcard)


def _superseded(old: Constraint, new: list[Constraint]) -> bool:
    return bool(old.name) and any(n.name == old.name for n in new if n.name) or \
        bool(old.text) and any(_c_name(n.text) == _c_name(old.text) and _c_name(old.text) for n in new)


def _c_name(text: str) -> str:
    from .brain import parse_constraint

    return parse_constraint(text).name


def _example_input(example_id: str, mode: str) -> PlanInput:
    ex = load_example(example_id)
    i = ex["input"]
    return plan_input(mode, goal=i.get("goal", ""), failed=i.get("failed", []), idea=i.get("idea", ""),
                      before=i.get("before"), constraints=i.get("constraints", []), example=example_id)


def example_input(example_id: str, mode: str) -> PlanInput:
    """The canned example's own input for a mode (Improve adds its stored test results + constraint)."""
    inp = _example_input(example_id, mode)
    if mode == "improve":
        imp = load_example(example_id).get("improve", {})
        inp.test_results = list(imp.get("test_results", []))
        inp.constraints = [Constraint(text=c) for c in imp.get("constraints", [])]
    return inp


def save_plan(plan: SearchPlan, path: str | Path) -> None:
    Path(path).write_text(plan.model_dump_json(indent=2), encoding="utf-8")


def load_plan(path: str | Path) -> SearchPlan:
    return SearchPlan.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
