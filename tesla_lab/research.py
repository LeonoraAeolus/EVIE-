"""Step 1: gather evidence with Tavily (search + extract).

API verified 2026-10-12:
  POST https://api.tavily.com/search   body {query, search_depth, max_results, ...}
  POST https://api.tavily.com/extract  body {urls, query?, ...}
  auth: header `Authorization: Bearer tvly-...`
  docs: https://docs.tavily.com/documentation/api-reference/endpoint/search
        https://docs.tavily.com/documentation/api-reference/endpoint/extract
Keyless testing (rate-limited, no account): header `X-Tavily-Access-Mode: keyless`
  https://docs.tavily.com/documentation/keyless  -- use your own key for the real submission.
"""
from __future__ import annotations

import requests

from .config import Settings
from .schema import Source

LOW_QUALITY = ("facebook.com", "quora.com", "chegg.com", "scribd.com", "pinterest.", "tiktok.com")


class TavilyClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.base = settings.tavily_base_url.rstrip("/")

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.s.tavily_api_key:
            h["Authorization"] = f"Bearer {self.s.tavily_api_key}"
        elif self.s.tavily_keyless:
            h["X-Tavily-Access-Mode"] = "keyless"
        else:
            raise RuntimeError("Set TAVILY_API_KEY (or TAVILY_KEYLESS=1 for quick tests)")
        return h

    def search(self, query: str, max_results: int = 5, depth: str = "advanced", **extra) -> dict:
        body = {"query": query, "max_results": max_results, "search_depth": depth, **extra}
        r = requests.post(f"{self.base}/search", json=body, headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json()

    def extract(self, urls: list[str], query: str | None = None) -> dict:
        body: dict = {"urls": urls[:20]}
        if query:
            body["query"] = query
        r = requests.post(f"{self.base}/extract", json=body, headers=self._headers(), timeout=90)
        r.raise_for_status()
        return r.json()


def research_queries(claim: str) -> list[str]:
    """Three angles on the claim: the original source, the modern science, the debunk."""
    return [
        f"{claim} original patent OR paper primary source",
        f"{claim} modern research measured efficiency physics",
        f"{claim} debunked OR criticism OR replication",
    ]


def gather(claim: str, settings: Settings, per_query: int = 5, extract_top: int = 3) -> list[Source]:
    client = TavilyClient(settings)
    sources: list[Source] = []
    seen: set[str] = set()
    for q in research_queries(claim):
        data = client.search(q, max_results=per_query)
        for r in data.get("results", []):
            url = r.get("url", "")
            if not url or url in seen or any(b in url for b in LOW_QUALITY):
                continue
            seen.add(url)
            sources.append(
                Source(id=f"S{len(sources) + 1}", url=url, title=r.get("title", url),
                       content=(r.get("content") or "")[:2000], score=r.get("score"))
            )
    # Deepen the best few with Extract so Nemotron reads the page, not just a snippet.
    top = sorted(sources, key=lambda s: -(s.score or 0))[:extract_top]
    if top:
        try:
            ext = client.extract([s.url for s in top], query=claim)
            by_url = {e.get("url"): e.get("raw_content") or "" for e in ext.get("results", [])}
            for s in top:
                if by_url.get(s.url):
                    s.content = by_url[s.url][:6000]
        except requests.HTTPError:
            pass  # snippets are still usable
    return sources
