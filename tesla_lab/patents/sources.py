"""Where planner hits come from (live): Tavily discovery restricted to patents.google.com, and the optional
BigQuery path (Google Patents Public Datasets). Both are injectable so tests never touch the network.

Tavily search accepts `include_domains` (Tavily search API docs, checked 2026-10-12/13). Results are filtered to
individual /patent/ URLs; search-result pages are never requested by Tesla Lab itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from .gpatents import normalize_pub, pub_from_url


@dataclass
class Seed:
    pub: str
    found_via: str
    title: str = ""
    snippet: str = ""
    score: float | None = None


class TavilyPatents:
    """Keyword discovery of patent pages. `client` is a research.TavilyClient (or a mock with .search())."""

    def __init__(self, settings=None, client=None):
        if client is None:
            from ..research import TavilyClient

            client = TavilyClient(settings)
        self.client = client

    def search(self, query: str, max_results: int = 8) -> list[Seed]:
        data = self.client.search(query, max_results=max_results, depth="advanced",
                                  include_domains=["patents.google.com"])
        out: list[Seed] = []
        for r in data.get("results", []):
            pub = pub_from_url(r.get("url", "").split("?")[0].split("#")[0])
            if not pub:
                continue
            if pub not in {s.pub for s in out}:
                out.append(Seed(pub=pub, found_via=f"tavily: {query}", title=r.get("title", ""),
                                snippet=(r.get("content") or "")[:400], score=r.get("score")))
        return out

    def web(self, query: str, max_results: int = 5) -> list[dict]:
        """Open-web search (Wildcard niches like biology / nature)."""
        data = self.client.search(query, max_results=max_results, depth="basic")
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": (r.get("content") or "")[:500]}
                for r in data.get("results", [])]


class BigQueryUnavailable(RuntimeError):
    pass


class BigQueryPatents:
    """OPTIONAL bulk path: patents-public-data on BigQuery (Google Cloud blog 2017-10-31; table docs at
    github.com/google/patents-public-data). Needs `pip install google-cloud-bigquery`, Google credentials
    (gcloud auth application-default login) and GOOGLE_CLOUD_PROJECT. UNTESTED live in this repo.

    Cost guard: every query is dry-run first and refused above `max_gb` scanned (the first 1 TiB/month is free on
    on-demand pricing; patents.publications is ~900 GB in full, so select few columns and filter by CPC).
    """

    def __init__(self, project: str | None, max_gb: float = 20.0, client=None):
        self.project = project
        self.max_gb = max_gb
        if client is None:
            try:
                from google.cloud import bigquery  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise BigQueryUnavailable("google-cloud-bigquery is not installed") from exc
            if not project:
                raise BigQueryUnavailable("set GOOGLE_CLOUD_PROJECT for the BigQuery path")
            try:
                client = bigquery.Client(project=project)
            except Exception as exc:  # noqa: BLE001
                raise BigQueryUnavailable(f"no Google credentials ({type(exc).__name__})") from exc
        self.client = client

    def _run(self, sql: str, params: dict) -> list[dict]:
        from google.cloud import bigquery  # type: ignore

        qp = []
        for k, v in params.items():
            if isinstance(v, list):
                qp.append(bigquery.ArrayQueryParameter(k, "STRING", v))
            else:
                qp.append(bigquery.ScalarQueryParameter(k, "STRING", v))
        dry = self.client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=qp, dry_run=True,
                                                                          use_query_cache=False))
        gb = (dry.total_bytes_processed or 0) / 1e9
        if gb > self.max_gb:
            raise BigQueryUnavailable(f"query would scan {gb:.0f} GB (> PATENTS_BQ_MAX_GB={self.max_gb:g})")
        job = self.client.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=qp, maximum_bytes_billed=int(self.max_gb * 1e9)))
        return [dict(r) for r in job.result()]

    SQL_CPC = """
SELECT p.publication_number, ANY_VALUE(t.text) AS title, ANY_VALUE(p.priority_date) AS priority_date
FROM `patents-public-data.patents.publications` p, UNNEST(p.cpc) c, UNNEST(p.title_localized) t
WHERE STARTS_WITH(c.code, @cpc_prefix) AND t.language = 'en'
  AND REGEXP_CONTAINS(LOWER(t.text), @pattern)
GROUP BY p.publication_number
LIMIT 20"""

    SQL_CITED_BY = """
SELECT c.publication_number
FROM `patents-public-data.google_patents_research.publications` r, UNNEST(r.cited_by) c
WHERE r.publication_number = @pub"""

    def search_cpc(self, cpc_code: str, words: list[str]) -> list[Seed]:
        """Title-word search inside one CPC group (sketch; tune before relying on it)."""
        import re

        pattern = "|".join(re.escape(w.lower()) for w in words if w)[:500] or "."
        rows = self._run(self.SQL_CPC, {"cpc_prefix": cpc_code.replace(" ", ""), "pattern": pattern})
        out = []
        for r in rows:
            pub = normalize_pub(r["publication_number"])
            if pub:
                out.append(Seed(pub=pub, found_via=f"bigquery: cpc {cpc_code}", title=r.get("title") or ""))
        return out

    def cited_by(self, pub_bq: str) -> list[str]:
        rows = self._run(self.SQL_CITED_BY, {"pub": pub_bq})
        return [p for p in (normalize_pub(r["publication_number"]) for r in rows) if p]
