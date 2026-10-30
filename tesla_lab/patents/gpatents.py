"""Google Patents: a robots.txt-respecting reader for individual patent pages + query links for the user.

What patents.google.com/robots.txt says (fetched 2026-10-13):

    User-agent: *
    Disallow: /*
    Allow: /$
    Allow: /advanced$
    Allow: /patent/
    Allow: /sitemap/

So a search results page (``/?q=...``) is OFF LIMITS to scripts. This module therefore:
  * only ever fetches ``https://patents.google.com/patent/<number>[/<lang>]`` (hard allowlist + a
    longest-match robots evaluator; anything else raises ``DisallowedURL`` before any network call);
  * reads claims, CPC codes, "Patent Citations" (backward), "Cited By" (forward), legal status and dates
    from that page;
  * builds clickable Google Patents *search* links for the human (``search_link``) but never opens them;
  * caches every page it reads (``cache/patents/<number>.json``) so a demo replays without the network.

Google's own note on the status line: it "is an assumption and is not a legal conclusion". Tesla Lab adds
"not legal advice" to every status it shows.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlencode, urlparse

HOST = "patents.google.com"
BASE = f"https://{HOST}"
UA = "TeslaLab/4.0 (patent research prototype; reads individual /patent/ pages only)"

# Snapshot of https://patents.google.com/robots.txt (2026-10-13). Used when the live file can't be read.
ROBOTS_SNAPSHOT = """User-agent: *
Disallow: /*
Allow: /$
Allow: /advanced$
Allow: /patent/
Allow: /sitemap/
"""
PATENT_PATH = re.compile(r"^/patent/([A-Z]{2}[A-Z0-9]{3,20})(?:/([a-z]{2}))?/?$")
PUB_RE = re.compile(r"\b([A-Z]{2})[-\s]?(RE|D|PP)?[-\s]?(\d[\d,]{2,12})[-\s]?([A-Z]\d?)?\b")

LEGAL_NOTE = "Not legal advice: status and expiry are Google's assumption, check with a patent attorney."


class DisallowedURL(ValueError):
    """Raised before any request is made for a URL robots.txt (or our allowlist) does not permit."""


# ----------------------------------------------------------------------------- robots
def parse_robots(text: str, agent: str = "*") -> list[tuple[str, str]]:
    """Return [(allow|disallow, pattern)] for the group that applies to `agent` (only '*' is used here)."""
    rules: list[tuple[str, str]] = []
    active = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        k = k.lower()
        if k == "user-agent":
            active = v == agent or v == "*"
        elif active and k in ("allow", "disallow") and v:
            rules.append((k, v))
    return rules


def _pattern_match(pattern: str, path: str) -> bool:
    rx = "^" + re.escape(pattern).replace(r"\*", ".*")
    if rx.endswith(r"\$"):
        rx = rx[:-2] + "$"
    return re.match(rx, path) is not None


def robots_allows(path_and_query: str, rules: list[tuple[str, str]]) -> bool:
    """Google's rule: the most specific (longest) matching pattern wins; Allow wins a tie. No match = allowed."""
    best: tuple[int, bool] | None = None
    for kind, pat in rules:
        if _pattern_match(pat, path_and_query):
            cand = (len(pat), kind == "allow")
            if best is None or cand > best:
                best = cand
    return True if best is None else best[1]


def check_url(url: str, rules: list[tuple[str, str]] | None = None) -> str:
    """Validate a URL for fetching. Returns the canonical publication number. Raises DisallowedURL."""
    u = urlparse(url)
    if u.scheme != "https" or u.netloc != HOST:
        raise DisallowedURL(f"not a {HOST} URL: {url}")
    pq = u.path + (("?" + u.query) if u.query else "")
    if u.query or u.fragment:
        raise DisallowedURL(f"query strings are search pages (disallowed by robots.txt): {url}")
    m = PATENT_PATH.match(u.path)
    if not m:
        raise DisallowedURL(f"only individual /patent/<number> pages are read: {url}")
    if not robots_allows(pq, rules if rules is not None else parse_robots(ROBOTS_SNAPSHOT)):
        raise DisallowedURL(f"robots.txt disallows {pq}")
    return m.group(1)


# ----------------------------------------------------------------------------- numbers + links
def normalize_pub(text: str) -> str | None:
    """'US 787,412 A' / 'US-787412-A' / 'US787412A' -> 'US787412A' (None if no number found)."""
    t = text.strip().upper().replace(",", "")
    m = re.match(r"^([A-Z]{2})[-\s]?([A-Z]{0,2}\d+)[-\s]?([A-Z]\d?)?$", t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3) or ''}"
    m = PUB_RE.search(t)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2) or ''}{m.group(3).replace(',', '')}{m.group(4) or ''}"


def patent_url(pub: str, lang: str = "en") -> str:
    return f"{BASE}/patent/{pub}/{lang}"


def pub_from_url(url: str) -> str | None:
    m = PATENT_PATH.match(urlparse(url).path)
    return m.group(1) if m else None


def bq_pub(pub: str) -> str:
    """'US787412A' -> 'US-787412-A' (the publication_number format in the BigQuery tables)."""
    m = re.match(r"^([A-Z]{2})([A-Z]{0,2}\d+)([A-Z]\d?)?$", pub)
    if not m:
        return pub
    return "-".join(x for x in (m.group(1), m.group(2), m.group(3)) if x)


def search_link(q: str = "", *, cpc: list[str] | None = None, before: str | None = None, after: str | None = None,
                country: str | None = None, status: str | None = None) -> str:
    """A clickable Google Patents search URL FOR THE USER. Tesla Lab never fetches it (robots.txt).

    Uses the URL parameters the Google Patents UI writes (q, cpc, before/after as priority:YYYYMMDD,
    country, status). The query syntax inside q follows Google Patents Help "Searching"
    (support.google.com/faqs/answer/7049475): OR, quotes, minus to negate, field prefixes.
    """
    params: list[tuple[str, str]] = []
    if q:
        params.append(("q", q))
    if cpc:
        params.append(("cpc", ",".join(c.replace(" ", "") for c in cpc)))
    if before:
        params.append(("before", f"priority:{before.replace('-', '')}"))
    if after:
        params.append(("after", f"priority:{after.replace('-', '')}"))
    if country:
        params.append(("country", country))
    if status:
        params.append(("status", status))
    return f"{BASE}/?" + urlencode(params, quote_via=quote, safe="():,/")


# ----------------------------------------------------------------------------- page model
@dataclass
class CitedDoc:
    pub: str
    title: str = ""
    priority_date: str = ""
    publication_date: str = ""
    assignee: str = ""
    examiner_cited: bool = False


@dataclass
class PatentPage:
    pub: str
    url: str
    title: str = ""
    abstract: str = ""
    claims: list[str] = field(default_factory=list)  # one string per claim, in order
    cpc: list[dict] = field(default_factory=list)  # [{code, title, first, leaf}]
    backward: list[CitedDoc] = field(default_factory=list)  # "Patent Citations"
    forward: list[CitedDoc] = field(default_factory=list)  # "Cited By"
    status_text: str = ""  # Google's legal status line, e.g. "Expired - Lifetime"
    priority_date: str = ""
    filing_date: str = ""
    publication_date: str = ""
    grant_date: str = ""
    expiration_date: str = ""  # "Anticipated expiration" when Google shows it
    assignee: str = ""
    inventors: list[str] = field(default_factory=list)
    fetched_at: str = ""
    source: str = "live"  # live | cache | fixture
    claims_from_ocr: bool = False  # True when claims came from the OCR'd description (pre-1920s patents)
    forward_count: int = 0  # true counts (fixtures keep only the first rows of each list)
    backward_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PatentPage":
        d = dict(d)
        d["backward"] = [CitedDoc(**c) for c in d.get("backward", [])]
        d["forward"] = [CitedDoc(**c) for c in d.get("forward", [])]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def claims_text(self) -> str:
        return "\n".join(self.claims)

    @property
    def cpc_codes(self) -> list[str]:
        return [c["code"] for c in self.cpc]


def _txt(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el is not None else ""


def _cited_rows(soup, prefixes: tuple[str, ...]) -> list[CitedDoc]:
    out: dict[str, CitedDoc] = {}
    for tr in soup.find_all("tr"):
        prop = tr.get("itemprop") or ""
        if not any(prop == p for p in prefixes):
            continue
        num = tr.find(attrs={"itemprop": "publicationNumber"})
        pub = _txt(num)
        if not pub or pub in out:
            continue
        out[pub] = CitedDoc(
            pub=pub, title=_txt(tr.find(attrs={"itemprop": "title"})),
            priority_date=_txt(tr.find(attrs={"itemprop": "priorityDate"})),
            publication_date=_txt(tr.find(attrs={"itemprop": "publicationDate"})),
            assignee=_txt(tr.find(attrs={"itemprop": "assigneeOriginal"})),
            examiner_cited=bool(tr.find(attrs={"itemprop": "examinerCited"})),
        )
    return list(out.values())


def parse_patent_html(html: str, pub: str | None = None, url: str = "") -> PatentPage:
    """Parse a patents.google.com/patent/... page (schema.org microdata: itemprop=...)."""
    from bs4 import BeautifulSoup  # lazy import

    soup = BeautifulSoup(html, "html.parser")
    pub = pub or _txt(soup.find(attrs={"itemprop": "publicationNumber"}))
    title = _txt(soup.find(attrs={"itemprop": "title"}))
    if not title:
        meta = soup.find("meta", attrs={"name": "DC.title"})
        title = (meta.get("content") or "").strip() if meta else ""
    abstract = _txt(soup.find(attrs={"itemprop": "abstract"}) or soup.find("abstract"))
    abstract = re.sub(r"^Abstract\s+", "", abstract)

    claims: list[str] = []
    sec = soup.find("section", attrs={"itemprop": "claims"})
    if sec is not None:
        blocks = sec.find_all("div", class_="claim")
        top = [b for b in blocks if not b.find_parent("div", class_="claim")]
        claims = [_txt(b) for b in top if _txt(b)]
        if not claims:
            claims = [c for c in re.split(r"(?=\b\d{1,3}\s*\.\s)", _txt(sec)) if c.strip()]
    claims_from_ocr = False
    if not claims:  # old patents: the claims sit at the end of the OCR'd description
        desc = _txt(soup.find(attrs={"itemprop": "description"}))
        cut = max(desc.rfind(k) for k in ("What I claim", "I claim", "We claim", "claim as my invention",
                                           "claim as our invention"))
        if cut >= 0:
            tail = desc[cut:cut + 6000]
            claims = [c.strip() for c in re.split(r"(?=\s\d{1,2}\.\s)", tail) if len(c.strip()) > 15]
            claims_from_ocr = bool(claims)

    cpc: list[dict] = []
    seen: set[str] = set()
    for li in soup.find_all("li", attrs={"itemprop": "classifications"}):
        code = _txt(li.find(attrs={"itemprop": "Code"}))
        if "/" not in code or code in seen:
            continue
        if li.find("meta", attrs={"itemprop": "IsCPC"}) is None:
            continue
        seen.add(code)
        cpc.append({"code": code, "title": _txt(li.find(attrs={"itemprop": "Description"})),
                    "first": li.find("meta", attrs={"itemprop": "FirstCode"}) is not None,
                    "leaf": li.find("meta", attrs={"itemprop": "Leaf"}) is not None})

    def first(prop: str) -> str:
        return _txt(soup.find(attrs={"itemprop": prop}))

    status = first("legalStatusIfi") or first("status")
    expiration = ""
    for ev in soup.find_all(attrs={"itemprop": "events"}):
        t = _txt(ev)
        if "expiration" in t.lower():
            m = re.search(r"\d{4}-\d{2}-\d{2}", t)
            expiration = m.group(0) if m else expiration
    grant = ""
    for ev in soup.find_all(attrs={"itemprop": "events"}):
        t = _txt(ev)
        if "granted" in t.lower() or "publication of" in t.lower() and "grant" in t.lower():
            m = re.search(r"\d{4}-\d{2}-\d{2}", t)
            grant = m.group(0) if m else grant
    inventors = [_txt(x) for x in soup.find_all(attrs={"itemprop": "inventor"})]
    backward = _cited_rows(soup, ("backwardReferencesOrig", "backwardReferences", "backwardReferencesFamily"))
    forward = _cited_rows(soup, ("forwardReferencesOrig", "forwardReferences", "forwardReferencesFamily"))
    return PatentPage(
        pub=pub, url=url or patent_url(pub), title=title, abstract=abstract[:3000], claims=claims, cpc=cpc,
        claims_from_ocr=claims_from_ocr, backward=backward, forward=forward, backward_count=len(backward), forward_count=len(forward),
        status_text=re.sub(r"\s+,", ",", status), priority_date=first("priorityDate"), filing_date=first("filingDate"),
        publication_date=first("publicationDate"), grant_date=grant, expiration_date=expiration,
        assignee=first("assigneeOriginal"), inventors=list(dict.fromkeys(i for i in inventors if i)),
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )


# ----------------------------------------------------------------------------- fetcher
class PatentFetcher:
    """Reads individual patent pages, politely, with a disk cache. Never fetches search pages.

    `http_get(url, headers, timeout) -> (status_code, text)` is injectable so tests never touch the network.
    """

    def __init__(self, cache_dir: str | Path = "cache/patents", http_get: Callable | None = None,
                 min_interval_s: float = 1.0, offline: bool = False, fixtures: dict[str, dict] | None = None,
                 max_age_days: float = 30.0):
        self.cache = Path(cache_dir)
        self.http_get = http_get or _requests_get
        self.min_interval_s = min_interval_s
        self.offline = offline
        self.fixtures = fixtures or {}
        self.max_age_days = max_age_days
        self._last = 0.0
        self._rules: list[tuple[str, str]] | None = None
        self.requested: list[str] = []  # every URL actually sent to the network (tests read this)
        self.log: list[str] = []

    # robots.txt is read once per fetcher; the snapshot is the fallback
    def rules(self) -> list[tuple[str, str]]:
        if self._rules is None:
            text = ROBOTS_SNAPSHOT
            if not self.offline:
                try:
                    url = f"{BASE}/robots.txt"
                    self.requested.append(url)
                    code, body = self.http_get(url, {"User-Agent": UA}, 20)
                    if code == 200 and "user-agent" in body.lower():
                        text = body
                except Exception as exc:  # noqa: BLE001
                    self.log.append(f"robots.txt unreachable ({exc}); using the 2026-10-13 snapshot")
            self._rules = parse_robots(text)
        return self._rules

    def available(self, pub: str) -> bool:
        """True when the page can be read without the network (fixture or cache)."""
        return pub in self.fixtures or self._cache_path(pub).exists()

    def _cache_path(self, pub: str) -> Path:
        return self.cache / f"{pub}.json"

    def get(self, pub_or_url: str) -> PatentPage | None:
        url = pub_or_url if pub_or_url.startswith("http") else patent_url(normalize_pub(pub_or_url) or pub_or_url)
        if self.offline:
            # still validate, so offline/mock runs prove the same URL discipline
            pub = check_url(url, parse_robots(ROBOTS_SNAPSHOT))
        else:
            pub = check_url(url, self.rules())
        if pub in self.fixtures:
            page = PatentPage.from_dict(self.fixtures[pub])
            page.source = "fixture"
            return page
        cp = self._cache_path(pub)
        if cp.exists():
            try:
                d = json.loads(cp.read_text(encoding="utf-8"))
                age_days = (time.time() - cp.stat().st_mtime) / 86400
                if age_days <= self.max_age_days or self.offline:
                    page = PatentPage.from_dict(d)
                    page.source = "cache"
                    return page
            except Exception:  # noqa: BLE001
                pass
        if self.offline:
            return None
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self.requested.append(url)
        self._last = time.monotonic()
        code, html = self.http_get(url, {"User-Agent": UA, "Accept-Language": "en"}, 30)
        if code != 200:
            self.log.append(f"{pub}: HTTP {code}")
            return None
        page = parse_patent_html(html, pub=pub, url=url)
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(page.to_dict(), ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return page


def _requests_get(url: str, headers: dict, timeout: int) -> tuple[int, str]:
    import requests

    r = requests.get(url, headers=headers, timeout=timeout)
    r.encoding = "utf-8"
    return r.status_code, r.text


# ----------------------------------------------------------------------------- legal status
def likely_expired_flag(page: PatentPage, today: dt.date | None = None) -> tuple[str, str]:
    """(flag, why). flag in {expired, likely_expired, active, pending, unknown}. A FLAG, not a legal conclusion.

    US utility patents generally run 20 years from the earliest non-provisional filing (35 U.S.C. 154),
    extended (term adjustment) or cut short (unpaid maintenance fees, terminal disclaimers).
    """
    today = today or dt.date.today()
    s = (page.status_text or "").lower()
    if any(w in s for w in ("expired", "lapsed", "withdrawn", "abandoned", "ceased", "revoked")):
        return "expired", f"Google shows “{page.status_text}”"
    if page.expiration_date:
        try:
            exp = dt.date.fromisoformat(page.expiration_date)
            if exp < today:
                return "likely_expired", f"anticipated expiration {page.expiration_date} has passed"
            return "active", f"Google shows “{page.status_text or 'active'}”, anticipated expiration {page.expiration_date}"
        except ValueError:
            pass
    base = page.filing_date or page.priority_date
    if base:
        try:
            y = dt.date.fromisoformat(base)
            if (today - y).days > 20 * 365.25 + 30:
                return "likely_expired", f"filed {base}: more than 20 years ago"
        except ValueError:
            pass
    if "pending" in s or "application" in s:
        return "pending", f"Google shows “{page.status_text}”"
    if "active" in s:
        return "active", f"Google shows “{page.status_text}”"
    return "unknown", "no status on the page"
