"""CPC (Cooperative Patent Classification): local title table, code parsing, distance, validation, and
cross-industry matching.

Code anatomy, e.g. F01N 1/023:
    F        section     (Mechanical engineering ...)
    F01      class
    F01N     subclass    (Gas-flow silencers or exhaust apparatus ...)
    F01N1/00 main group  (Silencing apparatus characterised by method of silencing)
    F01N1/023 subgroup   (... by using resonance — Helmholtz resonators)

Distance from the home class (used by the Wildcard's `surprise`, spec v2):
    same subclass 0 · same class 0.25 · same section 0.5 · other section (or non-patent) 1.0
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data" / "cpc_titles.json"
SECTIONS = {
    "A": "Human necessities", "B": "Performing operations; transporting", "C": "Chemistry; metallurgy",
    "D": "Textiles; paper", "E": "Fixed constructions",
    "F": "Mechanical engineering; lighting; heating; weapons; blasting", "G": "Physics", "H": "Electricity",
    "Y": "General tagging of new technological developments",
}
CODE_RE = re.compile(r"^([A-HY])(\d{2})([A-Z])\s*(?:(\d{1,4})\s*/\s*(\d{2,6}))?$")
STOP = set("""a an the of or and for in on to by with from as at into other than not e.g i.e using use used
thereof therefor such which wherein being be is are their its this that these those means devices device
apparatus apparatuses method methods systems system arrangement arrangements general specially adapted
characterised characterized provided groups group details""".split())


def normalize(code: str) -> str | None:
    """'F01N 1/023' / 'f01n1/023' / 'F01N' -> canonical 'F01N1/023' / 'F01N'. None if malformed."""
    c = re.sub(r"\s+", "", (code or "").upper())
    m = CODE_RE.match(c)
    if not m:
        return None
    sec, cls, sub, main, grp = m.groups()
    base = f"{sec}{cls}{sub}"
    return base if main is None else f"{base}{int(main)}/{grp}"


def pretty(code: str) -> str:
    """'F01N1/023' -> 'F01N 1/023'."""
    c = normalize(code) or code
    return c if len(c) <= 4 else f"{c[:4]} {c[4:]}"


def parts(code: str) -> dict:
    c = normalize(code)
    if not c:
        return {}
    d = {"section": c[0], "class": c[:3], "subclass": c[:4]}
    if len(c) > 4:
        d["main_group"] = c.split("/")[0] + "/00"
    return d


@lru_cache(maxsize=1)
def table() -> dict[str, dict]:
    """code -> {title, full, dots, source}. Full titles are rebuilt from the dot hierarchy."""
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    stack: dict[int, str] = {}
    for code, dots, title in raw["scheme"]:
        if dots < 0:
            out[code] = {"title": title, "full": title, "dots": -1, "source": "uspto"}
            stack = {}
            continue
        stack[dots] = title
        for k in [k for k in stack if k > dots]:
            del stack[k]
        full = " — ".join(stack[i] for i in range(dots + 1) if i in stack)
        out[code] = {"title": title, "full": full, "dots": dots, "source": "uspto"}
    for code, title in raw.get("extra", {}).items():
        out.setdefault(code, {"title": title, "full": title, "dots": None, "source": "google_patents_page"})
    return out


def learn(code: str, title: str) -> None:
    """Add a title read from a fetched patent page (in memory only)."""
    c = normalize(code)
    if c and title:
        table().setdefault(c, {"title": title, "full": title, "dots": None, "source": "google_patents_page"})


def title(code: str) -> str:
    c = normalize(code) or ""
    t = table()
    if c in t:
        return t[c]["full"]
    p = parts(c)
    if p.get("main_group") in t:
        return t[p["main_group"]]["full"] + " (subgroup title not in the local table)"
    if p.get("subclass") in t:
        return t[p["subclass"]]["title"].capitalize() + " (group title not in the local table)"
    return ""


def validate(code: str) -> tuple[str | None, str]:
    """(canonical code or None, note). Known codes pass; well-formed codes under a known subclass pass with a
    note; malformed codes and unknown subclasses are dropped (spec: 'unknown codes dropped')."""
    c = normalize(code)
    if not c:
        return None, f"dropped “{code}”: not a CPC code"
    t = table()
    if c in t:
        return c, ""
    p = parts(c)
    if p.get("main_group") in t:
        return c, "group title not in the local table (subgroup kept: its main group exists)"
    if len(c) == 4 and c in t:
        return c, ""
    return None, f"dropped “{pretty(c)}”: not in the local CPC table (add its subclass with scripts/fetch_cpc_titles.py)"


def similarity(a: str, b: str) -> float:
    """1 same main group · 0.75 same subclass · 0.5 same class · 0.25 same section · 0 otherwise."""
    pa, pb = parts(a), parts(b)
    if not pa or not pb:
        return 0.0
    if pa.get("main_group") and pa.get("main_group") == pb.get("main_group"):
        return 1.0
    if pa["subclass"] == pb["subclass"]:
        return 0.75
    if pa["class"] == pb["class"]:
        return 0.5
    if pa["section"] == pb["section"]:
        return 0.25
    return 0.0


def distance(code: str | None, home: list[str]) -> float:
    """Spec v2: same subclass 0 · same class 0.25 · same section 0.5 · other section / non-patent 1.0 (to the
    NEAREST home code)."""
    if not code or not normalize(code) or not home:
        return 1.0
    best = 1.0
    pc = parts(code)
    for h in home:
        ph = parts(h)
        if not ph:
            continue
        if pc["subclass"] == ph["subclass"]:
            d = 0.0
        elif pc["class"] == ph["class"]:
            d = 0.25
        elif pc["section"] == ph["section"]:
            d = 0.5
        else:
            d = 1.0
        best = min(best, d)
    return best


def cpc_match(hit_codes: list[str], plan_codes: list[str]) -> float:
    """Best similarity between any of a hit's codes and the plan's codes (ranking input, 0..1)."""
    best = 0.0
    for h in hit_codes:
        for p in plan_codes:
            best = max(best, similarity(h, p))
            if best == 1.0:
                return 1.0
    return best


# ----------------------------------------------------------------------------- cross-industry matching
def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ations", "ation", "ings", "ing", "ers", "er", "ies", "es", "s", "ed", "al"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "") if w.lower() not in STOP}


def match_titles(function_words: list[str], *, exclude_subclasses: set[str] = frozenset(),
                 only_sections: set[str] | None = None, top: int = 8, min_overlap: int = 2) -> list[dict]:
    """Rank table entries whose titles share the most (stemmed) words with the function description.

    Returns [{code, title, overlap, words}] — the 'same function in other industries' candidates.
    """
    q = set()
    for t in function_words:
        q |= words(t)
    scored = []
    for code, row in table().items():
        if len(code) <= 4 or row.get("dots") is None:
            continue
        p = parts(code)
        if p["subclass"] in exclude_subclasses:
            continue
        if only_sections and p["section"] not in only_sections:
            continue
        ws = words(row["full"])
        hit = q & ws
        if len(hit) >= min_overlap:
            # prefer specific (deeper) groups when overlap ties, and short titles (less dilution)
            scored.append((len(hit), len(hit) / (len(ws) ** 0.5 + 1), row.get("dots") or 0, code, sorted(hit)))
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    out, seen_sub = [], {}
    for n, _, _, code, hit in scored:
        sub = parts(code)["subclass"]
        if seen_sub.get(sub, 0) >= 2:  # diversity: at most 2 per subclass
            continue
        seen_sub[sub] = seen_sub.get(sub, 0) + 1
        out.append({"code": code, "title": table()[code]["full"], "overlap": n, "words": hit})
        if len(out) >= top:
            break
    return out
