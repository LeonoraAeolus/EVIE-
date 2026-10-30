"""💡 Wildcard: borrow the same physical principle from a completely unrelated niche (spec v2).

    principle  -> strip product/industry words, keep the physics and its numbers      [model]
    niches     -> far CPC sections/classes + non-patent biology/nature                  [code]
    search     -> Tavily (patents.google.com for CPC niches, open web + arXiv otherwise) [live]
    transfer   -> "why not try this?" + a sizing calculator and its params              [model]
    sizing     -> Python computes the number (calculator of record), OpenMath re-derives it
    rank       -> score = surprise × math_fit

surprise: CPC distance from the nearest home code — same subclass 0 · same class 0.25 · same section 0.5 ·
          other section or non-patent 1.0; ×0.25 when the same mechanism already appears in the home-class hits.
math_fit: 1 = sizes inside every constraint · 0.5 = works but stretches one limit (by ≤ 2×) ·
          0 = numbers don't work (physics doesn't apply, a limit broken by > 2×, or two limits stretched).
          math_fit 0 ideas are kept under "fun, but no".
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..schema import Constraint, SearchPlan, Sizing, WildcardIdea
from . import cpc as CPC
from .brain import MOCK_DIR, CannedBrain
from .sizing import CALCULATORS, describe, run_sizing

FAR_NICHES = [
    {"niche": "Music (wind and string instruments)", "cpc": "G10D", "web": False},
    {"niche": "Toys", "cpc": "A63H", "web": False},
    {"niche": "Kitchen tools", "cpc": "A47J", "web": False},
    {"niche": "Biology and medicine (heart, blood vessels)", "cpc": "A61F2/24", "web": True},
    {"niche": "Nature and biomimicry", "cpc": None, "web": True},
]
MECH_PENALTY = 0.25


def surprise(idea_cpc: str | None, home: list[str], mechanism_keywords: list[str], home_hit_texts: list[str]
             ) -> tuple[float, str]:
    d = CPC.distance(idea_cpc, home) if idea_cpc else 1.0
    if idea_cpc:
        why = {0.0: "same subclass as the home class", 0.25: "same class, different subclass",
               0.5: "same CPC section, different class", 1.0: "a different CPC section"}[d]
        why = f"{CPC.pretty(idea_cpc)}: {why}"
    else:
        why = "non-patent niche"
    kws = [k.lower() for k in mechanism_keywords if k.strip()]
    found = [k for k in kws if any(k in t for t in home_hit_texts)]
    if found:
        d *= MECH_PENALTY
        why += f"; ×{MECH_PENALTY} because “{found[0]}” already appears in the home-class hits"
    return round(d, 4), why


def apply_constraints(sizes: dict, constraints: list[Constraint]) -> list[dict]:
    """[{constraint, value, limit, ratio, verdict: ok|stretch|break}] for constraints that apply to these sizes."""
    out = []
    for c in constraints:
        if not c.applies_to or c.value is None or c.applies_to not in sizes:
            continue
        v = sizes[c.applies_to]
        val = float(v[0] if isinstance(v, (list, tuple)) else v)
        unit = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else c.unit
        lim = float(c.value)
        if c.op == ">=":
            ok = val >= lim
            ratio = (lim / val) if val > 0 else float("inf")
        else:
            ok = val <= lim
            ratio = (val / lim) if lim > 0 else float("inf")
        verdict = "ok" if ok else ("stretch" if ratio <= 2.0 else "break")
        out.append({"constraint": c.text, "value": val, "unit": unit, "limit": lim, "op": c.op,
                    "ratio": round(ratio, 3), "verdict": verdict})
    return out


def math_fit(physics_ok: bool, checks: list[dict], physics_note: str = "") -> tuple[float, str]:
    if not physics_ok:
        return 0.0, f"the physics doesn't apply: {physics_note}"
    breaks = [c for c in checks if c["verdict"] == "break"]
    stretches = [c for c in checks if c["verdict"] == "stretch"]

    def fmt(c):
        sym = ">" if c["op"] != ">=" else "<"
        return f"{c['value']:g} {c['unit']} {sym} {c['limit']:g} {c['unit']} ({c['constraint']})"

    if breaks:
        return 0.0, "breaks a limit by more than 2×: " + fmt(breaks[0])
    if len(stretches) >= 2:
        return 0.0, "stretches two limits: " + "; ".join(fmt(c) for c in stretches)
    if len(stretches) == 1:
        return 0.5, "works but stretches one limit: " + fmt(stretches[0])
    oks = [c for c in checks if c["verdict"] == "ok"]
    if oks:
        return 1.0, "inside every limit: " + "; ".join(f"{c['value']:g} {c['unit']} vs {c['limit']:g}" for c in oks)
    return 1.0, "no team limit applies to these sizes"


def _canned_derivations() -> dict:
    p = MOCK_DIR / "derivations.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _size(idea: dict, quantities: dict, settings) -> tuple[Sizing, dict | None]:
    calc = idea.get("calculator")
    params = {**{k: v for k, v in quantities.items()}, **(idea.get("params") or {})}
    sz = Sizing(calculator=calc or "custom", params=params)
    try:
        if calc in CALCULATORS:
            r = run_sizing(calc, params)
        elif idea.get("custom_code"):
            from ..sandbox import run_code

            res = run_code(idea["custom_code"], getattr(settings, "sandbox_backend", "local"))
            r = res.get("result") if isinstance(res, dict) and "result" in res else res
            if not isinstance(r, dict) or "value" not in r:
                raise ValueError(f"custom sizing code returned no result dict ({str(res)[:120]})")
            r.setdefault("physics_ok", True)
            r.setdefault("sizes", {})
            r.setdefault("physics_note", "")
            r.setdefault("formula", "model-written code (sandboxed)")
        else:
            raise ValueError(f"no sizing calculator for this idea ({calc!r})")
    except Exception as exc:  # noqa: BLE001
        sz.error = f"{type(exc).__name__}: {exc}"[:300]
        sz.physics_ok = False
        sz.physics_note = "sizing failed, so the idea cannot be scored"
        return sz, None
    sz.quantity = r.get("quantity", "")
    sz.value = r.get("value")
    sz.unit = r.get("unit", "")
    sz.formula = r.get("formula", "")
    sz.sizes = {k: list(v) if isinstance(v, tuple) else v for k, v in (r.get("sizes") or {}).items()}
    sz.physics_ok = bool(r.get("physics_ok", True))
    sz.physics_note = r.get("physics_note", "")
    sz.problem = (r.get("check") or {}).get("problem", "")
    return sz, r.get("check")


def _math_check(items: list[tuple[WildcardIdea, dict]], plan: SearchPlan, settings) -> None:
    """OpenMath re-derives each sizing number (canned derivations in mock mode)."""
    from .. import mathcheck as MC

    if not items:
        return
    mock = plan.data_mode == "mock"
    info = MC.info_for(settings, mock=mock)
    canned = _canned_derivations() if info.source == "canned" else {}
    client = None
    if info.source == "openmath":
        try:
            from ..openmath import OpenMathClient

            client = OpenMathClient(settings)
        except Exception as exc:  # noqa: BLE001
            for w, _ in items:
                w.sizing.math_check = MC.skipped(f"wildcard:{w.niche}", str(exc), "error")
            return

    def one(item):
        w, chk = item
        tid = f"wildcard:{w.sizing.calculator}"
        if not info.enabled:
            return w, MC.skipped(tid, info.reason)
        prob = {"problem": chk["problem"], "quantity": chk.get("quantity", ""), "unit": chk.get("unit", ""),
                "variables": chk.get("variables") or {}, "reference": chk.get("reference", ""),
                "target_metric": "value", "python_value": chk.get("python_value")}
        pv = chk.get("python_value")
        if info.source == "canned":
            entry = canned.get(w.sizing.calculator)
            if not entry or abs(float(entry.get("for_value", -1)) - float(pv or 0)) > 1e-9 * max(1, abs(pv or 0)) + 1e-9:
                return w, MC.skipped(tid, "no canned derivation for these inputs (mock mode)", "inconclusive", "canned")
            return w, MC.build_check(tid, prob, entry["response"], python_value=pv, tol=settings.math_check_tolerance,
                                     source="canned", model=info.model, origin="canned")
        try:
            comp = client.solve(prob["problem"])
        except Exception as exc:  # noqa: BLE001
            return w, MC.skipped(tid, f"OpenMath call failed: {type(exc).__name__}: {exc}"[:300], "error", "openmath")
        return w, MC.build_check(tid, prob, comp.text, reasoning=comp.reasoning, python_value=pv,
                                 tol=settings.math_check_tolerance, source="openmath",
                                 model=comp.model or settings.openmath_model, origin="design",
                                 latency_s=comp.latency_s, completion_tokens=comp.completion_tokens)

    workers = 1 if info.source != "openmath" else min(4, len(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for w, chk in ex.map(one, items):
            w.sizing.math_check = chk


def _niches_live(plan: SearchPlan, home: list[str]) -> list[dict]:
    home_secs = {CPC.parts(h).get("section") for h in home}
    out = [n for n in FAR_NICHES if not n["cpc"] or CPC.parts(n["cpc"]).get("section") not in home_secs
           or CPC.distance(n["cpc"], home) >= 0.5]
    extra = CPC.match_titles([plan.wildcard_principle], only_sections=set(CPC.SECTIONS) - home_secs - {"Y"}, top=1)
    for m in extra:
        out.append({"niche": f"Other CPC section: {CPC.SECTIONS[m['code'][0]].split(';')[0]}", "cpc": m["code"],
                    "web": False})
    return out[:6]


def _snippets_live(planner, niche: dict, principle: str) -> list[dict]:
    disc = planner.discovery
    snips: list[dict] = []
    if disc is None:
        return snips
    q = f"{niche['niche']} {principle}"[:380]
    try:
        if niche.get("web"):
            snips += disc.web(q, max_results=4)
            from ..gaps import arxiv_search

            for ld in arxiv_search(principle, max_results=2):
                snips.append({"title": ld.title, "url": ld.url, "content": ld.snippet if hasattr(ld, "snippet") else ""})
        else:
            for sd in disc.search(q, max_results=4):
                snips.append({"title": sd.title, "url": f"https://patents.google.com/patent/{sd.pub}/en",
                              "content": sd.snippet})
    except Exception as exc:  # noqa: BLE001
        plan_note = f"Wildcard search failed for {niche['niche']}: {type(exc).__name__}"
        snips.append({"title": plan_note, "url": "", "content": ""})
    return snips


def run_wildcard(plan: SearchPlan, planner, last: SearchPlan | None = None) -> None:
    brain = planner.brain
    settings = planner.s
    ab = plan.abstraction
    pr = brain.principle(ab)
    plan.wildcard_principle = pr["principle"]
    quantities = {**(ab.quantities or {}), **(pr.get("quantities") or {})}
    home = [c.code for c in plan.cpc if c.role == "home"]
    home_texts = [(h.title + " " + h.claim_quote + " " + h.fit_why).lower() for h in plan.hits if h.role == "home"]
    # the claims of home-class hits too (mechanism already used at home?)
    for h in plan.hits:
        if h.role == "home":
            try:
                page = planner.fetcher.get(h.pub)
                if page:
                    home_texts.append((page.abstract + " " + page.claims_text).lower())
            except Exception:  # noqa: BLE001
                pass
    if isinstance(brain, CannedBrain):
        niches = [{"niche": n, "cpc": None} for n in brain.niches()]
        snippets = {n["niche"]: [] for n in niches}
    else:
        niches = _niches_live(plan, home)
        snippets = {n["niche"]: _snippets_live(planner, n, plan.wildcard_principle) for n in niches}
    ideas: list[WildcardIdea] = []
    checks: list[tuple[WildcardIdea, dict]] = []
    for n in niches:
        try:
            raw = brain.transfer(plan.wildcard_principle, n["niche"], snippets.get(n["niche"], []), describe())
        except Exception as exc:  # noqa: BLE001
            plan.notes.append(f"Wildcard transfer failed for {n['niche']}: {type(exc).__name__}")
            continue
        for r in raw:
            code = CPC.normalize(r.get("cpc") or n.get("cpc") or "") if (r.get("cpc") or n.get("cpc")) else None
            ev = r.get("evidence") or [{"title": u, "url": u, "kind": "web"} for u in r.get("evidence_urls", [])]
            w = WildcardIdea(niche=r.get("niche") or n["niche"], cpc=code, mechanism=r.get("mechanism", ""),
                             transfer=r.get("transfer", ""), evidence=ev)
            w.surprise, w.surprise_why = surprise(code, home, r.get("mechanism_keywords", []), home_texts)
            sz, chk = _size(r, quantities, settings)
            cons = apply_constraints(sz.sizes, plan.input.constraints)
            sz.constraint_notes = [f"{c['verdict']}: {c['value']:g} {c['unit']} vs limit {c['op']} {c['limit']:g} "
                                   f"({c['constraint']})" for c in cons]
            w.math_fit, w.math_fit_why = math_fit(sz.physics_ok, cons, sz.physics_note)
            if sz.error:
                w.math_fit_why = f"sizing failed: {sz.error}"
            w.score = round(w.surprise * w.math_fit, 4)
            w.bucket = "try" if w.math_fit > 0 else "fun_but_no"
            w.sizing = sz
            ideas.append(w)
            if chk and not sz.error:
                checks.append((w, chk))
    _math_check(checks, plan, settings)
    ideas.sort(key=lambda w: (w.bucket != "try", -w.score, -w.surprise))
    plan.wildcard = ideas
