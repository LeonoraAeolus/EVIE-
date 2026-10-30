"""The Tesla Lab pipeline.

research (Tavily + Tesla Archive) -> claims (Nemotron) -> math (Python) -> verdicts
  -> gap search for every MISSING PIECE (Tavily + arXiv)       [v2 innovation loop]
  -> product concept for the WORKS part -> spec sheet computed by a design calculator,
     render prompt written from the computed numbers           [v2 product design engine]
  -> math checker: OpenMath-Nemotron re-derives each result step by step, compared with the
     Python number; agreement / disagreement is flagged on every verdict  [v3]
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .calc_runner import run_calc
from .calcs import describe_templates
from .config import Settings, get_settings
from .schema import ArchiveHit, CalcResult, Claim, ClaimVerdict, ProductConcept, ProductSpec, Report, Source

StepCallback = Callable[[str, str], None]  # (step_name, message)

VERDICT_RANK = {"WORKS": 0, "MISSING PIECE": 1, "BREAKS PHYSICS": 2}


def _noop(step: str, msg: str) -> None:
    pass


class _Fmt(dict):
    def __missing__(self, key):  # leave unknown placeholders visible instead of crashing
        return "{" + key + "}"


def apply_guardrails(model_verdict: str, calc: CalcResult) -> tuple[str, str | None]:
    """The math outranks the model. Returns (final_verdict, note_if_overruled)."""
    if calc.flags.get("violates_energy_conservation") or calc.flags.get("violates_second_law"):
        if model_verdict != "BREAKS PHYSICS":
            return "BREAKS PHYSICS", f"Model said {model_verdict}; the calculation shows a conservation/second-law violation."
        return model_verdict, None
    if not calc.ok and model_verdict == "WORKS":
        return "MISSING PIECE", "The calculation could not run, so the claim is not confirmed."
    if calc.suggested_verdict and VERDICT_RANK.get(model_verdict, 1) < VERDICT_RANK[calc.suggested_verdict]:
        return calc.suggested_verdict, f"Model said {model_verdict}; the numbers only support {calc.suggested_verdict}."
    return model_verdict, None


def overall_line(verdicts: list[ClaimVerdict]) -> str:
    counts = {v: sum(1 for x in verdicts if x.verdict == v) for v in VERDICT_RANK}
    parts = [f"{n} {v}" for v, n in counts.items() if n]
    return "Verdict: " + ", ".join(parts) + "."


def math_check_pass(verdicts: list[ClaimVerdict], spec: ProductSpec | None, s: Settings, step: StepCallback,
                    notes: list[str], *, demo_id: str | None = None, llm=None):
    """v3: run the OpenMath-Nemotron math checker. Never fails the run."""
    from .mathcheck import nemotron_problem_writer, run_checks

    try:
        writer = nemotron_problem_writer(llm) if llm is not None else None
        info = run_checks(verdicts, spec, s, demo_id=demo_id, problem_writer=writer, step=step)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Math checker failed: {exc}")
        return None
    if info.source == "canned":
        notes.append("MATH CHECKER: canned derivations (written in OpenMath-Nemotron's output format, not produced by "
                     "the model). Set OPENMATH_BASE_URL to run the real model; the Python comparison is live either way.")
    elif not info.enabled:
        notes.append(info.reason)
    for v in verdicts:
        if v.math_check and v.math_check.status == "disagreed":
            notes.append(f"{v.claim.id}: {v.math_check.headline}")
    if spec is not None and spec.math_check and spec.math_check.status == "disagreed":
        notes.append(f"Spec sheet: {spec.math_check.headline}")
    return info


def _with_checker_line(overall: str, info) -> str:
    return overall + (" " + info.line if info is not None and info.line else "")


def _concept_target(verdicts: list[ClaimVerdict]) -> ClaimVerdict | None:
    works = [v for v in verdicts if v.verdict == "WORKS"]
    return works[0] if works else None




# ----------------------------------------------------------------------------- v2 helpers
def archive_lookup(query: str, k: int = 2, min_score: float = 0.44) -> list[ArchiveHit]:
    """Tesla Archive hits for one query; never fails the run."""
    try:
        from .archive import search

        return search(query, k=k, min_score=min_score)
    except Exception:
        return []


def union_hits(*groups: list[ArchiveHit]) -> list[ArchiveHit]:
    seen, out = set(), []
    for g in groups:
        for h in g:
            if (h.doc_id, h.page) not in seen:
                seen.add((h.doc_id, h.page))
                out.append(h)
    return out


def design_product(concept: ProductConcept, target: ClaimVerdict, step: StepCallback) -> ProductSpec | None:
    """Run the design calculator for the WORKS claim and rewrite the render prompt from the spec."""
    from .design import build_spec, design_for_template

    design = concept.design or design_for_template(target.claim.calc.template)
    if not design:
        return None
    step("design", f"Sizing {concept.name} with the {design} calculator")
    spec = build_spec(design, concept.requirements, title=concept.name, based_on_claim=target.claim.id)
    if not spec.error and spec.render_prompt:
        concept.image_prompt = spec.render_prompt  # the render now follows the computed numbers
        concept.design = design
    return spec


# ----------------------------------------------------------------------------- mock
def _run_mock(claim_text: str, s: Settings, step: StepCallback) -> Report:
    from .mock import CAPTURED_NOTE, pick_demo

    t: dict[str, float] = {}
    demo, exact = pick_demo(claim_text)
    notes = ["MOCK MODE: no API calls. " + CAPTURED_NOTE,
             "v2: gap-search evidence was captured live from Tavily + arXiv on 2026-10-13; the design "
             "calculators, gap ranking and Tesla Archive search run live on every mock run."]
    if not exact:
        notes.append(f"Mock mode only knows 3 demo claims; showing the closest one: {demo['title']}.")

    t0 = time.perf_counter()
    step("research", f"Searching patents, papers and modern research ({len(demo['research'])} queries, canned)")
    sources = [Source(**r) for blk in demo["research"] for r in blk["results"]]
    t["research"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    archive_hits = archive_lookup(f"{demo['topic']} {claim_text}", k=4, min_score=0.25)
    step("archive", f"Tesla Archive: {len(archive_hits)} matching pages in local patents/files (real search)")
    from .archive import hits_as_sources
    sources += hits_as_sources(archive_hits)
    t["archive"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    step("extract", f"Nemotron (mock) pulled out {len(demo['claims'])} physical claims")
    claims = [Claim(**c) for c in demo["claims"]]
    t["extract"] = time.perf_counter() - t0

    verdicts: list[ClaimVerdict] = []
    t0 = time.perf_counter()
    for c in claims:
        step("math", f"{c.id}: running {c.calc.template or 'custom sandboxed code'}")
        calc = run_calc(c.calc, backend=s.sandbox_backend)
        model_v = calc.suggested_verdict or "MISSING PIECE"
        final, note = apply_guardrails(model_v, calc)
        expl = demo["explanations"][c.id].format_map(_Fmt(calc.metrics)) if calc.ok else f"Calculation failed: {calc.error}"
        verdicts.append(ClaimVerdict(claim=c, calc=calc, verdict=final, explanation=expl,
                                     source_ids=c.source_ids, guardrail_note=note,
                                     archive_refs=archive_lookup(c.archive_query or c.text)))
    t["math+verdicts"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gaps = []
    if any(v.verdict == "MISSING PIECE" for v in verdicts):
        from .gaps import gap_search_mock
        gaps = gap_search_mock(verdicts, demo)
        n = sum(len(g.requirements) for g in gaps)
        step("gaps", f"Gap search: {n} 'what would need to be true' items ranked (captured Tavily + arXiv results)")
    t["gaps"] = time.perf_counter() - t0

    concept, spec = None, None
    target = _concept_target(verdicts)
    if target:
        step("concept", f"Designing a product from {target.claim.id}")
        concept = ProductConcept(**demo["concept"])
        t0 = time.perf_counter()
        spec = design_product(concept, target, step)
        t["design"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    info = math_check_pass(verdicts, spec, s, step, notes, demo_id=demo["id"])
    t["math_check"] = time.perf_counter() - t0
    return Report(input_claim=claim_text, mode="mock", model=f"{s.model} (mocked)", topic=demo["topic"],
                  summary=demo["summary"], sources=sources, verdicts=verdicts, concept=concept,
                  overall=_with_checker_line(overall_line(verdicts), info),
                  timings_s={k: round(v, 3) for k, v in t.items()}, notes=notes,
                  spec=spec, gaps=gaps, archive_hits=union_hits(archive_hits, *[v.archive_refs for v in verdicts]),
                  math_checker=info)


# ----------------------------------------------------------------------------- live
def _run_live(claim_text: str, s: Settings, step: StepCallback, image_dir: str | None) -> Report:
    from . import prompts
    from .llm import NemotronClient
    from .research import gather

    t: dict[str, float] = {}
    notes: list[str] = []
    llm = NemotronClient(s)

    t0 = time.perf_counter()
    step("research", "Searching patents, papers and modern research with Tavily")
    sources = gather(claim_text, s)
    t["research"] = time.perf_counter() - t0
    if not sources:
        raise RuntimeError("Tavily returned no usable sources")
    archive_hits = archive_lookup(claim_text, k=4, min_score=0.25) if s.use_archive else []
    if archive_hits:
        from .archive import hits_as_sources, merge_evidence
        step("archive", f"Tesla Archive: {len(archive_hits)} matching pages in local documents")
        sources = merge_evidence(hits_as_sources(archive_hits), sources)

    t0 = time.perf_counter()
    step("extract", f"Nemotron ({s.model}) is reading {len(sources)} sources")
    extracted = llm.json_call(
        prompts.EXTRACT_SYSTEM.format(templates=describe_templates()) + prompts.EXTRACT_V2_ADDENDUM,
        f"User idea: {claim_text}\n\nSources:\n{prompts.sources_block(sources)}",
    )
    claims = [Claim(**c) for c in extracted.get("claims", [])][:4]
    t["extract"] = time.perf_counter() - t0
    if not claims:
        raise RuntimeError("Nemotron found no testable claims")

    verdicts: list[ClaimVerdict] = []
    t0 = time.perf_counter()
    src_ids = {x.id for x in sources}
    for c in claims:
        step("math", f"{c.id}: running {c.calc.template or 'custom sandboxed code'}")
        calc = run_calc(c.calc, backend=s.sandbox_backend)
        step("verdict", f"{c.id}: Nemotron weighs the numbers")
        v = llm.json_call(
            prompts.VERDICT_SYSTEM,
            f"Claim: {c.model_dump_json()}\n\nCalculation result: {calc.model_dump_json()}\n\n"
            f"Sources:\n{prompts.sources_block([x for x in sources if x.id in c.source_ids] or sources[:4])}",
            model=s.fast_model,
        )
        final, note = apply_guardrails(v.get("verdict", "MISSING PIECE"), calc)
        cited = [i for i in v.get("source_ids", c.source_ids) if i in src_ids]
        refs = archive_lookup(c.archive_query or c.text) if s.use_archive else []
        verdicts.append(ClaimVerdict(claim=c, calc=calc, verdict=final, explanation=v.get("explanation", ""),
                                     source_ids=cited, guardrail_note=note, archive_refs=refs))
    t["math+verdicts"] = time.perf_counter() - t0

    gaps = []
    if s.gap_search and any(v.verdict == "MISSING PIECE" for v in verdicts):
        from .gaps import gap_search_live
        t0 = time.perf_counter()
        gaps = gap_search_live(verdicts, s, llm=llm, notes=notes, step=step)
        t["gaps"] = time.perf_counter() - t0

    concept, spec = None, None
    target = _concept_target(verdicts)
    if target:
        t0 = time.perf_counter()
        step("concept", f"Designing a product from {target.claim.id}")
        from .design import describe_designs
        cj = llm.json_call(prompts.CONCEPT_SYSTEM + prompts.CONCEPT_V2_ADDENDUM.format(designs=describe_designs()),
                           f"Working claim: {target.model_dump_json()}\nOriginal idea: {claim_text}")
        cj["based_on_claim"] = target.claim.id
        cj["requirements"] = cj.get("requirements") or {}
        concept = ProductConcept(**cj)
        spec = design_product(concept, target, step)
        if s.generate_images and image_dir:
            try:
                step("render", "Rendering the product concept")
                Path(image_dir).mkdir(parents=True, exist_ok=True)
                concept.image_path = llm.image(concept.image_prompt, str(Path(image_dir) / f"concept-{int(time.time())}.png"))
            except Exception as e:
                notes.append(f"Image generation failed: {e}")
        t["concept"] = time.perf_counter() - t0
    else:
        notes.append("No claim fully WORKS, so no product concept was generated.")

    t0 = time.perf_counter()
    info = math_check_pass(verdicts, spec, s, step, notes, llm=llm)
    t["math_check"] = time.perf_counter() - t0
    return Report(input_claim=claim_text, mode="live", model=s.model, topic=extracted.get("topic", claim_text),
                  summary=extracted.get("summary", ""), sources=sources, verdicts=verdicts, concept=concept,
                  overall=_with_checker_line(overall_line(verdicts), info),
                  timings_s={k: round(v, 3) for k, v in t.items()}, notes=notes,
                  spec=spec, gaps=gaps, archive_hits=union_hits(archive_hits, *[v.archive_refs for v in verdicts]),
                  math_checker=info)


def run(claim_text: str, settings: Settings | None = None, on_step: StepCallback | None = None,
        image_dir: str | None = "outputs") -> Report:
    s = settings or get_settings()
    step = on_step or _noop
    if s.is_mock:
        return _run_mock(claim_text, s, step)
    return _run_live(claim_text, s, step, image_dir)
