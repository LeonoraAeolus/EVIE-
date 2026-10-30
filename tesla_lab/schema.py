"""The structured output of a Tesla Lab run. Everything the UI shows comes from here."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Verdict = Literal["WORKS", "MISSING PIECE", "BREAKS PHYSICS"]


class Source(BaseModel):
    id: str  # "S1", "S2", ...
    url: str
    title: str
    content: str = ""  # snippet / extracted text the model saw
    score: float | None = None


class Quantity(BaseModel):
    name: str
    value: float | None = None
    unit: str = ""
    source_id: str | None = None  # where the number came from; None = assumed
    note: str = ""


class CalcSpec(BaseModel):
    """Which calculation to run. Prefer a vetted template; custom code is sandboxed."""

    template: str | None = None  # name in tesla_lab.calcs.TEMPLATES
    params: dict[str, Any] = Field(default_factory=dict)
    custom_code: str | None = None  # small Python, must set `result = {...}`


class Claim(BaseModel):
    id: str  # "C1", ...
    text: str  # the concrete physical claim, in plain words
    source_ids: list[str] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    equation: str = ""  # the relevant physics, human-readable
    calc: CalcSpec = Field(default_factory=CalcSpec)
    archive_query: str = ""  # v2: keywords to look this claim up in the Tesla Archive


class CalcResult(BaseModel):
    ok: bool
    template: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    series: dict[str, list[float]] | None = None  # for charts: {"x": [...], "y": [...]}
    series_labels: dict[str, str] | None = None
    flags: dict[str, bool] = Field(default_factory=dict)  # e.g. violates_energy_conservation
    summary: str = ""
    suggested_verdict: Verdict | None = None
    error: str | None = None


class ClaimVerdict(BaseModel):
    claim: Claim
    calc: CalcResult
    verdict: Verdict
    explanation: str
    source_ids: list[str] = Field(default_factory=list)
    guardrail_note: str | None = None  # set when the math overruled the model
    archive_refs: list["ArchiveHit"] = Field(default_factory=list)  # v2: Tesla Archive citations
    math_check: "MathCheck | None" = None  # v3: OpenMath-Nemotron derivation cross-checked against Python


class ProductConcept(BaseModel):
    based_on_claim: str  # claim id
    name: str
    pitch: str
    who_its_for: str
    how_it_works: str
    honest_limits: str
    image_prompt: str
    image_path: str | None = None
    # v2: which design calculator sizes this product, and the requirements it is sized for
    design: str | None = None
    requirements: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    input_claim: str
    mode: Literal["mock", "live"]
    model: str
    topic: str
    summary: str
    sources: list[Source]
    verdicts: list[ClaimVerdict]
    concept: ProductConcept | None = None
    overall: str
    timings_s: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    # v2
    spec: "ProductSpec | None" = None
    gaps: list["GapAnalysis"] = Field(default_factory=list)
    archive_hits: list["ArchiveHit"] = Field(default_factory=list)
    # v3
    math_checker: "MathCheckerInfo | None" = None
    # v4: Patent Search Planner (+ Wildcard), when a planner run is attached to a claim report
    search_plan: "SearchPlan | None" = None


# ----------------------------------------------------------------------------- v2 additions
# Product Design Engine ---------------------------------------------------------------------
Basis = Literal["requirement", "computed", "assumed", "standard", "estimate"]


class SpecItem(BaseModel):
    group: str  # "Transmitter coil", "Performance", ...
    name: str
    value: float | int | str
    unit: str = ""
    basis: Basis = "computed"  # where the number comes from
    note: str = ""


class SafetyCheck(BaseModel):
    name: str
    value: float
    limit: float
    unit: str
    passes: bool
    source: str  # citation, e.g. "ICNIRP (2020) Table 5"
    url: str = ""
    approximate: bool = True  # screening estimate, not a compliance test
    note: str = ""


class BomItem(BaseModel):
    part: str
    qty: float = 1
    unit_cost_usd: float
    note: str = ""
    estimate: bool = True

    @property
    def total(self) -> float:
        return self.qty * self.unit_cost_usd


class Reference(BaseModel):
    title: str
    url: str = ""


class ProductSpec(BaseModel):
    design: str  # name in tesla_lab.design.DESIGNS
    title: str
    based_on_claim: str = ""
    requirements: dict[str, Any] = Field(default_factory=dict)
    headline: list[str] = Field(default_factory=list)  # 3-5 key numbers, plain words
    specs: list[SpecItem] = Field(default_factory=list)
    series: dict[str, list[float]] | None = None
    series_labels: dict[str, str] | None = None
    extra_series: dict[str, list[float]] | None = None  # optional second y-series on the same x
    safety: list[SafetyCheck] = Field(default_factory=list)
    bom: list[BomItem] = Field(default_factory=list)
    bom_total_usd: float = 0.0
    cost_note: str = "Rough estimates from typical list prices, not quotes."
    render_prompt: str = ""
    caveats: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    error: str | None = None
    math_check: "MathCheck | None" = None  # v3: the design's governing equation, re-derived by OpenMath


# Innovation loop / gap search ---------------------------------------------------------------
class Lead(BaseModel):
    """One piece of evidence that someone is working on a missing piece."""

    title: str
    url: str
    kind: Literal["arxiv", "web"]
    published: str | None = None  # ISO date when known
    authors: list[str] = Field(default_factory=list)
    snippet: str = ""
    relevance: float = 0.5  # 0..1 (Tavily score for web, query-term overlap for arXiv)


class Requirement(BaseModel):
    """'What would need to be true' for the missing piece to be filled."""

    statement: str
    target: str = ""  # a measurable bar, e.g. "Q > 1000"
    feasibility: float = 0.5  # 0..1, the model's (or canned) judgment
    feasibility_basis: str = ""
    web_query: str = ""
    arxiv_query: str = ""
    leads: list[Lead] = Field(default_factory=list)
    evidence: float = 0.0  # computed from leads (recency-weighted)
    score: float = 0.0  # computed: what the ranking sorts by
    rank: int = 0


class GapAnalysis(BaseModel):
    claim_id: str
    claim_text: str
    missing_piece: str
    requirements: list[Requirement] = Field(default_factory=list)
    ranking_rule: str = ""
    captured: str = ""  # when the evidence was fetched (mock mode shows captured data)


# Tesla Archive (RAG) -----------------------------------------------------------------------
class ArchiveHit(BaseModel):
    doc_id: str  # e.g. "US645576A"
    title: str  # e.g. "US 645,576 - System of transmission of electrical energy (1900)"
    page: int  # 1-based page in the PDF
    snippet: str
    score: float
    source_url: str = ""
    file: str = ""
    kind: str = "patent"  # patent | foia | fbi_vault | arxiv | other

    @property
    def cite(self) -> str:
        return f"{self.title}, page {self.page}"


# ----------------------------------------------------------------------------- v3 additions
# Math checker: OpenMath-Nemotron derives, Python computes, the two are compared ------------
CheckStatus = Literal["agreed", "disagreed", "inconclusive", "skipped", "error"]


class DerivationStep(BaseModel):
    n: int
    text: str  # Markdown with $...$ math, as the model wrote it


class MathCheck(BaseModel):
    """One cross-check: OpenMath-Nemotron's derivation vs the Python calculator's number."""

    target: str  # claim id ("C1") or "spec:<design>"
    status: CheckStatus
    headline: str  # "math checker agreed: 0.102 s vs Python 0.102 s"
    source: Literal["openmath", "canned", "none"] = "none"
    model: str = ""
    # what was asked
    problem: str = ""
    problem_origin: Literal["template", "design", "nemotron", "canned", ""] = ""
    quantity: str = ""  # plain-words name of the compared quantity
    target_metric: str = ""  # key in calc.metrics (or spec item name)
    unit: str = ""
    # what came back
    steps: list[DerivationStep] = Field(default_factory=list)
    key_equations: list[str] = Field(default_factory=list)  # LaTeX display equations, in order
    final_answer_latex: str = ""  # the \boxed{} content
    answer_expr: str = ""  # SymPy form of the boxed answer
    answer_symbolic: bool = False  # True when the model boxed a formula, evaluated at the inputs
    checker_value: float | None = None
    python_value: float | None = None
    scale_note: str = ""  # e.g. "checker answered as a fraction; x100 to compare with percent"
    rel_diff: float | None = None
    tolerance: float = 0.02
    symbolic_match: bool | None = None  # formula identical to the calculator's (SymPy simplify)
    reasoning_chars: int = 0  # length of the model's <think> scratch work (not shown)
    latency_s: float | None = None
    completion_tokens: int | None = None
    note: str = ""


class MathCheckerInfo(BaseModel):
    """Run-level summary of the math checker."""

    enabled: bool
    source: Literal["openmath", "canned", "none"]
    model: str = ""
    base_url: str = ""
    reason: str = ""  # why it was skipped / canned
    counts: dict[str, int] = Field(default_factory=dict)  # status -> n
    line: str = ""  # "Math checker: 8 agreed, 1 disagreed"


# ----------------------------------------------------------------------------- v4 additions
# Patent Search Planner + Wildcard (spec v2, "SearchPlan"). Everything the planner UI shows comes from here.
PlanMode = Literal["find", "check", "improve"]


class Constraint(BaseModel):
    """A team constraint or test-derived limit, e.g. 'max tube length 1.0 m'."""

    text: str
    name: str = ""  # max_tube_length_m | max_volume_l | max_material_temp_c | min_efficiency_pct | ...
    op: Literal["<=", ">=", ""] = ""
    value: float | None = None
    unit: str = ""
    applies_to: str = ""  # key in a sizing result's `sizes`


class PlanInput(BaseModel):
    goal: str = ""  # Find: what the team is trying to do
    failed: list[str] = Field(default_factory=list)  # Find: what was tried and failed
    idea: str = ""  # Check: the idea to vet
    before: str | None = None  # Check: the team's date (YYYY-MM-DD); older documents are prior art
    constraints: list[Constraint] = Field(default_factory=list)  # Improve (and any mode): limits
    test_results: list[str] = Field(default_factory=list)  # Improve: what the last prototype showed
    last_plan_id: str | None = None
    example: str | None = None  # mock fixture id (exhaust | tesla_wireless)


class Conflict(BaseModel):
    improve: str
    without: str


class Abstraction(BaseModel):
    function: str
    conflict: Conflict
    principle: str = ""  # bare physics, used by the Wildcard
    exclude: list[str] = Field(default_factory=list)  # failed approaches -> exclusions
    quantities: dict[str, float] = Field(default_factory=dict)  # numbers the physics needs (frequency_hz, ...)


class Term(BaseModel):
    term: str
    synonyms: list[str] = Field(default_factory=list)
    patent_speak: list[str] = Field(default_factory=list)


class CpcCode(BaseModel):
    code: str  # canonical, e.g. F01N1/023
    title: str = ""
    role: Literal["home", "cross"]
    why: str = ""
    origin: Literal["model", "title_match", "fixture", "page"] = "model"
    note: str = ""


class PlanQuery(BaseModel):
    engine: Literal["gpatents_link", "tavily", "bigquery"]
    text: str
    url: str | None = None  # gpatents_link: clickable for the user, never fetched by the app
    purpose: str = ""


class Hit(BaseModel):
    pub: str
    title: str = ""
    date: str = ""  # priority date
    filing_date: str = ""
    cpc: list[str] = Field(default_factory=list)
    found_via: str = ""  # "tavily: <query>" | "backward citation of US..." | "cited by US..." | "bigquery" | "fixture"
    claim_quote: str = ""
    quote_verified: bool = False
    fit: float = 0.0  # 0..1, claim fit to the conflict (0 when the quote is not on the page)
    fit_why: str = ""
    status_text: str = ""  # Google's legal status line
    status_flag: str = "unknown"  # expired | likely_expired | active | pending | unknown
    status_why: str = ""
    legal_note: str = "Not legal advice"
    forward_citations: int = 0
    backward_citations: int = 0
    cpc_match: float = 0.0
    recency: float = 0.0
    rank_score: float = 0.0
    rank: int = 0
    prior_art: bool | None = None  # Check mode: older than the idea's date
    is_new: bool = False  # Improve mode: not in the last run
    read: bool = True  # claims actually read (False = chained/listed only)
    url: str = ""
    role: Literal["home", "cross", ""] = ""


class ChainEdge(BaseModel):
    src: str
    dst: str
    kind: Literal["backward", "forward"]  # backward: src cites dst (older); forward: dst cites src (newer)
    dst_title: str = ""
    dst_date: str = ""


class Sizing(BaseModel):
    calculator: str
    params: dict[str, Any] = Field(default_factory=dict)
    problem: str = ""
    quantity: str = ""
    value: float | None = None
    unit: str = ""
    formula: str = ""
    sizes: dict[str, Any] = Field(default_factory=dict)
    physics_ok: bool = True
    physics_note: str = ""
    constraint_notes: list[str] = Field(default_factory=list)
    math_check: "MathCheck | None" = None
    error: str = ""


class WildcardIdea(BaseModel):
    niche: str  # "Music", "Biology (heart)", ...
    cpc: str | None = None
    mechanism: str
    transfer: str  # "why not try this?"
    evidence: list[dict] = Field(default_factory=list)  # [{title, url, kind: patent|paper|web}]
    sizing: Sizing | None = None
    surprise: float = 0.0
    surprise_why: str = ""
    math_fit: float = 0.0
    math_fit_why: str = ""
    score: float = 0.0
    bucket: Literal["try", "fun_but_no"] = "try"


class TrailStep(BaseModel):
    step: str  # "1 Abstract", ...
    did: str
    kept: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    source: str = ""  # model | canned | code | tavily | page | bigquery
    seconds: float | None = None


class PlanDiff(BaseModel):
    new: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    reranked: list[dict] = Field(default_factory=list)  # [{pub, from, to}]
    wildcard_changes: list[dict] = Field(default_factory=list)  # [{niche, from_score, to_score, why}]


class SearchPlan(BaseModel):
    id: str
    mode: PlanMode
    data_mode: Literal["mock", "live"]
    canned_note: str = ""  # non-empty in mock mode: what is canned
    input: PlanInput
    abstraction: Abstraction | None = None
    terms: list[Term] = Field(default_factory=list)
    cpc: list[CpcCode] = Field(default_factory=list)
    queries: list[PlanQuery] = Field(default_factory=list)
    hits: list[Hit] = Field(default_factory=list)
    chain: list[ChainEdge] = Field(default_factory=list)
    wildcard: list[WildcardIdea] = Field(default_factory=list)
    wildcard_principle: str = ""
    trail: list[TrailStep] = Field(default_factory=list)
    diff: PlanDiff | None = None
    watch: list[PlanQuery] = Field(default_factory=list)  # Improve: links for filings newer than the last run
    npl: list[dict] = Field(default_factory=list)  # Check (live): non-patent literature, e.g. arXiv
    notes: list[str] = Field(default_factory=list)
    legal_note: str = ("Legal status and expiry are flags from Google Patents' own (non-authoritative) status "
                       "line and the dates on the page. Not legal advice; freedom-to-operate needs a patent attorney.")
    created_at: str = ""
    timings_s: dict[str, float] = Field(default_factory=dict)


ClaimVerdict.model_rebuild()
ProductSpec.model_rebuild()
Report.model_rebuild()
Sizing.model_rebuild()
SearchPlan.model_rebuild()
