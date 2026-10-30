"""Tesla Lab web UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from tesla_lab.calc_runner import calc_source
from tesla_lab.design import design_source
from tesla_lab.spec_sheet import to_blender_script, to_markdown, to_pdf, to_stl, to_threejs_html
from tesla_lab.config import get_settings
from tesla_lab.mock import demo_inputs, pick_demo
from tesla_lab.pipeline import run
import streamlit.components.v1 as components

st.set_page_config(page_title="EVIE", page_icon="⚡", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Warm paper background */
  .stApp {
    background-color: #f5f2eb;
    color: #231e16;
  }

  /* Inked Retro Cards */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff !important;
    border: 1.5px solid #231e16 !important;
    border-radius: 12px !important;
    box-shadow: 3px 3px 0 #231e16 !important;
    margin-bottom: 12px !important;
  }

  /* Pill Buttons */
  button[kind="primary"] {
    background-color: #231e16 !important;
    color: #f5f2eb !important;
    border: 1.5px solid #231e16 !important;
    border-radius: 9999px !important;
    box-shadow: 2px 2px 0 rgba(35,30,22,0.3) !important;
    font-weight: 700 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
  }
  button[kind="primary"]:hover {
    transform: translate(-1px, -1px) !important;
    box-shadow: 3px 3px 0 rgba(35,30,22,0.4) !important;
  }
  button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #231e16 !important;
    border: 1.5px solid #231e16 !important;
    border-radius: 9999px !important;
    box-shadow: 2px 2px 0 #231e16 !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
  }
  button[kind="secondary"]:hover {
    transform: translate(-1px, -1px) !important;
    box-shadow: 3px 3px 0 #231e16 !important;
  }

  /* Monospace Tab Headers */
  button[data-baseweb="tab"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-weight: 700 !important;
    font-size: 0.88em !important;
    color: #231e16 !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2.5px solid #231e16 !important;
  }

  /* Text areas and inputs */
  textarea, input {
    border: 1.5px solid #231e16 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
    color: #231e16 !important;
  }

  /* Stage divider header */
  .stage-label {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 700;
    font-size: 11px;
    color: #78716c;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 28px 0 10px 0;
    padding-top: 16px;
    border-top: 1.5px solid #e7e5e4;
  }
</style>
""", unsafe_allow_html=True)

# ── Badge helpers ─────────────────────────────────────────────────────────────
def _nebius_badge(model: str, extra: str = "") -> str:
    txt = f"⚡ {model}" + (f" · {extra}" if extra else "") + " · Nebius"
    return (f"<span style='background:#1e3a5f;color:#93c5fd;padding:2px 9px;border-radius:10px;"
            f"font-size:0.78em;font-weight:600;font-family:monospace'>{txt}</span>")

def _tavily_badge(what: str = "") -> str:
    txt = "🔍 Tavily" + (f" · {what}" if what else "")
    return (f"<span style='background:#14532d;color:#86efac;padding:2px 9px;border-radius:10px;"
            f"font-size:0.78em;font-weight:600;font-family:monospace'>{txt}</span>")

# ── Constants ─────────────────────────────────────────────────────────────────
BADGE = {"WORKS": ("✅", "#16a34a"), "MISSING PIECE": ("🧩", "#d97706"), "BREAKS PHYSICS": ("⛔", "#dc2626")}
RENDERS = Path(__file__).parent / "tesla_lab" / "mock" / "renders"
CHECK_BADGE = {
    "agreed":      ("🧮 math checker agreed",       "#0f766e"),
    "disagreed":   ("🧮 math checker DISAGREED",    "#b91c1c"),
    "inconclusive":("🧮 math checker inconclusive", "#6b7280"),
    "error":       ("🧮 math checker error",         "#6b7280"),
    "skipped":     ("🧮 math checker skipped",       "#9ca3af"),
}


def check_chip(chk) -> str:
    label, color = CHECK_BADGE.get(chk.status, CHECK_BADGE["skipped"])
    if chk.source == "canned" and chk.status != "skipped":
        label += " (canned)"
    return (f"<span style='background:{color};color:white;padding:2px 9px;border-radius:12px;"
            f"font-size:0.85em;font-weight:600'>{label}</span>")


def _num(x) -> str:
    return "—" if x is None else f"{x:.4g}"


def render_derivation(chk) -> None:
    """OpenMath-Nemotron's step-by-step derivation and comparison with Python."""
    who = ("CANNED derivation (mock mode, written in OpenMath's format, not model output)"
           if chk.source == "canned"
           else f"Derived by {chk.model}")
    st.caption(who
               + (f" · {chk.latency_s:.1f} s" if chk.latency_s else "")
               + (f" · {chk.completion_tokens} tokens" if chk.completion_tokens else ""))
    st.markdown("**Problem given to the math checker**")
    st.caption(chk.problem)
    for s_ in chk.steps:
        st.markdown(s_.text)
    if chk.final_answer_latex:
        st.markdown(f"**Final answer:** $\\boxed{{{chk.final_answer_latex}}}$")
    unit = chk.unit
    st.dataframe(
        [{"quantity": chk.quantity,
          "math checker": f"{_num(chk.checker_value)} {unit}".strip(),
          "Python calculator": f"{_num(chk.python_value)} {unit}".strip(),
          "difference": "—" if chk.rel_diff is None else f"{100 * chk.rel_diff:.2f}%",
          "tolerance": f"{100 * chk.tolerance:.0f}%",
          "result": chk.status}],
        hide_index=True, use_container_width=True,
    )
    if chk.symbolic_match is not None:
        st.caption("Formula: " + ("same as the calculator's (SymPy simplify)" if chk.symbolic_match
                                  else "differs from the calculator's formula"))
    if chk.scale_note:
        st.caption(f"Unit note: {chk.scale_note}")
    if chk.note:
        st.caption(chk.note)


def _stage(label: str) -> None:
    """Render a styled stage-divider header."""
    st.markdown(f"<div class='stage-label'>{label}</div>", unsafe_allow_html=True)


# ── Settings ──────────────────────────────────────────────────────────────────
settings = get_settings()

# ── Sidebar: Tesla Archive ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Tesla Archive")
    try:
        from tesla_lab.archive import load_index
        _idx = load_index()
    except Exception:
        _idx = None
    if _idx:
        st.caption(
            f"{len(_idx.meta.get('docs', []))} documents · {_idx.meta.get('n_chunks')} chunks · "
            f"search: {_idx.meta.get('embedder')}"
        )
        for d in _idx.meta.get("docs", []):
            st.markdown(f"- {d['title']} ({d['pages']} p.)")
        st.caption("Add PDFs to archive/pdfs/ and run `python scripts/ingest_archive.py`.")
    else:
        st.caption("No archive index. Run `python scripts/ingest_archive.py`.")

# ── Hero header ───────────────────────────────────────────────────────────────
col_head_left, col_head_right = st.columns([5, 2])
with col_head_left:
    st.markdown(
        f"""
        <div style="font-family:ui-monospace,Menlo,monospace;font-size:11px;font-weight:700;
                    color:#78716c;letter-spacing:0.08em;text-transform:uppercase;">
          Nebius × NVIDIA AI Hackathon · Hardware AI Studio
        </div>
        <h1 style="margin:4px 0 4px 0;font-size:2.4em;font-weight:800;color:#231e16;letter-spacing:-0.02em;">
          ⚡ EVIE
        </h1>
        <div style="font-size:1.12em;color:#231e16;font-weight:700;margin-bottom:3px;">
          Hardware AI Co-Pilot · Inspired by Spider-Man's EV Agent
        </div>
        <div style="font-size:0.95em;color:#44403c;">
          Take a hardware idea or physics claim seriously. Before you order parts — let EVIE run the physics.
        </div>
        <div style="margin-top:8px">
          {_nebius_badge("Token Factory")} &nbsp; {_tavily_badge()}
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_head_right:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("✨ Launch EVIE Spatial UI", key="nav_nexa", use_container_width=True, type="primary"):
        st.session_state.show_nexa = not st.session_state.get("show_nexa", False)
        st.rerun()
    if st.button("🔎 Patent Search Planner", key="nav_planner", use_container_width=True):
        st.session_state.show_planner = True
        st.rerun()

# Mode notice
if settings.is_mock:
    st.info(
        "🧪 **Mock mode** — no API keys needed. Real physics math, canned research & model answers. "
        "Add `NEBIUS_API_KEY` and `TAVILY_API_KEY` to `.env` to switch to live.",
        icon="🧪",
    )
else:
    st.success(
        f"Live · Tavily research + {settings.model} on Nebius Token Factory",
        icon="🛰️",
    )

# ── Navigation states ─────────────────────────────────────────────────────────
if "show_nexa" not in st.session_state:
    st.session_state.show_nexa = False

if st.session_state.show_nexa:
    nexa_file = Path(__file__).parent / "tesla_lab" / "nexa_spatial_ui.html"
    col_back_nexa, col_info_nexa = st.columns([2, 5])
    with col_back_nexa:
        if st.button("← Return to Classic Lab", key="back_from_nexa", type="secondary"):
            st.session_state.show_nexa = False
            st.rerun()
    with col_info_nexa:
        st.markdown(
            "<span style='font-family:monospace; font-size:12px; color:#78716c;'>✨ "
            "<b>EVIE Robotics & Spatial Hardware Co-pilot</b> "
            "· Three.js 3D WebGL · KaTeX Mathematical Physics · Audio Haptics</span>",
            unsafe_allow_html=True,
        )
    if nexa_file.exists():
        with open(nexa_file, encoding="utf-8") as f:
            components.html(f.read(), height=940, scrolling=False)
    else:
        st.error(f"NEXA UI file not found at {nexa_file}")
    st.stop()

if "show_planner" not in st.session_state:
    st.session_state.show_planner = False

if st.session_state.show_planner:
    col_back, col_tag = st.columns([2, 5])
    with col_back:
        if st.button("← Back to Physics Lab", key="back_to_lab", type="primary"):
            st.session_state.show_planner = False
            st.rerun()
    with col_tag:
        st.markdown(
            f"<div style='padding-top:6px'>{_nebius_badge('Nemotron 3 Super', 'Classification & Reasoning')} "
            f"&nbsp; {_tavily_badge('include_domains=patents.google.com')}</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    from tesla_lab.ui_planner import render as render_planner
    render_planner(settings)
    st.stop()


def _to_planner(goal: str) -> None:
    """A MISSING PIECE becomes the goal of a Patent Search run."""
    st.session_state.show_planner = True
    st.session_state["pl_find_goal"] = goal
    st.session_state["pl_find_failed"] = ""
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# CLAIM LAB PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

# ── Demo buttons ──────────────────────────────────────────────────────────────
if "claim" not in st.session_state:
    st.session_state.claim = demo_inputs()[0][1]

demo_cols = st.columns(len(demo_inputs()))
for col, (title, text) in zip(demo_cols, demo_inputs()):
    if col.button(title, use_container_width=True):
        st.session_state.claim = text

# ── Input area ────────────────────────────────────────────────────────────────
claim = st.text_area(
    "The assumption your project depends on",
    key="claim",
    height=90,
    placeholder=(
        'e.g. "Our coil design assumes 90% efficiency at 2 metres" — '
        "or paste any bold claim to stress-test it against physics."
    ),
)
go = st.button("Test it against physics", type="primary")

# ── Run pipeline ──────────────────────────────────────────────────────────────
if go and claim.strip():
    with st.status("Running the lab...", expanded=True) as status:
        report = run(
            claim.strip(), settings,
            on_step=lambda step, msg: status.write(f"**{step}** · {msg}"),
        )
        status.update(label=report.overall, state="complete", expanded=False)
    st.session_state.report = report

report = st.session_state.get("report")
if not report:
    st.stop()

# ── Telemetry strip ───────────────────────────────────────────────────────────
mc_info = report.math_checker
_mc_label = ""
if mc_info and mc_info.enabled and mc_info.model:
    _mc_label = f" + {mc_info.model} (math checker)"
elif mc_info and mc_info.source == "canned":
    _mc_label = " + OpenMath-Nemotron 7B (canned/mock)"

_timing_str = "  ·  ".join(f"{k}: {v:.1f}s" for k, v in report.timings_s.items()) if report.timings_s else ""
st.markdown(
    f"""
<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;
            padding:7px 14px;margin:6px 0 18px 0;font-size:0.8em;
            color:#0c4a6e;font-family:monospace;line-height:1.7">
  <b>Nebius Token Factory:</b> {report.model}{_mc_label}
  &nbsp;·&nbsp;
  <b>Tavily:</b> {len(report.sources)} sources
  {"&nbsp;·&nbsp;<b>Timing:</b> " + _timing_str if _timing_str else ""}
</div>
""",
    unsafe_allow_html=True,
)

# ── Topic + summary ───────────────────────────────────────────────────────────
st.subheader(report.topic)
st.write(report.summary)
for n in report.notes:
    st.caption(n)
src = {s.id: s for s in report.sources}


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1 — VERDICT
# ════════════════════════════════════════════════════════════════════════════════
_stage("Stage 1 · Verdict — Does the physics allow it?")

for v in report.verdicts:
    icon, color = BADGE[v.verdict]
    with st.container(border=True):

        header_html = (
            f"<span style='background:{color};color:white;padding:3px 10px;"
            f"border-radius:12px;font-weight:700'>{icon} {v.verdict}</span>"
        )
        if not settings.is_mock:
            header_html += " &nbsp; " + _nebius_badge(settings.fast_model, "verdict")
        if v.math_check:
            header_html += " &nbsp; " + check_chip(v.math_check)

        st.markdown(
            header_html + f" &nbsp; **{v.claim.id}. {v.claim.text}**",
            unsafe_allow_html=True,
        )
        st.write(v.explanation)

        if v.guardrail_note:
            st.warning(f"Math guardrail: {v.guardrail_note}")
        if v.math_check and v.math_check.status == "disagreed":
            st.error(v.math_check.headline)
        elif v.math_check and v.math_check.status == "agreed":
            st.caption(v.math_check.headline)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Verified math** (computed in Python, not guessed by the model)")
            if v.claim.equation:
                st.code(v.claim.equation, language="text")
            st.write(v.calc.summary or v.calc.error)
            if v.calc.series:
                lab = v.calc.series_labels or {"x": "x", "y": "y"}
                st.line_chart(
                    {lab["x"]: v.calc.series["x"], lab["y"]: v.calc.series["y"]},
                    x=lab["x"], y=lab["y"],
                )
            if v.math_check and v.math_check.steps:
                with st.expander("🧮 Step-by-step derivation (OpenMath-Nemotron)"):
                    render_derivation(v.math_check)
            with st.expander("The code that produced these numbers"):
                st.code(calc_source(v.claim.calc), language="python")

        with right:
            st.markdown("**Numbers**")
            st.json(v.calc.metrics, expanded=True)
            st.markdown("**Inputs**")
            for q in v.claim.quantities:
                val = "" if q.value is None else f"{q.value:,.10g} {q.unit}".strip()
                tag = (f"[{q.source_id}]" + (f" ({q.note})" if q.note else "")
                       if q.source_id else f"_{q.note or 'assumed'}_")
                st.markdown(f"- {q.name}: {val} {tag}")
            sources_md = " · ".join(f"[{i}]({src[i].url})" for i in v.source_ids if i in src)
            if sources_md:
                st.markdown(
                    "**Sources:** " + sources_md + " &nbsp; " + _tavily_badge(),
                    unsafe_allow_html=True,
                )

        if v.archive_refs:
            with st.expander(f"📜 Tesla Archive: {len(v.archive_refs)} matching pages"):
                for h in v.archive_refs:
                    link = h.source_url or ""
                    ref_line = (f"[{h.title}]({link}), **page {h.page}**" if link
                                else f"{h.title}, **page {h.page}**")
                    st.markdown(f"📜 **Found in the Tesla Archive:** {ref_line}")
                    st.caption(f'"{h.snippet}" (OCR text of the original scan, match {h.score:.2f})')

        if v.verdict == "MISSING PIECE":
            st.markdown("---")
            bridge_col, _ = st.columns([3, 2])
            with bridge_col:
                st.button(
                    f"🔎 Search patents for this",
                    key=f"toplan_{v.claim.id}",
                    on_click=_to_planner,
                    args=(v.claim.text,),
                )


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2 — MATH CHECKER: Derivation vs Python
# ════════════════════════════════════════════════════════════════════════════════
checks = [(f"{v.claim.id}. {v.claim.text}", v.math_check)
          for v in report.verdicts if v.math_check]
if report.spec and report.spec.math_check:
    checks.append((f"Spec sheet: {report.spec.title}", report.spec.math_check))

if checks:
    st.divider()
    st.subheader("🧮 Math checker: derivation vs Python")
    mc = report.math_checker
    st.caption("Nemotron 3 Super leads. OpenMath-Nemotron (math specialist) derives each governing equation step by "
               "step; Python computes the number; EVIE compares the two. A disagreement is flagged, never hidden."
               + (f" {mc.line}" if mc and mc.line else ""))
    if mc and mc.source == "canned":
        st.warning("Canned derivations: written in OpenMath-Nemotron's format for the demo claims, not produced by the "
                   "model. The comparison with Python runs for real. Set OPENMATH_BASE_URL for real derivations.",
                   icon="🧪")
    st.dataframe([{"check": name.split(".")[0] if not name.startswith("Spec") else "Spec", "result": c.status,
                   "quantity": c.quantity,
                   "math checker": f"{_num(c.checker_value)} {c.unit}".strip(),
                   "Python": f"{_num(c.python_value)} {c.unit}".strip(),
                   "difference": "—" if c.rel_diff is None else f"{100 * c.rel_diff:.2f}%"} for name, c in checks],
                 hide_index=True, use_container_width=True)
    tabs = st.tabs([name.split(".")[0] if not name.startswith("Spec") else "Spec" for name, _ in checks])
    for tab, (name, c) in zip(tabs, checks):
        with tab:
            st.markdown(check_chip(c) + f" &nbsp; **{name}**", unsafe_allow_html=True)
            st.markdown(c.headline)
            if c.steps:
                render_derivation(c)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3 — INNOVATION LOOP: What would need to be true
# ════════════════════════════════════════════════════════════════════════════════
if report.gaps:
    st.divider()
    st.subheader("🔭 Innovation loop: what would need to be true")
    st.caption("For every MISSING PIECE, EVIE lists what would have to become true and searches "
               "Tavily (web, last year) and arXiv for who is working on it now.")
    for g in report.gaps:
        with st.container(border=True):
            st.markdown(f"**{g.claim_id}. {g.claim_text}**")
            st.write(g.missing_piece)
            st.button("🔎 Search patents for this", key=f"toplan_gap_{g.claim_id}", on_click=_to_planner,
                      args=(g.missing_piece,))
            for r in g.requirements:
                st.markdown(f"**#{r.rank} · {r.statement}**")
                a, b, c_col = st.columns([1, 1, 3])
                a.metric("score", f"{r.score:.2f}")
                b.metric("feasibility", f"{r.feasibility:.2f}")
                c_col.markdown(f"Target: _{r.target}_  \n{r.feasibility_basis}  \nEvidence weight: {r.evidence:.2f} "
                               f"from {len(r.leads)} sources")
                with st.expander(f"Who's working on it ({len(r.leads)} sources)"):
                    for l in r.leads:
                        tag = "arXiv" if l.kind == "arxiv" else "web"
                        date = f" · {l.published}" if l.published else ""
                        who = f" · {', '.join(l.authors[:2])}" if l.authors else ""
                        st.markdown(f"- `{tag}` [{l.title}]({l.url}){date}{who}")
            st.caption(f"Ranking: {g.ranking_rule}." + (f" Evidence captured {g.captured}." if g.captured else ""))


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 4 — BUILD IT: Concept, spec sheet & 3D model
# ════════════════════════════════════════════════════════════════════════════════
img = None
if report.concept:
    c = report.concept
    st.divider()
    st.subheader(f"💡 {c.name}")
    st.write(f"_{c.pitch}_")
    ca, cb = st.columns([3, 2])
    with ca:
        st.markdown(f"**Who it's for:** {c.who_its_for}")
        st.markdown(f"**How it works:** {c.how_it_works}")
        st.markdown(f"**Honest limits:** {c.honest_limits}")
        with st.expander("Image prompt (written from the spec numbers)" if report.spec else "Image prompt"):
            st.code(c.image_prompt, language="text")
    with cb:
        img = c.image_path
        if not img and report.mode == "mock":
            demo_pick, _ = pick_demo(report.input_claim)
            p = RENDERS / f"{demo_pick['id']}.jpg"
            img = str(p) if p.exists() else None
        if img and Path(img).exists():
            st.image(img, caption=f"Concept render: {c.name}")
        else:
            st.caption("Set GENERATE_IMAGES=1 in live mode to render this concept.")

spec = report.spec
if spec and not spec.error:
    st.subheader(f"📐 Spec sheet: {spec.title}")
    st.caption("Designed by the math: a vetted design calculator sized this product. "
               "calc = computed · req = your requirement · assumed = check on a prototype · est. = cost estimate")
    with st.container(border=True):
        for h in spec.headline:
            st.markdown(f"- **{h}**")

        # ── REORDERED TABS: Key Numbers first, 3D Model second ────────────────
        t_nums, t_3d, t_chart, t_safety, t_bom = st.tabs(
            ["📊 Key Numbers", "🎮 3D Model", "📈 Chart", "🛡️ Safety", "💰 BOM"]
        )

        with t_nums:
            st.dataframe([{"group": i.group, "spec": i.name, "value": f"{i.value} {i.unit}".strip(), "basis": i.basis,
                           "note": i.note} for i in spec.specs], hide_index=True, use_container_width=True)
            if spec.math_check:
                st.markdown("---")
                st.markdown(
                    "**Math check (OpenMath-Nemotron)** &nbsp; "
                    + _nebius_badge("OpenMath-Nemotron", "self-hosted Nebius GPU"),
                    unsafe_allow_html=True,
                )
                st.markdown(check_chip(spec.math_check) + " &nbsp; " + spec.math_check.headline, unsafe_allow_html=True)
                if spec.math_check.steps:
                    with st.expander("Step-by-step derivation"):
                        render_derivation(spec.math_check)
            else:
                st.caption("No math check for this design.")
            with st.expander("How were these numbers computed? (design calculator code)"):
                st.code(design_source(spec.design), language="python")

        with t_3d:
            st.markdown("##### 🎮 Interactive Hand-Inked 3D Concept & Blender CAD")
            st.caption("Turn it all the way round, toggle See-through to inspect internal coils/discs, and view labeled parts. Modeled directly from physical math.")
            c_nexa_launch, _ = st.columns([3, 2])
            with c_nexa_launch:
                if st.button("✨ Open in EVIE Industrial Spatial Co-pilot", key="open_nexa_from_3d"):
                    st.session_state.show_nexa = True
                    st.rerun()
            st.markdown("<div style='background:#1a1a2e;border-radius:10px;padding:6px;margin-bottom:6px'>", unsafe_allow_html=True)
            viewer_html = to_threejs_html(spec)
            components.html(viewer_html, height=480)
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("Show Procedural Blender Python Script (bpy)"):
                st.caption("Copy this code or download below, open Blender's Scripting workspace, paste and click 'Run Script' for full PBR studio materials, lighting, and camera.")
                st.code(to_blender_script(spec), language="python")

        with t_chart:
            if spec.series:
                lab = spec.series_labels or {"x": "x", "y": "y"}
                data = {lab["x"]: spec.series["x"], lab["y"]: spec.series["y"]}
                ys = [lab["y"]]
                for name, vals in (spec.extra_series or {}).items():
                    data[name] = vals
                    ys.append(name)
                st.line_chart(data, x=lab["x"], y=ys)

        with t_safety:
            if spec.safety:
                for sc in spec.safety:
                    icon = "✅" if sc.passes else "⚠️"
                    st.markdown(f"{icon} **{sc.name}**: {sc.value:g} {sc.unit} vs limit {sc.limit:g} {sc.unit} "
                                f"({'pass' if sc.passes else 'over the limit'}) · [{sc.source}]({sc.url})")
                    st.caption(spec.safety[0].note)
            else:
                st.caption("No field-exposure screen applies to this design; see caveats.")
            for cv in spec.caveats:
                st.markdown(f"- {cv}")

        with t_bom:
            st.dataframe([{"part": b.part, "qty": b.qty, "unit USD": b.unit_cost_usd, "total USD": round(b.total, 2),
                           "note": b.note} for b in spec.bom], hide_index=True, use_container_width=True)
            st.markdown(f"**Total (estimate): ${spec.bom_total_usd:,.0f}**")
            st.caption(spec.cost_note)

        st.markdown("---")
        claim_txt = next((v.claim.text for v in report.verdicts if v.claim.id == spec.based_on_claim), "")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.download_button("Spec sheet (Markdown)", to_markdown(spec, report.concept, claim_txt),
                           file_name=f"{spec.title.replace(' ', '_')}_spec.md", mime="text/markdown")
        try:
            img_for_pdf = img if (report.concept and img and Path(img).exists()) else None
            pdf_bytes = to_pdf(spec, report.concept, claim_txt, image_path=img_for_pdf)
            d2.download_button("Spec sheet (PDF)", pdf_bytes, file_name=f"{spec.title.replace(' ', '_')}_spec.pdf",
                               mime="application/pdf")
        except Exception as e:
            d2.caption(f"PDF export unavailable: {e}")
        d3.download_button("Spec (JSON)", spec.model_dump_json(indent=2),
                           file_name=f"{spec.title.replace(' ', '_')}_spec.json", mime="application/json")
        d4.download_button("Blender script (.py)", to_blender_script(spec),
                           file_name=f"{spec.title.replace(' ', '_')}_blender.py", mime="text/x-python")
        d5.download_button("3D Printable Mesh (.stl)", to_stl(spec),
                           file_name=f"{spec.title.replace(' ', '_')}_model.stl", mime="model/stl")
elif spec and spec.error:
    st.warning(f"Design calculator failed: {spec.error}")

# ── Cross-Industry Patent Solver CTA ──────────────────────────────────────────
st.markdown("""
<div style="background:#ffffff;border:1.5px solid #231e16;border-radius:12px;padding:16px 20px;margin-top:28px;margin-bottom:12px;box-shadow:3px 3px 0 #231e16">
  <div style="font-family:ui-monospace,Menlo,monospace;font-weight:700;font-size:11px;color:#78716c;text-transform:uppercase;letter-spacing:0.06em">
    Cross-Industry R&amp;D Discovery
  </div>
  <div style="font-weight:800;font-size:1.1em;color:#231e16;margin:4px 0">
    🔎 Stuck on an engineering trade-off? Search across unrelated industries
  </div>
  <div style="font-size:0.9em;color:#44403c;margin-bottom:12px;line-height:1.45">
    The most expensive R&amp;D mistake is spending months reinventing what another field (aerospace, medical devices, acoustics, oil pipelines) solved decades ago.
    The Patent Search Planner strips your problem to bare physics and surfaces unexpected mechanisms using the 💡 Wildcard engine.
  </div>
</div>
""", unsafe_allow_html=True)
if st.button("🔎 Open Cross-Industry Patent Planner →", key="open_planner_bottom", type="secondary"):
    st.session_state.show_planner = True
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
if report.archive_hits:
    with st.expander(f"📜 Tesla Archive: {len(report.archive_hits)} matching pages"):
        for h in report.archive_hits:
            st.markdown(f"**{h.title}, page {h.page}** · [open]({h.source_url}) · match {h.score:.2f}")
            st.caption(h.snippet)

with st.expander("All sources"):
    for s in report.sources:
        st.markdown(f"**[{s.id}]** [{s.title}]({s.url})")

st.download_button("Download JSON verdict", json.dumps(report.model_dump(), indent=2),
                   file_name="tesla-lab-report.json", mime="application/json")
