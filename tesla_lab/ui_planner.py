"""Streamlit view for the v4 Patent Search Planner (Find / Check / Improve) and the 💡 Wildcard.

Called from app.py when the "🔎 Patent Search Planner" view is selected. Everything shown comes from a
SearchPlan (tesla_lab/schema.py), so the same JSON drives the CLI, the UI and the download.
"""
from __future__ import annotations

import datetime as dt
import html

import streamlit as st

from .patents import cpc as CPC
from .patents.brain import list_examples
from .patents.planner import example_input, plan_input, run_plan
from .schema import SearchPlan

FLAG = {"expired": ("expired", "#15803d"), "likely_expired": ("likely expired", "#65a30d"),
        "active": ("active", "#b91c1c"), "pending": ("pending", "#d97706"), "unknown": ("status unknown", "#6b7280")}
ROLE = {"home": ("home class", "#1d4ed8"), "cross": ("other industry", "#7c3aed"), "": ("", "#6b7280")}
CHECK = {"agreed": ("🧮 math checker agreed", "#0f766e"), "disagreed": ("🧮 math checker DISAGREED", "#b91c1c"),
         "inconclusive": ("🧮 inconclusive", "#6b7280"), "error": ("🧮 error", "#6b7280"), "skipped": ("🧮 skipped", "#9ca3af")}


def chip(text: str, color: str) -> str:
    return (f"<span style='background:{color};color:white;padding:2px 9px;border-radius:12px;font-size:0.82em;"
            f"font-weight:600;margin-right:4px'>{html.escape(text)}</span>")


def _lines(s: str) -> list[str]:
    return [x.strip(" -•\t") for x in (s or "").splitlines() if x.strip(" -•\t")]


def _set_example(ex_id: str) -> None:
    for mode in ("find", "check", "improve"):
        inp = example_input(ex_id, mode)
        st.session_state[f"pl_{mode}_goal"] = inp.goal
        st.session_state[f"pl_{mode}_failed"] = "\n".join(inp.failed)
        st.session_state[f"pl_{mode}_idea"] = inp.idea
        st.session_state[f"pl_{mode}_cons"] = "\n".join(c.text for c in inp.constraints)
        st.session_state[f"pl_{mode}_tests"] = "\n".join(inp.test_results)
        if inp.before:
            st.session_state["pl_check_before"] = dt.date.fromisoformat(inp.before)
    st.session_state["pl_example"] = ex_id


def _run(mode: str, inp, settings, last=None, wildcard=True) -> SearchPlan:
    with st.status(f"Planning ({mode.title()} mode)…", expanded=True) as status:
        plan = run_plan(mode, inp, settings, last=last, wildcard=wildcard,
                        on_step=lambda step, msg: status.write(f"**{step}** · {msg[:220]}"))
        status.update(label=f"{mode.title()}: {len(plan.hits)} patents ranked, {len(plan.wildcard)} Wildcard ideas",
                      state="complete", expanded=False)
    st.session_state[f"plan_{mode}"] = plan
    st.session_state["plan_last"] = plan
    return plan


# ----------------------------------------------------------------------------- renderers
def render_trail(plan: SearchPlan) -> None:
    st.markdown("**Reasoning trail**")
    for i, t in enumerate(plan.trail):
        label = f"{t.step} · {t.did[:120]}{'…' if len(t.did) > 120 else ''}"
        with st.expander(label, expanded=(i == 0)):
            st.write(t.did)
            if t.step.startswith("1") and plan.abstraction:
                ab = plan.abstraction
                st.markdown(f"> **{ab.conflict.improve}** WITHOUT **{ab.conflict.without}**")
            if t.step.startswith("2") and plan.terms:
                st.dataframe([{"term": x.term, "synonyms": ", ".join(x.synonyms), "patent-speak": ", ".join(x.patent_speak)}
                              for x in plan.terms], hide_index=True, use_container_width=True)
            elif t.step.startswith("3") and plan.cpc:
                st.markdown(" ".join(chip(f"{CPC.pretty(c.code)}", ROLE[c.role][1]) for c in plan.cpc), unsafe_allow_html=True)
                st.dataframe([{"code": CPC.pretty(c.code), "role": c.role, "title": c.title, "why": c.why,
                               "from": c.origin} for c in plan.cpc], hide_index=True, use_container_width=True)
            elif t.kept:
                st.markdown("\n".join(f"- {html.escape(k)}" for k in t.kept[:14]))
            if t.dropped:
                st.caption("Dropped / skipped: " + " · ".join(t.dropped[:10]))
            st.caption(f"source: {t.source}" + (f" · {t.seconds:.2f} s" if t.seconds is not None else ""))


def render_chain(plan: SearchPlan) -> None:
    if not plan.chain:
        return
    read = {h.pub for h in plan.hits}
    lines = ["digraph G {", "rankdir=LR; node [shape=box, style=\"rounded,filled\", fontname=Helvetica, fontsize=10];"]
    nodes = {}
    for e in plan.chain:
        nodes.setdefault(e.src, None)
        nodes[e.dst] = e.dst_date[:4]
    for n, year in nodes.items():
        h = next((x for x in plan.hits if x.pub == n), None)
        y = (h.date[:4] if h and h.date else year) or ""
        fill = "#dbeafe" if n in read else "#f3f4f6"
        lines.append(f'"{n}" [label="{n}\\n{y}", fillcolor="{fill}"];')
    for e in plan.chain:
        if e.kind == "backward":  # older root -> seed
            lines.append(f'"{e.dst}" -> "{e.src}" [color="#64748b"];')
        else:
            lines.append(f'"{e.src}" -> "{e.dst}" [color="#7c3aed"];')
    lines.append("}")
    st.markdown("**Citation chain** (left = older roots, right = newer improvements; blue = claims read)")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


def render_hits(plan: SearchPlan, limit: int = 12) -> None:
    st.markdown(f"**Ranked patents** · rank = 0.5·fit + 0.2·CPC match + 0.2·citations + 0.1·recency"
                + (" · prior art first" if plan.mode == "check" else " · new first" if plan.mode == "improve" else ""))
    for h in plan.hits[:limit]:
        with st.container(border=True):
            flag, fcol = FLAG.get(h.status_flag, FLAG["unknown"])
            role, rcol = ROLE.get(h.role, ROLE[""])
            chips = chip(f"#{h.rank} · {h.rank_score:.2f}", "#111827") + chip(flag, fcol)
            if role:
                chips += chip(role, rcol)
            if h.is_new:
                chips += chip("NEW", "#db2777")
            if plan.mode == "check" and h.prior_art:
                chips += chip("prior art", "#b45309")
            st.markdown(chips + f" &nbsp; **[{h.pub}]({h.url})** {html.escape(h.title)}", unsafe_allow_html=True)
            a, b = st.columns([3, 1])
            with a:
                if h.claim_quote and h.quote_verified:
                    st.markdown(f"> “{h.claim_quote}”  \n✓ quote found in the claims")
                elif h.claim_quote:
                    st.markdown(f"> ~~“{h.claim_quote}”~~  \n⚠️ **unconfirmed**: not in this patent's claims, so fit = 0")
                st.caption(h.fit_why)
                st.caption(f"Found via {h.found_via} · priority {h.date or '?'} · CPC {', '.join(CPC.pretty(c) for c in h.cpc[:4])}"
                           f" · cited by {h.forward_citations}")
                st.caption(f"Status: {h.status_text or '?'} ({h.status_why}). {h.legal_note}")
            with b:
                st.metric("fit to conflict", f"{h.fit:.2f}")
                st.progress(min(1.0, h.fit))


def render_wildcard(plan: SearchPlan) -> None:
    if not plan.wildcard:
        st.caption("No Wildcard ideas in this run.")
        return
    st.markdown(f"**💡 Wildcard** · bare principle: _{html.escape(plan.wildcard_principle)}_")
    st.caption("score = surprise × math_fit. Surprise = CPC distance from your home class (×0.25 if your own class already "
               "uses the mechanism). math_fit = 1 inside every limit, 0.5 stretches one, 0 the numbers don't work. "
               "Every number below is computed in Python.")
    tries = [w for w in plan.wildcard if w.bucket == "try"]
    noes = [w for w in plan.wildcard if w.bucket != "try"]
    for w in tries:
        sz = w.sizing
        with st.container(border=True):
            st.markdown(f"<div style='border-left:6px solid #f97316;padding-left:10px'>"
                        f"{chip(f'score {w.score:.2f}', '#ea580c')}{chip(f'surprise {w.surprise:.2f}', '#9a3412')}"
                        f"{chip(f'math {w.math_fit:.1f}', '#c2410c')}"
                        + (chip(*[CHECK[sz.math_check.status][0] + (' (canned)' if sz.math_check.source == 'canned' else ''),
                                  CHECK[sz.math_check.status][1]]) if sz and sz.math_check else "")
                        + f"<br><b>{html.escape(w.niche)}</b>: why not try this? {html.escape(w.transfer)}</div>",
                        unsafe_allow_html=True)
            if sz and sz.value is not None:
                c1, c2 = st.columns([1, 2])
                c1.metric(sz.quantity or "computed", f"{sz.value:g} {sz.unit}")
                c2.code(sz.formula, language="text")
                c2.caption(sz.physics_note)
            st.caption(f"Surprise: {w.surprise_why} · Math: {w.math_fit_why}")
            if sz and sz.constraint_notes:
                st.caption("Limits: " + " · ".join(sz.constraint_notes))
            if w.evidence:
                st.caption("Borrowed from: " + " · ".join(f"[{html.escape(e.get('title', ''))[:80]}]({e.get('url')})"
                                                          for e in w.evidence if e.get("url")))
    if noes:
        with st.expander(f"Fun, but no ({len(noes)})"):
            for w in noes:
                sz = w.sizing
                num = f"{sz.value:g} {sz.unit}" if sz and sz.value is not None else "—"
                st.markdown(f"**{w.niche}**: {w.transfer}  \n{sz.quantity if sz else ''} = **{num}** → {w.math_fit_why}")


def render_plan(plan: SearchPlan, show_wildcard: bool = True) -> None:
    if plan.canned_note:
        st.warning(plan.canned_note, icon="🧪")
    if plan.diff:
        d = plan.diff
        st.markdown("**What changed since the last run**")
        c1, c2, c3 = st.columns(3)
        c1.metric("new", len(d.new), help=", ".join(d.new))
        c2.metric("dropped", len(d.dropped), help=", ".join(d.dropped))
        c3.metric("re-ranked", len(d.reranked))
        rows = [{"patent": r["pub"], "rank before": str(r["from"]), "rank now": str(r["to"])} for r in d.reranked]
        rows += [{"patent": p, "rank before": "new", "rank now": str(next((h.rank for h in plan.hits if h.pub == p), ""))}
                 for p in d.new]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        for c in d.wildcard_changes:
            st.caption(f"💡 {c['niche']}: score {c['from_score']:.2f} → {c['to_score']:.2f} ({c['why']})")
    render_trail(plan)
    render_hits(plan)
    render_chain(plan)
    if show_wildcard and plan.wildcard:
        render_wildcard(plan)
    links = [q for q in plan.queries if q.engine == "gpatents_link"] + plan.watch
    if links:
        st.markdown("**Search it yourself on Google Patents** (EVIE never opens search pages: robots.txt)")
        st.markdown("\n".join(f"- [{html.escape(q.purpose)}: {html.escape(q.text[:90])}]({q.url})" for q in links))
    for n in plan.notes:
        st.caption(n)
    if plan.npl:
        st.markdown("**Non-patent literature**  \n" + "\n".join(f"- [{x['title']}]({x['url']}) ({x.get('published', '')})" for x in plan.npl))
    st.caption(plan.legal_note)
    st.download_button("Download SearchPlan JSON", plan.model_dump_json(indent=2), file_name=f"searchplan-{plan.mode}.json",
                       mime="application/json", key=f"dl_{plan.mode}_{plan.id}")


# ----------------------------------------------------------------------------- page
def render(settings) -> None:
    st.subheader("🔎 Patent Search Planner")
    # Context banner: explain the R&D problem this solves
    st.markdown(
        """
<div style="background:#fff;border:1.5px solid #231e16;border-radius:10px;
            padding:13px 18px;margin-bottom:14px;box-shadow:3px 3px 0 #231e16">
  <div style="font-family:ui-monospace,Menlo,monospace;font-weight:700;font-size:11px;
              color:#231e16;letter-spacing:0.05em;margin-bottom:4px">
    WORKFLOW 02 · CROSS-INDUSTRY R&amp;D PROBLEM SOLVER
  </div>
  <div style="font-size:0.93em;color:#292524;line-height:1.5">
    Mid-project and stuck on an engineering trade-off?
    Instead of searching your own product category, the planner strips the jargon to bare physics
    and finds mechanisms in <b>unrelated industries</b> that solved the same problem decades ago.
    <br><span style="color:#78716c;font-size:0.9em">
    Abstract (function + conflict) → Expand (synonyms + patent-speak) → Classify (CPC codes,
    cross-industry) → Chain (backward + forward citations) → Read claims (fit score + verbatim
    quote check) → 💡 Wildcard (surprise × math_fit from unrelated niches).
    </span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if settings.is_mock:
        st.info(
            "Mock mode: real captured patent pages + CANNED model outputs (labelled). "
            "Quote checks, CPC validation, citation chains, ranking and all Wildcard math run for real.",
            icon="🧪",
        )
    exs = list_examples()
    if "pl_example" not in st.session_state:
        _set_example(exs[0]["id"])
    titles = {e["id"]: e["title"] for e in exs}
    cols = st.columns(len(exs) + 1)
    cols[0].caption("Examples:")
    for col, e in zip(cols[1:], exs):
        col.button(e["title"], key=f"ex_{e['id']}", on_click=_set_example, args=(e["id"],), use_container_width=True)
    st.caption(f"Loaded example: {titles.get(st.session_state.get('pl_example'), '—')}")

    t_find, t_check, t_imp, t_wild = st.tabs(["🔎 Find", "✅ Check", "🔁 Improve", "💡 Wildcard"])
    with t_find:
        st.caption("Mid-research and stuck. Start from the goal and what failed; the planner goes wide across industries.")
        goal = st.text_area("Goal / problem", key="pl_find_goal", height=80)
        failed = st.text_area("What you tried that failed (one per line)", key="pl_find_failed", height=60)
        cons = st.text_area("Limits (one per line, e.g. 'max volume 8 L')", key="pl_find_cons", height=80)
        wild = st.checkbox("Add 💡 Wildcard", value=True, key="pl_find_wild")
        if st.button("Run Find", type="primary", key="run_find"):
            inp = plan_input("find", goal=goal, failed=_lines(failed), constraints=_lines(cons),
                             example=st.session_state.get("pl_example") if settings.is_mock else None)
            _run("find", inp, settings, wildcard=wild)
        if st.session_state.get("plan_find"):
            render_plan(st.session_state["plan_find"], show_wildcard=False)
            if st.session_state["plan_find"].wildcard:
                st.caption("Wildcard ideas for this run are in the 💡 Wildcard tab.")
    with t_check:
        st.caption("Vet an idea before spending months: narrow and deep in your own class, every claim read, "
                   "anything older than your date is prior art.")
        idea = st.text_area("The idea", key="pl_check_idea", height=80)
        if "pl_check_before" not in st.session_state:
            st.session_state["pl_check_before"] = dt.date.today()
        before = st.date_input("Your date (prior art = older than this)", key="pl_check_before")
        cons_c = st.text_area("Limits (one per line)", key="pl_check_cons", height=60)
        if st.button("Run Check", type="primary", key="run_check"):
            inp = plan_input("check", idea=idea, before=before.isoformat(), constraints=_lines(cons_c),
                             goal=st.session_state.get("pl_find_goal", ""),
                             example=st.session_state.get("pl_example") if settings.is_mock else None)
            _run("check", inp, settings, wildcard=False)
        if st.session_state.get("plan_check"):
            render_plan(st.session_state["plan_check"], show_wildcard=False)
    with t_imp:
        last = st.session_state.get("plan_find") or st.session_state.get("plan_check")
        if last:
            st.caption(f"Starting from your last {last.mode.title()} run ({len(last.hits)} patents).")
        elif settings.is_mock:
            st.caption("No run yet: mock mode will start from the example's canned Find run.")
        else:
            st.caption("Run Find or Check first; Improve re-runs it with your new results.")
        tests = st.text_area("New test results (one per line)", key="pl_improve_tests", height=80)
        cons_i = st.text_area("New limits (one per line, e.g. 'max tube length 1.0 m')", key="pl_improve_cons", height=60)
        st.caption("🗺️ Roadmap, not built: a closed loop where 3D-printed prototypes are tested and the results flow back "
                   "in automatically. Today you type them here.")
        if st.button("Re-run with these results", type="primary", key="run_improve", disabled=not (last or settings.is_mock)):
            inp = plan_input("improve", test_results=_lines(tests), constraints=_lines(cons_i),
                             example=st.session_state.get("pl_example") if settings.is_mock else None)
            _run("improve", inp, settings, last=last)
        if st.session_state.get("plan_improve"):
            render_plan(st.session_state["plan_improve"], show_wildcard=True)
    with t_wild:
        plan = st.session_state.get("plan_last")
        if plan is None or not plan.wildcard:
            st.caption("Strip the problem to its bare physics and borrow from music, biology, toys, kitchens and nature. "
                       "Runs on the Find inputs.")
            if st.button("Run 💡 Wildcard", type="primary", key="run_wild"):
                inp = plan_input("find", goal=st.session_state.get("pl_find_goal", ""),
                                 failed=_lines(st.session_state.get("pl_find_failed", "")),
                                 constraints=_lines(st.session_state.get("pl_find_cons", "")),
                                 example=st.session_state.get("pl_example") if settings.is_mock else None)
                plan = _run("find", inp, settings, wildcard=True)
        if plan is not None and plan.wildcard:
            if plan.canned_note:
                st.warning("Wildcard ideas are CANNED in mock mode; the sizing numbers, limits, surprise and ranking are "
                           "computed for real.", icon="🧪")
            render_wildcard(plan)
