"""Command line: python -m tesla_lab "your wild claim" [--mock] [--out report.json]

v4 Patent Search Planner:
  python -m tesla_lab --mock --plan find --example exhaust --plan-out plan.json
  python -m tesla_lab --plan find --goal "..." --failed "..." [--constraint "max volume 8 L"]
  python -m tesla_lab --plan check --idea "..." --before 2026-10-01
  python -m tesla_lab --plan improve --last-plan plan.json --test-result "..." --constraint "max tube length 1.0 m"
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .pipeline import run

COLORS = {"WORKS": "\033[92m", "MISSING PIECE": "\033[93m", "BREAKS PHYSICS": "\033[91m"}


def main(argv: list[str] | None = None) -> int:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="tesla_lab", description="Test a wild tech claim against physics.")
    ap.add_argument("claim", nargs="?", default="Nikola Tesla's Wardenclyffe Tower could send free electric power wirelessly to anyone, anywhere on Earth.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="force mock mode (no API keys needed)")
    g.add_argument("--live", action="store_true", help="force live mode")
    ap.add_argument("--out", help="write the full JSON report here")
    ap.add_argument("--spec-out", help="write the spec sheet here (.md or .pdf)")
    ap.add_argument("--blender-out", help="write the Blender procedural Python script here (.py)")
    ap.add_argument("--stl-out", help="write the 3D printable STL mesh here (.stl)")
    ap.add_argument("--show-derivations", action="store_true", help="print the math checker's step-by-step derivations")
    pg = ap.add_argument_group("v4 Patent Search Planner")
    pg.add_argument("--plan", choices=["find", "check", "improve"], help="run the patent search planner instead of a claim check")
    pg.add_argument("--goal", default="", help="Find: what you are trying to do")
    pg.add_argument("--failed", action="append", default=[], help="Find: something you tried that failed (repeatable)")
    pg.add_argument("--idea", default="", help="Check: the idea to vet")
    pg.add_argument("--before", help="Check: your date (YYYY-MM-DD); older documents count as prior art")
    pg.add_argument("--constraint", action="append", default=[], help="a limit, e.g. 'max tube length 1.0 m' (repeatable)")
    pg.add_argument("--test-result", action="append", default=[], help="Improve: what the last prototype showed (repeatable)")
    pg.add_argument("--last-plan", help="Improve: the previous run's JSON (from --plan-out)")
    pg.add_argument("--example", help="mock mode: canned example to use (exhaust | tesla_wireless); its input fills empty fields")
    pg.add_argument("--no-wildcard", action="store_true", help="skip the Wildcard step")
    pg.add_argument("--plan-out", help="write the SearchPlan JSON here")
    args = ap.parse_args(argv)

    s = get_settings("mock" if args.mock else "live" if args.live else None)
    if args.plan:
        return run_planner_cli(args, s)
    print(f"Tesla Lab · mode={s.mode} · model={s.model}\n")
    report = run(args.claim, s, on_step=lambda step, msg: print(f"  [{step}] {msg}"))
    print(f"\n{report.topic}\n{report.summary}\n")
    for v in report.verdicts:
        c = COLORS.get(v.verdict, "")
        print(f"{c}{v.verdict}\033[0m  {v.claim.id}: {v.claim.text}")
        print(f"   math: {v.calc.summary or v.calc.error}")
        print(f"   why:  {v.explanation}")
        if v.guardrail_note:
            print(f"   guardrail: {v.guardrail_note}")
        print(f"   sources: {', '.join(v.source_ids)}")
        for h in v.archive_refs:
            print(f"   archive: found in {h.title}, page {h.page}")
        if v.math_check:
            print(f"   {v.math_check.headline}")
        print()
    for g in report.gaps:
        print(f"Innovation loop for {g.claim_id} (what would need to be true, ranked):")
        for r in g.requirements:
            print(f"  #{r.rank} [{r.score:.2f}] {r.statement}  ({len(r.leads)} sources)")
        print()
    if report.concept:
        print(f"Product concept: {report.concept.name} — {report.concept.pitch}")
    if report.spec and not report.spec.error:
        print("Spec sheet (computed):")
        for h in report.spec.headline:
            print(f"  - {h}")
        if report.spec.math_check:
            print(f"  {report.spec.math_check.headline}")
    if args.show_derivations:
        checks = [v.math_check for v in report.verdicts if v.math_check]
        if report.spec and report.spec.math_check:
            checks.append(report.spec.math_check)
        for chk in checks:
            if not chk.steps:
                continue
            print(f"--- derivation for {chk.target} ({'CANNED' if chk.source == 'canned' else chk.model}) ---")
            print(f"problem: {chk.problem}")
            for st in chk.steps:
                print(f"  {st.n}. {st.text}")
            print(f"  => {chk.headline}\n")
    print(f"\n{report.overall}")
    for n in report.notes:
        print(f"note: {n}")
    if args.spec_out and report.spec and not report.spec.error:
        from .spec_sheet import to_markdown, to_pdf
        claim = next((v.claim.text for v in report.verdicts if v.claim.id == report.spec.based_on_claim), "")
        if args.spec_out.lower().endswith(".pdf"):
            img = report.concept.image_path if report.concept else None
            if not img and report.mode == "mock":
                from pathlib import Path

                from .mock import pick_demo
                p = Path(__file__).parent / "mock" / "renders" / f"{pick_demo(report.input_claim)[0]['id']}.jpg"
                img = str(p) if p.exists() else None
            with open(args.spec_out, "wb") as f:
                f.write(to_pdf(report.spec, report.concept, claim, image_path=img))
        else:
            with open(args.spec_out, "w") as f:
                f.write(to_markdown(report.spec, report.concept, claim))
        print(f"Spec sheet written to {args.spec_out}")
    if args.blender_out and report.spec and not report.spec.error:
        from .spec_sheet import to_blender_script
        with open(args.blender_out, "w", encoding="utf-8") as f:
            f.write(to_blender_script(report.spec))
        print(f"Blender script written to {args.blender_out}")
    if args.stl_out and report.spec and not report.spec.error:
        from .spec_sheet import to_stl
        with open(args.stl_out, "w", encoding="utf-8") as f:
            f.write(to_stl(report.spec))
        print(f"3D STL mesh written to {args.stl_out}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        print(f"\nJSON report written to {args.out}")
    return 0


def run_planner_cli(args, s) -> int:
    from .patents import load_plan, run_plan, save_plan
    from .patents.planner import example_input, plan_input

    print(f"Tesla Lab · Patent Search Planner · mode={args.plan} · data={s.mode}\n")
    if args.example:
        inp = example_input(args.example, args.plan)
        for k in ("goal", "idea", "before"):
            if getattr(args, k):
                setattr(inp, k, getattr(args, k))
        if args.failed:
            inp.failed = args.failed
        if args.test_result:
            inp.test_results = args.test_result
        if args.constraint:
            from .schema import Constraint

            inp.constraints = [Constraint(text=c) for c in args.constraint]
    else:
        inp = plan_input(args.plan, goal=args.goal, failed=args.failed, idea=args.idea, before=args.before,
                         constraints=args.constraint, test_results=args.test_result)
        if s.is_mock and not (inp.goal or inp.idea or inp.test_results):
            inp = example_input("exhaust", args.plan)
    last = load_plan(args.last_plan) if args.last_plan else None
    plan = run_plan(args.plan, inp, s, last=last, wildcard=not args.no_wildcard,
                    on_step=lambda step, msg: print(f"  [{step}] {msg[:160]}"))
    if plan.canned_note:
        print(f"\n{plan.canned_note}")
    print("\nRanked patents (rank = 0.5 fit + 0.2 CPC match + 0.2 citations + 0.1 recency):")
    for h in plan.hits:
        tag = " NEW" if h.is_new else (" PRIOR ART" if h.prior_art and plan.mode == "check" else "")
        q = f"\n      “{h.claim_quote[:140]}”" if h.claim_quote and h.quote_verified else (
            "\n      quote NOT on the page → fit 0" if h.claim_quote else "")
        print(f"  #{h.rank} {h.pub} [{h.rank_score:.2f}] fit {h.fit:.2f} · {h.status_flag}{tag} · {h.title[:70]}"
              f"\n      via {h.found_via[:90]} · {h.url}{q}")
    if plan.diff:
        d = plan.diff
        moves = ", ".join(f"{r['pub']} {r['from']}→{r['to']}" for r in d.reranked) or "—"
        print(f"\nChanged since the last run: new {', '.join(d.new) or '—'} · dropped {', '.join(d.dropped) or '—'} · "
              f"re-ranked {moves}")
        for c in d.wildcard_changes:
            print(f"  Wildcard {c['niche']}: {c['from_score']:.2f} → {c['to_score']:.2f} ({c['why']})")
    if plan.wildcard:
        print(f"\n💡 Wildcard · principle: {plan.wildcard_principle}")
        for w in plan.wildcard:
            sz = w.sizing
            num = f"{sz.value:g} {sz.unit}" if sz and sz.value is not None else "no number"
            chk = f" · {sz.math_check.headline}" if sz and sz.math_check else ""
            print(f"  {'TRY ' if w.bucket == 'try' else 'FUN, BUT NO '}[{w.score:.2f}] {w.niche}: {w.transfer}\n"
                  f"      {sz.quantity if sz else ''} = {num} ({sz.formula if sz else ''}); surprise {w.surprise:.2f} "
                  f"({w.surprise_why}); math_fit {w.math_fit:.1f} ({w.math_fit_why}){chk}")
    print("\nSearch it yourself on Google Patents (links only; Tesla Lab never opens search pages):")
    for q in [q for q in plan.queries if q.engine == "gpatents_link"][:4] + plan.watch:
        print(f"  {q.purpose}: {q.url}")
    for n in plan.notes:
        print(f"note: {n}")
    print(f"\n{plan.legal_note}")
    if args.plan_out:
        save_plan(plan, args.plan_out)
        print(f"\nSearchPlan JSON written to {args.plan_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
