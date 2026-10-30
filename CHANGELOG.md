# Changelog

## v4 — 2026-10-13
**Added**
- Patent Search Planner (`tesla_lab/patents/`): Find / Check / Improve modes over 5 steps (Abstract → Expand →
  Classify → Chain → Read claims), ranked hits with verbatim claim quotes, legal status ("not legal advice"),
  clickable Google Patents search links, Improve diff (new / dropped / re-ranked) and a new-filings watch link.
- 💡 Wildcard: bare physical principle → unrelated niches → ranked by surprise × math_fit, with Python sizing
  calculators (quarter-wave tube, Helmholtz, Windkessel chamber, serrations, tunable tube, resonator chain, coil link)
  and the OpenMath cross-check when configured.
- Robots-respecting Google Patents page reader with cache (never requests search pages), Tavily patent discovery,
  optional BigQuery path behind `PATENTS_BIGQUERY`.
- New Streamlit view "🔎 Patent Search Planner" (Find, Check, Improve, Wildcard tabs + reasoning trail, citation
  chain, JSON download); "Search patents for this" button on MISSING PIECE items.
- CLI: `--plan find|check|improve`, `--goal`, `--failed`, `--idea`, `--before`, `--constraint`, `--test-result`,
  `--last-plan`, `--example`, `--no-wildcard`, `--plan-out`.
- Mock fixtures (labelled CANNED): exhaust example and Tesla wireless-power example, real captured patent pages.
- 38 tests (`tests/test_planner.py`, `tests/test_ui_planner.py`); suite is 120 passed / 1 skipped.

**Changed**
- `requirements.txt`: + beautifulsoup4. `config.py`, `schema.py` (SearchPlan models), `app.py`, `cli.py`, README.

**Not done / roadmap**
- Closed loop with 3D-printed prototypes (Improve takes typed results today).
- No live Tavily / Nemotron / BigQuery call has been made (no keys here).

## v3 — 2026-10-13
- OpenMath-Nemotron math checker (self-hosted vLLM on a Nebius GPU), cross-check badges, deploy scripts.

## v2 — 2026-10-13
- Product Design Engine + spec sheets, innovation loop / gap search, Tesla Archive RAG, 20-claim eval set.

## v1 — 2026-10-12
- Claim → research → extract → compute → verdict pipeline, Streamlit UI, mock mode.
