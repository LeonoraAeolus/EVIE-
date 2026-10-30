"""Refresh the mock-mode gap-search evidence with REAL Tavily + arXiv results.

    python scripts/capture_gap_mock.py            # uses TAVILY_API_KEY (or TAVILY_KEYLESS=1)

Reads the canned "what would need to be true" plans in tesla_lab/mock/gap_plans.json (written the
way Nemotron would write them), fetches live evidence for each, and writes tesla_lab/mock/gaps.json.
Mock mode then shows real, dated search results without needing keys.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tesla_lab.config import get_settings  # noqa: E402
from tesla_lab.gaps import arxiv_search, web_search  # noqa: E402

MOCK = Path(__file__).resolve().parents[1] / "tesla_lab" / "mock"


def main() -> int:
    s = get_settings("live")
    plans = json.loads((MOCK / "gap_plans.json").read_text())
    out = {}
    for demo_id, claims in plans.items():
        out[demo_id] = {"captured": dt.date.today().isoformat(), "claims": {}}
        for cid, plan in claims.items():
            reqs = []
            for r in plan["requirements"]:
                leads = []
                try:
                    leads += [l.model_dump() for l in web_search(r["web_query"], s, 4)]
                except Exception as e:
                    print(f"  ! tavily failed: {e}")
                try:
                    leads += [l.model_dump() for l in arxiv_search(r["arxiv_query"], 4)]
                except Exception as e:
                    print(f"  ! arxiv failed: {e}")
                print(f"{demo_id}/{cid}: {len(leads)} leads for: {r['statement'][:60]}")
                reqs.append({**r, "leads": leads})
            out[demo_id]["claims"][cid] = {"missing_piece": plan["missing_piece"], "requirements": reqs}
    (MOCK / "gaps.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {MOCK / 'gaps.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
