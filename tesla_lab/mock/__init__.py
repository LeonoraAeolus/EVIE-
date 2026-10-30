"""Canned data so the whole app runs with no API keys.

Research results were captured from the real Tavily API (2026-10-12). Gap-search evidence
(gaps.json) was captured from the real Tavily + arXiv APIs (2026-10-13, scripts/capture_gap_mock.py). Claim extraction,
verdict wording and concepts are hand-written stand-ins for Nemotron's output. The
physics calculations are NOT canned: they run for real on every mock run.
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA = json.loads((Path(__file__).parent / "demos.json").read_text(encoding="utf-8"))
DEMOS: list[dict] = _DATA["demos"]
CAPTURED_NOTE: str = _DATA["captured"]


def pick_demo(claim: str) -> tuple[dict, bool]:
    """Return (demo, exact_match). Unknown input falls back to the closest demo."""
    text = claim.lower()
    best, best_hits = DEMOS[0], 0
    for d in DEMOS:
        hits = sum(1 for k in d["keywords"] if k in text)
        if hits > best_hits:
            best, best_hits = d, hits
    return best, best_hits > 0


def demo_inputs() -> list[tuple[str, str]]:
    return [(d["title"], d["input"]) for d in DEMOS]


def load_gaps() -> dict:
    p = Path(__file__).parent / "gaps.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
