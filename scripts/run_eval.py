"""Eval: Tesla Lab vs the expected verdicts (and optionally vs a plain chatbot).

  python scripts/run_eval.py                 # mock if no keys: runs the 3 demo claims, marks the rest
  python scripts/run_eval.py --live          # every claim through the real pipeline (Tavily + Nemotron)
  python scripts/run_eval.py --baseline      # also ask a plain chat model (no tools, no math)
  python scripts/run_eval.py --only wardenclyffe nif_gain --limit 5

Scoring
  * Tesla Lab's headline verdict = the most severe verdict across the claims it extracted
    (one BREAKS PHYSICS part means the claim as stated does not hold up).
  * Numeric error = |value - reference| / |reference| when the eval item has a reference number and
    the run produced the matching metric (Tesla Lab: from its calculation; baseline: the number it states).
Results are written to evals/results/scoreboard-<mode>-<time>.md and .csv. Nothing is filled in by hand:
an item that was not run says so.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tesla_lab.config import get_settings  # noqa: E402
from tesla_lab.pipeline import VERDICT_RANK, run  # noqa: E402

EVALS = ROOT / "evals" / "claims.jsonl"
OUT = ROOT / "evals" / "results"

BASELINE_SYSTEM = """You are a helpful assistant. Judge the physical claim the user gives you.
Answer with JSON only: {"verdict": "WORKS" | "MISSING PIECE" | "BREAKS PHYSICS",
 "number": number or null, "unit": str, "explanation": str}
WORKS = physics allows it and it has been built/measured; MISSING PIECE = not forbidden but unproven or
impractical; BREAKS PHYSICS = violates conservation of energy or thermodynamics.
If asked for a quantity, give your best numeric value for it in "number"."""


def load_items(path: Path = EVALS) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def headline_verdict(report) -> str | None:
    vs = [v.verdict for v in report.verdicts]
    return max(vs, key=lambda v: VERDICT_RANK[v]) if vs else None


def find_metric(report, key: str | None):
    if not key:
        return None
    for v in report.verdicts:
        val = v.calc.metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def rel_err(value, ref) -> float | None:
    if value is None or ref is None or ref.get("value") in (None, 0):
        return None
    return abs(value - ref["value"]) / abs(ref["value"])


def run_tesla_lab(item: dict, settings) -> dict:
    if settings.is_mock and not item.get("mock_demo"):
        return {"status": "needs live run"}
    t0 = time.perf_counter()
    try:
        rep = run(item["claim"], settings, image_dir=None)
    except Exception as e:
        return {"status": f"error: {type(e).__name__}: {e}"[:200]}
    value = find_metric(rep, item.get("tesla_lab_metric"))
    return {"status": "mock (canned extraction, real math)" if settings.is_mock else "live",
            "verdict": headline_verdict(rep), "value": value, "seconds": round(time.perf_counter() - t0, 1),
            "parts": " / ".join(v.verdict for v in rep.verdicts)}


def run_baseline(item: dict, client, model: str) -> dict:
    from tesla_lab.llm import parse_json

    ask = item["claim"]
    if item.get("reference"):
        ask += f"\nAlso state: {item['reference']['quantity']} (unit: {item['reference']['unit'] or 'none'})."
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(model=model, temperature=0.2, max_tokens=800,
                                              messages=[{"role": "system", "content": BASELINE_SYSTEM},
                                                        {"role": "user", "content": ask}])
        j = parse_json(resp.choices[0].message.content or "")
    except Exception as e:
        return {"status": f"error: {type(e).__name__}: {e}"[:200]}
    num = j.get("number")
    return {"status": "live", "verdict": j.get("verdict"), "value": float(num) if isinstance(num, (int, float)) else None,
            "seconds": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tesla Lab eval")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true")
    g.add_argument("--live", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="also ask a plain chat model on Token Factory")
    ap.add_argument("--baseline-model", default=None, help="default: NEMOTRON_MODEL")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--evals", default=str(EVALS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    s = get_settings("mock" if a.mock else "live" if a.live else None)
    items = load_items(Path(a.evals))
    if a.only:
        items = [i for i in items if i["id"] in a.only]
    if a.limit:
        items = items[: a.limit]

    client, bmodel, bnote = None, a.baseline_model or s.model, ""
    if a.baseline:
        if not s.nebius_api_key:
            bnote = "baseline not run: set NEBIUS_API_KEY"
        else:
            from openai import OpenAI
            client = OpenAI(base_url=s.nebius_base_url, api_key=s.nebius_api_key)

    rows = []
    for it in items:
        print(f"- {it['id']}: ", end="", flush=True)
        tl = run_tesla_lab(it, s)
        bl = run_baseline(it, client, bmodel) if client else {"status": bnote or "not requested"}
        ref = it.get("reference")
        row = {
            "id": it["id"], "claim": it["claim"], "expected": it["expected"],
            "tesla_lab_status": tl["status"], "tesla_lab_verdict": tl.get("verdict", ""),
            "tesla_lab_parts": tl.get("parts", ""),
            "tesla_lab_correct": (tl.get("verdict") == it["expected"]) if tl.get("verdict") else "",
            "reference": f"{ref['value']} {ref['unit']}".strip() if ref else "",
            "reference_source": ref["source"] if ref else "",
            "tesla_lab_value": tl.get("value") if tl.get("value") is not None else "",
            "tesla_lab_rel_error": round(rel_err(tl.get("value"), ref), 3) if rel_err(tl.get("value"), ref) is not None else "",
            "baseline_status": bl["status"], "baseline_verdict": bl.get("verdict", ""),
            "baseline_correct": (bl.get("verdict") == it["expected"]) if bl.get("verdict") else "",
            "baseline_value": bl.get("value") if bl.get("value") is not None else "",
            "baseline_rel_error": round(rel_err(bl.get("value"), ref), 3) if rel_err(bl.get("value"), ref) is not None else "",
        }
        rows.append(row)
        print(tl["status"], tl.get("verdict", ""), "| baseline:", bl["status"], bl.get("verdict", ""))

    def summary(prefix: str) -> str:
        done = [r for r in rows if r[f"{prefix}_correct"] != ""]
        errs = [r[f"{prefix}_rel_error"] for r in rows if r[f"{prefix}_rel_error"] != ""]
        if not done:
            return "not run"
        acc = sum(1 for r in done if r[f"{prefix}_correct"]) / len(done)
        e = f", mean numeric error {100 * sum(errs) / len(errs):.0f}% over {len(errs)} items" if errs else ""
        return f"{sum(1 for r in done if r[f'{prefix}_correct'])}/{len(done)} verdicts correct ({100 * acc:.0f}%){e}"

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"scoreboard-{s.mode}-{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    md = [f"# Tesla Lab eval scoreboard ({s.mode}, {dt.datetime.now():%Y-%m-%d %H:%M})", "",
          f"- Tesla Lab ({s.model}{' mocked' if s.is_mock else ''}): {summary('tesla_lab')}",
          f"- Plain chatbot baseline ({bmodel if client else 'none'}): {summary('baseline')}",
          f"- Items: {len(rows)}; not run in this mode: {sum(1 for r in rows if r['tesla_lab_correct'] == '')}", "",
          "| id | expected | Tesla Lab | ok | ref | TL value | TL err | baseline | ok | BL value | BL err |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tlv = r["tesla_lab_verdict"] or r["tesla_lab_status"]
        blv = r["baseline_verdict"] or r["baseline_status"]
        ok = lambda x: "✅" if x is True else ("❌" if x is False else "")  # noqa: E731
        md.append(f"| {r['id']} | {r['expected']} | {tlv} | {ok(r['tesla_lab_correct'])} | {r['reference']} | "
                  f"{r['tesla_lab_value']} | {r['tesla_lab_rel_error']} | {blv} | {ok(r['baseline_correct'])} | "
                  f"{r['baseline_value']} | {r['baseline_rel_error']} |")
    md += ["", "Headline verdict = most severe verdict across the claims Tesla Lab extracted. "
           "Numeric error = |value - reference| / reference. Mock mode uses canned research and claim "
           "extraction for the 3 demo claims, with the real math."]
    md_path = out / f"scoreboard-{s.mode}-{stamp}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md[:5]))
    print(f"\nWrote {md_path}\n      {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
