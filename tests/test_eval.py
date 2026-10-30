from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eval_set_shape():
    import json

    items = [json.loads(l) for l in (ROOT / "evals" / "claims.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(items) >= 20
    assert {i["expected"] for i in items} == {"WORKS", "MISSING PIECE", "BREAKS PHYSICS"}
    for i in items:
        if i.get("reference"):
            assert i["reference"]["source"] and isinstance(i["reference"]["value"], (int, float))


def test_run_eval_mock_writes_scoreboard(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setenv("NEBIUS_API_KEY", "")
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--mock", "--baseline", "--out", str(tmp_path)]) == 0
    md = next(tmp_path.glob("scoreboard-mock-*.md")).read_text(encoding="utf-8")
    csv = next(tmp_path.glob("scoreboard-mock-*.csv")).read_text(encoding="utf-8")
    assert "3/3 verdicts correct" in md and "needs live run" in csv
    assert "baseline not run" in csv  # no key -> nothing invented
