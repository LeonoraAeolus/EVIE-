"""v4: the Patent Search Planner view runs in mock mode (Streamlit AppTest, no keys, no network)."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _planner_app(monkeypatch):
    monkeypatch.setenv("TESLA_LAB_MODE", "mock")
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    at.button(key="nav_planner").click().run()
    assert not at.exception
    return at


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader]
    parts += [w.value for w in at.warning] + [i.value for i in at.info]
    return " ".join(str(p) for p in parts)


def test_planner_find_and_wildcard_mock(monkeypatch):
    at = _planner_app(monkeypatch)
    assert any("Patent Search Planner" in s.value for s in at.subheader)
    assert at.text_area(key="pl_find_goal").value  # example pre-filled
    at.button(key="run_find").click().run()
    assert not at.exception
    text = _all_text(at)
    assert "US5365025A" in text  # top-ranked patent shown
    assert "patents.google.com/patent/" in text  # clickable patent page links
    assert "patents.google.com/?q=" in text  # clickable search links for the user
    assert "not legal advice" in text.lower()
    assert "CANNED" in text or "canned" in text
    assert "1.28 m" in text or "1.277 m" in text  # Wildcard quarter-wave tube, computed
    assert "8.83 L" in text or "8.8 L" in text  # Helmholtz volume


def test_planner_check_and_improve_mock(monkeypatch):
    at = _planner_app(monkeypatch)
    at.button(key="run_check").click().run()
    assert not at.exception
    assert "PRIOR ART" in _all_text(at).upper()
    at.text_area(key="pl_improve_cons").set_value("max tube length 1.0 m").run()
    at.button(key="run_improve").click().run()
    assert not at.exception
    text = _all_text(at)
    assert "1.0 m" in text
    assert "3D-printed" in text  # roadmap only, labelled
