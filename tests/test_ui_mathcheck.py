from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_streamlit_shows_math_checker_panel(monkeypatch):
    monkeypatch.setenv("TESLA_LAB_MODE", "mock")
    monkeypatch.setenv("OPENMATH_BASE_URL", "")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    at.button[-1].click().run()
    assert not at.exception
    assert any("Math checker" in s.value for s in at.subheader)  # v3 panel
    text = " ".join(m.value for m in at.markdown)
    assert "math checker agreed" in text  # badge on the verdicts
    assert "Step 1" in text  # derivation steps rendered
    assert any("Canned derivations" in w.value for w in at.warning)  # clearly labelled in mock mode
