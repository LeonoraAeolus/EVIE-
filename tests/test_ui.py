from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_streamlit_app_runs_in_mock_mode(monkeypatch):
    monkeypatch.setenv("TESLA_LAB_MODE", "mock")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[-1].click().run()  # "Test it against physics"
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "WORKS" in text and "BREAKS PHYSICS" in text
    assert "Found in the Tesla Archive" in text  # v2: archive citation with page
    assert any("Spec sheet" in s.value for s in at.subheader)  # v2: product design engine
    assert any("what would need to be true" in s.value for s in at.subheader)  # v2: innovation loop
    button_labels = [b.label for b in at.download_button]
    assert "Blender script (.py)" in button_labels
    assert "3D Printable Mesh (.stl)" in button_labels
