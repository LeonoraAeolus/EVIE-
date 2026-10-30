import pytest
from tesla_lab.design import build_spec
from tesla_lab.mesh_gen import generate_stl, generate_threejs_viewer_html


def test_generate_stl_mesh():
    spec = build_spec("wireless_power", {"target_power_w": 10.0, "distance_m": 0.5})
    stl_text = generate_stl(spec)
    assert stl_text.startswith("solid wireless_power")
    assert "facet normal" in stl_text
    assert "outer loop" in stl_text
    assert "vertex" in stl_text
    assert stl_text.strip().endswith("endsolid wireless_power")


def test_generate_threejs_html():
    spec = build_spec("wireless_power", {"target_power_w": 10.0, "distance_m": 0.5})
    html = generate_threejs_viewer_html(spec)
    assert "THREE.Scene" in html
    assert "OrbitControls" in html
    assert "TorusGeometry" in html
    assert "Tesla_Shelf" or "Bookshelf" in html
