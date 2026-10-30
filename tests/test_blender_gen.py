import ast
import pytest
from tesla_lab.blender_gen import generate_blender_script
from tesla_lab.design import build_spec


def test_blender_wpt_script_syntax_and_dimensions():
    spec = build_spec("wireless_power", {"target_power_w": 15.0, "distance_m": 0.6})
    script = generate_blender_script(spec)
    assert script.startswith("# ==============================================================================")
    # Must be 100% valid Python code
    tree = ast.parse(script)
    assert isinstance(tree, ast.Module)
    # Check that key physical dimensions and Blender operations exist
    assert "import bpy" in script
    assert "Transmitter_Coil" in script
    assert "Receiver_Coil" in script
    assert "0.6" in script  # distance
    assert "PolishedCopper" in script


def test_blender_disc_pump_script():
    spec = build_spec("disc_pump", {"flow_m3_h": 12.0, "head_m": 15.0})
    script = generate_blender_script(spec)
    tree = ast.parse(script)
    assert isinstance(tree, ast.Module)
    assert "Turbine_Shaft" in script
    assert "Disc_" in script
    assert "Clear_Casing_Volute" in script
    assert "BrushedSteel" in script


def test_blender_h2_storage_script():
    spec = build_spec("h2_storage", {"backup_kwh": 30.0, "h2_kg": 2.5})
    script = generate_blender_script(spec)
    tree = ast.parse(script)
    assert isinstance(tree, ast.Module)
    assert "H2_Tank_Body" in script
    assert "EndCap_" in script
    assert "CarbonFiberComposite" in script
