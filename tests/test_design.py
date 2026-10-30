import math

import pytest

from tesla_lab.design import DESIGNS, build_spec, design_for_template
from tesla_lab.design.disc_pump import design_disc_pump
from tesla_lab.design.h2_storage import LHV, design_h2_backup, h2_density
from tesla_lab.design.wpt import (EXPOSURE_LIMITS, MU0, coil_inductance, design_wireless_power, link_efficiency,
                                  loop_mutual)


def test_mutual_inductance_matches_far_field_limit():
    # far apart, two loops look like dipoles: M ~ mu0*pi*r1^2*r2^2 / (2 d^3)
    r, d = 0.1, 5.0
    approx = MU0 * math.pi * r**4 / (2 * (r * r + d * d) ** 1.5)
    assert loop_mutual(r, r, d) == pytest.approx(approx, rel=1e-3)


def test_single_loop_inductance_textbook():
    # 1-turn loop, r = 0.25 m, wire radius 3 mm: mu0*r*(ln(8r/a)-2) = 1.414 uH
    assert coil_inductance(0.25, 0.003, 1, 0.01) == pytest.approx(1.414e-6, rel=0.01)


def test_link_efficiency_formula():
    assert link_efficiency(0) == 0
    assert link_efficiency(1e6) == pytest.approx(1, abs=1e-5)
    assert link_efficiency(1.0) == pytest.approx(1 / (1 + math.sqrt(2)) ** 2)


def test_wireless_power_design_is_physical():
    r = design_wireless_power({"target_power_w": 10, "distance_m": 0.5})
    assert 0 < r["k"] < 1 and 0 < r["eta"] < 1
    ys = r["series"]["y"]
    assert all(a >= b - 1e-9 for a, b in zip(ys, ys[1:])), "efficiency must fall with distance"
    assert r["P_in"] > 10  # can't deliver more than you put in
    assert r["tx"]["wire_len"] < r["lambda"] / 10 or r["tx"]["n"] == 1


def test_icnirp_reference_levels_at_678mhz():
    assert EXPOSURE_LIMITS["icnirp2020_wholebody"]["h_limit"](6.78) == pytest.approx(0.3245, rel=1e-3)
    assert EXPOSURE_LIMITS["icnirp2020_local"]["h_limit"](6.78) == pytest.approx(0.7227, rel=1e-3)
    assert EXPOSURE_LIMITS["icnirp2010_nerve"]["h_limit"](1.0) == 21.0


def test_spec_render_prompt_comes_from_numbers():
    spec = build_spec("wireless_power", {"tx_diameter_m": 0.4}, title="Shelf")
    assert spec.error is None
    turns = next(s.value for s in spec.specs if s.group == "Transmitter coil" and s.name == "Turns")
    assert "40 cm diameter" in spec.render_prompt and f"{turns} turns" in spec.render_prompt
    assert spec.safety and all(s.approximate and s.source.startswith("ICNIRP") for s in spec.safety)
    assert spec.bom_total_usd == pytest.approx(sum(b.total for b in spec.bom), abs=0.01)
    assert all(b.estimate for b in spec.bom)


def test_disc_pump_sizing():
    r = design_disc_pump({"flow_lpm": 60, "head_m": 3, "rpm": 2850, "head_coefficient": 0.5})
    u = math.sqrt(9.81 * 3 / 0.5)
    assert r["D"] == pytest.approx(60 * u / (math.pi * 2850))
    assert r["P_h"] == pytest.approx(r["rho"] * 9.81 * 0.001 * 3)
    assert r["motor_w"] >= r["P_shaft"]
    spec = build_spec("disc_pump", {"flow_lpm": 60}, title="Pump")
    assert f"{r['n_discs']} thin" in spec.render_prompt


def test_h2_backup_energy_balance():
    r = design_h2_backup({"load_w": 100, "backup_hours": 24, "fuel_cell_eff": 0.5})
    assert r["kg"] == pytest.approx(2.4 / (LHV * 0.5))
    assert h2_density(30) == pytest.approx(2.40, rel=0.02)  # NIST: ~2.39 kg/m3 at 30 bar, 25 C
    assert 0 < r["rte"] < 0.5  # round trip loses most of the energy - never a free lunch


def test_build_spec_handles_bad_input():
    assert build_spec("nope").error
    assert build_spec("disc_pump", {"flow_lpm": "lots"}).error
    assert set(DESIGNS) == {"wireless_power", "disc_pump", "h2_backup"}
    assert design_for_template("resonant_coupling") == "wireless_power"
