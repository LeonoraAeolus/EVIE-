"""Product Design Engine: the math doesn't only judge the claim, it designs the product.

For the claim that WORKS, a vetted design calculator turns requirements (power, distance,
flow, backup hours...) into concrete specs, an exposure/safety screen, a rough bill of
materials, and an image prompt written FROM the computed numbers, so the render matches
the spec sheet.

  DESIGNS            name -> builder(requirements) -> ProductSpec
  design_for_template(calc_template) -> design name (or None)
  build_spec(design, requirements, title=..., based_on_claim=...) -> ProductSpec
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Callable

from ..schema import BomItem, ProductSpec, Reference, SafetyCheck, SpecItem
from .disc_pump import design_disc_pump
from .h2_storage import design_h2_backup
from .wpt import design_wireless_power

PRICE_NOTE = ("Rough estimates from typical 2025-26 distributor / retail list prices for one-off "
              "prototype quantities, not quotes. Verify before you buy.")


def _fmt(x: float, sig: int = 3) -> str:
    if x == 0:
        return "0"
    return f"{x:.{sig}g}" if abs(x) < 1e5 else f"{x:,.0f}"


def _si(x: float, unit: str) -> str:
    for p, s in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")):
        if abs(x) >= p:
            return f"{x / p:.3g} {s}{unit}"
    return f"{x:.3g} {unit}"


def _bom(items: list[tuple]) -> tuple[list[BomItem], float]:
    bom = [BomItem(part=p, qty=q, unit_cost_usd=c, note=n) for p, q, c, n in items]
    return bom, round(sum(b.total for b in bom), 2)


# --------------------------------------------------------------------------- wireless power
def spec_wireless_power(req: dict[str, Any], title: str = "Resonant wireless power", based_on_claim: str = "") -> ProductSpec:
    r = design_wireless_power(req)
    tx, rx, f = r["tx"], r["rx"], r["f"]
    d0 = r["req"]["distance_m"]
    P = r["req"]["target_power_w"]
    wb = next(s for s in r["safety"] if s["key"] == "icnirp2020_wholebody")
    loc = next(s for s in r["safety"] if s["key"] == "icnirp2020_local")
    p_cap_wb = P * (wb["limit"] / wb["value"]) ** 2 if wb["value"] > 0 else P
    p_cap_loc = P * (loc["limit"] / loc["value"]) ** 2 if loc["value"] > 0 else P
    cap_v_rating = math.ceil(r["Vc1"] * math.sqrt(2) * 2 / 500) * 500  # 2x margin on the peak

    S = SpecItem
    specs = [
        S(group="Requirements", name="Power into the receiver load", value=P, unit="W", basis="requirement"),
        S(group="Requirements", name="Design distance (face to face)", value=d0, unit="m", basis="requirement"),
        S(group="Requirements", name="Operating frequency", value=round(f / 1e6, 3), unit="MHz", basis="requirement",
          note="6.78 MHz ISM band used by AirFuel Resonant" if abs(f - 6.78e6) < 1e3 else ""),
        S(group="Transmitter coil", name="Diameter", value=round(2 * tx["r"], 3), unit="m", basis="requirement"),
        S(group="Transmitter coil", name="Turns", value=tx["n"], basis="computed", note="best efficiency at the design distance, wire < λ/10"),
        S(group="Transmitter coil", name="Conductor", value=f"{2000 * tx['a']:.0f} mm copper tube", basis="requirement"),
        S(group="Transmitter coil", name="Wire length", value=round(tx["wire_len"], 2), unit="m", basis="computed",
          note=f"λ = {r['lambda']:.1f} m, so the lumped model holds"),
        S(group="Transmitter coil", name="Inductance", value=_si(tx["L"], "H"), basis="computed"),
        S(group="Transmitter coil", name="Tuning capacitor", value=_si(tx["C"], "F"), basis="computed",
          note=f"{r['Vc1']:.0f} V rms across it at full power; spec ≥ {cap_v_rating} V, low-loss (C0G/porcelain)"),
        S(group="Transmitter coil", name="Quality factor Q", value=round(tx["Q"]), basis="computed",
          note=f"copper {tx['r_ohm']:.3f} Ω + radiation {tx['r_rad']:.4f} Ω + capacitor ESR {tx['r_cap']:.3f} Ω (Q_cap assumed)"),
        S(group="Receiver coil", name="Diameter", value=round(2 * rx["r"], 3), unit="m", basis="requirement"),
        S(group="Receiver coil", name="Turns", value=rx["n"], basis="computed"),
        S(group="Receiver coil", name="Inductance", value=_si(rx["L"], "H"), basis="computed"),
        S(group="Receiver coil", name="Tuning capacitor", value=_si(rx["C"], "F"), basis="computed"),
        S(group="Receiver coil", name="Quality factor Q", value=round(rx["Q"]), basis="computed"),
        S(group="Link at design distance", name="Coupling k", value=round(r["k"], 4), basis="computed"),
        S(group="Link at design distance", name="Figure of merit k·√(Q1·Q2)", value=round(r["U"], 2), basis="computed"),
        S(group="Link at design distance", name="Coil-to-coil efficiency (optimal load)", value=round(100 * r["eta"], 1), unit="%", basis="computed"),
        S(group="Link at design distance", name="Optimal load resistance", value=round(r["RL"], 2), unit="Ω", basis="computed"),
        S(group="Link at design distance", name="Amplifier output power", value=round(r["P_in"], 1), unit="W", basis="computed"),
        S(group="Link at design distance", name="Wall power (amplifier η assumed)", value=round(r["P_wall"], 1), unit="W", basis="computed",
          note=f"amplifier efficiency {100 * r['amp_eff']:.0f}% assumed (GaN class-E/D)"),
        S(group="Link at design distance", name="Transmitter coil current", value=round(r["I1"], 2), unit="A rms", basis="computed"),
        S(group="Range", name="Efficiency halves at", value=round(r["d_half_eff"], 2) if r["d_half_eff"] else "beyond chart", unit="m", basis="computed"),
        S(group="Range", name="Efficiency below 10% beyond", value=round(r["d_10pct"], 2) if r["d_10pct"] else "beyond chart", unit="m", basis="computed"),
        S(group="Exposure screen", name="Max power passing the whole-body screen at design distance", value=round(p_cap_wb, 2), unit="W", basis="computed",
          note="H ∝ √P; on-axis peak field, conservative"),
        S(group="Exposure screen", name="Max power passing the local-exposure screen", value=round(p_cap_loc, 2), unit="W", basis="computed"),
        S(group="Exposure screen", name="Keep-out distance on the coil axis at full power", value=wb["clearance_m"], unit="m", basis="computed",
          note="whole-body reference level"),
    ]
    safety = [SafetyCheck(**{k: v for k, v in s.items() if k in SafetyCheck.model_fields}) for s in r["safety"]]
    bom, total = _bom([
        ("6.78 MHz GaN amplifier board (class-E/D, ~20-30 W)", 1, 120.0, "e.g. a GaN eval/reference board"),
        ("Copper tube for tx coil (m)", round(tx["wire_len"] + 0.5, 1), 6.0, f"{2000 * tx['a']:.0f} mm soft copper"),
        (f"Tx tuning capacitors (≥{cap_v_rating} V, low-loss)", 2, 18.0, "porcelain/C0G RF caps, parallel for tuning"),
        ("Receiver coil + capacitor (per device)", 3, 6.0, "enamelled/litz wire on a printed former"),
        ("Receiver rectifier + DC-DC (per device)", 3, 9.0, "Schottky/GaN bridge + buck to 5 V"),
        ("Controller with BLE (power negotiation, foreign-object cut-off)", 1, 12.0, "ESP32/nRF-class module"),
        ("24 V power supply, 60 W", 1, 25.0, ""),
        ("Shelf body, back panel, ferrite-free mounting", 1, 80.0, "oak/plywood; keep metal away from the coil"),
        ("Near-field probe / test equipment share", 1, 40.0, "to verify the exposure screen on the prototype"),
    ])
    req_view = dict(r["req"])
    headline = [
        f"{P:g} W into the load at {d0:g} m with {100 * r['eta']:.0f}% coil-to-coil efficiency",
        f"{2 * tx['r'] * 100:.0f} cm {tx['n']}-turn copper coil at {f / 1e6:g} MHz, Q ≈ {tx['Q']:.0f}",
        f"Efficiency halves by {r['d_half_eff']:.2f} m and is below 10% beyond {r['d_10pct']:.2f} m" if r["d_half_eff"] and r["d_10pct"] else "Efficiency falls steeply with distance",
        (f"Exposure screen: {P:g} W fails the whole-body level on-axis at {d0:g} m; passes beyond {wb['clearance_m']} m "
         f"(or at ≤{p_cap_wb:.1f} W)") if not wb["passes"] else f"Exposure screen passes at {d0:g} m (whole-body level)",
        f"Prototype parts ≈ ${total:,.0f} (estimate)",
    ]
    render = (
        f"Photoreal product render, 3/4 view, warm studio light, clean background, no text. "
        f"A minimalist oak bookshelf about 1.2 m wide; a single {2 * tx['r'] * 100:.0f} cm diameter circular coil of "
        f"{tx['n']} turns of {2000 * tx['a']:.0f} mm brushed-copper tube is inlaid flush in the back panel, centred behind the middle shelf. "
        f"Two small cylindrical tuning capacitors and a palm-sized amplifier box sit at the coil's lower edge. "
        f"On the shelf, within {d0 * 100:.0f} cm of the coil face, a desk lamp whose round base hides a {2 * rx['r'] * 100:.0f} cm receiver coil "
        f"glows softly; a phone and a small plant sensor sit nearby. "
        f"Faint pale-blue magnetic field lines loop through the coil and fade out within about {(r['d_half_eff'] or d0 * 1.5):.2g} m. "
        f"A subtle floor marking shows a {wb['clearance_m']:g} m keep-clear arc in front of the coil. Shallow depth of field."
    )
    return ProductSpec(
        design="wireless_power", title=title, based_on_claim=based_on_claim, requirements=req_view,
        headline=headline, specs=specs, series=r["series"],
        series_labels={"x": "distance (m)", "y": "coil-to-coil efficiency (%)"},
        safety=safety, bom=bom, bom_total_usd=total, cost_note=PRICE_NOTE, render_prompt=render,
        caveats=[
            "Coaxial, aligned coils; lateral offset or tilt lowers k and efficiency.",
            "Quasi-static model (coils much smaller than the 44 m wavelength); no ferrite or nearby metal.",
            "Capacitor Q, proximity factor and amplifier efficiency are assumptions; measure them on the prototype.",
            "The exposure screen compares the on-axis peak H-field with ICNIRP reference levels. "
            "Real compliance needs measurement or SAR/induced-field simulation against the basic restrictions.",
        ],
        references=[
            Reference(title="Kurs et al., Wireless power transfer via strongly coupled magnetic resonances, Science 317 (2007)",
                      url="https://pubmed.ncbi.nlm.nih.gov/17556549"),
            Reference(title="ICNIRP (2020) RF guidelines, Tables 5-6", url="https://www.icnirp.org/cms/upload/publications/ICNIRPrfgdl2020.pdf"),
            Reference(title="ICNIRP (2010) low-frequency guidelines, Table 4", url="https://www.icnirp.org/cms/upload/publications/ICNIRPLFgdl.pdf"),
            Reference(title="Tesla, US 645,576 / 649,621 (1900) - the original tuned-circuit transmission patents",
                      url="https://patents.google.com/patent/US645576A/en"),
        ],
    )


# --------------------------------------------------------------------------- disc pump
def spec_disc_pump(req: dict[str, Any], title: str = "Disc pump", based_on_claim: str = "") -> ProductSpec:
    r = design_disc_pump(req)
    S = SpecItem
    lpm = r["Q"] * 60000
    specs = [
        S(group="Requirements", name="Flow", value=round(lpm, 1), unit="L/min", basis="requirement"),
        S(group="Requirements", name="Head", value=r["H"], unit="m", basis="requirement"),
        S(group="Requirements", name="Motor speed", value=int(r["rpm"]), unit="rpm", basis="requirement",
          note="2-pole induction motor on 50 Hz mains (Malaysia/Singapore)"),
        S(group="Rotor", name="Tip speed", value=round(r["u"], 2), unit="m/s", basis="computed", note=f"head coefficient ψ = {r['psi']} (assumed)"),
        S(group="Rotor", name="Disc diameter", value=round(r["D"] * 1000, 1), unit="mm", basis="computed"),
        S(group="Rotor", name="Inlet (eye) diameter", value=round(2 * r["r_in"] * 1000, 1), unit="mm", basis="computed"),
        S(group="Rotor", name="Boundary-layer thickness √(ν/ω)", value=round(r["delta"] * 1e6), unit="µm", basis="computed"),
        S(group="Rotor", name="Disc gap", value=round(r["gap"] * 1000, 2), unit="mm", basis="computed",
          note=f"set by {r['gap_limited_by']}" + (" - wider than the efficiency optimum, the price of not clogging" if r["gap_limited_by"] == "solids" else "")),
        S(group="Rotor", name="Number of discs", value=r["n_discs"], basis="computed", note=f"{r['n_gaps']} flow gaps"),
        S(group="Rotor", name="Stack length", value=round(r["stack"] * 1000, 1), unit="mm", basis="computed"),
        S(group="Rotor", name="Gap Reynolds number ωb²/ν", value=round(r["re_gap"]), basis="computed"),
        S(group="Power", name="Hydraulic power ρgQH", value=round(r["P_h"], 1), unit="W", basis="computed"),
        S(group="Power", name="Hydraulic efficiency", value=round(100 * r["eta"]), unit="%", basis="assumed", note="published range 20-30% for disc pumps"),
        S(group="Power", name="Shaft power", value=round(r["P_shaft"]), unit="W", basis="computed"),
        S(group="Power", name="Motor (next standard size, 15% margin)", value=r["motor_w"], unit="W", basis="computed"),
        S(group="Running cost", name="Energy per day", value=round(r["kwh_day"], 2), unit="kWh", basis="computed"),
        S(group="Running cost", name="Electricity per year", value=round(r["usd_year"]), unit="USD", basis="estimate",
          note=f"at ${r['tariff']}/kWh (assumed)"),
        S(group="Running cost", name=f"Same duty, centrifugal pump at {100 * r['eta_cf']:.0f}%", value=round(r["usd_year_cf"]), unit="USD/yr", basis="estimate"),
    ]
    bom, total = _bom([
        (f"{r['motor_w']} W 2-pole motor (IP55)", 1, 85.0, ""),
        ("Stainless 304 discs, laser-cut, 0.5 mm", r["n_discs"], 1.5, f"{r['D'] * 1000:.0f} mm OD, spoked eye"),
        ("Spacers / washers set", 1, 8.0, f"{r['gap'] * 1000:.2f} mm"),
        ("Shaft, bearings, mechanical seal", 1, 35.0, ""),
        ("Clear polycarbonate volute housing (machined)", 1, 60.0, "transparent so the stack is visible"),
        ("Hose fittings, clamps, mounting bracket", 1, 15.0, ""),
        ("Motor starter / switch, cable", 1, 20.0, ""),
    ])
    headline = [
        f"{lpm:.0f} L/min at {r['H']:g} m head from a {r['D'] * 1000:.0f} mm stack of {r['n_discs']} discs at {int(r['rpm'])} rpm",
        f"Gaps {r['gap'] * 1000:.1f} mm pass solids up to that size, no blades to clog or bruise fish",
        f"{r['P_shaft']:.0f} W shaft power, {r['motor_w']} W motor (at an assumed {100 * r['eta']:.0f}% efficiency)",
        f"Costs about ${r['usd_year'] - r['usd_year_cf']:.0f}/yr more electricity than a {100 * r['eta_cf']:.0f}% centrifugal pump (estimate)",
        f"Prototype parts ≈ ${total:,.0f} (estimate)",
    ]
    render = (
        f"Photoreal product render, 3/4 view, soft daylight, shallow depth of field, no text. "
        f"A compact bladeless disc pump: a clear polycarbonate housing about {r['D'] * 1000 * 1.6:.0f} mm across shows a stack of "
        f"{r['n_discs']} thin polished stainless-steel discs, each {r['D'] * 1000:.0f} mm in diameter with {r['gap'] * 1000:.1f} mm gaps, "
        f"total stack about {r['stack'] * 1000:.0f} mm long, driven by a small {r['motor_w']} W grey electric motor on the same shaft. "
        f"A central inlet hose and a tangential outlet hose; mounted beside a small aquaculture pond with water droplets on the housing."
    )
    return ProductSpec(
        design="disc_pump", title=title, based_on_claim=based_on_claim,
        requirements={"flow_lpm": round(lpm, 1), "head_m": r["H"], "rpm": r["rpm"]}, headline=headline, specs=specs,
        series=r["series"], series_labels={"x": "speed (rpm)", "y": "head (m)"}, safety=[], bom=bom,
        bom_total_usd=total, cost_note=PRICE_NOTE, render_prompt=render,
        caveats=["First-order sizing: the head coefficient and efficiency are assumptions to calibrate on a prototype.",
                 "Laminar-flow assumption between discs; check the gap Reynolds number.",
                 "Worth it only where clogging, shear or noise matter more than electricity cost."],
        references=[Reference(title="Tesla, US 1,061,206 Turbine (1913)", url="https://patents.google.com/patent/US1061206A/en"),
                    Reference(title="Infinity Turbine, Tesla disc pump review (20-30% hydraulic efficiency)",
                              url="https://www.infinityturbine.com/tesla-pump-review-article-by-infinity-turbine.html"),
                    Reference(title="Heng et al., Tesla bladed pump preliminary study, Energies 13(18):4873 (2020)",
                              url="https://www.mdpi.com/1996-1073/13/18/4873")],
    )


# --------------------------------------------------------------------------- hydrogen backup
def spec_h2_backup(req: dict[str, Any], title: str = "Solar-hydrogen backup", based_on_claim: str = "") -> ProductSpec:
    r = design_h2_backup(req)
    q = r["req"]
    S = SpecItem
    specs = [
        S(group="Requirements", name="Average critical load", value=q["load_w"], unit="W", basis="requirement"),
        S(group="Requirements", name="Peak load", value=q["peak_load_w"], unit="W", basis="requirement"),
        S(group="Requirements", name="Backup duration", value=q["backup_hours"], unit="h", basis="requirement"),
        S(group="Requirements", name="Refill the tank in", value=q["recharge_days"], unit="days", basis="requirement"),
        S(group="Energy", name="Energy delivered per outage", value=round(r["e_out"], 2), unit="kWh", basis="computed"),
        S(group="Energy", name="Fuel cell efficiency (LHV)", value=round(100 * q["fuel_cell_eff"]), unit="%", basis="assumed"),
        S(group="Energy", name="Hydrogen needed", value=round(r["kg"], 3), unit="kg", basis="computed", note="LHV 33.33 kWh/kg"),
        S(group="Storage", name="Storage pressure", value=q["storage_bar"], unit="bar", basis="requirement", note="electrolyser outlet pressure, no compressor"),
        S(group="Storage", name="H₂ density at that pressure (25 °C)", value=round(r["rho"], 3), unit="kg/m³", basis="computed"),
        S(group="Storage", name="Tank volume needed", value=round(r["vol_l"]), unit="L", basis="computed", note=f"{100 * r['usable']:.0f}% usable"),
        S(group="Storage", name=f"{r['tank_l']:.0f} L cylinders", value=r["tanks"], basis="computed"),
        S(group="Storage", name="Water consumed per refill", value=round(r["water_l"], 1), unit="L", basis="computed", note="8.94 kg water per kg H₂"),
        S(group="Charging", name="Electrolyser energy per refill", value=round(r["e_in"], 1), unit="kWh", basis="computed",
          note=f"{q['electrolyser_kwh_per_kg']} kWh/kg (assumed, small PEM)"),
        S(group="Charging", name="Round-trip efficiency", value=round(100 * r["rte"], 1), unit="%", basis="computed"),
        S(group="Charging", name="Electrolyser size", value=round(r["el_kw"], 2), unit="kW", basis="computed"),
        S(group="Charging", name="Extra solar PV needed", value=round(r["pv_kwp"], 2), unit="kWp", basis="computed",
          note=f"{q['peak_sun_hours']} peak-sun-hours/day (assumed, typical Malaysia), performance ratio 0.8"),
        S(group="Output", name="Fuel cell rating", value=r["fc_w"], unit="W", basis="computed"),
        S(group="Battery comparison", name="LFP battery for the same outage", value=round(r["batt_kwh"], 1), unit="kWh", basis="computed"),
        S(group="Battery comparison", name="Battery cost / H₂ system cost", value=f"${r['cost_batt']:,.0f} / ${r['cost_h2']:,.0f}", basis="estimate"),
        S(group="Battery comparison", name="H₂ becomes cheaper for outages longer than",
          value=(f"{r['crossover_h'] / 24:.1f}" if r["crossover_h"] else "> 30"), unit="days", basis="estimate"),
    ]
    bom, total = _bom([
        (f"PEM electrolyser ~{max(0.5, round(r['el_kw'], 1))} kW, 30 bar outlet", 1, 5500.0, "small-scale units are expensive"),
        (f"PEM fuel cell {r['fc_w']:.0f} W with DC-DC", 1, 2500.0, ""),
        (f"{r['tank_l']:.0f} L steel cylinder, 30 bar rated", r["tanks"], 450.0, ""),
        (f"Solar PV, {r['pv_kwp']:.1f} kWp extra", round(r["pv_kwp"], 2), 600.0, "per kWp, panels + mounting"),
        ("Water purifier / deioniser", 1, 250.0, ""),
        ("Ventilated outdoor cabinet, H₂ sensor, valves, controls", 1, 900.0, "H₂ lower flammability limit is 4% in air"),
    ])
    cross = f"{r['crossover_h'] / 24:.0f} days" if r["crossover_h"] else "more than 30 days"
    headline = [
        f"{q['backup_hours']:.0f} h of {q['load_w']:.0f} W backup needs {r['kg']:.2f} kg of hydrogen in {r['tanks']} × {r['tank_l']:.0f} L tanks at {q['storage_bar']:.0f} bar",
        f"Round trip {100 * r['rte']:.0f}%: {r['e_in']:.1f} kWh of solar in, {r['e_out']:.1f} kWh out",
        f"Refill in {q['recharge_days']:.0f} days with {r['pv_kwp']:.1f} kWp of extra solar and {r['water_l']:.1f} L of water",
        f"Honest comparison: a {r['batt_kwh']:.0f} kWh battery costs about ${r['cost_batt']:,.0f}; hydrogen only wins for outages over {cross}",
        f"Prototype parts ≈ ${total:,.0f} (estimate)",
    ]
    render = (
        f"Photoreal product render, 3/4 view, tropical late-afternoon light, palm trees softly blurred, no text. "
        f"A white and teal ventilated outdoor cabinet about 1.6 m tall beside a small rural clinic; through a side window "
        f"{r['tanks']} slim grey {r['tank_l']:.0f}-litre steel hydrogen cylinders stand upright. A compact electrolyser and a "
        f"{r['fc_w']:.0f} W fuel cell module sit on shelves above; a small water tank on the side. "
        f"{max(2, round(r['pv_kwp'] / 0.4))} solar panels on the clinic roof, a status panel glowing green."
    )
    return ProductSpec(
        design="h2_backup", title=title, based_on_claim=based_on_claim, requirements=q, headline=headline, specs=specs,
        series=r["series"], series_labels={"x": "backup duration (h)", "y": "hydrogen system cost (USD)"},
        extra_series={"battery system cost (USD)": r["batt_series"]}, safety=[], bom=bom, bom_total_usd=total,
        cost_note=PRICE_NOTE, render_prompt=render,
        caveats=["Never makes energy from water: about 70% of the solar energy is lost in the loop.",
                 "Component efficiencies and prices are assumptions for small (sub-kW) systems.",
                 "Hydrogen storage needs ventilation, leak sensing and local code approval."],
        references=[Reference(title="US DOE Hydrogen Program, H2 properties (LHV 33.3 kWh/kg)",
                              url="https://www.energy.gov/eere/fuelcells/hydrogen-storage"),
                    Reference(title="NIST Chemistry WebBook, hydrogen fluid properties", url="https://webbook.nist.gov/chemistry/fluid/")],
    )


DESIGNS: dict[str, Callable[..., ProductSpec]] = {
    "wireless_power": spec_wireless_power,
    "disc_pump": spec_disc_pump,
    "h2_backup": spec_h2_backup,
}
CALCULATORS = {"wireless_power": design_wireless_power, "disc_pump": design_disc_pump, "h2_backup": design_h2_backup}

TEMPLATE_TO_DESIGN = {
    "resonant_coupling": "wireless_power",
    "disc_turbine": "disc_pump",
    "energy_balance": "h2_backup",
    "electrolysis_loop": "h2_backup",
}


def design_for_template(template: str | None) -> str | None:
    return TEMPLATE_TO_DESIGN.get(template or "")


def describe_designs() -> str:
    lines = []
    for name, fn in CALCULATORS.items():
        doc = inspect.getdoc(fn) or ""
        lines.append(f"- {name}: {doc.splitlines()[0]}  " + " ".join(doc.splitlines()[1:]))
    return "\n".join(lines)


def design_source(design: str) -> str:
    """The code behind the spec numbers, shown in the UI."""
    import sys
    fn = CALCULATORS.get(design)
    return inspect.getsource(sys.modules[fn.__module__]) if fn else ""


def build_spec(design: str, requirements: dict[str, Any] | None = None, title: str = "", based_on_claim: str = "") -> ProductSpec:
    fn = DESIGNS.get(design)
    if fn is None:
        return ProductSpec(design=design, title=title or design, error=f"no design calculator named {design!r}")
    try:
        return fn(dict(requirements or {}), title=title or design, based_on_claim=based_on_claim)
    except Exception as e:  # bad requirement values from the model
        return ProductSpec(design=design, title=title or design, error=f"{type(e).__name__}: {e}")
