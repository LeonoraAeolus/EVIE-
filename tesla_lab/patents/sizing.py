"""Vetted sizing calculators for Wildcard ideas (Python is the calculator of record).

Each calculator takes a params dict and returns:
    {value, unit, quantity, formula, inputs, sizes: {name: (value, unit)}, physics_ok, physics_note,
     check: {problem, variables, reference, python_value, quantity, unit}}   # for the OpenMath checker

`sizes` are compared with the team's constraints to get math_fit; `physics_ok=False` means the mechanism
cannot act on the target at all (e.g. a feature 1000x smaller than the wavelength) -> math_fit 0.

Exhaust worked example (spec v2): 4-cylinder 4-stroke at 3,000 rpm -> firing frequency 100 Hz; exhaust
~400 °C, gamma 1.35, R 287.05 J/(kg K) -> c ≈ 510 m/s; quarter-wave tube c/(4f) ≈ 1.28 m; Helmholtz
resonator with a 40 mm diameter x 60 mm long neck (end correction 1.7 r) ≈ 8.8 L.
"""
from __future__ import annotations

import math
from typing import Any, Callable

R_AIR = 287.05  # J/(kg K)


def _f(p: dict, k: str, d: float) -> float:
    v = p.get(k, d)
    return float(d if v is None else v)


def speed_of_sound(gas_temp_c: float = 400.0, gamma: float = 1.35, r_specific: float = R_AIR) -> float:
    """c = sqrt(gamma R T)."""
    return math.sqrt(gamma * r_specific * (gas_temp_c + 273.15))


def firing_frequency(rpm: float = 3000, cylinders: int = 4, strokes: int = 4) -> float:
    """Pulses per second at the tailpipe: rpm/60 x cylinders / (strokes/2)."""
    return rpm / 60.0 * cylinders / (strokes / 2.0)


def _common(p: dict) -> tuple[float, float]:
    c = p.get("c_m_s")
    if c is None:
        c = speed_of_sound(_f(p, "gas_temp_c", 400), _f(p, "gamma", 1.35))
    f = p.get("frequency_hz")
    if f is None:
        f = firing_frequency(_f(p, "rpm", 3000), int(_f(p, "cylinders", 4)))
    return float(c), float(f)


def quarter_wave_tube(p: dict) -> dict:
    """Closed side branch a quarter wavelength long: L = c / (4 f). Coiled like a French horn."""
    c, f = _common(p)
    L = c / (4 * f)
    d = _f(p, "tube_diameter_m", 0.04)
    vol_l = math.pi * (d / 2) ** 2 * L * 1000
    f2 = _f(p, "second_frequency_hz", 2 * f)
    return {
        "value": round(L, 3), "unit": "m", "quantity": "quarter-wave tube length",
        "formula": "L = c / (4 f)", "inputs": {"c_m_s": round(c, 1), "frequency_hz": f, "tube_diameter_m": d},
        "sizes": {"tube_length_m": (round(L, 3), "m"), "volume_l": (round(vol_l, 2), "L"),
                  "gas_temp_c": (_f(p, "gas_temp_c", 400), "°C")},
        "physics_ok": True,
        "physics_note": f"{L:.2f} m at {f:g} Hz ({c / (4 * f2):.2f} m at {f2:g} Hz); a {d * 1000:.0f} mm tube "
                        f"holds {vol_l:.1f} L, so it fits when coiled",
        "check": {"problem": (f"A closed side-branch tube cancels a pressure pulse when its length is a quarter of the "
                              f"wavelength. The speed of sound in the gas is c = {c:.1f} m/s and the pulse frequency is "
                              f"f = {f:g} Hz. Derive the tube length L as a formula in c and f, then evaluate it in metres."),
                  "variables": {"c": round(c, 1), "f": f}, "reference": "c/(4*f)", "python_value": round(L, 3),
                  "quantity": "quarter-wave tube length", "unit": "m"},
    }


def helmholtz_resonator(p: dict) -> dict:
    """f = c/(2 pi) sqrt(A / (V L_eff)), L_eff = L + 1.7 r  ->  V = A c^2 / ((2 pi f)^2 L_eff)."""
    c, f = _common(p)
    d = _f(p, "neck_diameter_m", 0.04)
    ln = _f(p, "neck_length_m", 0.06)
    r = d / 2
    a = math.pi * r * r
    l_eff = ln + 1.7 * r
    v = a * c * c / ((2 * math.pi * f) ** 2 * l_eff)
    return {
        "value": round(v * 1000, 2), "unit": "L", "quantity": "Helmholtz cavity volume",
        "formula": "V = A c² / ((2π f)² (L + 1.7 r))",
        "inputs": {"c_m_s": round(c, 1), "frequency_hz": f, "neck_diameter_m": d, "neck_length_m": ln},
        "sizes": {"volume_l": (round(v * 1000, 2), "L"), "gas_temp_c": (_f(p, "gas_temp_c", 400), "°C")},
        "physics_ok": True,
        "physics_note": f"{v * 1000:.1f} L cavity for a {d * 1000:.0f} × {ln * 1000:.0f} mm neck",
        "check": {"problem": (f"A Helmholtz resonator has a neck of radius r = {r:g} m and length L = {ln:g} m, with "
                              f"effective length L + 1.7 r. The speed of sound is c = {c:.1f} m/s. Its resonance "
                              f"frequency is f = (c/(2 pi)) sqrt(A/(V (L + 1.7 r))) with A = pi r^2. Find the cavity "
                              f"volume V that tunes it to f = {f:g} Hz, in litres."),
                  "variables": {"c": round(c, 1), "f": f, "r": r, "L": ln}, "reference": "",
                  "python_value": round(v * 1000, 2), "quantity": "Helmholtz cavity volume", "unit": "L"},
    }


def windkessel_chamber(p: dict) -> dict:
    """Two-element Windkessel as an RC low-pass: |H| = 1/sqrt(1 + (w R C)^2).

    R = allowed backpressure / volume flow; C = V / (rho c^2) = V / (gamma p) (gas compliance).
    Solve for the chamber volume that gives `attenuation_db` at f.
    """
    c, f = _common(p)
    dp = _f(p, "backpressure_pa", 10_000)
    q = _f(p, "flow_m3_s", 0.112)
    att = _f(p, "attenuation_db", 10)
    gamma = _f(p, "gamma", 1.35)
    p_abs = _f(p, "pressure_pa", 101_325)
    r_ac = dp / q
    wrc = math.sqrt(10 ** (att / 10) - 1)
    cap = wrc / (2 * math.pi * f * r_ac)
    v = cap * gamma * p_abs
    return {
        "value": round(v * 1000, 2), "unit": "L", "quantity": "compliance chamber volume",
        "formula": "V = γ p · √(10^(dB/10) − 1) / (2π f R),  R = Δp / Q",
        "inputs": {"frequency_hz": f, "backpressure_pa": dp, "flow_m3_s": q, "attenuation_db": att},
        "sizes": {"volume_l": (round(v * 1000, 2), "L"), "gas_temp_c": (_f(p, "gas_temp_c", 400), "°C"),
                  "elastic_wall_temp_c": (_f(p, "gas_temp_c", 400), "°C")},
        "physics_ok": True,
        "physics_note": f"{v * 1000:.1f} L of gas compliance for {att:g} dB at {f:g} Hz with a {dp / 1000:g} kPa "
                        f"backpressure budget; an elastic wall (the aorta's trick) would see {_f(p, 'gas_temp_c', 400):g} °C",
        "check": {"problem": (f"A two-element Windkessel filter has resistance R = {r_ac:.4g} Pa s/m^3 and compliance "
                              f"C = V/(gamma p) with gamma = {gamma:g} and p = {p_abs:g} Pa. Its pressure transfer is "
                              f"|H| = 1/sqrt(1 + (2 pi f R C)^2). Find V in litres such that the attenuation at "
                              f"f = {f:g} Hz is {att:g} dB."),
                  "variables": {}, "reference": "", "python_value": round(v * 1000, 2),
                  "quantity": "compliance chamber volume", "unit": "L"},
    }


def serration_frequency(p: dict) -> dict:
    """A feature of pitch s acts on sound when it is ~ a quarter wavelength: f = c / (4 s)."""
    c, f = _common(p)
    s = _f(p, "feature_m", 0.002)
    f_act = c / (4 * s)
    ratio = f_act / f
    return {
        "value": round(f_act, 0), "unit": "Hz", "quantity": "frequency the serrations act on",
        "formula": "f = c / (4 s)", "inputs": {"c_m_s": round(c, 1), "feature_m": s, "target_hz": f},
        "sizes": {"acting_frequency_hz": (round(f_act, 0), "Hz")},
        "physics_ok": ratio < 10,
        "physics_note": f"{s * 1000:g} mm serrations act near {f_act / 1000:.0f} kHz, {ratio:.0f}× above the "
                        f"{f:g} Hz pulses: they quiet broadband flow noise, not engine pulses",
        "check": {"problem": (f"A serrated edge with tooth pitch s = {s:g} m interacts with sound whose quarter "
                              f"wavelength equals s. The speed of sound is c = {c:.1f} m/s. Derive that frequency f as a "
                              f"formula in c and s and evaluate it in hertz."),
                  "variables": {"c": round(c, 1), "s": s}, "reference": "c/(4*s)", "python_value": round(f_act, 0),
                  "quantity": "frequency the serrations act on", "unit": "Hz"},
    }


def tunable_quarter_wave(p: dict) -> dict:
    """Slide-whistle idea: a piston changes the side-branch length with rpm. Longest length at the lowest rpm."""
    c = p.get("c_m_s") or speed_of_sound(_f(p, "gas_temp_c", 400), _f(p, "gamma", 1.35))
    rpm_lo, rpm_hi = _f(p, "rpm_min", 1500), _f(p, "rpm_max", 6000)
    cyl = int(_f(p, "cylinders", 4))
    f_lo, f_hi = firing_frequency(rpm_lo, cyl), firing_frequency(rpm_hi, cyl)
    l_max, l_min = c / (4 * f_lo), c / (4 * f_hi)
    return {
        "value": round(l_max, 3), "unit": "m", "quantity": "longest tube length (lowest rpm)",
        "formula": "L(rpm) = c / (4 f(rpm)),  f = rpm/60 × cylinders/2",
        "inputs": {"c_m_s": round(c, 1), "rpm_min": rpm_lo, "rpm_max": rpm_hi, "cylinders": cyl},
        "sizes": {"tube_length_m": (round(l_max, 3), "m"), "stroke_m": (round(l_max - l_min, 3), "m"),
                  "gas_temp_c": (_f(p, "gas_temp_c", 400), "°C")},
        "physics_ok": True,
        "physics_note": f"a sliding piston covers {l_min:.2f}–{l_max:.2f} m for {rpm_lo:g}–{rpm_hi:g} rpm "
                        f"({f_lo:g}–{f_hi:g} Hz)",
        "check": {"problem": (f"A 4-stroke engine with {cyl} cylinders fires rpm/60 x {cyl}/2 pulses per second. A "
                              f"quarter-wave tube must be c/(4f) long, with c = {c:.1f} m/s. Find the tube length in "
                              f"metres at {rpm_lo:g} rpm."),
                  "variables": {}, "reference": "", "python_value": round(l_max, 3),
                  "quantity": "longest tube length (lowest rpm)", "unit": "m"},
    }


def relay_resonator_chain(p: dict) -> dict:
    """Sympathetic-string idea for wireless power: N passive tuned relay coils between Tx and Rx.

    Uses the same coupled-loop model as calcs.resonant_coupling (dipole mutual inductance, eta_max per hop),
    chain efficiency ≈ product of per-hop efficiencies (a simplification: it ignores cross-coupling).
    """
    from ..calcs import MU0

    r = _f(p, "coil_radius_m", 0.3)
    a = _f(p, "wire_radius_m", 0.003)
    q = _f(p, "q", 950)
    d = _f(p, "distance_m", 2.0)
    n_relays = int(_f(p, "relays", 3))

    def eta(dist: float) -> float:
        m = MU0 * math.pi * r ** 4 / (2 * (dist ** 2 + r ** 2) ** 1.5)
        l_ = MU0 * r * (math.log(8 * r / a) - 2)
        u = (m / l_) ** 2 * q * q
        return u / (1 + math.sqrt(1 + u)) ** 2

    hop = d / (n_relays + 1)
    chain = eta(hop) ** (n_relays + 1)
    direct = eta(d)
    return {
        "value": round(100 * chain, 2), "unit": "%", "quantity": f"efficiency over {d:g} m with {n_relays} relays",
        "formula": "η_chain = η(d/(N+1))^(N+1)", "inputs": {"distance_m": d, "relays": n_relays, "coil_radius_m": r, "q": q},
        "sizes": {"efficiency_pct": (round(100 * chain, 2), "%"), "relays": (n_relays, "coils")},
        "physics_ok": True,
        "physics_note": f"{100 * chain:.0f}% over {d:g} m through {n_relays} relay coils vs {100 * direct:.1f}% direct",
        "check": {"problem": (f"Each hop of a resonator chain transfers power with efficiency eta = u/(1+sqrt(1+u))^2. "
                              f"A chain has {n_relays + 1} identical hops, each with u = {(lambda dist: ((MU0 * math.pi * r ** 4 / (2 * (dist ** 2 + r ** 2) ** 1.5)) / (MU0 * r * (math.log(8 * r / a) - 2))) ** 2 * q * q)(hop):.4g}. "
                              f"Find the total efficiency in percent (the product of the hop efficiencies)."),
                  "variables": {}, "reference": "", "python_value": round(100 * chain, 2),
                  "quantity": "chain efficiency", "unit": "%"},
    }


def small_coil_link(p: dict) -> dict:
    """Induction-cooktop idea: cooktop-size coils (r ~ 0.1 m) over room distance. Same loop model."""
    from ..calcs import MU0

    r = _f(p, "coil_radius_m", 0.1)
    a = _f(p, "wire_radius_m", 0.002)
    q = _f(p, "q", 100)
    d = _f(p, "distance_m", 2.0)
    m = MU0 * math.pi * r ** 4 / (2 * (d ** 2 + r ** 2) ** 1.5)
    l_ = MU0 * r * (math.log(8 * r / a) - 2)
    u = (m / l_) ** 2 * q * q
    e = u / (1 + math.sqrt(1 + u)) ** 2
    return {
        "value": float(f"{100 * e:.3g}"), "unit": "%", "quantity": f"efficiency at {d:g} m",
        "formula": "η = u/(1+√(1+u))², u = k² Q²", "inputs": {"coil_radius_m": r, "q": q, "distance_m": d},
        "sizes": {"efficiency_pct": (float(f"{100 * e:.3g}"), "%")},
        "physics_ok": e > 1e-3,
        "physics_note": f"cooktop coils work at ~5 mm; at {d:g} m the link efficiency is {100 * e:.2g}%",
        "check": {"problem": (f"Two coupled resonators have u = k^2 Q^2 = {u:.4g}. The maximum link efficiency is "
                              f"u/(1+sqrt(1+u))^2. Evaluate it in percent."),
                  "variables": {}, "reference": "", "python_value": float(f"{100 * e:.3g}"),
                  "quantity": "link efficiency", "unit": "%"},
    }


CALCULATORS: dict[str, Callable[[dict], dict]] = {
    "quarter_wave_tube": quarter_wave_tube,
    "helmholtz_resonator": helmholtz_resonator,
    "windkessel_chamber": windkessel_chamber,
    "serration_frequency": serration_frequency,
    "tunable_quarter_wave": tunable_quarter_wave,
    "relay_resonator_chain": relay_resonator_chain,
    "small_coil_link": small_coil_link,
}


def run_sizing(name: str, params: dict[str, Any]) -> dict:
    if name not in CALCULATORS:
        raise KeyError(f"unknown sizing calculator {name!r}; known: {sorted(CALCULATORS)}")
    return CALCULATORS[name](params or {})


def describe() -> str:
    return "\n".join(f"- {n}: {(fn.__doc__ or '').strip().splitlines()[0]}" for n, fn in CALCULATORS.items())
