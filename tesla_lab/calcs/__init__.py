"""Vetted physics calculators.

Every template is plain, readable Python that the LLM can *select* and fill with
numbers (it never runs arbitrary code by default). If no template fits, the LLM can
write a tiny custom calculation, which runs in the sandbox (see sandbox.py).

Each template returns a dict matching schema.CalcResult.
"""
from __future__ import annotations

import math
from typing import Any, Callable

MU0 = 4e-7 * math.pi  # vacuum permeability, H/m
H2O_HHV_KJ_PER_MOL = 285.8  # enthalpy of formation of liquid water (higher heating value)
H2O_LHV_KJ_PER_MOL = 241.8  # water leaves an engine as vapour (lower heating value)
H2O_GIBBS_KJ_PER_MOL = 237.1  # minimum electrical work to split water at 25 C


def _f(params: dict, key: str, default: float | None = None) -> float:
    v = params.get(key, default)
    if v is None:
        raise ValueError(f"missing parameter: {key}")
    return float(v)


# --- 1. Near-field resonant coupling (Tesla coils, WiTricity, Qi) ---------------------
def resonant_coupling(p: dict) -> dict:
    """Max link efficiency of two resonant loops vs distance.

    Model: coaxial single-turn loops, dipole-approximation mutual inductance
        M = mu0*pi*r1^2*r2^2 / (2*(d^2 + r1^2)^1.5)
        L = mu0*r*(ln(8r/a) - 2)
        k = M / sqrt(L1*L2)
        eta_max = k^2 Q1 Q2 / (1 + sqrt(1 + k^2 Q1 Q2))^2
    Simplified on purpose: it shows the shape (efficiency collapses ~1/d^6 once d >> r).
    """
    r1 = _f(p, "r_tx_m", 0.3)
    r2 = _f(p, "r_rx_m", 0.3)
    a = _f(p, "wire_radius_m", 0.003)
    q1 = _f(p, "q_tx", 950)
    q2 = _f(p, "q_rx", 950)
    d_claim = _f(p, "claimed_distance_m", 2.0)
    d_max = float(p.get("d_max_m", max(4 * d_claim, 4.0)))

    def eta(d: float) -> float:
        m = MU0 * math.pi * r1**2 * r2**2 / (2 * (d**2 + r1**2) ** 1.5)
        l1 = MU0 * r1 * (math.log(8 * r1 / a) - 2)
        l2 = MU0 * r2 * (math.log(8 * r2 / a) - 2)
        k = m / math.sqrt(l1 * l2)
        u = k * k * q1 * q2
        return u / (1 + math.sqrt(1 + u)) ** 2

    xs = [round(d_max * i / 60, 4) for i in range(1, 61)]
    ys = [round(100 * eta(x), 4) for x in xs]
    e_claim = eta(d_claim)
    metrics: dict[str, Any] = {
        "efficiency_at_claimed_distance_pct": round(100 * e_claim, 2),
        "claimed_distance_m": d_claim,
        "distance_where_efficiency_drops_below_1pct_m": next((x for x, y in zip(xs, ys) if y < 1), None),
        "efficiency_at_10x_coil_radius_pct": round(100 * eta(10 * r1), 4),
    }
    if "observed_efficiency_pct" in p:
        metrics["observed_efficiency_pct"] = float(p["observed_efficiency_pct"])
    works = e_claim >= 0.10
    return {
        "ok": True,
        "template": "resonant_coupling",
        "metrics": metrics,
        "series": {"x": xs, "y": ys},
        "series_labels": {"x": "distance (m)", "y": "max link efficiency (%)"},
        "flags": {"violates_energy_conservation": False},
        "summary": (
            f"Simplified two-loop model: ~{100 * e_claim:.0f}% max efficiency at {d_claim:g} m, "
            f"falling below 1% by ~{metrics['distance_where_efficiency_drops_below_1pct_m']} m. "
            "Room-scale resonant power is real physics; it does not scale to continents."
        ),
        "suggested_verdict": "WORKS" if works else "MISSING PIECE",
    }


# --- 2. Broadcasting power in all directions ------------------------------------------
def broadcast_spreading(p: dict) -> dict:
    """Fraction of radiated power a receiver catches if power spreads over a hemisphere.

        P_rx / P_tx = A_rx / (2*pi*d^2)
    """
    p_tx = _f(p, "tx_power_w", 200e3)
    a_rx = _f(p, "rx_area_m2", 1.0)
    d_points = [float(x) for x in p.get("distances_km", [1, 10, 100, 1000, 5000])]
    rows = {}
    for d_km in d_points:
        d = d_km * 1000
        frac = a_rx / (2 * math.pi * d * d)
        rows[f"received_at_{d_km:g}km_w"] = p_tx * frac
    xs = [float(x) for x in d_points]
    ys = [math.log10(max(rows[f"received_at_{d:g}km_w"], 1e-300)) for d in d_points]
    far = rows[f"received_at_{d_points[-1]:g}km_w"]
    return {
        "ok": True,
        "template": "broadcast_spreading",
        "metrics": {"tx_power_w": p_tx, "rx_area_m2": a_rx, **{k: float(f"{v:.3g}") for k, v in rows.items()}},
        "series": {"x": xs, "y": ys},
        "series_labels": {"x": "distance (km)", "y": "log10 received power (W)"},
        "flags": {"violates_energy_conservation": False},
        "summary": (
            f"Spread over a hemisphere, {p_tx:,.0f} W leaves a {a_rx:g} m^2 receiver "
            f"{far:.2g} W at {d_points[-1]:g} km. Broadcast power is real but hopeless for supply."
        ),
        "suggested_verdict": "MISSING PIECE",
    }


# --- 3. The Earth as a resonator --------------------------------------------------------
def earth_resonator(p: dict) -> dict:
    """How long the Earth-ionosphere cavity can hold energy.

        Q = f0 / bandwidth ; energy ring-down time tau = Q / (2*pi*f0)
        loss per cycle = 2*pi / Q of the stored energy
    """
    f0 = _f(p, "f0_hz", 7.83)
    if "q_factor" in p:
        q = _f(p, "q_factor")
    else:
        q = 1 / _f(p, "fractional_bandwidth", 0.2)
    tau = q / (2 * math.pi * f0)
    loss_per_cycle = min(2 * math.pi / q, 1.0)
    return {
        "ok": True,
        "template": "earth_resonator",
        "metrics": {
            "f0_hz": f0,
            "q_factor": round(q, 2),
            "energy_ring_down_s": round(tau, 3),
            "ring_down_in_cycles": round(tau * f0, 2),
            "fraction_of_stored_energy_lost_per_cycle": round(loss_per_cycle, 3),
        },
        "flags": {"violates_energy_conservation": False},
        "summary": (
            f"With Q~{q:.0f}, the cavity loses its stored energy in ~{tau:.2f} s "
            f"(about {tau * f0:.1f} oscillation cycles). A leaky resonator cannot be a global power line: the station must "
            "re-supply the losses continuously, and no low-loss path is known."
        ),
        "suggested_verdict": "MISSING PIECE",
    }


# --- 4. Generic energy-conservation check --------------------------------------------------
def energy_balance(p: dict) -> dict:
    """Sum energy (or power) in vs out. Out > in means the claim needs free energy."""
    ins = {k: float(v) for k, v in (p.get("inputs") or {}).items()}
    outs = {k: float(v) for k, v in (p.get("outputs") or {}).items()}
    unit = p.get("unit", "W")
    tin, tout = sum(ins.values()), sum(outs.values())
    tol = float(p.get("tolerance", 0.01))
    violates = tout > tin * (1 + tol)
    ratio = (tout / tin) if tin > 0 else math.inf
    return {
        "ok": True,
        "template": "energy_balance",
        "metrics": {
            "total_in": tin,
            "total_out": tout,
            "unit": unit,
            "out_over_in": ratio if math.isfinite(ratio) else "infinite (nothing in)",
        },
        "flags": {"violates_energy_conservation": violates},
        "summary": (
            f"Out {tout:g} {unit} vs in {tin:g} {unit}: "
            + ("more comes out than goes in. That is a perpetual-motion claim." if violates else "balance is possible.")
        ),
        "suggested_verdict": "BREAKS PHYSICS" if violates else "WORKS",
    }


# --- 5. Water as fuel: electrolysis -> burn -> electricity loop -----------------------------
def electrolysis_loop(p: dict) -> dict:
    """Energy returned per mole of water if a car splits water and burns the hydrogen.

    Split:   E_in  = HHV / eta_electrolyser          (>= 237.1 kJ/mol even when perfect)
    Burn:    E_mech = LHV * eta_engine
    Return:  E_back = E_mech * eta_generator
    Loop gain = E_back / E_in   (must be > 1 to "run on water")
    """
    ee = _f(p, "electrolyser_eff", 0.70)
    en = _f(p, "engine_eff", 0.25)
    eg = _f(p, "generator_eff", 0.90)
    claimed = float(p.get("claimed_loop_gain", 1.0))
    e_in = H2O_HHV_KJ_PER_MOL / ee
    e_back = H2O_LHV_KJ_PER_MOL * en * eg
    gain = e_back / e_in
    best = H2O_LHV_KJ_PER_MOL / H2O_GIBBS_KJ_PER_MOL  # perfect machines, still below the 286 kJ heat-balanced split
    best_honest = H2O_LHV_KJ_PER_MOL / H2O_HHV_KJ_PER_MOL
    violates = claimed > 1.0 or claimed > best_honest
    return {
        "ok": True,
        "template": "electrolysis_loop",
        "metrics": {
            "energy_in_kj_per_mol": round(e_in, 1),
            "energy_back_kj_per_mol": round(e_back, 1),
            "loop_gain_realistic": round(gain, 3),
            "loop_gain_with_perfect_machines": round(best_honest, 3),
            "claimed_loop_gain": claimed,
            "gibbs_min_kj_per_mol": H2O_GIBBS_KJ_PER_MOL,
        },
        "flags": {"violates_energy_conservation": violates},
        "summary": (
            f"Each mole of water costs {e_in:.0f} kJ to split and returns {e_back:.0f} kJ "
            f"(gain {gain:.2f}). Even perfect machines return {best_honest:.2f}. "
            "Water is the ash of hydrogen, not a fuel."
        ),
        "suggested_verdict": "BREAKS PHYSICS" if violates else "MISSING PIECE",
        "_unused_best_gibbs_ratio": round(best, 3),
    }


# --- 6. Tesla's bladeless (boundary-layer) turbine ----------------------------------------
def disc_turbine(p: dict) -> dict:
    """Boundary-layer thickness on a spinning disc and the claimed-vs-measured gap.

        delta ~ 5 * sqrt(nu / omega)   (von Karman rotating disc)
        good disc spacing is roughly 2*delta: wider gaps let fluid slip through unused
    """
    rpm = _f(p, "rpm", 10000)
    nu = _f(p, "kinematic_viscosity_m2_s", 1.5e-5)  # air at ~20 C
    gap = _f(p, "disc_gap_m", 0.0004)
    claimed = _f(p, "claimed_efficiency", 0.97)
    lo, hi = [float(x) for x in p.get("measured_efficiency_range", [0.36, 0.41])]
    omega = rpm * 2 * math.pi / 60
    delta = 5 * math.sqrt(nu / omega)
    violates = claimed > 1.0
    gap_ratio = gap / (2 * delta)
    if violates:
        verdict = "BREAKS PHYSICS"
    elif claimed <= hi * 1.15:
        verdict = "WORKS"
    else:
        verdict = "MISSING PIECE"
    return {
        "ok": True,
        "template": "disc_turbine",
        "metrics": {
            "boundary_layer_mm": round(delta * 1000, 3),
            "disc_gap_mm": gap * 1000,
            "gap_vs_ideal_ratio": round(gap_ratio, 2),
            "claimed_efficiency_pct": round(claimed * 100, 1),
            "measured_efficiency_pct": f"{lo * 100:.0f}-{hi * 100:.0f}",
            "unexplained_gap_pct_points": round((claimed - hi) * 100, 1),
        },
        "flags": {"violates_energy_conservation": violates},
        "summary": (
            f"Boundary layer ~{delta * 1000:.2f} mm at {rpm:,.0f} rpm, so smooth discs really can drag the fluid. "
            + (f"Claimed {claimed * 100:.0f}% is within the measured {lo * 100:.0f}-{hi * 100:.0f}%."
               if claimed <= hi * 1.15 else
               f"Claimed {claimed * 100:.0f}% vs measured {lo * 100:.0f}-{hi * 100:.0f}%: "
               f"{(claimed - hi) * 100:.0f} points nobody has reproduced.")
        ),
        "suggested_verdict": verdict,
    }


# --- 7. Heat-engine ceiling ----------------------------------------------------------------
def carnot_limit(p: dict) -> dict:
    """No heat engine beats eta = 1 - Tc/Th (temperatures in kelvin)."""
    th = _f(p, "t_hot_k")
    tc = _f(p, "t_cold_k")
    claimed = _f(p, "claimed_efficiency")
    limit = 1 - tc / th
    violates = claimed > limit
    return {
        "ok": True,
        "template": "carnot_limit",
        "metrics": {"carnot_limit_pct": round(limit * 100, 2), "claimed_pct": round(claimed * 100, 2)},
        "flags": {"violates_energy_conservation": False, "violates_second_law": violates},
        "summary": f"Carnot ceiling is {limit * 100:.1f}%; claim is {claimed * 100:.1f}%.",
        "suggested_verdict": "BREAKS PHYSICS" if violates else "WORKS",
    }


TEMPLATES: dict[str, Callable[[dict], dict]] = {
    "resonant_coupling": resonant_coupling,
    "broadcast_spreading": broadcast_spreading,
    "earth_resonator": earth_resonator,
    "energy_balance": energy_balance,
    "electrolysis_loop": electrolysis_loop,
    "disc_turbine": disc_turbine,
    "carnot_limit": carnot_limit,
}


def describe_templates() -> str:
    """One line per template, fed to the LLM so it can pick one and fill its params."""
    lines = []
    for name, fn in TEMPLATES.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        lines.append(f"- {name}: {doc}")
    return "\n".join(lines)
