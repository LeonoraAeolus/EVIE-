"""Design calculator: Tesla (disc / boundary-layer) pump.

Model (first-order sizing, every assumption named):
  * Tip speed from head:   H = psi * u^2 / g   ->  u = sqrt(g*H/psi)
    psi (head coefficient) is ASSUMED ~0.5 for a disc rotor (ideal no-slip rotor gives
    psi -> 1; slip and exit losses roughly halve it). Calibrate against your prototype.
  * Disc diameter from tip speed and motor speed:  D = 60*u / (pi * n)
  * Boundary-layer (Ekman) thickness on a spinning disc: delta = sqrt(nu / omega)
    The disc gap is set to max(gap_factor*delta, the largest solid you must pass).
  * Flow per gap: Q_gap = 2*pi*r_in*b*v_r with the radial inlet velocity v_r = radial_ratio*u
  * Hydraulic power  P_h = rho*g*Q*H;  shaft power = P_h / eta, eta ASSUMED from the
    published range for disc pumps (about 20-30% hydraulic, up to ~40% in lab studies).
Sources: Tesla, US 1,061,206 (1913); Infinity Turbine disc-pump review (20-30% hydraulic
efficiency); Heng et al., Energies 13(18):4873 (2020).
"""
from __future__ import annotations

import math
from typing import Any

G = 9.81
STANDARD_MOTORS_W = [90, 120, 180, 250, 370, 550, 750, 1100, 1500, 2200]


def design_disc_pump(req: dict[str, Any]) -> dict[str, Any]:
    """req keys [defaults]: flow_lpm [60], head_m [3], rpm [2850 (2-pole motor on 50 Hz)],
    solids_mm [0.8] largest particle to pass, water_temp_c [28], hydraulic_efficiency [0.25],
    head_coefficient [0.5], radial_ratio [0.1], inlet_ratio [0.4], disc_thickness_mm [0.5],
    gap_factor [3], hours_per_day [24], electricity_usd_kwh [0.12], centrifugal_efficiency [0.5]"""
    Q = float(req.get("flow_lpm", 60)) / 60000.0  # m^3/s
    H = float(req.get("head_m", 3))
    n = float(req.get("rpm", 2850))
    solids = float(req.get("solids_mm", 0.8)) / 1000
    T = float(req.get("water_temp_c", 28))
    eta = float(req.get("hydraulic_efficiency", 0.25))
    psi = float(req.get("head_coefficient", 0.5))
    radial_ratio = float(req.get("radial_ratio", 0.1))
    inlet_ratio = float(req.get("inlet_ratio", 0.4))
    t_disc = float(req.get("disc_thickness_mm", 0.5)) / 1000
    gap_factor = float(req.get("gap_factor", 3))
    hours = float(req.get("hours_per_day", 24))
    tariff = float(req.get("electricity_usd_kwh", 0.12))
    eta_cf = float(req.get("centrifugal_efficiency", 0.5))

    # water kinematic viscosity (m^2/s), Vogel-type fit, good to ~2% for 0-60 C
    mu = 2.414e-5 * 10 ** (247.8 / (T + 273.15 - 140))
    rho = 1000 * (1 - (T + 288.9414) / (508929.2 * (T + 68.12963)) * (T - 3.9863) ** 2)
    nu = mu / rho

    omega = 2 * math.pi * n / 60
    u = math.sqrt(G * H / psi)
    D = 60 * u / (math.pi * n)
    r_out = D / 2
    r_in = inlet_ratio * r_out
    delta = math.sqrt(nu / omega)
    gap = max(gap_factor * delta, solids)
    v_r = radial_ratio * u
    q_gap = 2 * math.pi * r_in * gap * v_r
    n_gaps = max(1, math.ceil(Q / q_gap))
    n_discs = n_gaps + 1
    stack = n_discs * t_disc + n_gaps * gap
    re_gap = omega * gap * gap / nu  # rotational Reynolds number on the gap
    P_h = rho * G * Q * H
    P_shaft = P_h / eta
    motor = next((m for m in STANDARD_MOTORS_W if m >= P_shaft * 1.15), STANDARD_MOTORS_W[-1])
    kwh_day = P_shaft / 0.8 * hours / 1000  # 80% motor efficiency assumed
    kwh_day_cf = P_h / eta_cf / 0.8 * hours / 1000
    # Affinity law: head vs speed for this rotor (chart)
    rpms = [int(n * x / 20) for x in range(6, 31)]
    heads = [round(psi * (math.pi * D * r / 60) ** 2 / G, 3) for r in rpms]
    return {
        "Q": Q, "H": H, "rpm": n, "nu": nu, "rho": rho, "u": u, "D": D, "r_in": r_in,
        "delta": delta, "gap": gap, "gap_limited_by": "solids" if gap == solids else "boundary layer",
        "v_r": v_r, "n_gaps": n_gaps, "n_discs": n_discs, "stack": stack, "re_gap": re_gap,
        "P_h": P_h, "P_shaft": P_shaft, "motor_w": motor, "eta": eta, "psi": psi,
        "kwh_day": kwh_day, "kwh_day_cf": kwh_day_cf, "usd_year": kwh_day * 365 * tariff,
        "usd_year_cf": kwh_day_cf * 365 * tariff, "tariff": tariff, "eta_cf": eta_cf,
        "series": {"x": rpms, "y": heads},
    }
