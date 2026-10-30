"""Design calculator: solar-hydrogen backup (electrolyser -> tank -> fuel cell).

Physics constants (standard values):
  LHV of H2 = 33.33 kWh/kg, HHV = 39.41 kWh/kg; water needed 8.94 kg per kg H2 (stoichiometry).
  Gas density from the real-gas law  rho = p*M / (Z*R*T),  Z(H2) ~ 1 + 0.00062*p_bar at 25 C
  (fit to NIST REFPROP data, good to ~1% below 100 bar).
Assumed component performance (edit in the request): electrolyser 55 kWh per kg H2
(~61% LHV, small PEM), fuel cell 50% electrical (LHV), PV performance ratio 0.8.
Cost model: rough small-scale prices, marked as estimates.
"""
from __future__ import annotations

import math
from typing import Any

LHV = 33.33  # kWh/kg
HHV = 39.41
M_H2 = 2.016e-3  # kg/mol
R_GAS = 8.314


def h2_density(p_bar: float, t_c: float = 25.0) -> float:
    Z = 1 + 0.00062 * p_bar
    return p_bar * 1e5 * M_H2 / (Z * R_GAS * (t_c + 273.15))


def design_h2_backup(req: dict[str, Any]) -> dict[str, Any]:
    """req keys [defaults]: load_w [150] average critical load, peak_load_w [400],
    backup_hours [48], recharge_days [7], storage_bar [30], min_bar [2], tank_litres [50],
    electrolyser_kwh_per_kg [55], fuel_cell_eff [0.5], peak_sun_hours [4.5], pv_pr [0.8],
    battery_usd_kwh [350], battery_dod [0.9], h2_fixed_usd [9000] (electrolyser+fuel cell+BoP),
    tank_usd [450] per cylinder"""
    load = float(req.get("load_w", 150))
    peak = float(req.get("peak_load_w", 400))
    hours = float(req.get("backup_hours", 48))
    days = float(req.get("recharge_days", 7))
    p = float(req.get("storage_bar", 30))
    pmin = float(req.get("min_bar", 2))
    tank_l = float(req.get("tank_litres", 50))
    el = float(req.get("electrolyser_kwh_per_kg", 55))
    fc = float(req.get("fuel_cell_eff", 0.5))
    psh = float(req.get("peak_sun_hours", 4.5))
    pr = float(req.get("pv_pr", 0.8))
    batt_usd = float(req.get("battery_usd_kwh", 350))
    dod = float(req.get("battery_dod", 0.9))
    fixed = float(req.get("h2_fixed_usd", 9000))
    tank_usd = float(req.get("tank_usd", 450))

    e_out = load * hours / 1000  # kWh delivered
    kg = e_out / (LHV * fc)
    rho = h2_density(p)
    usable = (p - pmin) / p
    vol_l = kg / (rho * usable) * 1000
    tanks = math.ceil(vol_l / tank_l)
    e_in = kg * el
    rte = e_out / e_in
    kwh_day = e_in / days
    el_kw = kwh_day / psh
    pv_kwp = kwh_day / (psh * pr)
    water_l = kg * 8.94
    batt_kwh = e_out / dod
    cost_h2 = fixed + tanks * tank_usd
    cost_batt = batt_kwh * batt_usd

    def costs(h: float) -> tuple[float, float]:
        k = load * h / 1000 / (LHV * fc)
        t = math.ceil(k / (rho * usable) * 1000 / tank_l)
        return fixed + t * tank_usd, load * h / 1000 / dod * batt_usd

    xs = [12 * i for i in range(1, 61)]  # 12 h .. 30 days
    ch, cb = zip(*(costs(x) for x in xs))
    crossover = next((x for x, a, b in zip(xs, ch, cb) if a <= b), None)
    return {
        "e_out": e_out, "kg": kg, "rho": rho, "usable": usable, "vol_l": vol_l, "tanks": tanks,
        "tank_l": tank_l, "e_in": e_in, "rte": rte, "kwh_day": kwh_day, "el_kw": el_kw,
        "pv_kwp": pv_kwp, "fc_w": peak, "water_l": water_l, "batt_kwh": batt_kwh,
        "batt_mass_kg": batt_kwh / 0.12,  # ~120 Wh/kg LFP pack, assumed
        "cost_h2": cost_h2, "cost_batt": cost_batt, "crossover_h": crossover,
        "series": {"x": xs, "y": [round(c) for c in ch]}, "batt_series": [round(c) for c in cb],
        "req": dict(load_w=load, peak_load_w=peak, backup_hours=hours, recharge_days=days,
                    storage_bar=p, electrolyser_kwh_per_kg=el, fuel_cell_eff=fc, peak_sun_hours=psh),
    }
