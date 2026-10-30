from tesla_lab.calcs import TEMPLATES, electrolysis_loop, energy_balance, resonant_coupling


def test_resonant_efficiency_falls_with_distance():
    r = resonant_coupling({"claimed_distance_m": 2.0})
    ys = r["series"]["y"]
    assert ys[0] > ys[-1]
    assert 0 < r["metrics"]["efficiency_at_claimed_distance_pct"] < 100


def test_water_car_cannot_run_itself():
    r = electrolysis_loop({"claimed_loop_gain": 1.0})
    assert r["metrics"]["loop_gain_with_perfect_machines"] < 1
    assert r["metrics"]["loop_gain_realistic"] < 0.5
    assert r["suggested_verdict"] == "BREAKS PHYSICS"


def test_energy_balance_flags_free_energy():
    r = energy_balance({"inputs": {"in": 100}, "outputs": {"out": 150}})
    assert r["flags"]["violates_energy_conservation"]
    ok = energy_balance({"inputs": {"in": 100}, "outputs": {"out": 40}})
    assert not ok["flags"]["violates_energy_conservation"]


def test_every_template_runs_with_defaults_or_minimal_params():
    minimal = {"carnot_limit": {"t_hot_k": 600, "t_cold_k": 300, "claimed_efficiency": 0.6}}
    for name, fn in TEMPLATES.items():
        out = fn(minimal.get(name, {}))
        assert out["ok"] and out["summary"], name
