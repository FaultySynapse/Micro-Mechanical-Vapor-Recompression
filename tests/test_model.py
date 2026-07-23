"""Tests for the steady-state MVR model: sanity, conservation, and monotonicity."""

import math

import pytest

from mvr import DesignParameters, solve, overall_U
from mvr.model import Results


def test_baseline_runs_and_is_sane():
    r = solve(DesignParameters())
    assert r.distillate_rate > 0
    assert r.fan_power > 0
    assert 500.0 < r.overall_U < 20_000.0        # plate HX ballpark
    assert 0.1 < r.distillate_lph < 100.0         # bench-scale unit
    assert r.gain_output_ratio > 1.0              # MVR must beat single-effect


def test_overall_U_series_resistance():
    p = DesignParameters()
    U = overall_U(p)
    # U must be below the smallest individual conductance (series resistances).
    assert U < min(p.boiling_htc, p.condensing_htc)


def test_heat_duty_equals_latent_times_rate():
    from mvr import properties as props
    p = DesignParameters()
    r = solve(p)
    h_fg = props.latent_heat(r.boiling_temp_C)
    assert r.heat_duty == pytest.approx(r.distillate_rate * h_fg, rel=1e-9)


def test_energy_balance_closes():
    # Fan shaft work + makeup heat must equal all the sinks it feeds.
    p = DesignParameters()
    r = solve(p)
    shaft = r.fan_power * p.fan_motor_efficiency
    sinks = r.insulation_loss + r.unrecovered_stream_loss + r.feed_heating_duty
    assert (shaft + r.makeup_heat) == pytest.approx(sinks, rel=1e-9)


def test_mass_balance():
    p = DesignParameters()
    r = solve(p)
    assert r.feed_rate == pytest.approx(r.distillate_rate + r.concentrate_rate, rel=1e-9)
    assert r.distillate_rate == pytest.approx(r.feed_rate * p.recovery_ratio, rel=1e-9)


def test_larger_lift_increases_production():
    low = solve(DesignParameters(temp_lift=2.0))
    high = solve(DesignParameters(temp_lift=8.0))
    assert high.distillate_rate > low.distillate_rate


def test_larger_lift_costs_more_fan_power_per_kg():
    low = solve(DesignParameters(temp_lift=2.0))
    high = solve(DesignParameters(temp_lift=8.0))
    low_per_kg = low.fan_power / low.distillate_rate
    high_per_kg = high.fan_power / high.distillate_rate
    assert high_per_kg > low_per_kg


def test_better_insulation_lowers_makeup_heat():
    leaky = solve(DesignParameters(insulation_ua=2.0))
    tight = solve(DesignParameters(insulation_ua=0.05))
    assert tight.makeup_heat < leaky.makeup_heat


def test_better_feed_hx_lowers_energy():
    poor = solve(DesignParameters(feed_hx_effectiveness=0.5))
    good = solve(DesignParameters(feed_hx_effectiveness=0.95))
    assert good.specific_energy_kwh_per_l < poor.specific_energy_kwh_per_l


def test_boiling_point_elevation_raises_pressure_ratio():
    none = solve(DesignParameters(boiling_point_elevation=0.0))
    some = solve(DesignParameters(boiling_point_elevation=1.5))
    assert some.pressure_ratio > none.pressure_ratio
    # BPE does no useful heat transfer -> same duty, more compression work/kg.
    assert some.heat_duty == pytest.approx(none.heat_duty, rel=1e-9)
    assert (some.compression_work_specific > none.compression_work_specific)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        DesignParameters(temp_lift=-1.0)
    with pytest.raises(ValueError):
        DesignParameters(feed_hx_effectiveness=1.5)


def test_vacuum_operation_lower_temp():
    # Running under partial vacuum (lower cold-side temp) should still work.
    r = solve(DesignParameters(evaporator_temp_C=60.0))
    assert r.evap_pressure_pa < 101_325.0
    assert r.distillate_rate > 0
