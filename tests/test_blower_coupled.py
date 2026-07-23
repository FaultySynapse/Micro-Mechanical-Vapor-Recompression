"""Tests for the blower-coupled operating point (fan curve + power as inputs)."""

import pytest

from mvr import DesignParameters, fan
from mvr import properties as props
from mvr.masstransfer import (
    lift_from_compression_pressure,
    solve_at_power,
    solve_at_speed,
)


def _design_and_blower():
    p = DesignParameters(evaporator_temp_C=70.0, noncondensable_pressure=500.0,
                         hx_area=0.4, evap_area=0.4, condenser_area=0.4,
                         feed_hx_effectiveness=0.85)
    rho = props.vapor_density(70.0, props.sat_pressure(70.0))
    blower = fan.size_blower_for_duty(0.012, 18_000.0, rho)
    return p, blower


def test_lift_from_pressure_roundtrips_with_the_model_ratio():
    # A pressure rise mapped to a lift should reproduce that rise via the model's
    # ratio definition (P_sat(T+lift)/P_sat(T)).
    t, p_ncg = 70.0, 500.0
    lift = lift_from_compression_pressure(12_000.0, t, p_ncg)
    p_sat = props.sat_pressure(t)
    dp = (props.sat_pressure(t + lift) / p_sat - 1.0) * (p_sat + p_ncg)
    assert dp == pytest.approx(12_000.0, rel=0.02)


def test_solve_at_power_draws_the_requested_power():
    p, blower = _design_and_blower()
    _, op = solve_at_power(p, blower, 500.0)
    assert op["blower_power_w"] == pytest.approx(500.0, rel=1e-6)


def test_flow_and_lift_are_outputs_of_the_curve():
    p, blower = _design_and_blower()
    r, op = solve_at_speed(p, blower, 1.0)
    # At design speed the blower sits at its BEP flow and design pressure; the
    # lift is driven by the head remaining after the duct drop.
    assert op["flow_m3s"] == pytest.approx(blower.design_flow, rel=1e-9)
    expected_lift = lift_from_compression_pressure(
        blower.design_pressure - p.duct_pressure_drop_pa,
        p.evaporator_temp_C, p.noncondensable_pressure)
    assert op["lift_K"] == pytest.approx(expected_lift, rel=1e-9)


def test_more_power_gives_more_flow_lift_and_production():
    p, blower = _design_and_blower()
    r_lo, op_lo = solve_at_power(p, blower, 250.0)
    r_hi, op_hi = solve_at_power(p, blower, 750.0)
    assert op_hi["flow_m3s"] > op_lo["flow_m3s"]
    assert op_hi["lift_K"] > op_lo["lift_K"]
    assert r_hi.distillate_lph > r_lo.distillate_lph


def test_production_never_exceeds_delivery():
    # The blower cannot condense more vapor than it delivers: ratio >= 1.
    p, blower = _design_and_blower()
    for power in (200.0, 500.0, 900.0):
        _, op = solve_at_power(p, blower, power)
        assert op["delivery_ratio"] >= 1.0 - 1e-6


def test_speed_power_cubic_relation():
    # Best-efficiency power scales as speed^3.
    p, blower = _design_and_blower()
    _, op1 = solve_at_speed(p, blower, 1.0)
    _, op2 = solve_at_speed(p, blower, 2.0)
    assert op2["blower_power_w"] == pytest.approx(8 * op1["blower_power_w"], rel=1e-6)


def test_model_fan_power_matches_blower_draw():
    # Consolidation guarantee: the evaporative model's Q·Δp/η fan power equals the
    # blower curve's power for the same operating point (one power basis).
    p, blower = _design_and_blower()
    for speed in (0.7, 1.0, 1.3):
        r, op = solve_at_speed(p, blower, speed)
        assert r.fan_power == pytest.approx(op["blower_power_w"], rel=0.01)
