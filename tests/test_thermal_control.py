"""Tests for the economizer-bypass temperature controller."""

import pytest

from mvr import DesignParameters
from mvr.masstransfer import (
    balance_temperature_by_economizer,
    solve_evaporative,
)


def _surplus_design(**kw):
    # A well-insulated, high-recovery unit runs a heat surplus (makeup < 0).
    base = dict(evaporator_temp_C=80.0, noncondensable_pressure=500.0,
                feed_hx_effectiveness=0.85, insulation_ua=0.1,
                fan_volumetric_flow=0.012, temp_lift=10.0)
    base.update(kw)
    return DesignParameters(**base)


def test_controller_balances_a_surplus():
    p = _surplus_design()
    assert solve_evaporative(p).makeup_heat < 0        # there is a surplus
    tc = balance_temperature_by_economizer(p)
    assert tc.mode == "reject-surplus"
    assert tc.heat_to_reject_w > 0
    # It detunes the economizer below the design value and closes the balance.
    assert tc.balanced_effectiveness < tc.design_effectiveness
    assert 0.0 < tc.bypass_fraction < 1.0
    assert abs(tc.residual_makeup_w) < 1.0


def test_rejecting_heat_warms_the_clean_output():
    p = _surplus_design()
    r_full = solve_evaporative(p)
    tc = balance_temperature_by_economizer(p)
    # Detuning recovers less, so the feed arrives colder and the product leaves
    # warmer than at full recovery.
    assert tc.feed_preheat_temp_C < r_full.feed_preheat_temp_C
    assert tc.clean_output_temp_C > p.feed_temp_C


def test_no_bypass_when_no_surplus():
    # A leaky, low-recovery unit needs heat, not rejection: controller keeps
    # full recovery and flags heating.
    p = _surplus_design(insulation_ua=3.0, feed_hx_effectiveness=0.4)
    assert solve_evaporative(p).makeup_heat > 0
    tc = balance_temperature_by_economizer(p)
    assert tc.mode == "needs-heating"
    assert tc.bypass_fraction == 0.0
    assert tc.balanced_effectiveness == pytest.approx(p.feed_hx_effectiveness)


def test_more_fan_power_needs_more_bypass():
    low = balance_temperature_by_economizer(_surplus_design(temp_lift=5.0))
    high = balance_temperature_by_economizer(_surplus_design(temp_lift=14.0))
    assert high.heat_to_reject_w > low.heat_to_reject_w
    assert high.bypass_fraction >= low.bypass_fraction
