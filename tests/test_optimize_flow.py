"""Tests for the power-constrained flow-maximization optimizer.

Kept fast by using a single material and a small variable set.
"""

from dataclasses import replace

import pytest

from mvr import DesignParameters
from mvr.masstransfer import solve_evaporative
from mvr.optimize import (
    fan_flow_for_power_budget,
    maximize_flow,
    _pattern_search,
)

_STAINLESS = {"stainless_steel": {"wall_conductivity": 16.0, "wall_thickness": 0.0008}}
# Lift is derived from the fan curve + power; the free flow/lift lever is the
# operating flow, so the small search is over area and flow.
_SMALL_BOUNDS = {"plate_area": (0.1, 0.5), "fan_volumetric_flow": (0.004, 0.03)}


def _base(**kw) -> DesignParameters:
    d = dict(noncondensable_pressure=500.0, feed_hx_effectiveness=0.85)
    d.update(kw)
    return DesignParameters(**d)


def test_fan_flow_hits_the_budget():
    base = _base(evaporator_temp_C=55.0)
    vf = fan_flow_for_power_budget(base, 400.0)
    r = solve_evaporative(replace(base, fan_volumetric_flow=vf, transfer_from_flow=True))
    assert r.total_energy_input == pytest.approx(400.0, rel=0.02)


def test_fan_flow_monotonic_in_budget():
    base = _base(evaporator_temp_C=55.0)
    lo = fan_flow_for_power_budget(base, 200.0)
    hi = fan_flow_for_power_budget(base, 600.0)
    assert hi > lo


def test_pattern_search_maximizes_concave_quadratic():
    # Maximum of -(x-3)^2 - (y+1)^2 is at (3, -1).
    def score(v):
        return -((v[0] - 3.0) ** 2) - ((v[1] + 1.0) ** 2)
    x, fx = _pattern_search(score, [0.0, 0.0], [(-10.0, 10.0), (-10.0, 10.0)])
    assert x[0] == pytest.approx(3.0, abs=0.05)
    assert x[1] == pytest.approx(-1.0, abs=0.05)


def test_maximize_flow_respects_budget_and_produces():
    base = _base()
    params, r, info = maximize_flow(base, budget_w=600.0,
                                    bounds=_SMALL_BOUNDS, materials=_STAINLESS)
    assert r.distillate_rate > 0
    # Total power (blower + auxiliary heat) is spent to the budget by the coupling.
    assert info["total_power_w"] == pytest.approx(600.0, rel=0.02)
    assert info["material"] == "stainless_steel"
    # Lift is a derived output of the fan curve + power, not a free variable.
    assert info["lift_K"] > 0
    # The optimizer should exploit the whole size budget (bigger plate helps).
    assert params.hx_area == pytest.approx(0.5, abs=1e-3)


def test_more_power_budget_yields_more_flow():
    base = _base()
    _, r_low, _ = maximize_flow(base, budget_w=300.0,
                                bounds=_SMALL_BOUNDS, materials=_STAINLESS)
    _, r_high, _ = maximize_flow(base, budget_w=900.0,
                                 bounds=_SMALL_BOUNDS, materials=_STAINLESS)
    assert r_high.distillate_lph > r_low.distillate_lph


def test_optimum_beats_a_naive_baseline():
    from mvr.optimize import solve_at_budget
    base = _base()
    params, r_opt, _ = maximize_flow(base, budget_w=600.0,
                                     bounds=_SMALL_BOUNDS, materials=_STAINLESS)
    # A naive mid-box design solved the same way (fan curve + power) should not
    # beat the optimum.
    naive = replace(base, wall_conductivity=16.0, wall_thickness=0.0008,
                    hx_area=0.3, evap_area=0.3, condenser_area=0.3,
                    fan_volumetric_flow=0.010, transfer_from_flow=True)
    r_naive, _, _, _ = solve_at_budget(naive, 600.0)
    assert r_opt.distillate_lph >= r_naive.distillate_lph - 1e-6
