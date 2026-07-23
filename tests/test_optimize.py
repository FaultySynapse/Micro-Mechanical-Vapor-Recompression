"""Tests for the optimization helpers."""

import pytest

from mvr import DesignParameters, solve
from mvr.optimize import (
    grid_search,
    minimize_scalar,
    specific_energy_objective,
)


def test_minimize_scalar_finds_interior_optimum():
    # With a production floor, there is an interior best lift: too small starves
    # production (penalty), too large wastes fan power.
    base = DesignParameters()
    obj = specific_energy_objective(min_distillate_lph=4.0)
    best_lift, best_val = minimize_scalar(base, "temp_lift", 0.5, 15.0, obj)
    assert 0.5 < best_lift < 15.0
    # The optimum should not be worse than the interval endpoints.
    assert best_val <= obj(DesignParameters(temp_lift=0.5)) + 1e-9
    assert best_val <= obj(DesignParameters(temp_lift=15.0)) + 1e-9


def test_grid_search_picks_best():
    base = DesignParameters()
    obj = specific_energy_objective(min_distillate_lph=0.0)
    grid = {
        "feed_hx_effectiveness": [0.6, 0.8, 0.95],
        "insulation_ua": [0.05, 0.5],
    }
    best, val = grid_search(base, grid, obj)
    # Lower energy comes from the best economizer and tightest shell.
    assert best.feed_hx_effectiveness == 0.95
    assert best.insulation_ua == 0.05


def test_objective_penalizes_low_production():
    obj = specific_energy_objective(min_distillate_lph=100.0)
    # A tiny unit cannot reach 100 L/h -> objective carries a penalty.
    penalized = obj(DesignParameters(hx_area=0.01))
    plain = solve(DesignParameters(hx_area=0.01)).specific_energy_kwh_per_l
    assert penalized > plain
