"""Tests for the standalone hull-length cost optimizer."""

from dataclasses import replace

import pytest

from mvr.hull_cost import HullCostModel, optimize_length, tube_cost_breakeven


def test_tube_count_decreases_with_length():
    m = HullCostModel()
    assert m.tube_count(0.20) > m.tube_count(0.40)


def test_cost_breakdown_sums_to_total():
    r = HullCostModel().evaluate(0.24)
    assert r["total_cost"] == pytest.approx(
        r["tube_cost"] + r["insulation_cost"] + r["hull_cost"] + r["fixed_cost"])


def test_regime_boundary_at_stock_height():
    m = HullCostModel()
    short = m.evaluate(m.stock_max_height_m - m.overhead_m - 0.01)
    tall = m.evaluate(m.stock_max_height_m - m.overhead_m + 0.01)
    assert short["regime"] == "stock"
    assert tall["regime"] == "custom"
    assert tall["hull_cost"] > short["hull_cost"]


def test_default_optimum_is_stock():
    # With cheap tubes, staying in the pot wins.
    assert optimize_length(HullCostModel())["best"]["regime"] == "stock"


def test_expensive_tubes_flip_to_custom():
    # Costly sealed joints make the taller, fewer-tube custom hull worth it.
    res = optimize_length(replace(HullCostModel(), cost_per_tube=60.0))
    assert res["best"]["regime"] == "custom"


def test_breakeven_brackets_the_decision():
    m = HullCostModel()
    be = tube_cost_breakeven(m)
    assert m.cost_per_tube < be                                    # currently stock
    assert optimize_length(replace(m, cost_per_tube=be + 5))["best"]["regime"] == "custom"
    assert optimize_length(replace(m, cost_per_tube=be - 5))["best"]["regime"] == "stock"


def test_within_stock_longer_is_cheaper():
    # In the stock regime the hull is fixed, so fewer tubes (longer) always wins.
    m = HullCostModel()
    assert m.evaluate(0.24)["total_cost"] < m.evaluate(0.16)["total_cost"]
