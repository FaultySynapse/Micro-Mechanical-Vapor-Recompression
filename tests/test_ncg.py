"""Tests for the non-condensable-gas load and purge model."""

import math

import pytest

from mvr import ncg


def test_release_rate_scales_with_feed_and_gas():
    r1 = ncg.ncg_release_rate(0.004, ncg.AIR_SATURATED_MOL_L)
    r2 = ncg.ncg_release_rate(0.008, ncg.AIR_SATURATED_MOL_L)
    r3 = ncg.ncg_release_rate(0.004, 2 * ncg.AIR_SATURATED_MOL_L)
    assert r2 == pytest.approx(2 * r1)
    assert r3 == pytest.approx(2 * r1)


def test_air_saturated_load_is_sub_millimolar():
    # Air-saturated water carries a bit under 1 mmol/L of dissolved gas.
    assert 0.0005 < ncg.AIR_SATURATED_MOL_L < 0.0015


def test_purge_water_loss_rises_as_ncg_falls():
    rel = ncg.ncg_release_rate(0.004)
    p_water = 40_000.0
    hi = ncg.purge_cost(rel, p_ncg=1000.0, p_water=p_water)
    lo = ncg.purge_cost(rel, p_ncg=100.0, p_water=p_water)
    # A leaner (lower) NCG partial pressure loses more water per mole removed.
    assert lo["water_lost_kg_s"] > hi["water_lost_kg_s"]
    # Limit check: water lost per NCG mole ~ p_water / p_ncg.
    assert lo["water_lost_mol_s"] == pytest.approx(rel * p_water / 100.0, rel=0.02)


def test_purge_removes_ncg_at_release_rate():
    rel = ncg.ncg_release_rate(0.004)
    cost = ncg.purge_cost(rel, p_ncg=500.0, p_water=40_000.0)
    ncg_removed = cost["purge_total_mol_s"] - cost["water_lost_mol_s"]
    assert ncg_removed == pytest.approx(rel, rel=1e-9)


def test_pump_power_only_under_vacuum():
    rel = ncg.ncg_release_rate(0.004)
    none = ncg.purge_cost(rel, 500.0, 40_000.0)  # no operating pressure given
    vac = ncg.purge_cost(rel, 500.0, 40_000.0, operating_pressure_pa=15_000.0)
    assert none["pump_power_w"] == 0.0
    assert vac["pump_power_w"] > 0.0


def test_zero_ncg_pressure_is_infinite_cost():
    rel = ncg.ncg_release_rate(0.004)
    cost = ncg.purge_cost(rel, p_ncg=0.0, p_water=40_000.0)
    assert math.isinf(cost["water_lost_kg_s"])


def test_feed_volumetric_conversion():
    # 1 kg/s of ~room-temp water is ~1.0 L/s.
    assert ncg.feed_volumetric_l_s(1.0, 20.0) == pytest.approx(1.0, rel=0.01)
