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


# --- Integrated bleed valve --------------------------------------------------

def _bleed(**kw):
    args = dict(feed_rate_kg_s=0.004, feed_temp_C=20.0,
                distillate_rate_kg_s=0.0032, evap_temp_C=80.0,
                p_ncg=500.0, cond_bulk_pv_pa=68_000.0, cond_total_pa=72_000.0,
                makeup_heat_w=-100.0)
    args.update(kw)
    return ncg.bleed_balance(**args)


def test_bleed_vents_ncg_at_release_rate():
    b = _bleed()
    # The bleed carries the NCG out at the release rate, plus its steam.
    assert b.bleed_total_mol_s == pytest.approx(b.ncg_release_mol_s + b.steam_in_bleed_mol_s)
    assert b.steam_in_bleed_mol_s > b.ncg_release_mol_s     # bleed is mostly steam


def test_feed_condenser_recovers_most_steam_and_cuts_pump():
    direct = _bleed(to_feed_condenser=False)
    recov = _bleed(to_feed_condenser=True)
    assert recov.steam_recovered_mol_s > 0.8 * recov.steam_in_bleed_mol_s
    assert recov.water_lost_kg_s < direct.water_lost_kg_s
    assert recov.pump_power_w < direct.pump_power_w
    assert recov.heat_recovered_w > 0
    assert recov.net_distillate_lph > direct.net_distillate_lph


def test_recovery_matters_more_for_gassy_feed():
    # CO2-rich feed loses much more without recovery; recovery closes most of it.
    direct_air = _bleed(dissolved_gas_mol_l=ncg.AIR_SATURATED_MOL_L, to_feed_condenser=False)
    direct_co2 = _bleed(dissolved_gas_mol_l=8 * ncg.AIR_SATURATED_MOL_L, to_feed_condenser=False)
    recov_co2 = _bleed(dissolved_gas_mol_l=8 * ncg.AIR_SATURATED_MOL_L, to_feed_condenser=True)
    assert direct_co2.water_lost_kg_s > 5 * direct_air.water_lost_kg_s
    assert recov_co2.net_distillate_lph > direct_co2.net_distillate_lph


def test_self_vents_above_ambient():
    below = _bleed(cond_total_pa=72_000.0)
    above = _bleed(cond_total_pa=110_000.0)
    assert below.self_vents is False and below.pump_power_w > 0
    assert above.self_vents is True and above.pump_power_w == 0.0


def test_solve_with_bleed_wrapper():
    from mvr import DesignParameters
    from mvr.masstransfer import solve_with_bleed
    p = DesignParameters(evaporator_temp_C=80.0, noncondensable_pressure=500.0,
                         bleed_to_feed_condenser=True)
    r, b = solve_with_bleed(p)
    assert 0 < b.net_distillate_lph <= b.gross_distillate_lph
    assert b.net_distillate_lph == pytest.approx(b.gross_distillate_lph, rel=0.02)
