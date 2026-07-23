"""Tests for the coupled evaporative (mass-transfer) model."""

import math

import pytest

from mvr import DesignParameters, solve
from mvr.masstransfer import (
    EvaporativeResults,
    film_mass_flow,
    mass_transfer_coeff_from_htc,
    solve_evaporative,
)
from mvr import properties as props


def _lowtemp(**overrides) -> DesignParameters:
    base = dict(evaporator_temp_C=50.0, temp_lift=6.0, noncondensable_pressure=1000.0)
    base.update(overrides)
    return DesignParameters(**base)


def _fixed_coeff(**overrides) -> DesignParameters:
    """Low-temp params in fixed-coefficient mode (transfer_from_flow off)."""
    base = dict(transfer_from_flow=False)
    base.update(overrides)
    return _lowtemp(**base)


def test_runs_and_is_sane():
    r = solve_evaporative(_lowtemp())
    assert isinstance(r, EvaporativeResults)
    assert r.distillate_rate > 0
    assert r.cond_surface_temp_C > r.evap_temp_C          # wall must run "uphill"
    assert 0.0 < r.mass_transfer_effectiveness <= 1.0
    assert r.limiting_mechanism in ("evaporation", "condensation", "wall-heat")


def test_film_flow_zero_and_infinite_limits():
    # No driving force -> zero flow.
    assert film_mass_flow(0.02, 0.3, 323.0, 15_000.0, 5_000.0, 5_000.0) == 0.0
    # Source partial pressure reaches total (no NCG to block) -> unbounded.
    assert film_mass_flow(0.02, 0.3, 323.0, 15_000.0, 15_000.0, 5_000.0) == math.inf
    # Normal case is positive and finite.
    q = film_mass_flow(0.02, 0.3, 323.0, 15_000.0, 8_000.0, 5_000.0)
    assert 0.0 < q < math.inf


def test_recovers_heat_limit_when_ncg_zero_and_mt_fast():
    # With negligible NCG, fast mass transfer, and the gas-phase sensible load
    # switched off (tiny condenser_gas_htc), the coupled model must collapse onto
    # the heat-transfer-limited boiling result.
    p = _fixed_coeff(noncondensable_pressure=1e-3,
                     evap_mass_transfer_coeff=5.0, condenser_mass_transfer_coeff=5.0,
                     condenser_gas_htc=1e-9)
    r_evap = solve_evaporative(p)
    r_boil = solve(p)
    assert r_evap.mass_transfer_effectiveness > 0.98
    assert r_evap.distillate_rate == pytest.approx(r_boil.distillate_rate, rel=0.02)
    assert r_evap.limiting_mechanism == "wall-heat"


def test_compression_superheat_is_positive_and_grows_with_ratio():
    # Higher lift -> larger pressure ratio -> more compression superheat.
    low = solve_evaporative(_lowtemp(temp_lift=2.0))
    high = solve_evaporative(_lowtemp(temp_lift=10.0))
    assert low.condenser_superheat_K > 0
    assert high.condenser_superheat_K > low.condenser_superheat_K
    assert high.discharge_temp_C > high.evap_temp_C


def test_lower_fan_efficiency_raises_superheat():
    # A less efficient fan reheats the vapor more.
    eff = solve_evaporative(_lowtemp(fan_isentropic_efficiency=0.9))
    ineff = solve_evaporative(_lowtemp(fan_isentropic_efficiency=0.4))
    assert ineff.condenser_superheat_K > eff.condenser_superheat_K


def test_gas_sensible_load_reduces_production():
    # Modeling gas-phase sensible heat (finite gas HTC) must not *increase*
    # production versus ignoring it (negligible gas HTC): the shed superheat is
    # an extra load on the wall.
    with_gas = solve_evaporative(_lowtemp(condenser_gas_htc=200.0))
    without_gas = solve_evaporative(_lowtemp(condenser_gas_htc=1e-9))
    assert with_gas.distillate_rate <= without_gas.distillate_rate
    assert with_gas.sensible_duty > without_gas.sensible_duty


def test_sensible_duty_is_minor_fraction_of_wall_duty():
    # For this device latent heat dominates: the sensible (desuperheat) load
    # should be a small fraction of the total condenser duty.
    r = solve_evaporative(_lowtemp())
    assert 0.0 <= r.sensible_fraction < 0.15


def test_more_ncg_reduces_production():
    low = solve_evaporative(_lowtemp(noncondensable_pressure=200.0))
    high = solve_evaporative(_lowtemp(noncondensable_pressure=4000.0))
    assert high.distillate_rate < low.distillate_rate
    assert high.limiting_mechanism == "condensation"


def test_faster_mass_transfer_increases_production():
    slow = solve_evaporative(_fixed_coeff(evap_mass_transfer_coeff=0.005,
                                          condenser_mass_transfer_coeff=0.005))
    fast = solve_evaporative(_fixed_coeff(evap_mass_transfer_coeff=0.2,
                                          condenser_mass_transfer_coeff=0.2))
    assert fast.distillate_rate > slow.distillate_rate
    assert fast.mass_transfer_effectiveness > slow.mass_transfer_effectiveness


def test_larger_surfaces_increase_production_when_mt_limited():
    small = solve_evaporative(_fixed_coeff(evap_area=0.1, condenser_area=0.1))
    large = solve_evaporative(_fixed_coeff(evap_area=1.0, condenser_area=1.0))
    assert large.distillate_rate > small.distillate_rate


def test_production_never_exceeds_heat_limit():
    for ncg in (10.0, 500.0, 3000.0):
        for hm in (0.01, 0.1, 1.0):
            r = solve_evaporative(_fixed_coeff(noncondensable_pressure=ncg,
                                               evap_mass_transfer_coeff=hm,
                                               condenser_mass_transfer_coeff=hm))
            assert r.distillate_rate <= r.heat_limited_rate * 1.02


def test_lift_budget_partitions_nominal_lift():
    r = solve_evaporative(_lowtemp())
    budget = (r.lift_ncg_penalty + r.lift_evaporation
              + r.lift_condensation + r.lift_useful_wall)
    assert budget == pytest.approx(r.lift_nominal, abs=0.05)
    assert all(x >= -1e-6 for x in
               (r.lift_ncg_penalty, r.lift_evaporation,
                r.lift_condensation, r.lift_useful_wall))


def test_energy_balance_closes():
    p = _lowtemp()
    r = solve_evaporative(p)
    shaft = r.fan_power * p.fan_motor_efficiency
    sinks = r.insulation_loss + r.unrecovered_stream_loss + r.feed_heating_duty
    assert (shaft + r.makeup_heat) == pytest.approx(sinks, rel=1e-9)


def test_mass_balance():
    p = _lowtemp()
    r = solve_evaporative(p)
    assert r.feed_rate == pytest.approx(r.distillate_rate + r.concentrate_rate, rel=1e-9)


def test_chilton_colburn_helper_reasonable():
    # A ~30 W/m^2K convective coefficient should map to an O(0.01-0.05 m/s) h_m.
    h_m = mass_transfer_coeff_from_htc(30.0)
    assert 0.005 < h_m < 0.1


def test_condensation_surface_pressure_consistency():
    # The condensation surface saturation pressure must not exceed the condenser
    # total pressure (otherwise it could not condense).
    r = solve_evaporative(_lowtemp())
    assert props.sat_pressure(r.cond_surface_temp_C) <= r.cond_total_pressure_pa * 1.001


# --- Fan-driven transport (transfer coefficients from the fan flow) ----------

def test_flow_mode_computes_coefficients_and_velocities():
    r = solve_evaporative(_lowtemp())            # transfer_from_flow defaults True
    assert r.evap_velocity > 0 and r.cond_velocity > 0
    assert r.evap_reynolds > 0 and r.cond_reynolds > 0
    assert r.evap_htc_mass > 0 and r.cond_htc_mass > 0
    # The fan sweeps more gas than the net vapor it condenses.
    assert r.circulation_ratio >= 1.0


def test_more_fan_flow_increases_production_and_transfer():
    low = solve_evaporative(_lowtemp(fan_volumetric_flow=0.004))
    high = solve_evaporative(_lowtemp(fan_volumetric_flow=0.040))
    assert high.evap_velocity > low.evap_velocity
    assert high.evap_htc_mass > low.evap_htc_mass       # faster sweep, thinner film
    assert high.distillate_rate > low.distillate_rate


def test_more_fan_flow_costs_more_power_per_liter():
    # In this regime fan power grows ~linearly with flow but production only
    # ~sqrt, so pushing more flow worsens specific energy: the fan-sizing tension.
    low = solve_evaporative(_lowtemp(fan_volumetric_flow=0.004))
    high = solve_evaporative(_lowtemp(fan_volumetric_flow=0.040))
    assert high.fan_power > low.fan_power
    assert high.specific_energy_kwh_per_l > low.specific_energy_kwh_per_l


def test_flow_coefficient_scaling_is_sqrt_like():
    # Laminar flat plate: h_m ~ velocity^0.5.  A 4x flow should raise h_m ~2x.
    r1 = solve_evaporative(_lowtemp(fan_volumetric_flow=0.005))
    r2 = solve_evaporative(_lowtemp(fan_volumetric_flow=0.020))
    ratio = r2.evap_htc_mass / r1.evap_htc_mass
    assert 1.7 < ratio < 2.3


def test_lower_pressure_raises_diffusivity_and_transfer():
    # Same fan flow, lower evaporator temperature -> lower pressure -> higher
    # diffusivity and (lower density ->) higher sweep velocity, so a larger h_m.
    warm = solve_evaporative(_lowtemp(evaporator_temp_C=70.0))
    cool = solve_evaporative(_lowtemp(evaporator_temp_C=40.0))
    assert cool.evap_htc_mass > warm.evap_htc_mass


def test_flow_mode_capped_by_deliverable_vapor():
    # A tiny fan flow cannot deliver much vapor; production stays bounded by the
    # saturated vapor mass the fan can carry.
    fan_flow = 1e-4
    r = solve_evaporative(_lowtemp(fan_volumetric_flow=fan_flow))
    deliverable = props.vapor_density(50.0, props.sat_pressure(50.0)) * fan_flow
    assert r.distillate_rate <= deliverable * 1.001


def test_transport_flat_plate_coefficients_positive():
    from mvr import transport
    h_m, h_g, diag = transport.surface_transfer_coefficients(
        velocity=1.0, length=0.5, temp_C=50.0, pressure_pa=13_000.0,
        density=0.09, gas_cp=1900.0)
    assert h_m > 0 and h_g > 0
    assert diag["reynolds"] > 0
    assert diag["regime"] in ("laminar", "turbulent")
    # Zero velocity -> no transfer.
    h_m0, h_g0, _ = transport.surface_transfer_coefficients(
        0.0, 0.5, 50.0, 13_000.0, 0.09, 1900.0)
    assert h_m0 == 0.0 and h_g0 == 0.0
