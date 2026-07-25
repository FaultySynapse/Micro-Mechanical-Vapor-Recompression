"""Tests for the metal-wool / mesh packing model and the lateral conduction link."""

import math

import pytest

from mvr import DesignParameters, solve, overall_U, link_cross_section_for_duty
from mvr.masstransfer import solve_evaporative
from mvr import packing
from mvr.packing import PackingSpec, apply_packing


def _spec(**kw):
    base = dict(fiber_diameter_m=60e-6, void_fraction=0.97, conductivity=385.0)
    base.update(kw)
    return PackingSpec(**base)


def test_specific_area_matches_cylinder_formula():
    s = _spec(fiber_diameter_m=50e-6, void_fraction=0.98)
    assert s.specific_area() == pytest.approx(4 * (1 - 0.98) / 50e-6, rel=1e-9)
    # Finer fiber -> more area per volume.
    assert _spec(fiber_diameter_m=30e-6).specific_area() > _spec(fiber_diameter_m=90e-6).specific_area()


def test_interfacial_area_scales_with_volume_and_dwarfs_a_plate():
    s = _spec()
    assert s.interfacial_area(0.006) == pytest.approx(2 * s.interfacial_area(0.003))
    # A few liters of fine wool presents many m^2 -- far more than a 0.6 m^2 plate.
    assert s.interfacial_area(0.003) > 1.0


def test_fiber_mass_transfer_coeff_positive_and_grows_with_velocity():
    s = _spec()
    lo, _ = packing.fiber_mass_transfer_coeff(s, 0.2, 90.0, 90_000.0, 0.5)
    hi, _ = packing.fiber_mass_transfer_coeff(s, 1.0, 90.0, 90_000.0, 0.5)
    assert 0.0 < lo < hi


def test_fin_efficiency_bounded_and_falls_with_thickness():
    thin = packing.fin_efficiency(_spec(mat_thickness_m=0.002), 5000.0)
    thick = packing.fin_efficiency(_spec(mat_thickness_m=0.03), 5000.0)
    assert 0.0 < thick < thin <= 1.0


def test_wall_htc_enhancement_at_least_unity():
    s = _spec()
    mult = packing.wall_htc_enhancement(s, 5000.0, 0.3, 0.003)
    assert mult >= 1.0


def test_pressure_drop_grows_with_velocity_and_finer_fiber():
    coarse = packing.packing_pressure_drop(_spec(fiber_diameter_m=120e-6), 0.7, 0.15, 90.0, 0.5)
    fine = packing.packing_pressure_drop(_spec(fiber_diameter_m=30e-6), 0.7, 0.15, 90.0, 0.5)
    slow = packing.packing_pressure_drop(_spec(), 0.2, 0.15, 90.0, 0.5)
    fast = packing.packing_pressure_drop(_spec(), 1.4, 0.15, 90.0, 0.5)
    assert fine > coarse
    assert fast > slow


def test_apply_packing_gives_large_area_fixed_coeff_and_solves():
    base = DesignParameters(evaporator_temp_C=90.0, temp_lift=8.0, hx_area=0.6,
                            evap_area=0.6, condenser_area=0.6)
    p = apply_packing(base, _spec(), packed_volume_m3=0.003, superficial_velocity=0.6)
    assert p.transfer_from_flow is False
    assert p.evap_area > 1.0 and p.condenser_area > 1.0
    assert p.condensing_htc >= base.condensing_htc      # fin enhancement
    r = solve_evaporative(p)
    assert r.distillate_rate > 0


# --- Lateral conduction link -------------------------------------------------


def test_link_cross_section_formula():
    assert link_cross_section_for_duty(9000.0, 0.05, 385.0, 2.0) == pytest.approx(
        9000.0 * 0.05 / (385.0 * 2.0))


def test_link_lowers_U_and_material_matters():
    base = DesignParameters(evaporator_temp_C=90.0, temp_lift=8.0, hx_area=0.6)
    u_plate = overall_U(base)
    # A sparse mesh over a lateral gap adds a dominant resistance.
    copper = base.__class__(**{**base.__dict__, "conduction_link_length_m": 0.02,
                               "conduction_link_area_ratio": 0.1,
                               "conduction_link_conductivity": 385.0})
    alum = base.__class__(**{**base.__dict__, "conduction_link_length_m": 0.02,
                             "conduction_link_area_ratio": 0.1,
                             "conduction_link_conductivity": 205.0})
    assert overall_U(copper) < u_plate
    # Conduction-limited: unlike the plate, material now matters a lot.
    assert overall_U(copper) > overall_U(alum)


def test_short_link_barely_hurts_but_long_link_does():
    base = DesignParameters(evaporator_temp_C=90.0, temp_lift=8.0, hx_area=0.6,
                            evap_area=0.6, condenser_area=0.6)
    r_plate = solve_evaporative(base)
    short = solve_evaporative(base.__class__(**{**base.__dict__,
        "conduction_link_length_m": 0.005, "conduction_link_area_ratio": 0.1,
        "conduction_link_conductivity": 385.0}))
    long = solve_evaporative(base.__class__(**{**base.__dict__,
        "conduction_link_length_m": 0.02, "conduction_link_area_ratio": 0.1,
        "conduction_link_conductivity": 385.0}))
    assert short.distillate_lph > long.distillate_lph
    assert short.distillate_lph > 0.9 * r_plate.distillate_lph      # 5mm copper mesh ~ok
