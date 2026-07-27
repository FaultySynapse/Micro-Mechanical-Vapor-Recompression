"""Tests for the concentric-cylinder geometry model."""

import math

import pytest

from mvr import DesignParameters
from mvr import geometry
from mvr.geometry import (
    ConcentricCylinder,
    TubeBundle,
    size_tube_count,
    apply_bundle,
    hoop_thickness,
    size_length_for_area,
    apply_geometry,
)


def test_heat_transfer_area_is_tube_lateral_area():
    c = ConcentricCylinder(inner_diameter_m=0.2, length_m=0.95)
    assert c.heat_transfer_area() == pytest.approx(math.pi * 0.2 * 0.95)


def test_size_length_for_area_is_inverse():
    L = size_length_for_area(0.25, 0.60)
    c = ConcentricCylinder(inner_diameter_m=0.25, length_m=L)
    assert c.heat_transfer_area() == pytest.approx(0.60)


def test_envelope_exceeds_chambers_and_fits_budget():
    c = ConcentricCylinder(inner_diameter_m=0.25, length_m=0.76)
    assert c.envelope_volume() > c.bore_volume() + c.annulus_volume()
    assert c.envelope_volume() < 0.5           # both chambers under half a m^3


def test_holdup_is_small_without_storage():
    c = ConcentricCylinder(inner_diameter_m=0.25, length_m=0.76, sump_depth_m=0.01)
    assert c.liquid_holdup_l() < 5.0           # falling film + thin sump only


def test_pressure_walls_are_sub_fab_minimum_at_2bar():
    # 1 bar gauge on a ~0.3 m shell needs far less than the fab minimum wall.
    t = hoop_thickness(1.0e5, 0.15)
    assert t < geometry.MIN_FAB_WALL_M
    # Higher pressure or bigger radius needs more metal (monotone).
    assert hoop_thickness(2.0e5, 0.15) > t
    assert hoop_thickness(1.0e5, 0.30) > t


def test_size_tube_count_reaches_target_area():
    n = size_tube_count(0.6, 0.032, 0.22)
    b = TubeBundle(tube_inner_diameter_m=0.032, tube_count=n, active_length_m=0.22)
    assert b.heat_transfer_area() >= 0.6
    # One fewer tube would fall short.
    short = TubeBundle(tube_inner_diameter_m=0.032, tube_count=n - 1, active_length_m=0.22)
    assert short.heat_transfer_area() < 0.6


def test_bundle_fit_and_perimeter():
    b = TubeBundle(tube_inner_diameter_m=0.032, tube_count=28, active_length_m=0.22)
    assert b.wetted_perimeter() == pytest.approx(28 * math.pi * 0.032)
    # A 28-tube 32 mm bundle fits a 26 cm pot but not a 20 cm one.
    assert b.fits_in_shell(0.26) is True
    assert b.fits_in_shell(0.20) is False
    assert 0.0 < b.metal_fraction(0.26) < 1.0


def test_more_or_bigger_tubes_grow_the_envelope():
    small = TubeBundle(tube_inner_diameter_m=0.025, tube_count=20, active_length_m=0.22)
    more = TubeBundle(tube_inner_diameter_m=0.025, tube_count=35, active_length_m=0.22)
    bigger = TubeBundle(tube_inner_diameter_m=0.050, tube_count=20, active_length_m=0.22)
    assert more.bundle_envelope_diameter() > small.bundle_envelope_diameter()
    assert bigger.bundle_envelope_diameter() > small.bundle_envelope_diameter()


def test_apply_bundle_sets_areas():
    base = DesignParameters(evaporator_temp_C=110.0, temp_lift=10.0)
    b = TubeBundle(tube_inner_diameter_m=0.032, tube_count=28, active_length_m=0.22)
    p = apply_bundle(base, b)
    assert p.hx_area == pytest.approx(b.heat_transfer_area())
    assert p.evap_area == pytest.approx(b.heat_transfer_area())


def test_apply_geometry_sets_all_areas():
    base = DesignParameters(evaporator_temp_C=110.0, temp_lift=10.0)
    c = ConcentricCylinder(inner_diameter_m=0.2, length_m=0.95, tube_wall_m=0.0012)
    p = apply_geometry(base, c)
    a = c.heat_transfer_area()
    assert p.hx_area == pytest.approx(a)
    assert p.evap_area == pytest.approx(a)
    assert p.condenser_area == pytest.approx(a)
    assert p.wall_thickness == pytest.approx(0.0012)
