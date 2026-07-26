"""Tests for the concentric-cylinder geometry model."""

import math

import pytest

from mvr import DesignParameters
from mvr import geometry
from mvr.geometry import (
    ConcentricCylinder,
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


def test_apply_geometry_sets_all_areas():
    base = DesignParameters(evaporator_temp_C=110.0, temp_lift=10.0)
    c = ConcentricCylinder(inner_diameter_m=0.2, length_m=0.95, tube_wall_m=0.0012)
    p = apply_geometry(base, c)
    a = c.heat_transfer_area()
    assert p.hx_area == pytest.approx(a)
    assert p.evap_area == pytest.approx(a)
    assert p.condenser_area == pytest.approx(a)
    assert p.wall_thickness == pytest.approx(0.0012)
