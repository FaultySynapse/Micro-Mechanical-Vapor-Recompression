"""Tests for water/steam property correlations against known reference values."""

import math

import pytest

from mvr import properties as p


def test_sat_pressure_at_boiling():
    # Water boils at 101.325 kPa at 100 C.
    assert p.sat_pressure(100.0) == pytest.approx(101_325.0, rel=0.01)


def test_sat_pressure_room_temperature():
    # ~3.17 kPa at 25 C.
    assert p.sat_pressure(25.0) == pytest.approx(3_170.0, rel=0.03)


def test_sat_pressure_monotonic():
    temps = range(10, 150, 5)
    values = [p.sat_pressure(t) for t in temps]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_sat_pressure_smooth_across_blend():
    # Across the 95-105 C blend band the curve must stay smooth: the second
    # difference (curvature proxy) should be tiny compared with the first
    # difference (the genuine ~3600 Pa/K slope), i.e. no kink or jump.
    dt = 0.01
    for t0 in (95.0, 99.0, 100.0, 101.0, 105.0):
        f0 = p.sat_pressure(t0 - dt)
        f1 = p.sat_pressure(t0)
        f2 = p.sat_pressure(t0 + dt)
        first_diff = abs(f2 - f0) / 2.0          # ~slope * dt
        second_diff = abs(f2 - 2 * f1 + f0)       # curvature * dt^2
        assert second_diff < 0.01 * first_diff


def test_sat_temperature_roundtrip():
    for t in (40.0, 80.0, 100.0, 120.0):
        assert p.sat_temperature(p.sat_pressure(t)) == pytest.approx(t, abs=0.05)


def test_latent_heat_reference_points():
    assert p.latent_heat(100.0) == pytest.approx(2_256_500.0, rel=0.005)
    assert p.latent_heat(25.0) == pytest.approx(2_442_000.0, rel=0.02)
    assert p.latent_heat(0.0) == pytest.approx(2_501_000.0, rel=0.02)


def test_latent_heat_decreases_with_temperature():
    assert p.latent_heat(50.0) > p.latent_heat(100.0)


def test_vapor_density_saturated_at_100C():
    # Saturated steam at 100 C ~0.598 kg/m^3; ideal gas ~0.59.
    assert p.vapor_density(100.0) == pytest.approx(0.598, rel=0.03)


def test_liquid_cp_range():
    assert p.liquid_cp(25.0) == pytest.approx(4181.0, rel=0.01)
    assert p.liquid_cp(100.0) == pytest.approx(4217.0, rel=0.01)


def test_liquid_density_range():
    assert p.liquid_density(4.0) == pytest.approx(1000.0, rel=0.005)
    assert p.liquid_density(100.0) == pytest.approx(958.0, rel=0.01)


def test_clausius_clapeyron_matches_finite_difference():
    # Analytic dP/dT should match a numerical derivative of sat_pressure.
    t = 100.0
    dt = 0.01
    numeric = (p.sat_pressure(t + dt) - p.sat_pressure(t - dt)) / (2 * dt)
    analytic = p.clausius_clapeyron_dPdT(t)
    assert analytic == pytest.approx(numeric, rel=0.05)
