"""Tests for the falling-film hydrodynamics/heat-transfer model."""

import math

import pytest

from mvr import film


def test_properties_trend_with_temperature():
    assert film.liquid_viscosity(20.0) > film.liquid_viscosity(110.0)   # thins when hot
    assert film.surface_tension(20.0) > film.surface_tension(110.0)     # falls when hot
    assert film.liquid_viscosity(110.0) > 0 and film.surface_tension(110.0) > 0


def test_film_reynolds_definition():
    assert film.film_reynolds(0.008, 110.0) == pytest.approx(
        4.0 * 0.008 / film.liquid_viscosity(110.0))


def test_thickness_and_htc_move_oppositely_with_flow():
    thin = film.film_thickness(0.002, 110.0)
    thick = film.film_thickness(0.02, 110.0)
    assert thick > thin > 0
    # More flow -> thicker film -> lower coefficient.
    assert film.falling_film_htc(0.02, 110.0) < film.falling_film_htc(0.002, 110.0)


def test_falling_film_htc_is_high():
    # Thin films give high coefficients (comparable to the assumed wall films).
    h = film.falling_film_htc(0.006, 110.0)
    assert 3000.0 < h < 20000.0


def test_minimum_wetting_rate_scales_with_threshold():
    assert film.minimum_wetting_rate(110.0, re_min=30.0) == pytest.approx(
        30.0 * film.liquid_viscosity(110.0) / 4.0)
    assert film.minimum_wetting_rate(110.0, 50.0) > film.minimum_wetting_rate(110.0, 30.0)


def test_wetting_check_wet_and_dry_regimes():
    perim = math.pi * 0.22
    # Our feed (~5 g/s) wets the wall with margin.
    wet = film.wetting_check(perim, 0.005, 110.0)
    assert wet["wetted"] is True
    assert wet["recirculation_ratio"] <= 1.0
    assert wet["reynolds"] > 30.0
    # A trickle does not, and reports the recirculation needed.
    dry = film.wetting_check(perim, 0.0002, 110.0)
    assert dry["wetted"] is False
    assert dry["recirculation_ratio"] > 1.0
    assert dry["htc"] > 0
