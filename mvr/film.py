"""Falling-film hydrodynamics and heat transfer for the wetted-wall design.

In the no-storage layout each phase-change wall is wetted by a thin liquid film
running down it under gravity, rather than a pool:

  * evaporator face -- feed (usually recirculated to stay wet) falls as a film and
    partially evaporates;
  * condenser face  -- vapor condenses and the condensate film drains down.

For a film of mass flow per unit wetted width ``Gamma`` (kg/m/s):

    film Reynolds      Re = 4 Gamma / mu
    laminar thickness  delta = (3 mu Gamma / (rho^2 g))^(1/3)
    film coefficient   h ~ k_liquid / delta            (Nusselt laminar; ~conduction
                                                         across a thin film)

Falling films give high coefficients (thin films), so they keep ``U`` high -- but
only while the wall stays *wetted*.  Below a critical wetting rate the film breaks
into rivulets and bares part of the wall, throwing away area.  The threshold is
usually written as a minimum film Reynolds number ``Re_min`` (~25-50 for water on
metal); greywater's surfactants lower the surface tension and help wetting, so the
threshold is, if anything, easier to meet than for clean water.

Pure standard library (+ the package's ``properties``).
"""

from __future__ import annotations

import math

from . import properties as props

G = 9.80665


def liquid_viscosity(temp_C: float) -> float:
    """Dynamic viscosity of liquid water, Pa.s (Vogel fit, ~20-130 C)."""
    t_K = props.to_kelvin(temp_C)
    return 2.414e-5 * 10.0 ** (247.8 / (t_K - 140.0))


def surface_tension(temp_C: float) -> float:
    """Surface tension of water against its vapor, N/m (IAPWS form)."""
    tau = 1.0 - props.to_kelvin(temp_C) / 647.096
    return 0.2358 * tau ** 1.256 * (1.0 - 0.625 * tau)


def liquid_conductivity(temp_C: float) -> float:
    """Thermal conductivity of liquid water, W/(m K) (mild peak ~130 C)."""
    return 0.5706 + 1.756e-3 * temp_C - 6.46e-6 * temp_C ** 2


def film_reynolds(gamma: float, temp_C: float) -> float:
    """Film Reynolds number ``Re = 4 Gamma / mu``."""
    return 4.0 * gamma / liquid_viscosity(temp_C)


def film_thickness(gamma: float, temp_C: float) -> float:
    """Laminar falling-film thickness (m): ``(3 mu Gamma / (rho^2 g))^(1/3)``."""
    mu = liquid_viscosity(temp_C)
    rho = props.liquid_density(temp_C)
    return (3.0 * mu * gamma / (rho ** 2 * G)) ** (1.0 / 3.0)


def falling_film_htc(gamma: float, temp_C: float) -> float:
    """Falling-film heat-transfer coefficient, W/(m^2 K).

    Laminar Nusselt ``h = k / delta`` with a wavy-laminar enhancement factor
    (~1.1) that mildly raises it; adequate for both the evaporating and the
    condensing film at these low Reynolds numbers.
    """
    delta = film_thickness(gamma, temp_C)
    if delta <= 0.0:
        return 0.0
    return 1.1 * liquid_conductivity(temp_C) / delta


def minimum_wetting_rate(temp_C: float, re_min: float = 30.0) -> float:
    """Minimum wetting rate ``Gamma_min = re_min mu / 4`` (kg/m/s).

    Below this the film breaks into rivulets and bares the wall.  ``re_min`` is a
    wettability-dependent threshold (~25-50 for water on metal; lower with
    surfactant-laden greywater).
    """
    return re_min * liquid_viscosity(temp_C) / 4.0


def wetting_check(perimeter_m: float, film_rate_kg_s: float, temp_C: float,
                  re_min: float = 30.0) -> dict:
    """Is a wall of wetted width ``perimeter_m`` kept wet by ``film_rate_kg_s``?

    Returns the achieved ``Gamma``, its Reynolds number, the threshold, whether
    it is wetted, and the recirculation ratio (film flow / supplied flow) needed
    to reach the threshold if it is not.
    """
    gamma = film_rate_kg_s / perimeter_m if perimeter_m > 0 else 0.0
    gamma_min = minimum_wetting_rate(temp_C, re_min)
    needed_flow = gamma_min * perimeter_m
    return {
        "gamma": gamma,
        "reynolds": film_reynolds(gamma, temp_C),
        "gamma_min": gamma_min,
        "wetted": gamma >= gamma_min,
        "recirculation_ratio": needed_flow / film_rate_kg_s if film_rate_kg_s > 0 else math.inf,
        "htc": falling_film_htc(max(gamma, gamma_min), temp_C),
    }
