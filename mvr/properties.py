"""Thermophysical properties of water and steam.

All correlations here are pure-Python (no third-party dependencies) so the core
model runs in any environment.  They are engineering correlations chosen for a
good balance of accuracy and simplicity over the temperature range relevant to a
small mechanical-vapor-recompression (MVR) still: roughly 20 C to 150 C.

Accuracy notes (vs. IAPWS / steam tables):
  * Saturation pressure (Antoine): within ~0.2 % over 1-100 C, ~1 % to 150 C.
  * Latent heat (Watson):          within ~1.5 % over 0-150 C.
  * Vapor density (ideal gas):     within ~2 % near atmospheric saturation.

Units convention throughout this package: SI unless a name says otherwise.
Temperatures are in degrees Celsius when a symbol ends in ``_C`` and in Kelvin
when it ends in ``_K``.  Pressures are in pascal, energies in joule.
"""

from __future__ import annotations

import math

# --- Physical constants -------------------------------------------------------

R_VAPOR = 461.5          #: Specific gas constant of water vapor, J/(kg K)
GAMMA_VAPOR = 1.33       #: Isentropic exponent (cp/cv) of low-pressure steam
T_CRITICAL_K = 647.096   #: Critical temperature of water, K
ZERO_C_IN_K = 273.15     #: 0 C expressed in kelvin

#: Vapor specific heat at constant pressure implied by the ideal-gas relation
#: cp = gamma/(gamma-1) * R.  ~1860 J/(kg K); steam tables give ~2010 at 100 C.
CP_VAPOR = GAMMA_VAPOR / (GAMMA_VAPOR - 1.0) * R_VAPOR


def to_kelvin(temp_C: float) -> float:
    """Convert Celsius to Kelvin."""
    return temp_C + ZERO_C_IN_K


# --- Saturation pressure ------------------------------------------------------
#
# Antoine equation: log10(P_mmHg) = A - B / (C + T_C).  Two coefficient sets are
# blended so the function is smooth and reasonable from ~1 C to ~200 C.

# Valid ~1-100 C (NIST / Bridgeman & Aldrich).
_ANTOINE_LOW = (8.07131, 1730.63, 233.426)
# Valid ~99-374 C.
_ANTOINE_HIGH = (8.14019, 1810.94, 244.485)
_MMHG_TO_PA = 133.322387415


def _antoine_pa(temp_C: float, coeffs: tuple[float, float, float]) -> float:
    a, b, c = coeffs
    return 10.0 ** (a - b / (c + temp_C)) * _MMHG_TO_PA


def sat_pressure(temp_C: float) -> float:
    """Saturation pressure of water, in pascal, at ``temp_C`` (Celsius).

    Uses the low-temperature Antoine set up to 100 C and the high-temperature
    set above, blended linearly across 95-105 C so the result is continuous and
    smoothly differentiable through the boiling point (important for the
    compressor pressure-ratio calculation, which straddles 100 C).
    """
    low = _antoine_pa(temp_C, _ANTOINE_LOW)
    if temp_C <= 95.0:
        return low
    high = _antoine_pa(temp_C, _ANTOINE_HIGH)
    if temp_C >= 105.0:
        return high
    # Smoothstep blend across the 95-105 C overlap band.  The 3s^2 - 2s^3 weight
    # has zero slope at both ends, so the blended curve is C1-continuous (no
    # derivative kink at 95 or 105 C) -- important for the compressor
    # pressure-ratio math and for gradient-free optimizers.
    s = (temp_C - 95.0) / 10.0
    w = s * s * (3.0 - 2.0 * s)
    return (1.0 - w) * low + w * high


def sat_temperature(pressure_pa: float) -> float:
    """Inverse of :func:`sat_pressure`: boiling temperature (C) at a pressure.

    Solved numerically by bisection over 1-250 C.  Useful for expressing an
    operating point by pressure instead of temperature.
    """
    if pressure_pa <= 0.0:
        raise ValueError("pressure must be positive")
    lo, hi = 1.0, 250.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sat_pressure(mid) < pressure_pa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- Latent heat of vaporization ---------------------------------------------

_HFG_REF_JKG = 2_256_500.0        # latent heat at 100 C, J/kg
_HFG_REF_T_K = to_kelvin(100.0)
_WATSON_EXPONENT = 0.38


def latent_heat(temp_C: float) -> float:
    """Latent heat of vaporization of water, J/kg, at ``temp_C``.

    Watson correlation referenced to 2256.5 kJ/kg at 100 C.
    """
    temp_K = to_kelvin(temp_C)
    if temp_K >= T_CRITICAL_K:
        return 0.0
    ratio = (T_CRITICAL_K - temp_K) / (T_CRITICAL_K - _HFG_REF_T_K)
    return _HFG_REF_JKG * ratio ** _WATSON_EXPONENT


# --- Vapor density ------------------------------------------------------------


def vapor_density(temp_C: float, pressure_pa: float | None = None) -> float:
    """Density of water vapor, kg/m^3, treated as an ideal gas.

    If ``pressure_pa`` is omitted, the saturation pressure at ``temp_C`` is used
    (i.e. saturated-vapor density).
    """
    if pressure_pa is None:
        pressure_pa = sat_pressure(temp_C)
    return pressure_pa / (R_VAPOR * to_kelvin(temp_C))


# --- Liquid water properties --------------------------------------------------


def liquid_cp(temp_C: float) -> float:
    """Specific heat of liquid water, J/(kg K).

    Shallow quadratic fit that captures the mild rise from ~4180 near room
    temperature to ~4217 at 100 C; adequate for sensible-heat bookkeeping.
    """
    return 4209.0 - 1.31 * temp_C + 0.014 * temp_C * temp_C


def liquid_density(temp_C: float) -> float:
    """Density of liquid water, kg/m^3, over ~0-100 C (quadratic fit)."""
    return 1000.6 - 0.0574 * temp_C - 0.0034 * temp_C * temp_C


def clausius_clapeyron_dPdT(temp_C: float) -> float:
    """Slope dP/dT of the saturation curve, Pa/K, from Clausius-Clapeyron.

    dP/dT = h_fg * P / (R_vapor * T^2).  Handy for quick temperature-lift <->
    pressure-rise estimates without evaluating the Antoine curve twice.
    """
    temp_K = to_kelvin(temp_C)
    return latent_heat(temp_C) * sat_pressure(temp_C) / (R_VAPOR * temp_K * temp_K)
