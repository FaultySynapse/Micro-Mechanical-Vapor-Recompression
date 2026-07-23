"""Gas-phase transport properties and boundary-layer transfer correlations.

These turn a *fan-driven surface velocity* into mass- and heat-transfer
coefficients, so the evaporative model's transfer rates are computed from the
geometry and the fan flow rather than taken as free inputs.

Flow over each surface is modeled as parallel flow over a flat plate of length
``L`` (the flow-direction length), giving average Sherwood/Nusselt numbers:

    Sh = 0.664 Re^0.5 Sc^(1/3)   (laminar, Re < 5e5)
    Sh = 0.037 Re^0.8 Sc^(1/3)   (turbulent)

and analogously Nu with the Prandtl number.  ``h_m = Sh * D_AB / L`` and
``h_g = Nu * k_gas / L``.  Because both use the same Reynolds dependence, the
Chilton-Colburn (Lewis) analogy between them holds automatically.

All properties are for the low-pressure water-vapor / non-condensable gas
mixture in the vapor space; both the kinematic viscosity and the diffusivity
rise sharply as pressure drops, which is why low-pressure operation helps mass
transfer.  Pure standard library.
"""

from __future__ import annotations

from . import properties as props

#: Prandtl number of the vapor-space gas (~steam); weak function of state.
GAS_PRANDTL = 1.0

# Reference binary diffusivity of water vapor in air (25 C, 1 atm).
_D_AB_REF = 2.5e-5          # m^2/s
_D_AB_REF_T_K = 298.15
_D_AB_REF_P = 101_325.0
_LAMINAR_RE_LIMIT = 5.0e5


def vapor_viscosity(temp_C: float) -> float:
    """Dynamic viscosity of water vapor, Pa.s.

    Gas viscosity is essentially pressure-independent and rises mildly with
    temperature; linear fit (~1.0e-5 at 50 C, ~1.2e-5 at 100 C).
    """
    return (8.02 + 0.0407 * temp_C) * 1e-6


def vapor_diffusivity(temp_C: float, pressure_pa: float) -> float:
    """Binary diffusivity of water vapor through the gas, m^2/s.

    Fuller-type scaling ``D ~ T^1.75 / P``: at reduced pressure the diffusivity
    is much larger than at 1 atm, easing mass transfer.
    """
    temp_K = props.to_kelvin(temp_C)
    return _D_AB_REF * (temp_K / _D_AB_REF_T_K) ** 1.75 * (_D_AB_REF_P / pressure_pa)


def _plate_dimensionless(Re: float, schmidt_or_prandtl: float) -> float:
    """Average flat-plate Sherwood/Nusselt number (same functional form)."""
    if Re <= 0.0:
        return 0.0
    if Re < _LAMINAR_RE_LIMIT:
        return 0.664 * Re ** 0.5 * schmidt_or_prandtl ** (1.0 / 3.0)
    return 0.037 * Re ** 0.8 * schmidt_or_prandtl ** (1.0 / 3.0)


def surface_transfer_coefficients(velocity: float, length: float, temp_C: float,
                                  pressure_pa: float, density: float,
                                  gas_cp: float) -> tuple[float, float, dict]:
    """Mass- and heat-transfer coefficients for parallel flow over a surface.

    Returns ``(h_m, h_g, diagnostics)`` where ``h_m`` is the gas-side
    mass-transfer coefficient (m/s), ``h_g`` the gas-side sensible
    heat-transfer coefficient (W/m^2 K), and ``diagnostics`` carries the
    velocity, Reynolds/Schmidt numbers and flow regime.
    """
    if velocity <= 0.0 or density <= 0.0 or length <= 0.0:
        return 0.0, 0.0, {"velocity": max(velocity, 0.0), "reynolds": 0.0,
                          "schmidt": 0.0, "regime": "none"}
    mu = vapor_viscosity(temp_C)
    nu = mu / density
    d_ab = vapor_diffusivity(temp_C, pressure_pa)
    schmidt = nu / d_ab
    reynolds = velocity * length / nu

    sherwood = _plate_dimensionless(reynolds, schmidt)
    nusselt = _plate_dimensionless(reynolds, GAS_PRANDTL)
    h_m = sherwood * d_ab / length
    k_gas = mu * gas_cp / GAS_PRANDTL          # from Pr = mu*cp/k
    h_g = nusselt * k_gas / length

    return h_m, h_g, {
        "velocity": velocity,
        "reynolds": reynolds,
        "schmidt": schmidt,
        "regime": "laminar" if reynolds < _LAMINAR_RE_LIMIT else "turbulent",
    }
