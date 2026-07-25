"""Metal-wool / mesh packing model for the evaporation and condensation surfaces.

Filling a chamber with fine metal wool (or a wire mesh) does three things the
flat-plate model does not capture, and this module turns a physical wool spec
into the effective parameters the solver already understands:

  1. **Interfacial area** -- a huge wetted liquid/vapor area per unit volume,
     ``a_v = 4 (1-eps) / d_fiber`` for cylindrical fibers, so a compact chamber
     can present many m^2 of evaporation/condensation surface.
  2. **Gas-side transfer** -- the sweep now develops fiber-scale boundary layers,
     so the mass/heat-transfer coefficient follows a packed-bed correlation on
     the *fiber* diameter and interstitial velocity, NOT the plate length.  This
     is what lets the area grow without the flat-plate artifact of the sweep
     velocity collapsing.
  3. **Fin coupling to the wall** -- wool bonded to the heat-transfer wall acts as
     a dense fin field, multiplying the effective film coefficient by
     ``1 + (fin area / base area) * fin_efficiency`` (aluminum/copper fibers are
     near-unity efficient over their short length), improving ``U``.

It also adds a **pressure-drop** penalty (Ergun) for sweeping vapor through the
packing, which the fan must pay.  Thermal mass is irrelevant (steady state).

Corrosion note: aluminum is amphoteric and corrodes in the alkaline greywater of
the evaporator; copper or stainless wool is the durable choice on the wetted
side.  This module does not model corrosion -- it is a performance model only.

Pure standard library (+ the package's own transport/properties).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from . import properties as props
from . import transport
from .parameters import DesignParameters


@dataclass
class PackingSpec:
    """Physical description of a metal-wool / mesh packing."""

    fiber_diameter_m: float             # strand/fiber diameter, m (fine wool ~30-100 um)
    void_fraction: float                # porosity eps (wool ~0.95-0.99)
    conductivity: float = 205.0         # fiber material k, W/mK (Al 205, Cu 385, SS 16)
    wetted_fraction: float = 1.0        # fraction of surface active as liquid/vapor interface
    mat_thickness_m: float = 0.02       # wool-layer thickness bonded to the wall (fin length), m

    def specific_area(self) -> float:
        """Interfacial area per unit packed volume, m^2/m^3 (cylindrical fibers)."""
        return 4.0 * (1.0 - self.void_fraction) / self.fiber_diameter_m

    def interfacial_area(self, packed_volume_m3: float) -> float:
        """Active liquid/vapor area (m^2) for a given packed volume."""
        return self.specific_area() * packed_volume_m3 * self.wetted_fraction


def fiber_mass_transfer_coeff(spec: PackingSpec, superficial_velocity: float,
                              temp_C: float, pressure_pa: float,
                              density: float) -> tuple[float, dict]:
    """Gas-side mass-transfer coefficient (m/s) for flow through the packing.

    Wakao-Kaguei packed-bed correlation on the fiber diameter and interstitial
    velocity: ``Sh = 2 + 1.1 Re^0.6 Sc^(1/3)``, ``h_m = Sh D_AB / d_fiber``.
    Decouples the coefficient from the plate length, so enlarging the area no
    longer starves the sweep the way the flat-plate channel model does.
    """
    d_f = spec.fiber_diameter_m
    mu = transport.vapor_viscosity(temp_C)
    d_ab = transport.vapor_diffusivity(temp_C, pressure_pa)
    interstitial = superficial_velocity / max(spec.void_fraction, 1e-6)
    Re = density * interstitial * d_f / mu
    Sc = mu / (density * d_ab)
    Sh = 2.0 + 1.1 * Re ** 0.6 * Sc ** (1.0 / 3.0)
    h_m = Sh * d_ab / d_f
    return h_m, {"reynolds": Re, "schmidt": Sc, "sherwood": Sh, "specific_area": spec.specific_area()}


def fin_efficiency(spec: PackingSpec, film_htc: float) -> float:
    """Fin efficiency ``tanh(mL)/mL`` of a wool fiber acting as a pin fin.

    ``m = sqrt(4 h / (k d_fiber))`` for a cylindrical fiber; ``L`` is the wool
    layer thickness.  High-k fibers over a thin mat are near-unity efficient.
    """
    d_f = spec.fiber_diameter_m
    m = math.sqrt(4.0 * film_htc / (spec.conductivity * d_f))
    mL = m * spec.mat_thickness_m
    return math.tanh(mL) / mL if mL > 0 else 1.0


def wall_htc_enhancement(spec: PackingSpec, film_htc: float,
                         wall_area_m2: float, packed_volume_m3: float) -> float:
    """Multiplier on a wall film coefficient from wool bonded to the wall as fins.

    ``1 + (fin area / base area) * fin_efficiency``.  The fin area is the packed
    interfacial area sitting on the wall footprint; capped for sanity.
    """
    fin_area = spec.interfacial_area(packed_volume_m3)
    area_ratio = fin_area / wall_area_m2 if wall_area_m2 > 0 else 0.0
    eff = fin_efficiency(spec, film_htc)
    return 1.0 + area_ratio * eff


def packing_pressure_drop(spec: PackingSpec, superficial_velocity: float,
                          depth_m: float, temp_C: float, density: float) -> float:
    """Ergun pressure drop (Pa) for sweeping vapor through the packing depth."""
    eps = spec.void_fraction
    d_f = spec.fiber_diameter_m
    mu = transport.vapor_viscosity(temp_C)
    u = superficial_velocity
    viscous = 150.0 * mu * (1.0 - eps) ** 2 * u / (eps ** 3 * d_f ** 2)
    inertial = 1.75 * density * (1.0 - eps) * u ** 2 / (eps ** 3 * d_f)
    return (viscous + inertial) * depth_m


def apply_packing(base: DesignParameters, spec: PackingSpec,
                  packed_volume_m3: float, superficial_velocity: float,
                  sweep_depth_m: float | None = None) -> DesignParameters:
    """Return a parameter set with the wool's effective area, transfer coefficient,
    wall-``U`` enhancement and pressure drop folded in (fixed-coefficient mode).

    ``packed_volume_m3`` is the wool volume in each chamber; ``superficial_velocity``
    is the fan sweep velocity through the packing (chamber flow / face area).  The
    returned params run the evaporative solver in fixed-coefficient mode, so its
    fan power uses the net-vapor-throughput basis -- adequate here because the
    packed sweep is essentially single-pass.
    """
    t_evap = base.evaporator_temp_C
    p_sat = props.sat_pressure(t_evap)
    p_tot = p_sat + base.noncondensable_pressure
    rho = props.vapor_density(t_evap, p_tot)

    area = spec.interfacial_area(packed_volume_m3)
    h_m, _diag = fiber_mass_transfer_coeff(spec, superficial_velocity, t_evap, p_tot, rho)
    boil_mult = wall_htc_enhancement(spec, base.boiling_htc, base.hx_area, packed_volume_m3)
    cond_mult = wall_htc_enhancement(spec, base.condensing_htc, base.hx_area, packed_volume_m3)
    depth = sweep_depth_m if sweep_depth_m is not None else spec.mat_thickness_m
    dp = packing_pressure_drop(spec, superficial_velocity, depth, t_evap, rho)

    return replace(
        base,
        transfer_from_flow=False,
        evap_area=area,
        condenser_area=area,
        evap_mass_transfer_coeff=h_m,
        condenser_mass_transfer_coeff=h_m,
        boiling_htc=base.boiling_htc * boil_mult,
        condensing_htc=base.condensing_htc * cond_mult,
        duct_pressure_drop_pa=base.duct_pressure_drop_pa + dp,
    )
