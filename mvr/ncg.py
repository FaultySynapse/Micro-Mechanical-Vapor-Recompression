"""Non-condensable gas (NCG): load from the feed, and the purge trade-off.

Greywater arrives air-saturated (and often CO2-rich), so dissolved gas flashes
out of solution under the still's reduced operating pressure and accumulates in
the vapor space, where it blankets the condenser (see :mod:`mvr.masstransfer`).
Steady operation therefore requires a continuous purge, and the purge itself
costs product: the bled gas is mostly water vapor.

This module estimates
  * the NCG **load** -- how fast dissolved gas enters with the feed, and
  * the purge **cost** -- the water vapor lost to hold a target NCG partial
    pressure, plus a rough purge-pump power.

The central trade-off: a lower NCG partial pressure means better condensation
(more gross flow) but a leaner purge that carries away more water vapor per mole
of gas removed.  There is an interior optimum in *net* flow.

Pure standard library.
"""

from __future__ import annotations

from . import properties as props

#: Universal gas constant, J/(mol K).
R_UNIVERSAL = 8.314462618
#: Molar mass of water, kg/mol.
M_WATER = 0.018015

# Dissolved-gas load of air-saturated water near room temperature (mol per
# litre), from Henry's law at atmospheric air composition.  N2 + O2 + Ar.
_AIR_N2_MOL_L = 0.00053
_AIR_O2_MOL_L = 0.00028
_AIR_AR_MOL_L = 0.00001
#: Air-saturated dissolved gas, mol/L (~0.8 mmol/L).  CO2/bicarbonate in real
#: greywater can add several times this; pass a larger value to explore it.
AIR_SATURATED_MOL_L = _AIR_N2_MOL_L + _AIR_O2_MOL_L + _AIR_AR_MOL_L


def ncg_release_rate(feed_volumetric_l_s: float,
                     dissolved_gas_mol_l: float = AIR_SATURATED_MOL_L,
                     release_fraction: float = 1.0) -> float:
    """Molar rate (mol/s) at which NCG enters the vapor space with the feed.

    ``release_fraction`` is the share of dissolved gas that actually degasses at
    operating conditions; under vacuum and heat it approaches 1.
    """
    return feed_volumetric_l_s * dissolved_gas_mol_l * release_fraction


def purge_cost(ncg_release_mol_s: float, p_ncg: float, p_water: float,
               operating_pressure_pa: float | None = None,
               pump_efficiency: float = 0.3,
               ambient_pressure_pa: float = 101_325.0) -> dict:
    """Purge required to hold ``p_ncg``, and what it costs.

    At steady state the purge must remove NCG at exactly the release rate.  The
    bled gas has the local composition, so its water-vapor fraction is
    ``p_water / (p_water + p_ncg)`` and the water lost per mole of NCG removed is
    ``p_water / p_ncg``.  Returns water lost (kg/s) and a rough isothermal
    pump power (W) to lift the purge from the operating pressure to ambient.
    """
    if p_ncg <= 0.0:
        return {"purge_total_mol_s": float("inf"), "water_lost_kg_s": float("inf"),
                "water_lost_mol_s": float("inf"), "pump_power_w": float("inf")}
    # Total purge gas rate so that its NCG fraction carries the release rate.
    ncg_fraction = p_ncg / (p_ncg + p_water)
    purge_total_mol_s = ncg_release_mol_s / ncg_fraction
    water_lost_mol_s = purge_total_mol_s - ncg_release_mol_s
    water_lost_kg_s = water_lost_mol_s * M_WATER

    pump_power_w = 0.0
    if operating_pressure_pa and operating_pressure_pa < ambient_pressure_pa:
        import math
        # Isothermal compression work to lift the purge to ambient, /efficiency.
        temp_k = 320.0  # representative vapor-space temperature
        pump_power_w = (purge_total_mol_s * R_UNIVERSAL * temp_k
                        * math.log(ambient_pressure_pa / operating_pressure_pa)
                        / pump_efficiency)
    return {
        "purge_total_mol_s": purge_total_mol_s,
        "water_lost_mol_s": water_lost_mol_s,
        "water_lost_kg_s": water_lost_kg_s,
        "pump_power_w": pump_power_w,
    }


def feed_volumetric_l_s(feed_rate_kg_s: float, temp_C: float = 20.0) -> float:
    """Convert a feed mass rate (kg/s) to volumetric flow (L/s)."""
    return feed_rate_kg_s / props.liquid_density(temp_C) * 1000.0
