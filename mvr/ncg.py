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

import math
from dataclasses import dataclass

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


# --- Integrated bleed valve --------------------------------------------------


@dataclass
class BleedResult:
    """Steady-state bleed (purge) balance for the whole unit."""

    ncg_release_mol_s: float          # dissolved gas entering with the feed
    bleed_total_mol_s: float          # total gas bled to hold the NCG setpoint
    steam_in_bleed_mol_s: float       # water vapor carried into the bleed
    steam_recovered_mol_s: float      # condensed back in the feed-cooled condenser
    steam_lost_mol_s: float           # vented to atmosphere
    water_lost_kg_s: float            # net product lost with the vent
    heat_recovered_w: float           # latent heat returned to the feed
    pump_power_w: float               # vacuum-pump work for the vent (0 if self-vents)
    self_vents: bool                  # vent pressure >= ambient
    gross_distillate_lph: float
    net_distillate_lph: float         # gross minus vented steam


def bleed_balance(*, feed_rate_kg_s: float, feed_temp_C: float,
                  distillate_rate_kg_s: float, evap_temp_C: float,
                  p_ncg: float, cond_bulk_pv_pa: float, cond_total_pa: float,
                  makeup_heat_w: float,
                  dissolved_gas_mol_l: float = AIR_SATURATED_MOL_L,
                  release_fraction: float = 1.0,
                  to_feed_condenser: bool = False,
                  condenser_approach_C: float = 5.0,
                  pump_efficiency: float = 0.3,
                  ambient_pressure_pa: float = 101_325.0) -> BleedResult:
    """Steady-state bleed balance, with an optional feed-cooled recovery condenser.

    A bleed valve holds the vapor-space NCG partial pressure at ``p_ncg`` by
    venting gas at the condenser bulk composition.  The vented gas is mostly
    steam, so it is a real product loss and a real pump load.  Routing it through
    a **feed-cooled condenser** recovers most of that steam (back to product) and
    its latent heat (to the feed), and leaves only the residual non-condensables
    to vent -- which self-vents when the condenser pressure exceeds ambient.
    """
    feed_lps = feed_volumetric_l_s(feed_rate_kg_s, feed_temp_C)
    release = ncg_release_rate(feed_lps, dissolved_gas_mol_l, release_fraction)

    # Bleed drawn at the condenser bulk: NCG mole fraction there sets how much
    # steam rides along with each mole of NCG removed.
    p_water_bleed = max(cond_bulk_pv_pa, 1.0)
    ncg_fraction = p_ncg / (p_ncg + p_water_bleed)
    bleed_total = release / ncg_fraction if ncg_fraction > 0 else float("inf")
    steam = bleed_total - release

    if to_feed_condenser:
        # Cool the bleed toward the feed temperature; steam condenses until the
        # water partial pressure reaches saturation at the cold exit.
        t_cold = feed_temp_C + condenser_approach_C
        recover_frac = max(0.0, 1.0 - props.sat_pressure(t_cold) / p_water_bleed)
        steam_recovered = steam * recover_frac
        steam_lost = steam - steam_recovered
        pump_moles = release + steam_lost         # only the residual is vented
        heat_recovered = steam_recovered * M_WATER * props.latent_heat(evap_temp_C)
    else:
        steam_recovered = 0.0
        steam_lost = steam
        pump_moles = bleed_total                  # pump the whole steam-laden bleed
        heat_recovered = 0.0

    # Vent from the condenser side; a vacuum pump is needed only below ambient.
    self_vents = cond_total_pa >= ambient_pressure_pa
    if self_vents or pump_moles <= 0:
        pump_power = 0.0
    else:
        temp_k = props.to_kelvin(evap_temp_C)
        pump_power = (pump_moles * R_UNIVERSAL * temp_k
                      * math.log(ambient_pressure_pa / cond_total_pa)
                      / pump_efficiency)

    water_lost_kg_s = steam_lost * M_WATER
    rho_prod = props.liquid_density(evap_temp_C)
    gross_lph = distillate_rate_kg_s / rho_prod * 1000.0 * 3600.0
    net_kg_s = max(distillate_rate_kg_s - water_lost_kg_s, 0.0)
    net_lph = net_kg_s / rho_prod * 1000.0 * 3600.0

    return BleedResult(
        ncg_release_mol_s=release,
        bleed_total_mol_s=bleed_total,
        steam_in_bleed_mol_s=steam,
        steam_recovered_mol_s=steam_recovered,
        steam_lost_mol_s=steam_lost,
        water_lost_kg_s=water_lost_kg_s,
        heat_recovered_w=heat_recovered,
        pump_power_w=pump_power,
        self_vents=self_vents,
        gross_distillate_lph=gross_lph,
        net_distillate_lph=net_lph,
    )
