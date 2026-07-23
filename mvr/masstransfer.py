"""Coupled mass-transfer model for a sub-boiling, fan-swept evaporative MVR still.

When the unit runs *below* the boiling point, water leaves a warm free surface
by evaporation into the vapor space, the fan sweeps and compresses that vapor to
the condenser, and it condenses on the cooled wall.  Production is then no longer
just ``wall_heat / h_fg`` -- it can be throttled by how fast vapor can cross the
gas films on either surface, and by non-condensable gas (NCG: dissolved air/CO2
flashed out of the greywater) that blankets the condenser.

Resistance chain (a single mass flow ``m_dot`` threads all of it)
-----------------------------------------------------------------
    evaporator liquid surface   p_v = P_sat(T_evap)
        │  evaporation mass transfer  (gas film, throttled by NCG)
    evaporator bulk vapor       p_v = P_v_evap
        │  FAN: compresses total pressure by ratio r -> water p_v scales by r
    condenser bulk vapor        p_v = P_v_cond = r * P_v_evap
        │  condensation mass transfer (gas film, throttled by NCG)
    condensation surface        p_v = P_sat(T_surf_cond)
        │  WALL: latent heat conducts back to the evaporator (U * A * dT)
    evaporator liquid           T_evap

Mass transfer uses the stagnant-film (Stefan-flow) law

    m_dot = h_m * A * (P_tot / (R_v * T)) * ln[(P_tot - p_v_sink)/(P_tot - p_v_source)]

which correctly stiffens as NCG -> 0 (the log term diverges, the film resistance
vanishes) so the whole model collapses onto the heat-transfer-limited boiling
result of :func:`mvr.model.solve`.  With NCG present, evaporation and
condensation each "spend" part of the fan's compression lift, leaving less to
drive the wall -- the :class:`EvaporativeResults` lift budget makes that split
explicit.

The fan compression ratio is derived from ``temp_lift`` as the pure-vapor
pressure ratio ``P_sat(T_evap + temp_lift) / P_sat(T_evap)``, so ``temp_lift``
keeps the same intuitive meaning as in the boiling model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import properties as props
from .model import overall_U, stream_metrics
from .parameters import DesignParameters


@dataclass
class EvaporativeResults:
    """Steady-state operating point of the evaporative (mass-transfer) model."""

    # Operating point
    evap_temp_C: float
    cond_surface_temp_C: float
    evap_total_pressure_pa: float
    cond_total_pressure_pa: float
    pressure_ratio: float
    noncondensable_pressure_pa: float

    # Partial pressures of water vapor around the loop
    evap_bulk_pv_pa: float
    cond_bulk_pv_pa: float

    # Production and the limiting mechanism
    distillate_rate: float           # kg/s
    distillate_lph: float
    heat_limited_rate: float         # kg/s if mass transfer were infinite
    mass_transfer_effectiveness: float   # actual / heat-limited (0..1)
    limiting_mechanism: str          # "evaporation" | "condensation" | "wall-heat"

    # Lift budget (K): how the nominal temp_lift is spent
    lift_nominal: float
    lift_ncg_penalty: float
    lift_evaporation: float
    lift_condensation: float
    lift_useful_wall: float

    # Fan
    overall_U: float
    vapor_volume_flow: float
    compression_work_specific: float
    fan_power: float

    # Feed / streams / energy (from the shared helper)
    feed_rate: float
    concentrate_rate: float
    feed_preheat_temp_C: float
    feed_heating_duty: float
    insulation_loss: float
    unrecovered_stream_loss: float
    makeup_heat: float
    total_energy_input: float
    specific_energy_kwh_per_l: float
    gain_output_ratio: float


# --- Mass-transfer primitives -------------------------------------------------


def film_mass_flow(h_m: float, area: float, temp_K: float, p_total: float,
                   p_v_source: float, p_v_sink: float) -> float:
    """Water-vapor mass flow (kg/s) across a stagnant-NCG gas film.

    Vapor moves from the ``p_v_source`` face (higher water partial pressure) to
    the ``p_v_sink`` face (lower).  Returns ``inf`` when the source partial
    pressure reaches the total pressure (no NCG left to diffuse through -> the
    film stops limiting), and ``0`` when there is no driving force.
    """
    denom = p_total - p_v_source
    numer = p_total - p_v_sink
    if denom <= 0.0 or numer <= 0.0:
        return math.inf
    if numer <= denom:
        return 0.0
    return h_m * area * p_total / (props.R_VAPOR * temp_K) * math.log(numer / denom)


def mass_transfer_coeff_from_htc(heat_transfer_coeff: float,
                                 gas_density: float = 0.8,
                                 gas_cp: float = 1900.0,
                                 lewis_number: float = 0.85) -> float:
    """Estimate a gas-side mass-transfer coefficient (m/s) from a convective
    heat-transfer coefficient via the Chilton-Colburn analogy::

        h_m = h_heat / (rho_gas * cp_gas * Le^(2/3))

    Defaults suit a low-pressure water-vapor/air film.  A convenience for
    choosing ``evap_mass_transfer_coeff`` / ``condenser_mass_transfer_coeff``.
    """
    return heat_transfer_coeff / (gas_density * gas_cp * lewis_number ** (2.0 / 3.0))


def _invert_evaporation(p: DesignParameters, m_dot: float,
                        t_evap_C: float, p_evap_tot: float) -> float:
    """Bulk water partial pressure that sustains evaporation ``m_dot`` (Pa)."""
    p_surf = props.sat_pressure(t_evap_C)
    t_K = props.to_kelvin(t_evap_C)

    def flow(p_v_bulk: float) -> float:
        return film_mass_flow(p.evap_mass_transfer_coeff, p.evap_area, t_K,
                              p_evap_tot, p_surf, p_v_bulk)

    # flow decreases as the bulk pressure rises toward the surface value.
    if flow(0.0) <= m_dot:
        return 0.0                      # evaporation saturated: cannot go faster
    lo, hi = 0.0, p_surf
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if flow(mid) > m_dot:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _invert_condensation(p: DesignParameters, m_dot: float,
                         t_surf_C: float, p_cond_tot: float) -> float:
    """Bulk water partial pressure needed to condense ``m_dot`` onto a surface
    at ``t_surf_C`` (Pa)."""
    p_surf = props.sat_pressure(t_surf_C)
    t_K = props.to_kelvin(t_surf_C)
    if p_surf >= p_cond_tot:
        return p_cond_tot               # surface too hot to condense: infeasible

    def flow(p_v_bulk: float) -> float:
        return film_mass_flow(p.condenser_mass_transfer_coeff, p.condenser_area,
                              t_K, p_cond_tot, p_v_bulk, p_surf)

    # flow increases as the bulk pressure rises above the surface value.
    lo, hi = p_surf, p_cond_tot
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if flow(mid) < m_dot:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_evaporative(p: DesignParameters) -> EvaporativeResults:
    """Solve the coupled evaporative operating point for a parameter set."""
    t_evap = p.evaporator_temp_C
    t_evap_K = props.to_kelvin(t_evap)
    p_sat_evap = props.sat_pressure(t_evap)
    p_ncg = p.noncondensable_pressure

    # Fan compression ratio from the nominal lift (pure-vapor pressure ratio).
    ratio = props.sat_pressure(t_evap + p.temp_lift) / p_sat_evap
    p_evap_tot = p_sat_evap + p_ncg
    p_cond_tot = ratio * p_evap_tot

    U = overall_U(p)
    UA = U * p.hx_area
    h_fg = props.latent_heat(t_evap)

    # Upper bracket on m_dot: the wall cannot push the condensation surface above
    # the condenser saturation temperature.
    t_surf_ceiling = props.sat_temperature(p_cond_tot)
    m_dot_max = max(UA * (t_surf_ceiling - t_evap) / h_fg, 1e-9) * 0.999

    def residual(m_dot: float) -> float:
        t_surf = t_evap + m_dot * h_fg / UA
        pv_cond = _invert_condensation(p, m_dot, t_surf, p_cond_tot)
        pv_evap_from_fan = pv_cond / ratio          # fan scales water p_v by ratio
        pv_evap_required = _invert_evaporation(p, m_dot, t_evap, p_evap_tot)
        return pv_evap_from_fan - pv_evap_required

    # residual(0) < 0 and residual(m_dot_max) > 0 -> unique bracketed root.
    lo, hi = 0.0, m_dot_max
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    m_dot = 0.5 * (lo + hi)

    # --- Reconstruct the converged state -------------------------------------
    t_surf_cond = t_evap + m_dot * h_fg / UA
    pv_cond = _invert_condensation(p, m_dot, t_surf_cond, p_cond_tot)
    pv_evap = pv_cond / ratio

    # Lift budget (all in saturation-temperature/K terms).
    tb_evap = props.sat_temperature(max(pv_evap, 1.0))
    tb_cond = props.sat_temperature(max(pv_cond, 1.0))
    lift_evap = max(t_evap - tb_evap, 0.0)          # evaporation MT drop
    lift_cond = max(tb_cond - t_surf_cond, 0.0)     # condensation MT drop
    lift_wall = max(t_surf_cond - t_evap, 0.0)      # useful wall dT
    lift_effective = tb_cond - tb_evap              # sat-temp lift the fan delivers
    lift_ncg = max(p.temp_lift - lift_effective, 0.0)

    # Which resistance binds?  Compare each "spent" lift to the nominal.
    mechanisms = {
        "evaporation": lift_evap,
        "condensation": lift_cond,
        "wall-heat": lift_wall,
    }
    limiting_mechanism = max(mechanisms, key=mechanisms.get)

    heat_limited_rate = UA * p.temp_lift / h_fg
    mt_effectiveness = m_dot / heat_limited_rate if heat_limited_rate > 0 else 0.0

    # --- Fan power (compress water vapor from evaporator to condenser) --------
    p_discharge = p_cond_tot + p.duct_pressure_drop_pa
    exponent = (props.GAMMA_VAPOR - 1.0) / props.GAMMA_VAPOR
    w_specific = (props.CP_VAPOR * t_evap_K
                  * ((p_discharge / p_evap_tot) ** exponent - 1.0)
                  / p.fan_isentropic_efficiency)
    shaft_power = m_dot * w_specific
    fan_power = shaft_power / p.fan_motor_efficiency
    rho_v = props.vapor_density(t_evap, max(pv_evap, 1.0))
    vapor_volume_flow = m_dot / rho_v

    # --- Streams / energy balance / figures of merit (shared helper) ---------
    metrics = stream_metrics(
        p, m_dot, h_fg,
        distillate_temp_C=t_surf_cond, concentrate_temp_C=t_evap,
        feed_target_temp_C=t_evap, loss_ref_temp_C=t_evap,
        shaft_power=shaft_power, fan_power=fan_power,
    )

    return EvaporativeResults(
        evap_temp_C=t_evap,
        cond_surface_temp_C=t_surf_cond,
        evap_total_pressure_pa=p_evap_tot,
        cond_total_pressure_pa=p_cond_tot,
        pressure_ratio=ratio,
        noncondensable_pressure_pa=p_ncg,
        evap_bulk_pv_pa=pv_evap,
        cond_bulk_pv_pa=pv_cond,
        distillate_rate=m_dot,
        distillate_lph=metrics.distillate_lph,
        heat_limited_rate=heat_limited_rate,
        mass_transfer_effectiveness=mt_effectiveness,
        limiting_mechanism=limiting_mechanism,
        lift_nominal=p.temp_lift,
        lift_ncg_penalty=lift_ncg,
        lift_evaporation=lift_evap,
        lift_condensation=lift_cond,
        lift_useful_wall=lift_wall,
        overall_U=U,
        vapor_volume_flow=vapor_volume_flow,
        compression_work_specific=w_specific,
        fan_power=fan_power,
        feed_rate=metrics.feed_rate,
        concentrate_rate=metrics.concentrate_rate,
        feed_preheat_temp_C=metrics.feed_preheat_temp_C,
        feed_heating_duty=metrics.feed_heating_duty,
        insulation_loss=metrics.insulation_loss,
        unrecovered_stream_loss=metrics.unrecovered_stream_loss,
        makeup_heat=metrics.makeup_heat,
        total_energy_input=metrics.total_energy_input,
        specific_energy_kwh_per_l=metrics.specific_energy_kwh_per_l,
        gain_output_ratio=metrics.gain_output_ratio,
    )


def report_evaporative(p: DesignParameters, r: EvaporativeResults) -> str:
    """Human-readable summary of an evaporative operating point."""
    heater_note = "external heating" if r.makeup_heat >= 0 else "surplus / needs cooling"
    lines = [
        "=" * 64,
        " Micro-MVR greywater still  --  EVAPORATIVE (mass-transfer) model",
        "=" * 64,
        " Operating point",
        f"   evaporator temp ........... {r.evap_temp_C:8.2f} C",
        f"   condensation surface temp . {r.cond_surface_temp_C:8.2f} C",
        f"   evap / cond total press ... {r.evap_total_pressure_pa/1000:7.2f} / {r.cond_total_pressure_pa/1000:.2f} kPa",
        f"   non-condensable pressure .. {r.noncondensable_pressure_pa/1000:8.3f} kPa",
        f"   evap / cond vapor p_v ..... {r.evap_bulk_pv_pa/1000:7.2f} / {r.cond_bulk_pv_pa/1000:.2f} kPa",
        f"   pressure ratio ............ {r.pressure_ratio:8.3f}",
        "",
        " Production & limiting mechanism",
        f"   distillate rate ........... {r.distillate_rate*1000:8.3f} g/s  ({r.distillate_lph:.2f} L/h)",
        f"   heat-transfer limit ....... {r.heat_limited_rate*1000:8.3f} g/s",
        f"   mass-transfer effectiveness {r.mass_transfer_effectiveness*100:7.1f} %",
        f"   limiting resistance ....... {r.limiting_mechanism:>8}",
        "",
        " Compression-lift budget (K)",
        f"   nominal lift .............. {r.lift_nominal:8.3f}",
        f"   spent on NCG dilution ..... {r.lift_ncg_penalty:8.3f}",
        f"   spent on evaporation MT ... {r.lift_evaporation:8.3f}",
        f"   spent on condensation MT .. {r.lift_condensation:8.3f}",
        f"   useful across the wall .... {r.lift_useful_wall:8.3f}",
        "",
        " Fan / vapor compression",
        f"   vapor volume flow ......... {r.vapor_volume_flow*1000:8.2f} L/s",
        f"   compression work .......... {r.compression_work_specific/1000:8.2f} kJ/kg",
        f"   fan electrical power ...... {r.fan_power:8.1f} W",
        "",
        " Energy balance",
        f"   insulation loss ........... {r.insulation_loss:8.1f} W",
        f"   unrecovered stream loss ... {r.unrecovered_stream_loss:8.1f} W",
        f"   feed heating duty ......... {r.feed_heating_duty:8.1f} W",
        f"   makeup heat ............... {r.makeup_heat:8.1f} W  ({heater_note})",
        f"   total energy input ........ {r.total_energy_input:8.1f} W",
        "",
        " Figures of merit",
        f"   specific energy ........... {r.specific_energy_kwh_per_l*1000:8.2f} kWh/m^3",
        f"   gain output ratio (GOR) ... {r.gain_output_ratio:8.2f}",
        "=" * 64,
    ]
    return "\n".join(lines)
