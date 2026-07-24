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
from dataclasses import dataclass, replace

from . import properties as props
from . import transport
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
    limiting_mechanism: str          # "evaporation"|"condensation"|"wall-heat"|"fan-throughput"
    heat_transfer_ceiling: float     # kg/s the wall UA can condense at full lift
    fan_delivery_ceiling: float      # kg/s the fan can carry (inf if not flow-mode)

    # Gas-phase sensible heat transfer at the condenser
    discharge_temp_C: float          # superheated vapor leaving the fan
    condenser_superheat_K: float     # discharge temp above condenser saturation
    desuperheat_effectiveness: float # fraction of superheat shed at the interface (0..1)
    sensible_duty: float             # W of sensible (desuperheat) load on the wall
    sensible_fraction: float         # sensible / total condenser duty

    # Lift budget (K): how the nominal temp_lift is spent
    lift_nominal: float
    lift_ncg_penalty: float
    lift_evaporation: float
    lift_condensation: float
    lift_useful_wall: float

    # Fan / fan-driven transport
    overall_U: float
    vapor_volume_flow: float          # net vapor volume flow (evaporator conds)
    compression_work_specific: float
    fan_power: float
    fan_circulation_flow: float       # gas the fan sweeps over the surfaces, m^3/s
    circulation_ratio: float          # circulated gas mass / net vapor mass
    evap_velocity: float              # sweep velocity over the evaporator, m/s
    cond_velocity: float              # sweep velocity over the condenser, m/s
    evap_reynolds: float
    cond_reynolds: float
    evap_htc_mass: float              # mass-transfer coeff used at evaporator, m/s
    cond_htc_mass: float              # mass-transfer coeff used at condenser, m/s

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


def htc_from_mass_transfer_coeff(mass_transfer_coeff: float,
                                 gas_density: float = 0.1,
                                 gas_cp: float = 1900.0,
                                 lewis_number: float = 0.85) -> float:
    """Inverse of :func:`mass_transfer_coeff_from_htc`: gas-side sensible
    heat-transfer coefficient (W/m^2 K) implied by a mass-transfer coefficient,

        h_heat = h_m * rho_gas * cp_gas * Le^(2/3)

    so the same gas film governs both sensible heat and mass transfer.
    """
    return mass_transfer_coeff * gas_density * gas_cp * lewis_number ** (2.0 / 3.0)


def _invert_evaporation(p: DesignParameters, m_dot: float,
                        t_evap_C: float, p_evap_tot: float, h_m: float) -> float:
    """Bulk water partial pressure that sustains evaporation ``m_dot`` (Pa)."""
    p_surf = props.sat_pressure(t_evap_C)
    t_K = props.to_kelvin(t_evap_C)

    def flow(p_v_bulk: float) -> float:
        return film_mass_flow(h_m, p.evap_area, t_K,
                              p_evap_tot, p_surf, p_v_bulk)

    # flow decreases as the bulk pressure rises toward the surface value.
    if flow(0.0) <= m_dot:
        return 0.0                      # evaporation saturated: cannot go faster
    lo, hi = 0.0, p_surf
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if flow(mid) > m_dot:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _invert_condensation(p: DesignParameters, m_dot: float,
                         t_surf_C: float, p_cond_tot: float, h_m: float) -> float:
    """Bulk water partial pressure needed to condense ``m_dot`` onto a surface
    at ``t_surf_C`` (Pa)."""
    p_surf = props.sat_pressure(t_surf_C)
    t_K = props.to_kelvin(t_surf_C)
    if p_surf >= p_cond_tot:
        return p_cond_tot               # surface too hot to condense: infeasible

    def flow(p_v_bulk: float) -> float:
        return film_mass_flow(h_m, p.condenser_area,
                              t_K, p_cond_tot, p_v_bulk, p_surf)

    # flow increases as the bulk pressure rises above the surface value.
    lo, hi = p_surf, p_cond_tot
    for _ in range(45):
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

    # --- Transfer coefficients: from the fan-driven sweep, or fixed inputs ----
    t_cond_ref = t_evap + p.temp_lift
    rho_evap = props.vapor_density(t_evap, p_evap_tot)
    rho_cond = props.vapor_density(t_cond_ref, p_cond_tot)
    if p.transfer_from_flow:
        # Fan circulates fan_volumetric_flow over the surfaces (evaporator inlet
        # conditions); mass continuity sets the denser, slower condenser sweep.
        circulated_mass = rho_evap * p.fan_volumetric_flow
        xsec_evap = (p.evap_area / p.channel_length) * p.channel_gap
        xsec_cond = (p.condenser_area / p.channel_length) * p.channel_gap
        vel_evap = p.fan_volumetric_flow / xsec_evap
        vel_cond = (circulated_mass / rho_cond) / xsec_cond
        h_m_e, _hg_e, diag_e = transport.surface_transfer_coefficients(
            vel_evap, p.channel_length, t_evap, p_evap_tot, rho_evap, p.gas_specific_heat)
        h_m_c, h_g_flow, diag_c = transport.surface_transfer_coefficients(
            vel_cond, p.channel_length, t_cond_ref, p_cond_tot, rho_cond, p.gas_specific_heat)
    else:
        circulated_mass = None          # fan power falls back to net throughput
        h_m_e = p.evap_mass_transfer_coeff
        h_m_c = p.condenser_mass_transfer_coeff
        h_g_flow = None
        vel_evap = vel_cond = 0.0
        diag_e = diag_c = {"reynolds": 0.0}

    # --- Gas-phase sensible heat transfer at the condenser -------------------
    # The fan delivers the vapor superheated.  Actual (not just isentropic)
    # discharge temperature, including the reheat from fan inefficiency:
    exponent = (props.GAMMA_VAPOR - 1.0) / props.GAMMA_VAPOR
    isentropic_factor = ratio ** exponent
    discharge_K = t_evap_K * (1.0 + (isentropic_factor - 1.0) / p.fan_isentropic_efficiency)
    discharge_C = discharge_K - props.ZERO_C_IN_K
    cp_gas = p.gas_specific_heat
    # Gas-side sensible coefficient: an explicit override wins; otherwise use the
    # flat-plate value from the fan sweep (flow mode) or the Lewis analogy from
    # the fixed mass-transfer coefficient (fallback).  Same gas film either way.
    if p.condenser_gas_htc:
        h_g = p.condenser_gas_htc
    elif h_g_flow is not None:
        h_g = h_g_flow
    else:
        h_g = htc_from_mass_transfer_coeff(
            p.condenser_mass_transfer_coeff, p.gas_density, cp_gas, p.lewis_number)

    def _desuperheat_eff(m_dot: float) -> float:
        """NTU effectiveness for cooling the vapor toward the interface."""
        mcp = m_dot * cp_gas
        if mcp <= 0.0:
            return 0.0
        ntu = h_g * p.condenser_area / mcp
        return 1.0 - math.exp(-min(ntu, 60.0))

    def _interface_temp(m_dot: float) -> float:
        """Condensation-interface temperature from the interface energy balance

            U*A*(T_i - T_evap) = m_dot*h_fg + m_dot*cp*eff*(T_discharge - T_i)

        i.e. the wall must reject the latent heat *plus* the shed superheat.
        Reduces to the pure-latent result when there is no superheat.
        """
        cap = m_dot * cp_gas * _desuperheat_eff(m_dot)   # sensible capacity, W/K
        num = UA * t_evap + m_dot * h_fg + cap * discharge_C
        return num / (UA + cap)

    # Upper bracket on m_dot: two hard ceilings cap production regardless of how
    # fast the gas films can move vapor.
    #   * heat-transfer ceiling: the wall cannot push the condensation interface
    #     above the condenser saturation temperature (UA * dT_max / h_fg).
    #   * fan-delivery ceiling: in flow mode the fan cannot carry more vapor than
    #     it sweeps (rho_sat * volumetric flow); infinite otherwise.
    # The binding one sets the bracket; which of the three (either ceiling, or an
    # interior mass-transfer equilibrium) actually holds is classified below.
    t_surf_ceiling = props.sat_temperature(p_cond_tot)
    m_heat_ceiling = UA * (t_surf_ceiling - t_evap) / h_fg
    m_deliver_ceiling = (props.vapor_density(t_evap, p_sat_evap) * p.fan_volumetric_flow
                         if p.transfer_from_flow else math.inf)
    m_dot_ceiling = min(m_heat_ceiling, m_deliver_ceiling)
    m_dot_max = max(m_dot_ceiling, 1e-9) * 0.999

    def residual(m_dot: float) -> float:
        t_surf = _interface_temp(m_dot)
        pv_cond = _invert_condensation(p, m_dot, t_surf, p_cond_tot, h_m_c)
        pv_evap_from_fan = pv_cond / ratio          # fan scales water p_v by ratio
        pv_evap_required = _invert_evaporation(p, m_dot, t_evap, p_evap_tot, h_m_e)
        return pv_evap_from_fan - pv_evap_required

    # residual(0) < 0 and residual(m_dot_max) > 0 -> unique bracketed root.
    lo, hi = 0.0, m_dot_max
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    m_dot = 0.5 * (lo + hi)

    # --- Reconstruct the converged state -------------------------------------
    t_surf_cond = _interface_temp(m_dot)
    pv_cond = _invert_condensation(p, m_dot, t_surf_cond, p_cond_tot, h_m_c)
    pv_evap = pv_cond / ratio

    # Gas-phase sensible (desuperheat) diagnostics.
    superheat = max(discharge_C - props.sat_temperature(p_cond_tot), 0.0)
    eff_ds = _desuperheat_eff(m_dot)
    sensible_duty = m_dot * cp_gas * eff_ds * max(discharge_C - t_surf_cond, 0.0)
    wall_duty = UA * (t_surf_cond - t_evap)
    sensible_fraction = sensible_duty / wall_duty if wall_duty > 0 else 0.0

    # Lift budget (all in saturation-temperature/K terms).
    tb_evap = props.sat_temperature(max(pv_evap, 1.0))
    tb_cond = props.sat_temperature(max(pv_cond, 1.0))
    lift_evap = max(t_evap - tb_evap, 0.0)          # evaporation MT drop
    lift_cond = max(tb_cond - t_surf_cond, 0.0)     # condensation MT drop
    lift_wall = max(t_surf_cond - t_evap, 0.0)      # useful wall dT
    lift_effective = tb_cond - tb_evap              # sat-temp lift the fan delivers
    lift_ncg = max(p.temp_lift - lift_effective, 0.0)

    # Which constraint actually binds at the solved point?
    #
    # The solve terminates one of two ways.  If the pressure-matching residual is
    # still negative at the ceiling, the mass-transfer equilibrium *wants* more
    # vapor than a hard ceiling allows -> the ceiling binds, and it is whichever
    # ceiling is lower (fan delivery vs wall heat).  Otherwise the root is
    # interior -> the gas-film mass transfer itself binds, and the dominant film
    # is whichever "spends" more lift (evaporation vs condensation).  The old
    # scheme just reported the biggest lift bucket, which is nearly always the
    # useful wall dT and so hid the real limiter (typically fan throughput).
    ceiling_binds = residual(m_dot_max) < 0.0
    if ceiling_binds:
        limiting_mechanism = ("fan-throughput"
                              if m_deliver_ceiling < m_heat_ceiling
                              else "wall-heat")
    else:
        limiting_mechanism = "evaporation" if lift_evap >= lift_cond else "condensation"

    heat_limited_rate = UA * p.temp_lift / h_fg
    mt_effectiveness = m_dot / heat_limited_rate if heat_limited_rate > 0 else 0.0

    # --- Fan power (fan-curve model: air power Q*dP over efficiency) ----------
    # The fan moves a volumetric flow Q against the pressure rise dP; the ideal
    # "air power" is Q*dP, the shaft work is that over the aerodynamic
    # efficiency, and the electrical draw is the shaft work over the motor
    # efficiency.  This is the same Q*dP/eta the blower curve uses, so every
    # reported figure is on one basis.  In flow mode Q is the circulated sweep
    # flow; otherwise it is the net vapor volume flow.
    rho_v = props.vapor_density(t_evap, max(pv_evap, 1.0))
    dp_rise = (p_cond_tot + p.duct_pressure_drop_pa) - p_evap_tot
    fan_flow = (p.fan_volumetric_flow if p.transfer_from_flow
                else m_dot / rho_v)
    air_power = fan_flow * dp_rise
    shaft_power = air_power / p.fan_isentropic_efficiency
    fan_power = shaft_power / p.fan_motor_efficiency
    pumped_mass = props.vapor_density(t_evap, p_evap_tot) * fan_flow
    w_specific = shaft_power / pumped_mass if pumped_mass > 0 else 0.0
    vapor_volume_flow = m_dot / rho_v
    circulation_ratio = (pumped_mass / m_dot) if m_dot > 0 else 0.0

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
        heat_transfer_ceiling=m_heat_ceiling,
        fan_delivery_ceiling=m_deliver_ceiling,
        discharge_temp_C=discharge_C,
        condenser_superheat_K=superheat,
        desuperheat_effectiveness=eff_ds,
        sensible_duty=sensible_duty,
        sensible_fraction=sensible_fraction,
        lift_nominal=p.temp_lift,
        lift_ncg_penalty=lift_ncg,
        lift_evaporation=lift_evap,
        lift_condensation=lift_cond,
        lift_useful_wall=lift_wall,
        overall_U=U,
        vapor_volume_flow=vapor_volume_flow,
        compression_work_specific=w_specific,
        fan_power=fan_power,
        fan_circulation_flow=(p.fan_volumetric_flow if p.transfer_from_flow else vapor_volume_flow),
        circulation_ratio=circulation_ratio,
        evap_velocity=vel_evap,
        cond_velocity=vel_cond,
        evap_reynolds=diag_e.get("reynolds", 0.0),
        cond_reynolds=diag_c.get("reynolds", 0.0),
        evap_htc_mass=h_m_e,
        cond_htc_mass=h_m_c,
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


def _fmt_ceiling(rate: float) -> str:
    """Format a mass-flow ceiling, showing 'n/a' for the infinite (no-limit) case."""
    return "     n/a" if math.isinf(rate) else f"{rate*1000:8.3f} g/s"


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
        f"   evap / cond vapor p_v ..... {r.evap_bulk_pv_pa/1000:7.2f} / {r.cond_bulk_pv_pa/1000:.2f} kPa  (bulk)",
        f"   evap / cond NCG partial ... {(r.evap_total_pressure_pa-r.evap_bulk_pv_pa)/1000:7.2f} / {(r.cond_total_pressure_pa-r.cond_bulk_pv_pa)/1000:.2f} kPa  (bulk; = total - p_v)",
        f"   NCG partial at surface .... {r.noncondensable_pressure_pa/1000:8.3f} kPa  (setpoint)",
        f"   pressure ratio ............ {r.pressure_ratio:8.3f}",
        "",
        " Production & limiting mechanism",
        f"   distillate rate ........... {r.distillate_rate*1000:8.3f} g/s  ({r.distillate_lph:.2f} L/h)",
        f"   heat-transfer ceiling ..... {r.heat_transfer_ceiling*1000:8.3f} g/s  (wall UA at full lift)",
        f"   fan-delivery ceiling ...... {_fmt_ceiling(r.fan_delivery_ceiling)}  (fan sweep throughput)",
        f"   mass-transfer effectiveness {r.mass_transfer_effectiveness*100:7.1f} %  (vs nominal-lift heat limit)",
        f"   binding constraint ........ {r.limiting_mechanism:>14}",
        "",
        " Compression-lift budget (K)",
        f"   nominal lift .............. {r.lift_nominal:8.3f}",
        f"   spent on NCG dilution ..... {r.lift_ncg_penalty:8.3f}",
        f"   spent on evaporation MT ... {r.lift_evaporation:8.3f}",
        f"   spent on condensation MT .. {r.lift_condensation:8.3f}",
        f"   useful across the wall .... {r.lift_useful_wall:8.3f}",
        "",
        " Fan-driven transport",
        f"   fan circulation flow ...... {r.fan_circulation_flow*1000:8.2f} L/s",
        f"   circulation ratio ......... {r.circulation_ratio:8.1f} x net vapor",
        f"   evap / cond sweep vel ..... {r.evap_velocity:7.2f} / {r.cond_velocity:.2f} m/s",
        f"   evap / cond Reynolds ...... {r.evap_reynolds:7.0f} / {r.cond_reynolds:.0f}",
        f"   evap / cond h_m (mass) .... {r.evap_htc_mass*1000:7.2f} / {r.cond_htc_mass*1000:.2f} mm/s",
        "",
        " Fan / vapor compression",
        f"   net vapor volume flow ..... {r.vapor_volume_flow*1000:8.2f} L/s",
        f"   compression work .......... {r.compression_work_specific/1000:8.2f} kJ/kg",
        f"   fan electrical power ...... {r.fan_power:8.1f} W",
        "",
        " Gas-phase sensible heat (condenser desuperheat)",
        f"   fan discharge temp ........ {r.discharge_temp_C:8.2f} C",
        f"   condenser superheat ....... {r.condenser_superheat_K:8.2f} K",
        f"   desuperheat effectiveness . {r.desuperheat_effectiveness*100:7.1f} %",
        f"   sensible duty ............. {r.sensible_duty:8.1f} W  ({r.sensible_fraction*100:.1f}% of wall duty)",
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


# --- Vessel pressure spec (sizing) -------------------------------------------


@dataclass
class VesselSpec:
    """Pressure-containment spec for the chamber, derived from an operating point.

    The two chambers share one shell; the condenser side sits at the highest
    pressure and the evaporator at the lowest, so the shell must contain the
    condenser total pressure against ambient (positive gauge = pressure service)
    and resist collapse when the evaporator runs sub-atmospheric (vacuum
    service).  ``design_gauge_pressure_pa`` adds a sizing margin over the worst
    operating gauge for the relief-valve set point / wall thickness.
    """

    max_abs_pressure_pa: float        # highest absolute pressure (condenser)
    min_abs_pressure_pa: float        # lowest absolute pressure (evaporator)
    gauge_pressure_pa: float          # max_abs - ambient (shell load, >=0)
    vacuum_gauge_pressure_pa: float   # ambient - min_abs (collapse load, >=0)
    service: str                      # "pressure" | "vacuum" | "atmospheric"
    design_gauge_pressure_pa: float   # gauge with margin (relief set point)
    saturation_temp_C: float          # T_sat at max pressure (relief context)
    self_venting: bool                # condenser >= ambient -> bleed self-vents


def vessel_pressure_spec(r: EvaporativeResults, ambient_pa: float = 101_325.0,
                         margin_frac: float = 0.10,
                         margin_floor_pa: float = 50_000.0) -> VesselSpec:
    """Containment spec from an :class:`EvaporativeResults` operating point.

    The design gauge pressure follows the usual pressure-vessel rule of the
    larger of a fractional margin and an absolute floor over the operating gauge
    (defaults: +10% or +50 kPa).  ``self_venting`` flags the happy regime where
    the condenser sits at or above atmospheric, so the non-condensable bleed
    pushes itself out with no vacuum pump.
    """
    p_hi = r.cond_total_pressure_pa
    p_lo = r.evap_total_pressure_pa
    gauge = max(p_hi - ambient_pa, 0.0)
    vac = max(ambient_pa - p_lo, 0.0)
    if gauge > 0.0:
        service = "pressure"
        design_gauge = max(gauge * (1.0 + margin_frac), gauge + margin_floor_pa)
    elif vac > 0.0:
        service = "vacuum"
        design_gauge = 0.0            # full-vacuum rating; no positive set point
    else:
        service = "atmospheric"
        design_gauge = margin_floor_pa
    return VesselSpec(
        max_abs_pressure_pa=p_hi,
        min_abs_pressure_pa=p_lo,
        gauge_pressure_pa=gauge,
        vacuum_gauge_pressure_pa=vac,
        service=service,
        design_gauge_pressure_pa=design_gauge,
        saturation_temp_C=props.sat_temperature(p_hi),
        self_venting=p_hi >= ambient_pa,
    )


# --- Temperature control by economizer bypass --------------------------------


@dataclass
class ThermalControl:
    """Nominal setting of the temperature controller (economizer bypass).

    When the fan work exceeds the losses (a surplus, ``makeup < 0``) the unit
    would run hot; a controller rejects the surplus by *detuning the economizer*
    -- opening a bypass solenoid, or biasing the hot side so the clean output
    leaves warmer -- so less heat is recovered and more leaves with the streams.
    """

    mode: str                       # "reject-surplus" | "needs-heating" | "balanced"
    heat_to_reject_w: float         # surplus the controller must shed (0 if none)
    design_effectiveness: float     # the economizer's full-recovery value
    balanced_effectiveness: float   # effective value the controller settles at
    bypass_fraction: float          # 1 - balanced/design (fraction detuned)
    feed_preheat_temp_C: float      # feed inlet temp after the (detuned) economizer
    clean_output_temp_C: float      # product exit temp (hotter when rejecting heat)
    residual_makeup_w: float        # leftover imbalance (~0 when controllable)


def balance_temperature_by_economizer(p: DesignParameters,
                                      min_effectiveness: float = 0.02) -> ThermalControl:
    """Find the economizer setting that holds the target temperature (makeup=0).

    Assumes a controller that trims the feed-HX effectiveness to reject any
    surplus (or leaves it at full recovery and flags that heating is needed).
    Returns the nominal :class:`ThermalControl` settings for sizing the bypass.
    """
    design_eff = p.feed_hx_effectiveness
    full = solve_evaporative(p)

    def makeup_at(eff: float) -> float:
        return solve_evaporative(replace(p, feed_hx_effectiveness=max(eff, 1e-4))).makeup_heat

    if full.makeup_heat >= 0.0:
        # No surplus: the controller keeps full recovery (and may need heating).
        eff_star = design_eff
        mode = "balanced" if full.makeup_heat < 1.0 else "needs-heating"
        r = full
    else:
        # Surplus: makeup rises as effectiveness falls; find the zero crossing.
        lo, hi = min_effectiveness, design_eff
        if makeup_at(lo) < 0.0:
            eff_star = lo                       # even full bypass can't shed it all
        else:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if makeup_at(mid) < 0.0:
                    hi = mid
                else:
                    lo = mid
            eff_star = 0.5 * (lo + hi)
        r = solve_evaporative(replace(p, feed_hx_effectiveness=eff_star))
        mode = "reject-surplus"

    hot_inlet = ((r.distillate_rate * r.cond_surface_temp_C
                  + r.concentrate_rate * r.evap_temp_C) / r.feed_rate)
    clean_output = hot_inlet - eff_star * (hot_inlet - p.feed_temp_C)
    return ThermalControl(
        mode=mode,
        heat_to_reject_w=max(-full.makeup_heat, 0.0),
        design_effectiveness=design_eff,
        balanced_effectiveness=eff_star,
        bypass_fraction=max(1.0 - eff_star / design_eff, 0.0) if design_eff > 0 else 0.0,
        feed_preheat_temp_C=r.feed_preheat_temp_C,
        clean_output_temp_C=clean_output,
        residual_makeup_w=r.makeup_heat,
    )


# --- Blower-coupled operating point ------------------------------------------
#
# The fan curve, not the designer, sets the flow<->lift relation: at a given
# speed the blower delivers a flow and a pressure rise (-> compression lift), and
# the *power* selects the speed.  So the true parameters are the blower and the
# power; flow, lift and production are outputs.


def lift_from_compression_pressure(dp_pa: float, t_evap_C: float,
                                   p_ncg: float) -> float:
    """Compression lift (K) whose evaporator->condenser pressure rise is ``dp_pa``."""
    if dp_pa <= 0.0:
        return 0.0
    p_sat = props.sat_pressure(t_evap_C)
    ratio = 1.0 + dp_pa / (p_sat + p_ncg)
    return max(props.sat_temperature(ratio * p_sat) - t_evap_C, 0.0)


def solve_at_speed(p: DesignParameters, blower, speed_ratio: float,
                   duct_loss_coeff: float = 0.0):
    """Solve the operating point for a blower run at ``speed_ratio`` × design speed.

    The blower (any object exposing ``at_speed``, ``design_flow``,
    ``design_pressure``, ``efficiency`` and ``power``) delivers a best-efficiency
    flow ``Q`` and pressure rise; the pressure sets the compression lift and the
    flow sets the sweep.  Returns ``(EvaporativeResults, operating_point)`` where
    the operating point carries the blower flow, compression pressure, lift,
    efficiency, blower electrical power, and the delivery/production ratio (≈1 =
    single-pass, >1 = the blower over-delivers and vapor recirculates).
    """
    b = blower.at_speed(speed_ratio)
    flow = b.design_flow
    # The blower head covers the useful compression plus the duct drop; the model
    # re-adds the duct drop internally, so remove it here to avoid double-counting.
    compression_pa = (b.design_pressure - p.duct_pressure_drop_pa
                      - duct_loss_coeff * flow * flow)
    lift = lift_from_compression_pressure(max(compression_pa, 0.0),
                                          p.evaporator_temp_C,
                                          p.noncondensable_pressure)
    # Make the model's Q·Δp/η fan power use the blower's own efficiency, so its
    # reported fan_power equals the blower draw (one consistent power basis).
    isen = min(b.efficiency(flow) / p.fan_motor_efficiency, 1.0)
    result = solve_evaporative(replace(p, fan_volumetric_flow=flow,
                                       temp_lift=max(lift, 0.05),
                                       fan_isentropic_efficiency=isen,
                                       transfer_from_flow=True))
    rho_sat = props.vapor_density(p.evaporator_temp_C,
                                  props.sat_pressure(p.evaporator_temp_C))
    delivery_ratio = (rho_sat * flow / result.distillate_rate
                      if result.distillate_rate > 0 else float("inf"))
    op = {
        "speed_ratio": speed_ratio,
        "flow_m3s": flow,
        "compression_pa": compression_pa,
        "lift_K": lift,
        "efficiency": b.efficiency(flow),
        "blower_power_w": b.power(flow),
        "delivery_ratio": delivery_ratio,
    }
    return result, op


def solve_with_bleed(p: DesignParameters):
    """Solve the evaporative operating point and the integrated bleed balance.

    Returns ``(EvaporativeResults, BleedResult)``.  The bleed balance turns the
    gross production into a **net** production (after the steam vented with the
    non-condensables) and reports the vent pump power and any recovered water and
    heat -- so the costs of holding ``noncondensable_pressure`` are accounted for
    rather than assumed away.
    """
    from . import ncg

    r = solve_evaporative(p)
    bleed = ncg.bleed_balance(
        feed_rate_kg_s=r.feed_rate, feed_temp_C=p.feed_temp_C,
        distillate_rate_kg_s=r.distillate_rate, evap_temp_C=p.evaporator_temp_C,
        p_ncg=p.noncondensable_pressure, cond_bulk_pv_pa=r.cond_bulk_pv_pa,
        cond_total_pa=r.cond_total_pressure_pa, makeup_heat_w=r.makeup_heat,
        dissolved_gas_mol_l=p.dissolved_gas_mol_per_l,
        release_fraction=p.gas_release_fraction,
        to_feed_condenser=p.bleed_to_feed_condenser,
        condenser_approach_C=p.feed_condenser_approach_C,
        pump_efficiency=p.purge_pump_efficiency,
    )
    return r, bleed


def solve_at_power(p: DesignParameters, blower, power_w: float,
                   duct_loss_coeff: float = 0.0):
    """Solve the operating point for a blower drawing ``power_w`` electrical.

    Best-efficiency power scales as speed^3, so the required speed follows in
    closed form; flow, lift and production then come from :func:`solve_at_speed`.
    Makes power (and the blower) the true parameters.
    """
    base_power = blower.design_flow * blower.design_pressure / blower.peak_efficiency
    speed_ratio = (power_w / base_power) ** (1.0 / 3.0) if base_power > 0 else 1.0
    return solve_at_speed(p, blower, speed_ratio, duct_loss_coeff)
