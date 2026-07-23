"""Steady-state model of a micro MVR greywater still.

Physical picture
----------------
Two partially water-filled chambers share a heat-exchange wall.  Water boils in
the evaporator chamber; a fan draws the vapor into the condenser chamber,
raising its pressure and therefore its saturation (condensing) temperature.  The
vapor condenses on the far side of the shared wall, and the latent heat it
releases is conducted straight back through the wall to sustain boiling.  That
recycling of latent heat is what makes MVR efficient: the fan only supplies the
*compression* work plus whatever the system leaks, not the full latent heat.

Feed greywater is pre-heated in a counter-flow economizer by the outgoing hot
distillate and concentrate before entering the evaporator.  Contaminants stay
behind in the concentrate; the condensate is the purified product.

Temperature stack (bottom to top)
---------------------------------
    T_evap                      cold-side nominal boiling temp (design input)
    T_boil  = T_evap + BPE      actual boiling-liquid temp (solutes elevate it)
    T_cond  = T_boil + lift     condensing-vapor temp (drives wall heat transfer)

The compressor must raise the vapor from ``P_sat(T_evap)`` to ``P_sat(T_cond)``;
only ``lift`` does useful heat transfer, the boiling-point-elevation part is a
pure loss.

Key results
-----------
  * distillate_rate ..... kg/s (and L/h) of purified water
  * fan_power ........... electrical W drawn by the vapor mover
  * makeup_heat ......... external heat needed to close the energy balance
                          (negative => the unit self-heats and needs cooling)
  * specific_energy ..... total electrical input per liter of product (kWh/L)
  * gain_output_ratio ... latent heat produced / total energy input (GOR)

Everything is pure-Python and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import properties as props
from .parameters import DesignParameters


@dataclass
class Results:
    """Computed steady-state operating point (all SI unless noted)."""

    # Temperatures / pressures of the operating point
    boiling_temp_C: float
    condensing_temp_C: float
    evap_pressure_pa: float
    cond_pressure_pa: float
    pressure_ratio: float

    # Heat transfer
    overall_U: float          # W/(m^2 K)
    heat_duty: float          # W conducted through the wall
    distillate_rate: float    # kg/s
    distillate_lph: float     # L/h (convenience)

    # Fan / compression
    vapor_volume_flow: float  # m^3/s at evaporator conditions
    compression_work_specific: float  # J/kg of vapor (shaft, incl. duct losses)
    fan_power: float          # electrical W

    # Feed / streams
    feed_rate: float          # kg/s
    concentrate_rate: float   # kg/s
    feed_preheat_temp_C: float
    feed_heating_duty: float  # W to raise pre-heated feed to boiling

    # Energy balance
    insulation_loss: float    # W lost through the shell
    unrecovered_stream_loss: float  # W of sensible heat leaving in products
    makeup_heat: float        # W external heat to close the balance (may be <0)
    total_energy_input: float # W electrical-equivalent input

    # Performance figures of merit
    specific_energy_kwh_per_l: float
    gain_output_ratio: float


def overall_U(p: DesignParameters) -> float:
    """Overall heat-transfer coefficient of the shared wall, W/(m^2 K).

    Series resistances: boiling film, fouling, wall conduction, condensing film.
    """
    resistance = (
        1.0 / p.boiling_htc
        + p.fouling_resistance
        + p.wall_thickness / p.wall_conductivity
        + 1.0 / p.condensing_htc
    )
    return 1.0 / resistance


def _compression_work_specific(p: DesignParameters,
                               p_evap: float, p_cond: float,
                               t_evap_C: float) -> float:
    """Shaft compression work per kg of vapor, J/kg.

    Ideal-gas isentropic compression from ``p_evap`` to ``p_cond`` (with the
    extra duct pressure drop stacked on the discharge), divided by the fan's
    isentropic efficiency.  For the small pressure ratios of an MVR still this
    is very close to the incompressible estimate ``dP / rho_vapor``.
    """
    p_discharge = p_cond + p.duct_pressure_drop_pa
    exponent = (props.GAMMA_VAPOR - 1.0) / props.GAMMA_VAPOR
    isentropic = (
        props.CP_VAPOR
        * props.to_kelvin(t_evap_C)
        * ((p_discharge / p_evap) ** exponent - 1.0)
    )
    return isentropic / p.fan_isentropic_efficiency


@dataclass
class StreamMetrics:
    """Economizer, energy-balance and figure-of-merit results shared by both
    the boiling and the evaporative models (all SI unless noted)."""

    feed_rate: float
    concentrate_rate: float
    feed_preheat_temp_C: float
    feed_heating_duty: float
    insulation_loss: float
    unrecovered_stream_loss: float
    makeup_heat: float
    total_energy_input: float
    distillate_lph: float
    specific_energy_kwh_per_l: float
    gain_output_ratio: float


def stream_metrics(p: DesignParameters, distillate_rate: float, h_fg: float,
                   distillate_temp_C: float, concentrate_temp_C: float,
                   feed_target_temp_C: float, loss_ref_temp_C: float,
                   shaft_power: float, fan_power: float) -> StreamMetrics:
    """Feed economizer, whole-unit energy balance, and figures of merit.

    Shared by :func:`solve` (boiling) and the evaporative model so both account
    for streams and energy identically.  Temperatures name the physical points:
    the distillate leaves at ``distillate_temp_C`` (condenser), the concentrate
    at ``concentrate_temp_C`` (evaporator), the feed must be raised to
    ``feed_target_temp_C``, and shell loss is referenced to ``loss_ref_temp_C``.
    """
    feed_rate = distillate_rate / p.recovery_ratio
    concentrate_rate = feed_rate - distillate_rate
    cp = props.liquid_cp(loss_ref_temp_C)

    # Economizer: hot products pre-heat the incoming feed.  The effective hot
    # side is a mass-weighted blend of distillate and concentrate temperatures.
    hot_inlet_C = (distillate_rate * distillate_temp_C
                   + concentrate_rate * concentrate_temp_C) / feed_rate
    feed_preheat_temp_C = (
        p.feed_temp_C + p.feed_hx_effectiveness * (hot_inlet_C - p.feed_temp_C)
    )
    feed_heating_duty = feed_rate * cp * max(feed_target_temp_C - feed_preheat_temp_C, 0.0)

    # Sensible heat still riding out with the products after the economizer.
    unrecovered = (1.0 - p.feed_hx_effectiveness) * feed_rate * cp * (hot_inlet_C - p.feed_temp_C)

    # Whole-unit energy balance.  All fan shaft work dissipates into the vapor
    # loop and helps close the balance; motor losses do not enter the fluid.
    insulation_loss = p.insulation_ua * (loss_ref_temp_C - p.ambient_temp_C)
    makeup_heat = insulation_loss + unrecovered + feed_heating_duty - shaft_power
    total_energy_input = fan_power + max(makeup_heat, 0.0)

    rho_product = props.liquid_density(distillate_temp_C)
    distillate_lph = distillate_rate / rho_product * 1000.0 * 3600.0
    if distillate_rate > 0 and total_energy_input > 0:
        # (W / (kg/s)) = J/kg of product; * (kg/L) = J/L; / 3.6e6 = kWh/L.
        energy_per_kg = total_energy_input / distillate_rate
        specific_energy_kwh_per_l = energy_per_kg * (rho_product / 1000.0) / 3.6e6
        gain_output_ratio = distillate_rate * h_fg / total_energy_input
    else:
        specific_energy_kwh_per_l = float("inf")
        gain_output_ratio = 0.0

    return StreamMetrics(
        feed_rate=feed_rate,
        concentrate_rate=concentrate_rate,
        feed_preheat_temp_C=feed_preheat_temp_C,
        feed_heating_duty=feed_heating_duty,
        insulation_loss=insulation_loss,
        unrecovered_stream_loss=unrecovered,
        makeup_heat=makeup_heat,
        total_energy_input=total_energy_input,
        distillate_lph=distillate_lph,
        specific_energy_kwh_per_l=specific_energy_kwh_per_l,
        gain_output_ratio=gain_output_ratio,
    )


def solve(p: DesignParameters) -> Results:
    """Solve the steady-state operating point in the *boiling* regime.

    Phase change is assumed heat-transfer-limited: the distillate rate equals
    the wall heat duty divided by the latent heat.  For a low-temperature,
    fan-swept *evaporative* unit where mass transfer can bind instead, use
    :func:`mvr.masstransfer.solve_evaporative`.
    """

    # --- Temperature / pressure stack ----------------------------------------
    t_evap = p.evaporator_temp_C
    t_boil = t_evap + p.boiling_point_elevation
    t_cond = t_boil + p.temp_lift

    p_evap = props.sat_pressure(t_evap)
    p_cond = props.sat_pressure(t_cond)

    # --- Heat transfer through the shared wall -------------------------------
    U = overall_U(p)
    heat_duty = U * p.hx_area * p.temp_lift           # W, driven by useful lift
    h_fg = props.latent_heat(t_boil)
    distillate_rate = heat_duty / h_fg                # kg/s of vapor == product

    # --- Fan / vapor compression ---------------------------------------------
    rho_v = props.vapor_density(t_evap, p_evap)
    vapor_volume_flow = distillate_rate / rho_v
    w_specific = _compression_work_specific(p, p_evap, p_cond, t_evap)
    shaft_power = distillate_rate * w_specific
    fan_power = shaft_power / p.fan_motor_efficiency

    # --- Feed, streams, energy balance, figures of merit (shared helper) ------
    m = stream_metrics(
        p, distillate_rate, h_fg,
        distillate_temp_C=t_cond, concentrate_temp_C=t_boil,
        feed_target_temp_C=t_boil, loss_ref_temp_C=t_boil,
        shaft_power=shaft_power, fan_power=fan_power,
    )

    return Results(
        boiling_temp_C=t_boil,
        condensing_temp_C=t_cond,
        evap_pressure_pa=p_evap,
        cond_pressure_pa=p_cond,
        pressure_ratio=p_cond / p_evap,
        overall_U=U,
        heat_duty=heat_duty,
        distillate_rate=distillate_rate,
        distillate_lph=m.distillate_lph,
        vapor_volume_flow=vapor_volume_flow,
        compression_work_specific=w_specific,
        fan_power=fan_power,
        feed_rate=m.feed_rate,
        concentrate_rate=m.concentrate_rate,
        feed_preheat_temp_C=m.feed_preheat_temp_C,
        feed_heating_duty=m.feed_heating_duty,
        insulation_loss=m.insulation_loss,
        unrecovered_stream_loss=m.unrecovered_stream_loss,
        makeup_heat=m.makeup_heat,
        total_energy_input=m.total_energy_input,
        specific_energy_kwh_per_l=m.specific_energy_kwh_per_l,
        gain_output_ratio=m.gain_output_ratio,
    )


def report(p: DesignParameters, r: Results) -> str:
    """Render a human-readable summary of an operating point."""
    heater_note = "external heating" if r.makeup_heat >= 0 else "surplus / needs cooling"
    lines = [
        "=" * 62,
        " Micro-MVR greywater still  --  steady-state operating point",
        "=" * 62,
        " Operating point",
        f"   evaporator (cold) temp .... {p.evaporator_temp_C:8.2f} C",
        f"   boiling-liquid temp ....... {r.boiling_temp_C:8.2f} C  (+BPE {p.boiling_point_elevation:.2f} K)",
        f"   condensing temp ........... {r.condensing_temp_C:8.2f} C  (+lift {p.temp_lift:.2f} K)",
        f"   evap / cond pressure ...... {r.evap_pressure_pa/1000:7.2f} / {r.cond_pressure_pa/1000:.2f} kPa",
        f"   pressure ratio ............ {r.pressure_ratio:8.3f}",
        "",
        " Heat transfer & production",
        f"   overall U ................. {r.overall_U:8.1f} W/m^2K",
        f"   heat duty ................. {r.heat_duty:8.1f} W",
        f"   distillate rate ........... {r.distillate_rate*1000:8.3f} g/s  ({r.distillate_lph:.2f} L/h)",
        "",
        " Fan / vapor compression",
        f"   vapor volume flow ......... {r.vapor_volume_flow*1000:8.2f} L/s",
        f"   compression work .......... {r.compression_work_specific/1000:8.2f} kJ/kg",
        f"   fan electrical power ...... {r.fan_power:8.1f} W",
        "",
        " Feed & streams",
        f"   feed rate ................. {r.feed_rate*1000:8.3f} g/s",
        f"   concentrate rate .......... {r.concentrate_rate*1000:8.3f} g/s",
        f"   feed pre-heat temp ........ {r.feed_preheat_temp_C:8.2f} C",
        f"   feed heating duty ......... {r.feed_heating_duty:8.1f} W",
        "",
        " Energy balance",
        f"   insulation loss ........... {r.insulation_loss:8.1f} W",
        f"   unrecovered stream loss ... {r.unrecovered_stream_loss:8.1f} W",
        f"   makeup heat ............... {r.makeup_heat:8.1f} W  ({heater_note})",
        f"   total energy input ........ {r.total_energy_input:8.1f} W",
        "",
        " Figures of merit",
        f"   specific energy ........... {r.specific_energy_kwh_per_l:8.4f} kWh/L",
        f"   gain output ratio (GOR) ... {r.gain_output_ratio:8.2f}",
        "=" * 62,
    ]
    return "\n".join(lines)
