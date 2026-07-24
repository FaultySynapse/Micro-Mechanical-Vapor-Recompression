"""Design goal: maximize distillate flow for 600 W (fan + auxiliary heating).

Maximizes production over the design variables (plate area, temperature lift,
operating temperature, channel geometry, insulation, wall material) subject to
the total power budget and size/material limits.  The fan flow is set so the
design spends exactly the budget.

    python examples/optimize_600W.py
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvr import DesignParameters, fan
from mvr import properties as props
from mvr.optimize import maximize_flow, WALL_MATERIALS, DEFAULT_BOUNDS
from mvr.masstransfer import balance_temperature_by_economizer, vessel_pressure_spec

BUDGET_W = 600.0


def main() -> None:
    base = DesignParameters(
        noncondensable_pressure=500.0,
        feed_hx_effectiveness=0.85,
        recovery_ratio=0.80,
        ambient_temp_C=20.0,
        feed_temp_C=20.0,
    )

    params, r, info = maximize_flow(base, budget_w=BUDGET_W)

    print(f"Design goal: maximize flow for {BUDGET_W:.0f} W (fan + aux heating)\n")
    print("Optimal design (geometry + blower selected together)")
    print(f"  wall material ....... {info['material']} (k={params.wall_conductivity:.0f} W/mK)")
    print(f"  plate area .......... {params.hx_area:.3f} m^2")
    print(f"  evaporator temp ..... {params.evaporator_temp_C:.1f} C")
    print(f"  channel gap ......... {params.channel_gap*1000:.1f} mm")
    print(f"  channel length ...... {params.channel_length:.2f} m")
    print(f"  insulation UA ....... {params.insulation_ua:.3f} W/K")
    print()
    print("Blower operating point (flow & lift are OUTPUTS of the fan curve + power)")
    print(f"  operating flow ...... {info['fan_volumetric_flow']*1000:.1f} L/s")
    print(f"  compression lift .... {info['lift_K']:.2f} K   (derived)")
    print(f"  blower efficiency ... {info['efficiency']:.2f} (overall)")
    print(f"  blower power ........ {info['blower_power_w']:.0f} W")
    print(f"  auxiliary heating ... {info['auxiliary_heat_w']:.0f} W")
    print(f"  bleed vent pump ..... {info['pump_power_w']:.1f} W")
    print(f"  total power ......... {info['total_power_w']:.0f} W  (budget {BUDGET_W:.0f} W)")
    print()
    print("Performance")
    print(f"  NET PRODUCTION ...... {info['net_distillate_lph']:.2f} L/h "
          f"(gross {info['gross_distillate_lph']:.2f}, {info['water_lost_pct']:.2f}% vented)")
    print(f"  limited by .......... {r.limiting_mechanism} "
          f"(MT effectiveness {r.mass_transfer_effectiveness*100:.0f}%)")
    print(f"  sweep velocity ...... {r.evap_velocity:.2f} m/s (Re {r.evap_reynolds:.0f})")
    print(f"  specific energy ..... {r.specific_energy_kwh_per_l*1000:.1f} kWh/m^3, "
          f"GOR {r.gain_output_ratio:.1f}")
    print()

    # Ceiling utilization: which hard limit the design is pressed against, and
    # where the slack is.  The binding ceiling sits near 100%; the other shows
    # unused headroom (so you know whether more flow/power or more area buys output).
    print("Ceiling utilization (what the winning design is pressed against)")
    print(f"  fan-delivery ceiling  {info['fan_ceiling_use']*100:5.1f}% used "
          f"({r.fan_delivery_ceiling*1000:.2f} g/s cap)")
    print(f"  wall-heat ceiling ... {info['heat_ceiling_use']*100:5.1f}% used "
          f"({r.heat_transfer_ceiling*1000:.2f} g/s cap)")
    # Advise off the utilizations, not just the discrete label: when a ceiling is
    # near 100% that hardware is the lever; when both have slack the gas films
    # bind (interior root) and sweep/NCG is the lever.
    top = max(info["fan_ceiling_use"], info["heat_ceiling_use"])
    if top < 0.85:
        lever = "sweep flow / lower NCG (mass-transfer-limited; both ceilings have slack)"
    elif info["fan_ceiling_use"] >= info["heat_ceiling_use"]:
        lever = "fan flow/power (delivery-bound; wall has slack)"
    else:
        lever = "heat-exchange area (wall-bound; fan has slack)"
    print(f"  -> binding mechanism: {info['limiting_mechanism']};  lever: add {lever}")
    print()

    # When delivery-bound, "add fan power" is trivially true; the useful question
    # is whether a better-matched blower gives more air power per watt, or the
    # design is already at the efficiency ceiling (so only power / a cheaper duty
    # helps).  Air power = Q*dP at the operating point; efficiency headroom vs the
    # duty's specific speed answers it.
    if info["fan_ceiling_use"] >= max(info["heat_ceiling_use"], 0.85):
        p_evap_tot = r.evap_total_pressure_pa
        dp_pa = (r.cond_total_pressure_pa + params.duct_pressure_drop_pa) - p_evap_tot
        rho = props.vapor_density(r.evap_temp_C, p_evap_tot)
        hr = fan.efficiency_headroom(info["fan_volumetric_flow"], dp_pa, rho,
                                     info["efficiency"])
        print("Fan-delivery diagnosis (is it the curve, or just power?)")
        print(f"  duty specific speed . {hr['specific_speed']:.3f} -> {hr['machine']}")
        print(f"  assumed / achievable  {hr['assumed_overall_efficiency']:.2f} / "
              f"{hr['achievable_overall_efficiency']:.2f} overall efficiency")
        if hr["at_efficiency_ceiling"]:
            print(f"  -> at the efficiency ceiling for this duty class: a different fan")
            print(f"     curve won't help.  More output needs more POWER, or a cheaper")
            print(f"     duty (hotter vapor = denser sweep; lower NCG/lift = less dP).")
        else:
            print(f"  -> {hr['relative_gain']*100:.0f}% more air power (hence flow) is available")
            print(f"     from a better-matched blower at the same {info['blower_power_w']:.0f} W.")
        print()

    # Vessel spec: the shell must contain the condenser (highest) pressure; the
    # design gauge carries a sizing margin.  Positive gauge => the bleed
    # self-vents (no vacuum pump); sub-atmospheric => vacuum service.
    vs = vessel_pressure_spec(r)
    print("Vessel pressure spec (containment sizing)")
    print(f"  service ............. {vs.service} "
          f"({'self-venting' if vs.self_venting else 'needs vacuum pump'})")
    print(f"  evap / cond abs ..... {vs.min_abs_pressure_pa/1e5:.3f} / "
          f"{vs.max_abs_pressure_pa/1e5:.3f} bar-abs")
    if vs.service == "vacuum":
        print(f"  vacuum load ......... {vs.vacuum_gauge_pressure_pa/1e5:.3f} bar below ambient")
    else:
        print(f"  operating gauge ..... {vs.gauge_pressure_pa/1e5:.3f} bar-g")
    print(f"  design gauge ........ {vs.design_gauge_pressure_pa/1e5:.3f} bar-g (with margin)")
    print(f"  T_sat at max press .. {vs.saturation_temp_C:.1f} C")
    print()

    # Temperature controller: the fan work exceeds losses (a surplus), so the
    # controller detunes the economizer to reject it and hold temperature.
    tc = balance_temperature_by_economizer(params)
    if tc.mode == "reject-surplus":
        print("Temperature control (economizer-bypass to hold T, sizing)")
        print(f"  heat to reject ...... {tc.heat_to_reject_w:.0f} W")
        print(f"  economizer eff ...... {tc.design_effectiveness:.2f} -> {tc.balanced_effectiveness:.2f} "
              f"({tc.bypass_fraction*100:.0f}% bypass)")
        print(f"  feed preheat ........ {tc.feed_preheat_temp_C:.1f} C")
        print(f"  clean output temp ... {tc.clean_output_temp_C:.1f} C")
        print()

    # Bleed recovery: route the vent through a feed-cooled condenser to recover
    # its steam (back to product) and heat, and shrink the vent pump load.
    print("Bleed handling (net L/h | vent pump W | water vented %):")
    for gas, glabel in ((0.00082, "air-saturated feed"), (0.00656, "CO2-rich feed (8x)")):
        for recover, rlabel in ((False, "direct vent "), (True, "feed-cooled recovery")):
            b = replace(base, dissolved_gas_mol_per_l=gas, bleed_to_feed_condenser=recover)
            _, _, i = maximize_flow(b, budget_w=BUDGET_W,
                                    materials={"stainless_steel": WALL_MATERIALS["stainless_steel"]})
            print(f"  {glabel:18s} {rlabel}:  {i['net_distillate_lph']:6.2f} | "
                  f"{i['pump_power_w']:5.1f} | {i['water_lost_pct']:4.2f}")
    print()

    # Material matters little: the wall resistance is tiny next to the films, so
    # the durable (non-corroding) choice costs almost nothing in flow.
    print("Flow by wall material (durable choices lose almost nothing):")
    for name, mat in WALL_MATERIALS.items():
        _, rm, im = maximize_flow(base, budget_w=BUDGET_W, materials={name: mat})
        print(f"  {name:16s} k={mat['wall_conductivity']:5.0f} W/mK -> {im['net_distillate_lph']:6.2f} L/h")
    print()

    # How the achievable flow scales with the power budget.
    print("Flow vs power budget:")
    for budget in (300.0, 450.0, 600.0, 900.0):
        _, rb, ib = maximize_flow(base, budget_w=budget,
                                  materials={"stainless_steel": WALL_MATERIALS["stainless_steel"]})
        print(f"  {budget:6.0f} W -> {ib['net_distillate_lph']:6.2f} L/h")
    print()

    # Temperature trade-off: higher T improves both flow and efficiency; ~90 C is
    # the sweet spot where the condenser rises above atmospheric and the bleed
    # self-vents.  Above ~100 C the evaporator too goes positive (pressure vessel).
    print("Operating-temperature trade-off (everything else optimized, 600 W):")
    print(f"  {'T C':>4} | {'L/h':>6} | {'kWh/m3':>7} | {'GOR':>5} | {'self-vent':>9} | {'limited by':>12}")
    bounds_no_temp = {k: v for k, v in DEFAULT_BOUNDS.items() if k != "evaporator_temp_C"}
    stainless = {"stainless_steel": WALL_MATERIALS["stainless_steel"]}
    for t_op in (60.0, 70.0, 80.0, 90.0, 100.0):
        b = replace(base, evaporator_temp_C=t_op)
        _, rt, _ = maximize_flow(b, budget_w=600.0, bounds=bounds_no_temp,
                                 materials=stainless)
        sv = "yes" if rt.cond_total_pressure_pa >= 101_325.0 else "no"
        flag = "  <- cap" if t_op == 90.0 else ""
        print(f"  {t_op:4.0f} | {rt.distillate_lph:6.2f} | "
              f"{rt.specific_energy_kwh_per_l*1000:7.1f} | {rt.gain_output_ratio:5.1f} | "
              f"{sv:>9} | {rt.limiting_mechanism:>12}{flag}")


if __name__ == "__main__":
    main()
