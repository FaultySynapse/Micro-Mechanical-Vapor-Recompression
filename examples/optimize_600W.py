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

from mvr import DesignParameters
from mvr.optimize import maximize_flow, WALL_MATERIALS, DEFAULT_BOUNDS

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
    print(f"  total power ......... {info['total_power_w']:.0f} W  (budget {BUDGET_W:.0f} W)")
    print()
    print("Performance")
    print(f"  PRODUCTION .......... {r.distillate_lph:.2f} L/h")
    print(f"  limited by .......... {r.limiting_mechanism} "
          f"(MT effectiveness {r.mass_transfer_effectiveness*100:.0f}%)")
    print(f"  sweep velocity ...... {r.evap_velocity:.2f} m/s (Re {r.evap_reynolds:.0f})")
    print(f"  specific energy ..... {r.specific_energy_kwh_per_l*1000:.1f} kWh/m^3, "
          f"GOR {r.gain_output_ratio:.1f}")
    print()

    # Material matters little: the wall resistance is tiny next to the films, so
    # the durable (non-corroding) choice costs almost nothing in flow.
    print("Flow by wall material (durable choices lose almost nothing):")
    for name, mat in WALL_MATERIALS.items():
        _, rm, im = maximize_flow(base, budget_w=BUDGET_W, materials={name: mat})
        print(f"  {name:16s} k={mat['wall_conductivity']:5.0f} W/mK -> {im['distillate_lph']:6.2f} L/h")
    print()

    # How the achievable flow scales with the power budget.
    print("Flow vs power budget:")
    for budget in (300.0, 450.0, 600.0, 900.0):
        _, rb, ib = maximize_flow(base, budget_w=budget,
                                  materials={"stainless_steel": WALL_MATERIALS["stainless_steel"]})
        print(f"  {budget:6.0f} W -> {ib['distillate_lph']:6.2f} L/h")
    print()

    # Cold-side temperature trade-off: the design goal keeps T_cold < 80 C, but
    # this shows what running hotter or colder would cost/buy at 600 W.
    print("Cold-side temperature trade-off (everything else optimized, 600 W):")
    print(f"  {'T_cold C':>8} | {'L/h':>6} | {'kWh/m3':>7} | {'GOR':>5} | {'limited by':>12}")
    bounds_no_temp = {k: v for k, v in DEFAULT_BOUNDS.items() if k != "evaporator_temp_C"}
    stainless = {"stainless_steel": WALL_MATERIALS["stainless_steel"]}
    for t_cold in (50.0, 60.0, 70.0, 80.0, 85.0):
        b = replace(base, evaporator_temp_C=t_cold)
        _, rt, _ = maximize_flow(b, budget_w=600.0, bounds=bounds_no_temp,
                                 materials=stainless)
        flag = "  <- cap" if t_cold == 80.0 else ""
        print(f"  {t_cold:8.0f} | {rt.distillate_lph:6.2f} | "
              f"{rt.specific_energy_kwh_per_l*1000:7.1f} | {rt.gain_output_ratio:5.1f} | "
              f"{rt.limiting_mechanism:>12}{flag}")


if __name__ == "__main__":
    main()
