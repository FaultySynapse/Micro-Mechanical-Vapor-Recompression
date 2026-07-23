"""Blower operating band: one selectable blower covers a broad duty by speed.

Sizes a blower to the optimized 600 W / 80 C duty, then sweeps its speed
(affinity laws Q~N, dp~N^2, efficiency preserved) and maps the delivered
pressure back to a compression lift and a production rate.  Shows the design is
not pinned to a narrow operating band: speed control alone spans a wide range of
flow and power at constant efficiency, and the fixed-speed curve accommodates
varying system resistance.

    python examples/blower_band.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvr import DesignParameters, maximize_flow, fan
from mvr import properties as props
from mvr.masstransfer import solve_at_power, solve_at_speed


def main() -> None:
    base = DesignParameters(noncondensable_pressure=500.0, feed_hx_effectiveness=0.85,
                            recovery_ratio=0.8, ambient_temp_C=20.0, feed_temp_C=20.0)
    design, r0, _ = maximize_flow(base, budget_w=600.0)

    t = design.evaporator_temp_C
    q_star = design.fan_volumetric_flow
    dp_star = r0.cond_total_pressure_pa - r0.evap_total_pressure_pa + design.duct_pressure_drop_pa
    rho = props.vapor_density(t, r0.evap_total_pressure_pa)

    blower = fan.size_blower_for_duty(q_star, dp_star, rho)
    duty = fan.characterize_duty(q_star, dp_star, rho)
    print(f"Blower sized to the 600 W / {t:.0f} C duty:")
    print(f"  best-efficiency point  {q_star*1000:.1f} L/s at {dp_star/1000:.1f} kPa")
    print(f"  class                  {duty['machine']}")
    print(f"  peak overall eff       {blower.peak_efficiency:.2f}")
    print(f"  fixed-speed range      shut-off {blower.pressure(0)/1000:.1f} kPa,"
          f" free delivery {blower.max_flow*1000:.1f} L/s")
    print()

    # POWER is the true input; flow and lift come out of the fan curve.
    print("Flow & lift as OUTPUTS of the fan curve, selected by power:")
    print(f"  {'power W':>7} | {'speed %':>7} | {'Q L/s':>6} | {'lift K':>6} | "
          f"{'eff':>5} | {'L/h':>6} | {'deliv/prod':>10}")
    for power in (200.0, 400.0, 600.0, 800.0):
        r, op = solve_at_power(design, blower, power)
        tag = "  <- 600 W target" if power == 600.0 else ""
        print(f"  {power:7.0f} | {op['speed_ratio']*100:7.0f} | {op['flow_m3s']*1000:6.2f} | "
              f"{op['lift_K']:6.2f} | {op['efficiency']:5.2f} | {r.distillate_lph:6.2f} | "
              f"{op['delivery_ratio']:10.2f}{tag}")
    print()

    # The same picture by speed (affinity preserves efficiency across the band).
    print("Operating band by speed control:")
    print(f"  {'speed %':>7} | {'Q L/s':>6} | {'lift K':>6} | {'L/h':>6} | {'blower W':>8}")
    for n in (0.6, 0.8, 1.0, 1.2):
        r, op = solve_at_speed(design, blower, n)
        print(f"  {n*100:7.0f} | {op['flow_m3s']*1000:6.2f} | {op['lift_K']:6.2f} | "
              f"{r.distillate_lph:6.2f} | {op['blower_power_w']:8.0f}")


if __name__ == "__main__":
    main()
