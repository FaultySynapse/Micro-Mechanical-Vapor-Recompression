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
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvr import DesignParameters, maximize_flow, fan
from mvr import properties as props
from mvr.masstransfer import solve_evaporative


def lift_for_pressure(dp_pa: float, t_evap_C: float, p_ncg: float) -> float:
    """Compression lift (K) whose pressure rise equals ``dp_pa``."""
    p_sat = props.sat_pressure(t_evap_C)
    ratio = 1.0 + dp_pa / (p_sat + p_ncg)
    return props.sat_temperature(ratio * p_sat) - t_evap_C


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
    print("Operating band by speed control (efficiency preserved by affinity):")
    print(f"  {'speed %':>7} | {'Q L/s':>6} | {'dp kPa':>6} | {'lift K':>6} | "
          f"{'eff':>5} | {'L/h':>6} | {'fan W':>6}")
    for n in (0.6, 0.75, 0.9, 1.0, 1.1, 1.25):
        b = blower.at_speed(n)
        q, dp = b.design_flow, b.design_pressure
        lift = lift_for_pressure(dp, t, design.noncondensable_pressure)
        r = solve_evaporative(replace(design, temp_lift=lift, fan_volumetric_flow=q))
        tag = "  <- 600 W design" if abs(n - 1.0) < 1e-9 else ""
        print(f"  {n*100:7.0f} | {q*1000:6.2f} | {dp/1000:6.1f} | {lift:6.2f} | "
              f"{b.efficiency(q):5.2f} | {r.distillate_lph:6.2f} | {r.fan_power:6.0f}{tag}")


if __name__ == "__main__":
    main()
