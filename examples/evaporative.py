"""Evaporative (mass-transfer) example for a low-temperature, fan-swept unit.

Shows the compression-lift budget and how non-condensable gas throttles a
sub-boiling still.  Run with::

    python examples/evaporative.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvr import DesignParameters
from mvr.masstransfer import solve_evaporative, report_evaporative


def main() -> None:
    # A small ~50 C unit running under partial vacuum, with residual air that a
    # purge has knocked down to ~0.5 kPa.
    params = DesignParameters(
        evaporator_temp_C=50.0,
        temp_lift=6.0,
        noncondensable_pressure=500.0,
        evap_area=0.30,
        condenser_area=0.30,
        # transfer coefficients are computed from the fan sweep + channel geometry
        fan_volumetric_flow=0.010,   # 10 L/s circulated over the surfaces
        channel_length=0.5,
        channel_gap=0.02,
    )
    result = solve_evaporative(params)
    print(report_evaporative(params, result))
    print()

    # Show the payoff of purging non-condensables.
    print("Effect of the non-condensable purge:")
    print(f"  {'NCG (kPa)':>10} | {'L/h':>7} | {'limited by':>12} | {'MT eff %':>8}")
    for ncg in (5000.0, 2000.0, 1000.0, 500.0, 100.0):
        r = solve_evaporative(
            DesignParameters(evaporator_temp_C=50.0, temp_lift=6.0,
                             noncondensable_pressure=ncg)
        )
        print(f"  {ncg/1000:10.2f} | {r.distillate_lph:7.2f} | "
              f"{r.limiting_mechanism:>12} | {r.mass_transfer_effectiveness*100:8.1f}")
    print()

    # The fan-sizing trade-off: more circulation sweeps the surfaces faster
    # (better transfer, more production) but costs fan power that grows faster
    # than the production it buys.
    print("Fan-flow trade-off (transfer coefficients computed from the sweep):")
    print(f"  {'V_fan L/s':>9} | {'sweep m/s':>9} | {'h_m mm/s':>8} | "
          f"{'L/h':>6} | {'fan W':>7} | {'kWh/m3':>7}")
    for v_fan in (0.002, 0.005, 0.010, 0.020, 0.040):
        r = solve_evaporative(
            DesignParameters(evaporator_temp_C=50.0, temp_lift=6.0,
                             noncondensable_pressure=500.0, fan_volumetric_flow=v_fan)
        )
        print(f"  {v_fan*1000:9.0f} | {r.evap_velocity:9.2f} | {r.evap_htc_mass*1000:8.2f} | "
              f"{r.distillate_lph:6.2f} | {r.fan_power:7.1f} | "
              f"{r.specific_energy_kwh_per_l*1000:7.1f}")


if __name__ == "__main__":
    main()
