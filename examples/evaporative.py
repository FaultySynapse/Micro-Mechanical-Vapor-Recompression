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
        evap_mass_transfer_coeff=0.03,
        condenser_mass_transfer_coeff=0.03,
    )
    result = solve_evaporative(params)
    print(report_evaporative(params, result))
    print()

    # Show the payoff of purging non-condensables.
    print("Effect of the non-condensable purge:")
    print(f"  {'NCG (kPa)':>10} | {'L/h':>7} | {'limited by':>12} | {'MT eff %':>8}")
    for ncg in (5000.0, 2000.0, 1000.0, 500.0, 100.0):
        r = solve_evaporative(
            DesignParameters(
                evaporator_temp_C=50.0, temp_lift=6.0, noncondensable_pressure=ncg,
                evap_mass_transfer_coeff=0.03, condenser_mass_transfer_coeff=0.03,
            )
        )
        print(f"  {ncg/1000:10.2f} | {r.distillate_lph:7.2f} | "
              f"{r.limiting_mechanism:>12} | {r.mass_transfer_effectiveness*100:8.1f}")


if __name__ == "__main__":
    main()
