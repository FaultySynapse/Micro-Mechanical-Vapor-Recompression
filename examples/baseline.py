"""Baseline example: solve the default design and optimize the temperature lift.

Run with::

    python examples/baseline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvr import DesignParameters, solve, report
from mvr.optimize import minimize_scalar, specific_energy_objective


def main() -> None:
    base = DesignParameters()

    print(report(base, solve(base)))
    print()

    # Find the temperature lift that minimizes specific energy while keeping
    # production at or above 4 L/h.
    objective = specific_energy_objective(min_distillate_lph=4.0)
    best_lift, best_obj = minimize_scalar(base, "temp_lift", 0.5, 15.0, objective)

    tuned = DesignParameters(temp_lift=best_lift)
    tuned_result = solve(tuned)

    print(f"Optimized temperature lift: {best_lift:.2f} K")
    print(f"  distillate .......... {tuned_result.distillate_lph:.2f} L/h")
    print(f"  specific energy ..... {tuned_result.specific_energy_kwh_per_l*1000:.2f} kWh/m^3")
    print(f"  GOR ................. {tuned_result.gain_output_ratio:.2f}")


if __name__ == "__main__":
    main()
