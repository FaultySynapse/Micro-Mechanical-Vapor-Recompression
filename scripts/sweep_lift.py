"""Plot the temperature-lift trade-off (production vs. specific energy).

Requires matplotlib (see requirements.txt).  Saves a PNG next to the script.

    python scripts/sweep_lift.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from mvr import DesignParameters, solve


def main() -> None:
    lifts = [0.5 + 0.25 * i for i in range(0, 48)]  # 0.5 .. 12.25 K
    production = []
    sec = []
    fan = []
    for lift in lifts:
        r = solve(DesignParameters(temp_lift=lift))
        production.append(r.distillate_lph)
        sec.append(r.specific_energy_kwh_per_l * 1000.0)  # kWh/m^3
        fan.append(r.fan_power)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lifts, production, color="tab:blue", label="distillate (L/h)")
    ax1.set_xlabel("temperature lift across HX wall (K)")
    ax1.set_ylabel("distillate (L/h)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(lifts, sec, color="tab:red", label="specific energy (kWh/m^3)")
    ax2.set_ylabel("specific energy (kWh/m^3)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("MVR still: temperature-lift trade-off")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "sweep_lift.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
