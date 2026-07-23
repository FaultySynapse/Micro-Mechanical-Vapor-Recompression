"""Plot the non-condensable-gas sensitivity and purge trade-off.

Shows, versus NCG partial pressure and at two operating temperatures:
  * gross vs net distillate flow (net = gross - water lost to the purge),
  * water lost to the purge and the purge-pump power.

Reveals that the useful purge target depends on the operating regime: deep
purging pays off only while condensation is the bottleneck (lower temperature),
not once the wall heat flow limits (higher temperature).

Requires matplotlib.  Writes scripts/ncg_sensitivity.png.

    python scripts/ncg_sensitivity.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mvr import DesignParameters
from mvr.masstransfer import solve_evaporative
from mvr.optimize import fan_flow_for_power_budget, maximize_flow
from mvr import ncg

BUDGET_W = 600.0
P_NCG = [2000, 1500, 1000, 700, 500, 350, 200, 120, 70, 40, 20]


def _curves(design: DesignParameters, dissolved_gas_mol_l: float):
    gross, net, water_pct, pump = [], [], [], []
    for pncg in P_NCG:
        p = replace(design, noncondensable_pressure=float(pncg))
        vf = fan_flow_for_power_budget(p, BUDGET_W)
        r = solve_evaporative(replace(p, fan_volumetric_flow=vf))
        feed_lps = ncg.feed_volumetric_l_s(r.feed_rate, design.feed_temp_C)
        rel = ncg.ncg_release_rate(feed_lps, dissolved_gas_mol_l)
        cost = ncg.purge_cost(rel, pncg, r.cond_bulk_pv_pa,
                              operating_pressure_pa=r.evap_total_pressure_pa)
        gross_lph = r.distillate_lph
        net_lph = gross_lph - cost["water_lost_kg_s"] / 998.0 * 1000.0 * 3600.0
        gross.append(gross_lph)
        net.append(net_lph)
        water_pct.append(cost["water_lost_kg_s"] / r.distillate_rate * 100.0)
        pump.append(cost["pump_power_w"])
    return gross, net, water_pct, pump


def main() -> None:
    base = DesignParameters(feed_hx_effectiveness=0.85, recovery_ratio=0.8,
                            ambient_temp_C=20.0, feed_temp_C=20.0)

    # Optimize the geometry once at the 80 C cap, then reuse it at both temps.
    design, _, _ = maximize_flow(base, budget_w=BUDGET_W)
    hot = design                                       # ~80 C (wall-limited)
    cool = replace(design, evaporator_temp_C=55.0)     # 55 C (condensation-limited)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for design_pt, label, color in ((hot, "80 C", "tab:red"),
                                    (cool, "55 C", "tab:blue")):
        gross, net, _, _ = _curves(design_pt, ncg.AIR_SATURATED_MOL_L)
        ax1.plot(P_NCG, gross, color=color, label=f"{label} gross")
        ax1.plot(P_NCG, net, color=color, ls="--", label=f"{label} net")
    ax1.set_xscale("log")
    ax1.set_xlabel("NCG partial pressure (Pa)")
    ax1.set_ylabel("distillate (L/h)")
    ax1.set_title("Gross vs net flow (net = gross - purge loss)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Purge cost at the 80 C design for air-saturated and CO2-rich feed.
    for gas, glabel, color in ((ncg.AIR_SATURATED_MOL_L, "air-saturated", "tab:green"),
                               (8 * ncg.AIR_SATURATED_MOL_L, "CO2-rich (8x)", "tab:orange")):
        _, _, water_pct, pump = _curves(hot, gas)
        ax2.plot(P_NCG, water_pct, color=color, label=f"{glabel}: water lost %")
        ax2.plot(P_NCG, [pw / 10.0 for pw in pump], color=color, ls=":",
                 label=f"{glabel}: pump W/10")
    ax2.set_xscale("log")
    ax2.set_xlabel("NCG partial pressure (Pa)")
    ax2.set_ylabel("purge cost")
    ax2.set_title("Purge cost (water lost %, pump W/10)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "ncg_sensitivity.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
