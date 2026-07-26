"""Size the 2 bar concentric-cylinder MVR chamber pair.

Falling-film evaporator in the bore, condensing annulus, outer pressure shell.
Storage is external, so holdup is tiny.  Sizes several bores to the wall area the
600 W / 110 C operating point needs, and checks envelope volume, liquid holdup,
and pressure-wall thickness against the 0.5 m^3 / fabrication limits.

    python examples/cylinder_2bar.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvr import DesignParameters, geometry
from mvr.optimize import maximize_flow, solve_at_budget, WALL_MATERIALS, DEFAULT_BOUNDS
from mvr.geometry import ConcentricCylinder, size_length_for_area, hoop_thickness, apply_geometry


def main() -> None:
    base = DesignParameters(noncondensable_pressure=500.0, feed_hx_effectiveness=0.85,
                            recovery_ratio=0.80, ambient_temp_C=20.0, feed_temp_C=20.0,
                            evaporator_temp_C=110.0)
    bounds = {k: v for k, v in DEFAULT_BOUNDS.items() if k != "evaporator_temp_C"}
    params, r, info = maximize_flow(base, 600.0, bounds=bounds,
                                    materials={"stainless_steel": WALL_MATERIALS["stainless_steel"]})
    need_area = params.hx_area
    gauge = r.cond_total_pressure_pa - 101_325.0
    dp_chambers = r.cond_total_pressure_pa - r.evap_total_pressure_pa

    print("2 bar concentric-cylinder sizing (falling-film bore, condensing annulus)")
    print(f"  target: {need_area:.2f} m^2 wall for {r.distillate_lph:.1f} L/h at 600 W "
          f"(wall {info['heat_ceiling_use']*100:.0f}% used -> has margin)")
    print(f"  pressure: {r.cond_total_pressure_pa/1e5:.2f} bar-abs "
          f"({gauge/1e5:.2f} bar gauge), inter-chamber {dp_chambers/1e5:.2f} bar\n")

    hdr = f"  {'bore':>5} {'length':>7} {'OD':>6} {'envelope':>9} {'holdup':>7} {'shell t':>8} {'tube t':>7} {'L/h':>6}"
    print(hdr)
    for D in (0.15, 0.20, 0.25, 0.30):
        L = size_length_for_area(D, need_area)
        cyl = ConcentricCylinder(inner_diameter_m=D, length_m=L)
        rr, *_ , bleed = solve_at_budget(apply_geometry(params, cyl), 600.0)
        t_shell = hoop_thickness(gauge, cyl.outer_diameter() / 2.0)
        t_tube = hoop_thickness(dp_chambers, D / 2.0)
        fit = "OK" if cyl.envelope_volume() < 0.5 else "OVER"
        print(f"  {D*100:4.0f}cm {L:6.2f}m {cyl.outer_diameter()*100:5.0f}cm "
              f"{cyl.envelope_volume():8.3f}m3 {cyl.liquid_holdup_l():6.1f}L "
              f"{t_shell*1000:7.3f}mm {t_tube*1000:6.3f}mm {bleed.net_distillate_lph:6.2f}  {fit}")

    print(f"\n  Stress-required walls are sub-mm; the {geometry.MIN_FAB_WALL_M*1000:.0f} mm "
          f"fabrication minimum governs -- pressure is structurally trivial here.")
    print("  Envelope is ~10x under the 0.5 m^3 ceiling, so bore/length is free to")
    print("  choose for manufacturing and packaging (or spend the headroom on a wider")
    print("  annulus, bigger economizer, or a cooler lower-pressure variant).")


if __name__ == "__main__":
    main()
