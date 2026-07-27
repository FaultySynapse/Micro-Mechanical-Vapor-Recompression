"""Standalone hull-length cost optimizer.

Balances the length-dependent costs of the vertical tube-bundle MVR hull:

  * tube cost  -- a short hull needs many tubes (N = A / (pi d L)), each with two
    sealed tube-sheet joints and a distributor insert, so cost rises as 1/L;
  * insulation -- the insulated outer surface grows with hull height, so cost
    rises with L;
  * the hull itself -- a *step change*: an off-the-shelf pressure cooker is cheap
    but caps the height; going taller forces a custom rated vessel.

Everything else (fan, pumps, controls) is a fixed cost that does not move the
optimum; it is carried only so the totals are realistic.  Pure geometry -- no
thermodynamic solve -- so it runs standalone.  All coefficients are explicit and
meant to be edited.

    python -m mvr.hull_cost
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import size_tube_count


@dataclass
class HullCostModel:
    """Explicit, editable cost coefficients (USD) and fixed geometry."""

    # --- cost coefficients (order-of-magnitude placeholders) -----------------
    cost_per_tube: float = 12.0            # tube + 2 sealed joints + distributor insert
    cost_per_insulation_m2: float = 40.0   # insulation blanket over the outer surface
    cost_stock_hull: float = 50.0          # off-the-shelf pressure cooker (caps height)
    cost_custom_hull: float = 500.0        # custom rated pressure vessel (any length)
    cost_fixed: float = 250.0              # fan + pumps + controls (constant; context)

    # --- fixed geometry / target ---------------------------------------------
    shell_diameter_m: float = 0.26
    tube_inner_diameter_m: float = 0.032
    overhead_m: float = 0.06               # header + drain/sump (not active tube)
    target_area_m2: float = 0.60           # wall area production needs
    stock_max_height_m: float = 0.30       # tallest common stock pressure cooker (interior)

    # -------------------------------------------------------------------------
    def tube_count(self, active_length_m: float) -> int:
        return size_tube_count(self.target_area_m2, self.tube_inner_diameter_m, active_length_m)

    def outer_surface_m2(self, hull_height_m: float) -> float:
        d = self.shell_diameter_m
        return math.pi * d * hull_height_m + 2.0 * math.pi * (d / 2.0) ** 2

    def evaluate(self, active_length_m: float) -> dict:
        """Cost breakdown for a given active tube length."""
        height = active_length_m + self.overhead_m
        n = self.tube_count(active_length_m)
        is_stock = height <= self.stock_max_height_m

        tube = self.cost_per_tube * n
        insulation = self.cost_per_insulation_m2 * self.outer_surface_m2(height)
        hull = self.cost_stock_hull if is_stock else self.cost_custom_hull
        total = tube + insulation + hull + self.cost_fixed
        return {
            "active_length_m": active_length_m,
            "hull_height_m": height,
            "tube_count": n,
            "regime": "stock" if is_stock else "custom",
            "tube_cost": tube,
            "insulation_cost": insulation,
            "hull_cost": hull,
            "fixed_cost": self.cost_fixed,
            "total_cost": total,
        }


def optimize_length(model: HullCostModel, lo: float = 0.12, hi: float = 0.80,
                    step: float = 0.005) -> dict:
    """Minimize total cost over active tube length by a simple sweep.

    Returns the overall best plus the best point in each hull regime, so the
    stock-vs-custom step trade is explicit.
    """
    n = int(round((hi - lo) / step))
    best = best_stock = best_custom = None
    for i in range(n + 1):
        r = model.evaluate(lo + i * step)
        if best is None or r["total_cost"] < best["total_cost"]:
            best = r
        if r["regime"] == "stock" and (best_stock is None or r["total_cost"] < best_stock["total_cost"]):
            best_stock = r
        if r["regime"] == "custom" and (best_custom is None or r["total_cost"] < best_custom["total_cost"]):
            best_custom = r
    return {"best": best, "best_stock": best_stock, "best_custom": best_custom}


def tube_cost_breakeven(model: HullCostModel) -> float:
    """Per-tube cost at which the best custom hull ties the best stock hull.

    Above this, the tube savings of a taller (custom) hull repay the hull step;
    below it, staying in the stock pot wins.  Isolates the single number the
    decision hinges on.
    """
    res = optimize_length(model)
    s, c = res["best_stock"], res["best_custom"]
    if s is None or c is None:
        return float("nan")
    # total = (non-tube) + cost_per_tube * N; solve for the crossing per-tube cost.
    non_tube_s = s["total_cost"] - s["tube_cost"]
    non_tube_c = c["total_cost"] - c["tube_cost"]
    dn = s["tube_count"] - c["tube_count"]
    return (non_tube_c - non_tube_s) / dn if dn > 0 else float("inf")


def main() -> None:
    model = HullCostModel()
    print("Hull-length cost sweep (fixed: 26 cm Ø, 32 mm tubes, 0.60 m^2 target)\n")
    print(f"  {'L act':>6} {'H hull':>6} {'tubes':>5} {'regime':>6} "
          f"{'tube$':>6} {'insul$':>6} {'hull$':>6} {'total$':>7}")
    for L in (0.16, 0.20, 0.24, 0.28, 0.34, 0.44, 0.60):
        r = model.evaluate(L)
        print(f"  {r['active_length_m']:5.2f}m {r['hull_height_m']:5.2f}m {r['tube_count']:5d} "
              f"{r['regime']:>6} {r['tube_cost']:6.0f} {r['insulation_cost']:6.0f} "
              f"{r['hull_cost']:6.0f} {r['total_cost']:7.0f}")

    res = optimize_length(model)
    b, s, c = res["best"], res["best_stock"], res["best_custom"]
    print(f"\n  best overall : L {b['active_length_m']:.2f} m ({b['regime']}), "
          f"{b['tube_count']} tubes, ${b['total_cost']:.0f}")
    print(f"  best stock   : L {s['active_length_m']:.2f} m, {s['tube_count']} tubes, ${s['total_cost']:.0f}")
    print(f"  best custom  : L {c['active_length_m']:.2f} m, {c['tube_count']} tubes, ${c['total_cost']:.0f}")
    print(f"\n  decision hinges on per-tube cost: custom wins only above "
          f"${tube_cost_breakeven(model):.0f}/tube (now ${model.cost_per_tube:.0f}).")


if __name__ == "__main__":
    main()
