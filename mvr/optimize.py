"""Lightweight design optimization for the MVR still.

The physics model is cheap to evaluate, so we can optimize by direct search
without any third-party solver.  Helpers provided:

  * :func:`minimize_scalar` -- golden-section search over one parameter.
  * :func:`grid_search`     -- brute-force scan over several parameters.
  * :func:`maximize_flow`   -- maximize distillate production under a total
    power budget (fan + auxiliary heating) and size/material limits.  This is
    the project's headline design problem.

The first two take an *objective* callable ``(DesignParameters) -> float``
(smaller is better) so callers can trade off specific energy, production, capital
proxies, etc.  :func:`specific_energy_objective` minimizes kWh/L while penalizing
designs below a minimum production target.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from typing import Callable, Iterable

from .parameters import DesignParameters
from .model import solve
from . import properties as props
from . import fan

Objective = Callable[[DesignParameters], float]

_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618...


def specific_energy_objective(min_distillate_lph: float = 0.0) -> Objective:
    """Objective: specific energy (kWh/L), penalized below a production floor.

    Designs producing less than ``min_distillate_lph`` are pushed uphill in
    proportion to the shortfall so the search steers back into the feasible
    region rather than collapsing production to make energy/L look good.
    """

    def objective(p: DesignParameters) -> float:
        try:
            r = solve(p)
        except ValueError:
            return math.inf
        penalty = 0.0
        if r.distillate_lph < min_distillate_lph:
            shortfall = (min_distillate_lph - r.distillate_lph) / max(min_distillate_lph, 1e-9)
            penalty = 10.0 * shortfall  # kWh/L-scale penalty
        return r.specific_energy_kwh_per_l + penalty

    return objective


def minimize_scalar(base: DesignParameters, param: str, lo: float, hi: float,
                    objective: Objective, tol: float = 1e-4,
                    max_iter: int = 100) -> tuple[float, float]:
    """Golden-section search of ``param`` over ``[lo, hi]``.

    Returns ``(best_value, best_objective)``.  Assumes the objective is roughly
    unimodal on the interval (true for the lift/energy trade-off); for
    multi-modal objectives use :func:`grid_search` to seed the bracket.
    """
    def f(x: float) -> float:
        return objective(replace(base, **{param: x}))

    a, b = lo, hi
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = f(d)
    best_x = 0.5 * (a + b)
    return best_x, f(best_x)


def grid_search(base: DesignParameters, grid: dict[str, Iterable[float]],
                objective: Objective) -> tuple[DesignParameters, float]:
    """Exhaustively evaluate the Cartesian product of ``grid`` values.

    ``grid`` maps parameter names to iterables of candidate values.  Returns the
    best ``(DesignParameters, objective_value)`` found.  Intended for a handful
    of parameters with modest resolution; the model is cheap but the product
    still grows multiplicatively.
    """
    names = list(grid)
    best_params: DesignParameters | None = None
    best_val = math.inf
    for combo in product(*(list(grid[n]) for n in names)):
        candidate = replace(base, **dict(zip(names, combo)))
        val = objective(candidate)
        if val < best_val:
            best_val, best_params = val, candidate
    assert best_params is not None
    return best_params, best_val


# --- Power-constrained flow maximization -------------------------------------
#
# Design goal: maximize distillate flow for a fixed total power budget
# (fan + auxiliary heating), within size and material limits.

#: Candidate wall materials: thermal conductivity (W/m K) and a representative
#: minimum wall thickness (m).  Aluminum and copper conduct far better but
#: corrode in greywater; stainless steel and titanium are the durable choices.
WALL_MATERIALS = {
    "stainless_steel": {"wall_conductivity": 16.0, "wall_thickness": 0.0008},
    "titanium":        {"wall_conductivity": 22.0, "wall_thickness": 0.0008},
    "aluminum":        {"wall_conductivity": 205.0, "wall_thickness": 0.0010},
    "copper":          {"wall_conductivity": 385.0, "wall_thickness": 0.0008},
}

#: Default box constraints for the flow-maximization search.  ``plate_area`` is a
#: synthetic variable that sets the wall, evaporation and condensation areas
#: together (one plate, two wetted faces).  ``fan_volumetric_flow`` is the
#: blower's operating flow; the compression lift is NOT a free variable -- it is
#: derived from the blower curve and the power (flow · Δp = η · blower power), so
#: choosing the flow chooses the flow/lift split along the power budget.  All
#: express a "small unit".
DEFAULT_BOUNDS = {
    "plate_area": (0.05, 0.60),               # m^2   (size limit)
    "fan_volumetric_flow": (0.003, 0.05),     # m^3/s (blower operating flow)
    "evaporator_temp_C": (35.0, 80.0),        # C     (cold-side kept below 80 C)
    "channel_gap": (0.005, 0.040),            # m     (size limit)
    "channel_length": (0.20, 1.00),           # m     (size limit)
    "insulation_ua": (0.03, 0.50),            # W/K   (better insulation costs size)
}


def _expand(overrides: dict) -> dict:
    """Expand the synthetic ``plate_area`` into the three surface areas."""
    out = dict(overrides)
    if "plate_area" in out:
        area = out.pop("plate_area")
        out["hx_area"] = area
        out["evap_area"] = area
        out["condenser_area"] = area
    return out


def fan_flow_for_power_budget(base: DesignParameters, budget_w: float,
                              lo: float = 1e-4, hi: float = 0.3,
                              iters: int = 20) -> float:
    """Fan volumetric flow (m^3/s) whose total power draw equals ``budget_w``.

    Total power (fan + auxiliary heating) rises monotonically with, and is very
    nearly linear in, fan flow, so bracketed false-position converges in a
    handful of solves.  If even the minimum flow overspends (loss-dominated)
    returns ``lo``; if the maximum flow still underspends returns ``hi`` (the
    budget is not the binding constraint).
    """
    from .masstransfer import solve_evaporative

    def excess(vf: float) -> float:
        return solve_evaporative(
            replace(base, fan_volumetric_flow=vf, transfer_from_flow=True)
        ).total_energy_input - budget_w

    f_lo = excess(lo)
    if f_lo >= 0.0:
        return lo
    f_hi = excess(hi)
    if f_hi <= 0.0:
        return hi
    mid = hi
    for _ in range(iters):
        mid = lo - f_lo * (hi - lo) / (f_hi - f_lo)   # false position
        f_mid = excess(mid)
        if abs(f_mid) < 1e-3 * budget_w:
            break
        if f_mid < 0.0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return mid


def _pattern_search(score, x0, bounds, rel_tol: float = 2e-3,
                    max_iter: int = 200):
    """Maximize ``score(x)`` over box ``bounds`` by Hooke-Jeeves pattern search.

    ``score`` returns a float (``-inf`` for infeasible/failed points).  Pure,
    derivative-free, and deterministic.
    """
    n = len(bounds)

    def clamp(x):
        return [min(max(x[i], bounds[i][0]), bounds[i][1]) for i in range(n)]

    x = clamp(list(x0))
    fx = score(x)
    step = [(bounds[i][1] - bounds[i][0]) * 0.25 for i in range(n)]
    spans = [(bounds[i][1] - bounds[i][0]) or 1.0 for i in range(n)]

    for _ in range(max_iter):
        if max(step[i] / spans[i] for i in range(n)) < rel_tol:
            break
        improved = False
        for i in range(n):
            for direction in (+1.0, -1.0):
                trial = list(x)
                trial[i] = min(max(x[i] + direction * step[i],
                                   bounds[i][0]), bounds[i][1])
                ft = score(trial)
                if ft > fx:
                    x, fx = trial, ft
                    improved = True
                    break
        if not improved:
            step = [s * 0.5 for s in step]
    return x, fx


def _blower_compression(flow_m3s: float, blower_power_w: float,
                        gas_density: float) -> tuple[float, float]:
    """Compression pressure (Pa) and efficiency for best-efficiency operation.

    At the blower's best-efficiency point ``flow · Δp = η · power``.  The
    efficiency comes from the duty's specific speed (a couple of fixed-point
    iterations, since η barely moves in this low-specific-speed regime).
    """
    eff = 0.48
    dp = eff * blower_power_w / flow_m3s
    for _ in range(3):
        ns = fan.specific_speed(flow_m3s, dp, gas_density, 3000.0)
        eff = fan.achievable_efficiency(ns)
        dp = eff * blower_power_w / flow_m3s
    return dp, eff


def solve_at_budget(cand: DesignParameters, budget_w: float):
    """Solve a candidate at ``budget_w`` total power via the blower coupling.

    The operating flow is ``cand.fan_volumetric_flow``; the compression lift is
    derived from the blower curve and the power (not a free input).  The blower
    power is ``budget − auxiliary heating``, resolved by a short fixed-point
    iteration (auxiliary heating is usually zero at good designs).  Returns
    ``(EvaporativeResults, lift_K, efficiency, blower_power_w)``.
    """
    from .masstransfer import solve_evaporative, lift_from_compression_pressure

    t = cand.evaporator_temp_C
    p_ncg = cand.noncondensable_pressure
    rho = props.vapor_density(t, props.sat_pressure(t) + p_ncg)
    flow = cand.fan_volumetric_flow

    blower_power = budget_w
    result = lift = eff = None
    for _ in range(5):
        dp, eff = _blower_compression(flow, blower_power, rho)
        lift = lift_from_compression_pressure(dp, t, p_ncg)
        result = solve_evaporative(replace(cand, temp_lift=max(lift, 0.05),
                                           transfer_from_flow=True))
        aux = max(result.makeup_heat, 0.0)
        new_power = max(budget_w - aux, 1.0)
        if abs(new_power - blower_power) < 0.5:
            blower_power = new_power
            break
        blower_power = new_power
    return result, lift, eff, blower_power


def maximize_flow(base: DesignParameters, budget_w: float = 600.0,
                  bounds: dict | None = None,
                  materials: dict | None = None):
    """Maximize distillate flow (L/h) under a total power budget and limits.

    The optimizer searches geometry (areas, channel), operating temperature,
    insulation, the wall material, and the **blower operating flow**; the
    compression lift is derived from the blower curve and the power, so choosing
    the flow chooses the flow/lift split along the power budget (fan + auxiliary
    heating).  Pattern search with deterministic restarts guards against local
    optima.

    Returns ``(best_params, best_results, info)`` where ``info`` records the
    winning material, achieved production/power, and the blower operating point.
    """
    bounds = bounds or DEFAULT_BOUNDS
    materials = materials or WALL_MATERIALS
    names = list(bounds)
    box = [bounds[k] for k in names]

    best = {"flow": -math.inf}
    restarts = (0.35, 0.65)

    for mat_name, mat in materials.items():
        mat_base = replace(base, transfer_from_flow=True, **mat)

        def build(vec):
            return replace(mat_base, **_expand(dict(zip(names, vec))))

        def score(vec):
            try:
                r, lift, eff, power = solve_at_budget(build(vec), budget_w)
            except (ValueError, ZeroDivisionError):
                return -math.inf
            if r.total_energy_input > budget_w * 1.05 or r.distillate_rate <= 0:
                return -math.inf
            return r.distillate_lph

        for frac in restarts:
            x0 = [lo + frac * (hi - lo) for (lo, hi) in box]
            x_best, flow = _pattern_search(score, x0, box)
            if flow > best["flow"]:
                cand = build(x_best)
                r, lift, eff, power = solve_at_budget(cand, budget_w)
                best = {
                    "flow": flow, "material": mat_name,
                    "params": replace(cand, temp_lift=lift), "result": r,
                    "lift": lift, "efficiency": eff, "blower_power": power,
                }

    result = best.get("result")
    aux = max(result.makeup_heat, 0.0) if result else 0.0
    info = {
        "material": best.get("material"),
        "distillate_lph": best["flow"],
        # Power per the fan curve (authoritative): blower draw + auxiliary heat.
        "blower_power_w": best.get("blower_power"),
        "auxiliary_heat_w": aux,
        "total_power_w": (best.get("blower_power") or 0.0) + aux,
        # The model's own (isentropic) compression estimate, a cross-check.
        "model_fan_power_w": result.fan_power if result else None,
        "lift_K": best.get("lift"),
        "efficiency": best.get("efficiency"),
        "fan_volumetric_flow": best["params"].fan_volumetric_flow if best.get("params") else None,
        "budget_w": budget_w,
    }
    return best.get("params"), result, info


def budget_constrained_flow(base: DesignParameters, budget_w: float) -> float:
    """Distillate flow (L/h) with the fan flow set to spend ``budget_w``."""
    from .masstransfer import solve_evaporative
    vf = fan_flow_for_power_budget(base, budget_w)
    return solve_evaporative(replace(base, fan_volumetric_flow=vf,
                                     transfer_from_flow=True)).distillate_lph


def flow_sensitivity(base: DesignParameters, param_names, budget_w: float = 600.0,
                     rel: float = 0.1) -> dict:
    """Elasticity of budget-constrained flow to each parameter.

    Returns ``{name: d ln(flow) / d ln(param)}`` by central difference at fixed
    power budget (the fan flow is re-set to the budget at every perturbation).
    An elasticity of 1.0 means "10 % more of this gives 10 % more flow"; the
    ranking says where extra geometry/hardware actually pays off.
    """
    f0 = budget_constrained_flow(base, budget_w)
    out = {}
    for name in param_names:
        v = getattr(base, name)
        up = budget_constrained_flow(replace(base, **{name: v * (1 + rel)}), budget_w)
        dn = budget_constrained_flow(replace(base, **{name: v * (1 - rel)}), budget_w)
        out[name] = (up - dn) / (2.0 * rel * f0) if f0 > 0 else 0.0
    return out
