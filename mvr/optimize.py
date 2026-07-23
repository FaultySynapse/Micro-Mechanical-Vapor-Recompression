"""Lightweight design optimization for the MVR still.

The physics model in :mod:`mvr.model` is cheap to evaluate, so we can optimize
by direct search without any third-party solver.  Two helpers are provided:

  * :func:`minimize_scalar` -- golden-section search over one parameter.
  * :func:`grid_search`     -- brute-force scan over several parameters.

Both take an *objective* callable ``(DesignParameters) -> float`` (smaller is
better) so callers can trade off specific energy, production, capital proxies,
etc.  :func:`specific_energy_objective` is a ready-made objective that minimizes
kWh/L while penalizing designs that fall below a minimum production target.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from typing import Callable, Iterable

from .parameters import DesignParameters
from .model import solve

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
