"""Micro Mechanical Vapor Recompression (MVR) greywater still — modeling toolkit.

Public API::

    from mvr import DesignParameters, solve, report

    params = DesignParameters(temp_lift=5.0, hx_area=0.3)
    result = solve(params)
    print(report(params, result))
"""

from __future__ import annotations

from .parameters import DesignParameters
from .model import Results, overall_U, report, solve, stream_metrics
from .masstransfer import (
    EvaporativeResults,
    report_evaporative,
    solve_evaporative,
    mass_transfer_coeff_from_htc,
)
from . import properties

__all__ = [
    "DesignParameters",
    # boiling / heat-transfer-limited model
    "Results",
    "solve",
    "report",
    "overall_U",
    "stream_metrics",
    # evaporative / mass-transfer-limited model
    "EvaporativeResults",
    "solve_evaporative",
    "report_evaporative",
    "mass_transfer_coeff_from_htc",
    "properties",
]

__version__ = "0.1.0"
