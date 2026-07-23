"""Micro Mechanical Vapor Recompression (MVR) greywater still — modeling toolkit.

Public API::

    from mvr import DesignParameters, solve, report

    params = DesignParameters(temp_lift=5.0, hx_area=0.3)
    result = solve(params)
    print(report(params, result))
"""

from __future__ import annotations

from .parameters import DesignParameters
from .model import Results, overall_U, report, solve
from . import properties

__all__ = [
    "DesignParameters",
    "Results",
    "solve",
    "report",
    "overall_U",
    "properties",
]

__version__ = "0.1.0"
