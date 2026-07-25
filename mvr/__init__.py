"""Micro Mechanical Vapor Recompression (MVR) greywater still — modeling toolkit.

Public API::

    from mvr import DesignParameters, solve, report

    params = DesignParameters(temp_lift=5.0, hx_area=0.3)
    result = solve(params)
    print(report(params, result))
"""

from __future__ import annotations

from .parameters import DesignParameters
from .model import (
    Results,
    overall_U,
    link_cross_section_for_duty,
    report,
    solve,
    stream_metrics,
)
from .optimize import (
    maximize_flow,
    solve_at_budget,
    fan_flow_for_power_budget,
    budget_constrained_flow,
    flow_sensitivity,
    WALL_MATERIALS,
    DEFAULT_BOUNDS,
)
from .masstransfer import (
    EvaporativeResults,
    report_evaporative,
    solve_evaporative,
    solve_with_bleed,
    solve_at_speed,
    solve_at_power,
    balance_temperature_by_economizer,
    ThermalControl,
    vessel_pressure_spec,
    VesselSpec,
    lift_from_compression_pressure,
    mass_transfer_coeff_from_htc,
    htc_from_mass_transfer_coeff,
)
from . import properties
from . import transport
from . import ncg
from . import fan
from . import packing

__all__ = [
    "DesignParameters",
    # boiling / heat-transfer-limited model
    "Results",
    "solve",
    "report",
    "overall_U",
    "link_cross_section_for_duty",
    "stream_metrics",
    # evaporative / mass-transfer-limited model
    "EvaporativeResults",
    "solve_evaporative",
    "solve_with_bleed",
    "solve_at_speed",
    "solve_at_power",
    "balance_temperature_by_economizer",
    "ThermalControl",
    "vessel_pressure_spec",
    "VesselSpec",
    "lift_from_compression_pressure",
    "report_evaporative",
    "mass_transfer_coeff_from_htc",
    "htc_from_mass_transfer_coeff",
    # design optimization
    "maximize_flow",
    "solve_at_budget",
    "fan_flow_for_power_budget",
    "budget_constrained_flow",
    "flow_sensitivity",
    "WALL_MATERIALS",
    "DEFAULT_BOUNDS",
    "properties",
    "transport",
    "ncg",
    "fan",
    "packing",
]

__version__ = "0.1.0"
