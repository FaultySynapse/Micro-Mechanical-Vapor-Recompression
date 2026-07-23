"""Dimensionless fan/blower characterization for the vapor mover.

A turbomachine's behavior collapses onto dimensionless groups, so the *duty*
(volumetric flow Q and pressure rise Δp) alone tells you what class of machine
fits and what efficiency is realistically achievable -- independent of size or
speed within Reynolds-independence.

Dimensionless groups used here (SI, ω in rad/s):

    specific speed     Ns = ω · Q^0.5 / (Δp/ρ)^0.75
    specific diameter  Ds = D · (Δp/ρ)^0.25 / Q^0.5
    flow coefficient   φ  = Q / (ω D^3)
    pressure coeff.    ψ  = Δp / (ρ ω^2 D^2)
    fan power          P  = Q · Δp / η

Specific speed is the key selector: low Ns (high pressure, low flow) suits
positive-displacement / regenerative blowers at modest efficiency; mid Ns suits
centrifugals; high Ns suits axials.  A small MVR still runs a *high pressure
ratio at low flow*, i.e. very low Ns -- genuinely blower/compressor duty, not a
"fan" -- so the achievable efficiency is modest (~0.4-0.5), which this module
makes explicit.

Pure standard library.
"""

from __future__ import annotations

import math


def specific_speed(flow_m3s: float, pressure_rise_pa: float, gas_density: float,
                   speed_rpm: float) -> float:
    """Dimensionless specific speed ``Ns = ω Q^0.5 / (Δp/ρ)^0.75``."""
    omega = speed_rpm * 2.0 * math.pi / 60.0
    head = pressure_rise_pa / gas_density                     # specific work, J/kg
    return omega * flow_m3s ** 0.5 / head ** 0.75


def specific_diameter(diameter_m: float, flow_m3s: float, pressure_rise_pa: float,
                      gas_density: float) -> float:
    """Dimensionless specific diameter ``Ds = D (Δp/ρ)^0.25 / Q^0.5``."""
    head = pressure_rise_pa / gas_density
    return diameter_m * head ** 0.25 / flow_m3s ** 0.5


def recommended_machine(specific_speed_value: float) -> str:
    """Machine class best matched to a dimensionless specific speed."""
    ns = specific_speed_value
    if ns < 0.3:
        return "positive-displacement / regenerative blower"
    if ns < 1.0:
        return "centrifugal (backward-curved)"
    if ns < 3.0:
        return "mixed-flow"
    return "axial"


def achievable_efficiency(specific_speed_value: float) -> float:
    """Realistic best *overall* efficiency (aerodynamic × mechanical) for a duty.

    A log-normal "turbomachine potential" peaking near Ns~1.5 (η~0.86), with a
    positive-displacement floor (~0.48) that dominates at the low Ns of a
    high-pressure/low-flow blower duty.  Being an *overall* efficiency, ~0.48 at
    low Ns corresponds to ~0.55 aerodynamic × ~0.85 motor -- i.e. it validates,
    rather than contradicts, the model's default efficiencies.
    """
    ns = max(specific_speed_value, 1e-6)
    turbo = 0.86 * math.exp(-(math.log(ns / 1.5)) ** 2 / (2.0 * 0.9 ** 2))
    pd_floor = 0.48 if ns < 0.6 else 0.0
    return max(turbo, pd_floor)


# --- A representative dimensionless characteristic (for off-design) -----------
# Backward-curved blower: falling head with flow, efficiency peaking at a
# best-efficiency flow coefficient.  Normalized so phi/phi_bep = 1 is the BEP.

def head_coefficient(phi_ratio: float) -> float:
    """Normalized pressure-coefficient curve ``ψ/ψ_bep`` vs ``φ/φ_bep``.

    Roughly ``ψ = ψ0 (1 - a φ^2)``; returns the fraction of shut-off head.
    """
    return max(1.15 - 0.15 * phi_ratio - 0.40 * phi_ratio ** 2, 0.0)


def efficiency_offdesign(phi_ratio: float, peak_efficiency: float) -> float:
    """Efficiency at an off-design flow, as a fraction of the peak.

    Parabolic drop-off away from the best-efficiency point (φ/φ_bep = 1).
    """
    frac = max(1.0 - 0.9 * (phi_ratio - 1.0) ** 2, 0.0)
    return peak_efficiency * frac


def characterize_duty(flow_m3s: float, pressure_rise_pa: float,
                      gas_density: float, speed_rpm: float = 3000.0,
                      diameter_m: float | None = None) -> dict:
    """Summarize the fan/blower duty: specific speed, class, efficiency, power.

    ``diameter_m`` is optional; if given, the specific diameter is reported too.
    The returned ``efficiency`` is the achievable best efficiency for the duty
    and ``power_w`` the shaft power ``Q Δp / η``.
    """
    ns = specific_speed(flow_m3s, pressure_rise_pa, gas_density, speed_rpm)
    eff = achievable_efficiency(ns)
    # Q*Δp is the (incompressible) ideal work rate; /overall efficiency gives an
    # electrical-power estimate.  The main model uses a compressible isentropic
    # work form, so treat this as an independent ballpark cross-check.
    out = {
        "specific_speed": ns,
        "machine": recommended_machine(ns),
        "overall_efficiency": eff,
        "electrical_power_w": flow_m3s * pressure_rise_pa / eff if eff > 0 else float("inf"),
        "pressure_rise_pa": pressure_rise_pa,
    }
    if diameter_m is not None:
        out["specific_diameter"] = specific_diameter(diameter_m, flow_m3s,
                                                     pressure_rise_pa, gas_density)
    return out
