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
from dataclasses import dataclass


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


def efficiency_headroom(flow_m3s: float, pressure_rise_pa: float,
                        gas_density: float, assumed_overall_eff: float,
                        speed_rpm_band=(3_000.0, 30_000.0),
                        samples: int = 7) -> dict:
    """Is a fan-delivery-bound design leaving efficiency on the table?

    When the fan-delivery ceiling binds, "add fan power" is trivially true and
    unhelpful.  The useful question is whether a *better-matched blower* could
    deliver more air power (Q·Δp) for the same electrical watts -- i.e. whether
    the assumed efficiency is below the best achievable at this duty.

    Distillate at a fixed power budget tracks the delivered air power
    ``eff * P_electrical``, so the achievable relative gain from re-matching is
    just ``headroom / assumed_eff``.  Specific speed scales with shaft speed, so
    the best achievable efficiency is taken as the max over a plausible speed
    band (a micro-MVR's high-pressure/low-flow duty stays low-Ns -- PD/blower
    territory -- across any sane speed, so the verdict is robust to the exact
    rpm).  ``headroom <= ~0`` means the design is already at the efficiency
    ceiling: the delivery limit is a genuine power/duty limit, not a fan-choice
    one.
    """
    lo, hi = speed_rpm_band
    best_ns = 0.0
    best_eff = 0.0
    for i in range(samples):
        rpm = lo * (hi / lo) ** (i / (samples - 1)) if samples > 1 else lo
        ns = specific_speed(flow_m3s, pressure_rise_pa, gas_density, rpm)
        eff = achievable_efficiency(ns)
        if eff > best_eff:
            best_eff, best_ns = eff, ns
    headroom = best_eff - assumed_overall_eff
    relative_gain = headroom / assumed_overall_eff if assumed_overall_eff > 0 else 0.0
    return {
        "specific_speed": best_ns,
        "machine": recommended_machine(best_ns),
        "assumed_overall_efficiency": assumed_overall_eff,
        "achievable_overall_efficiency": best_eff,
        "headroom": headroom,
        "relative_gain": relative_gain,
        # A sub-5% re-match gain is negligible next to the power lever (output
        # tracks eff*power identically), so treat it as "at the ceiling".
        "at_efficiency_ceiling": relative_gain <= 0.05,
    }


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


# --- A selectable blower with a broad operating band -------------------------


@dataclass
class BlowerCurve:
    """A blower characterized by its best-efficiency (design) point and a broad
    dimensionless shape, usable across a range of flows and (via speed) duties.

    At a fixed speed the head falls roughly linearly with flow from a shut-off
    value to zero, and the efficiency is a *broad* hump about the design flow
    (regenerative/PD blowers are forgiving) -- so the design is not pinned to a
    narrow operating band.  Changing speed scales the whole curve by the affinity
    laws (Q ~ N, Δp ~ N^2, P ~ N^3) while preserving efficiency.
    """

    design_flow: float          # m^3/s at the best-efficiency point
    design_pressure: float      # Pa pressure rise at the BEP
    peak_efficiency: float = 0.48   # overall (aero x motor)
    shutoff_ratio: float = 1.60     # head at zero flow / design head
    efficiency_breadth: float = 0.5  # smaller = broader plateau

    @property
    def max_flow(self) -> float:
        """Free-delivery flow where the head falls to zero (fixed speed)."""
        # Linear head line through (Q_design, 1) with intercept shutoff_ratio.
        return self.design_flow * self.shutoff_ratio / (self.shutoff_ratio - 1.0)

    def pressure(self, flow_m3s: float) -> float:
        """Delivered pressure rise (Pa) at ``flow_m3s`` (fixed speed)."""
        q = flow_m3s / self.design_flow
        frac = self.shutoff_ratio - (self.shutoff_ratio - 1.0) * q
        return max(frac, 0.0) * self.design_pressure

    def efficiency(self, flow_m3s: float) -> float:
        """Overall efficiency at ``flow_m3s`` -- a broad hump about the BEP."""
        q = flow_m3s / self.design_flow
        return self.peak_efficiency * max(1.0 - self.efficiency_breadth * (q - 1.0) ** 2, 0.05)

    def power(self, flow_m3s: float) -> float:
        """Electrical power (W) at ``flow_m3s`` = Q·Δp / efficiency."""
        eff = self.efficiency(flow_m3s)
        return flow_m3s * self.pressure(flow_m3s) / eff if eff > 0 else float("inf")

    def at_speed(self, speed_ratio: float) -> "BlowerCurve":
        """A new curve for the same blower run at ``speed_ratio`` × design speed
        (affinity laws: Q ~ N, Δp ~ N^2; efficiency preserved)."""
        return BlowerCurve(
            design_flow=self.design_flow * speed_ratio,
            design_pressure=self.design_pressure * speed_ratio ** 2,
            peak_efficiency=self.peak_efficiency,
            shutoff_ratio=self.shutoff_ratio,
            efficiency_breadth=self.efficiency_breadth,
        )

    def operating_point(self, compression_pressure_pa: float,
                        duct_loss_coeff: float = 0.0) -> dict:
        """Where this blower settles against a system needing
        ``compression_pressure_pa`` (≈ flow-independent) plus a duct loss
        ``duct_loss_coeff · Q^2``.

        Returns the operating flow, pressure, efficiency and power, or a
        ``feasible=False`` flag if the blower cannot even reach the compression
        pressure at shut-off.
        """
        if self.pressure(0.0) <= compression_pressure_pa:
            return {"feasible": False, "flow_m3s": 0.0,
                    "pressure_pa": self.pressure(0.0), "efficiency": 0.0,
                    "power_w": float("inf")}

        def excess(q):
            return self.pressure(q) - (compression_pressure_pa + duct_loss_coeff * q * q)

        lo, hi = 0.0, self.max_flow
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if excess(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        q_op = 0.5 * (lo + hi)
        return {
            "feasible": True,
            "flow_m3s": q_op,
            "pressure_pa": self.pressure(q_op),
            "efficiency": self.efficiency(q_op),
            "power_w": self.power(q_op),
        }


def size_blower_for_duty(flow_m3s: float, pressure_rise_pa: float,
                         gas_density: float, speed_rpm: float = 3000.0) -> BlowerCurve:
    """Build a :class:`BlowerCurve` whose best-efficiency point is the given duty,
    with the achievable efficiency implied by the duty's specific speed."""
    ns = specific_speed(flow_m3s, pressure_rise_pa, gas_density, speed_rpm)
    return BlowerCurve(design_flow=flow_m3s, design_pressure=pressure_rise_pa,
                       peak_efficiency=achievable_efficiency(ns))
