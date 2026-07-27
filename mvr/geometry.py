"""Cylindrical pressure-shell geometry for the 2 bar MVR concept.

The hot design runs at ~2 bar-abs, which wants a *curved* pressure boundary, and
the heat must still cross a thin, full-area wall (the plate's superpower).  Both
are satisfied by a **concentric-tube** arrangement:

    outer shell  ── pressure boundary (holds the gauge pressure to atmosphere)
      annulus    ── condenser: vapor condenses on the OUTER face of the inner
                    tube, condensate drains to the product outlet
      inner tube ── the heat-transfer wall (thin, full area, perpendicular path)
      bore       ── evaporator: feed falls as a film down the INNER face, so
                    holdup is tiny (no pool) -- ideal now that storage is external

Latent heat released in the annulus conducts straight through the inner-tube wall
to drive the falling film in the bore -- the same short, full-area path a flat
plate gives, wrapped into a tube.  The fan lifts vapor from bore to annulus.

Two pressure loads, both carried as efficient hoop stress in tubes:
  * outer shell: gauge pressure (P_cond - atmosphere) vs the outside world
  * inner tube:  the inter-chamber compression rise (P_cond - P_evap)

This module sizes the envelope, holdup and wall thicknesses, and hands the wall
area to the solver.  Pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .parameters import DesignParameters

#: Representative allowable stresses (Pa) for thin-wall pressure design, and a
#: girth/long-seam joint efficiency.  Conservative room-for-derating values for
#: austenitic stainless at MVR temperatures; titanium similar order.
STRUCTURAL_MATERIALS = {
    "stainless_304": 100e6,
    "stainless_316": 115e6,
    "titanium_gr2": 100e6,
}
JOINT_EFFICIENCY = 0.85
#: Practical minimum wall a small vessel can be fabricated/handled at, m.
MIN_FAB_WALL_M = 0.001


@dataclass
class ConcentricCylinder:
    """A concentric-tube MVR chamber pair (falling-film evaporator in the bore,
    condensing annulus, outer pressure shell)."""

    inner_diameter_m: float          # bore diameter (evaporator wall)
    length_m: float                  # active tube length
    annulus_gap_m: float = 0.03      # radial condensing gap
    tube_wall_m: float = 0.001       # inner tube (heat-transfer wall)
    shell_wall_m: float = 0.0015     # outer pressure shell
    sump_depth_m: float = 0.01       # shallow concentrate sump (no storage)

    # --- areas ---------------------------------------------------------------
    def heat_transfer_area(self) -> float:
        """Active wall area (m^2) = inner-tube lateral area."""
        return math.pi * self.inner_diameter_m * self.length_m

    # --- volumes -------------------------------------------------------------
    def bore_volume(self) -> float:
        return math.pi * (self.inner_diameter_m / 2.0) ** 2 * self.length_m

    def annulus_volume(self) -> float:
        r_i = self.inner_diameter_m / 2.0 + self.tube_wall_m
        r_o = r_i + self.annulus_gap_m
        return math.pi * (r_o ** 2 - r_i ** 2) * self.length_m

    def outer_diameter(self) -> float:
        return self.inner_diameter_m + 2.0 * (self.tube_wall_m + self.annulus_gap_m
                                              + self.shell_wall_m)

    def envelope_volume(self) -> float:
        """Total external swept volume (m^3), the number to keep under 0.5."""
        return math.pi * (self.outer_diameter() / 2.0) ** 2 * self.length_m

    def liquid_holdup_l(self) -> float:
        """Working liquid inventory (L): only the thin concentrate sump plus a
        falling-film wetting layer -- storage is external."""
        sump = self.inner_diameter_m * self.length_m * self.sump_depth_m  # ~chord x L x depth
        film = self.heat_transfer_area() * 0.0003                          # ~0.3 mm film
        return (sump + film) * 1000.0


@dataclass
class TubeBundle:
    """A vertical falling-film tube bundle sized to fit inside a hull.

    Each tube is a single-wall heat-transfer surface: feed evaporates as a film
    on the inside, compressed vapor condenses on the outside (shell side).  The
    bundle packs the wall area into a short, wide envelope -- the shape a
    pressure-cooker hull imposes.
    """

    tube_inner_diameter_m: float
    tube_count: int
    active_length_m: float
    tube_wall_m: float = 0.0008
    pitch_ratio: float = 1.35        # center spacing / tube OD (bundle tightness)

    def tube_outer_diameter(self) -> float:
        return self.tube_inner_diameter_m + 2.0 * self.tube_wall_m

    def heat_transfer_area(self) -> float:
        """Total film-side (inner) wall area, m^2."""
        return self.tube_count * math.pi * self.tube_inner_diameter_m * self.active_length_m

    def wetted_perimeter(self) -> float:
        """Total falling-film wetted width (inner circumferences), m."""
        return self.tube_count * math.pi * self.tube_inner_diameter_m

    def bundle_envelope_diameter(self) -> float:
        """Approximate circle (m) enclosing the hex-packed bundle."""
        pitch = self.pitch_ratio * self.tube_outer_diameter()
        footprint = self.tube_count * (math.sqrt(3.0) / 2.0) * pitch ** 2
        return math.sqrt(4.0 * footprint / math.pi)

    def fits_in_shell(self, shell_inner_diameter_m: float) -> bool:
        return self.bundle_envelope_diameter() <= shell_inner_diameter_m

    def metal_fraction(self, shell_inner_diameter_m: float) -> float:
        """Fraction of the shell cross-section occupied by tube metal+bore."""
        tube_area = self.tube_count * math.pi * (self.tube_outer_diameter() / 2.0) ** 2
        return tube_area / (math.pi * (shell_inner_diameter_m / 2.0) ** 2)


def size_tube_count(target_area_m2: float, tube_inner_diameter_m: float,
                    active_length_m: float) -> int:
    """Number of tubes to reach ``target_area_m2`` of film-side wall."""
    per_tube = math.pi * tube_inner_diameter_m * active_length_m
    return max(1, math.ceil(target_area_m2 / per_tube))


def apply_bundle(base: DesignParameters, bundle: TubeBundle) -> DesignParameters:
    """Set the solver's wall/phase-change areas from a tube bundle."""
    a = bundle.heat_transfer_area()
    return replace(base, hx_area=a, evap_area=a, condenser_area=a,
                   wall_thickness=bundle.tube_wall_m)


def hoop_thickness(pressure_pa: float, radius_m: float,
                   allowable_stress_pa: float = 100e6,
                   joint_efficiency: float = JOINT_EFFICIENCY) -> float:
    """Thin-wall hoop thickness (m) for an internal pressure: ``t = P r / (S E)``.

    Returns the stress-required thickness; compare against ``MIN_FAB_WALL_M`` --
    at this scale the fabrication minimum almost always governs, i.e. pressure is
    structurally trivial.
    """
    return pressure_pa * radius_m / (allowable_stress_pa * joint_efficiency)


def size_length_for_area(inner_diameter_m: float, target_area_m2: float) -> float:
    """Tube length (m) giving ``target_area_m2`` of wall at a chosen bore."""
    return target_area_m2 / (math.pi * inner_diameter_m)


def apply_geometry(base: DesignParameters, cyl: ConcentricCylinder) -> DesignParameters:
    """Set the solver's wall/phase-change areas from the cylinder's tube area."""
    a = cyl.heat_transfer_area()
    return replace(base, hx_area=a, evap_area=a, condenser_area=a,
                   wall_thickness=cyl.tube_wall_m)
