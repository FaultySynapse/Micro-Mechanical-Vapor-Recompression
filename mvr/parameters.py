"""Design and operating parameters for a micro-scale MVR greywater still.

A single :class:`DesignParameters` dataclass gathers every knob the simulation
understands.  Defaults describe a plausible bench-top unit producing on the
order of a few liters of distillate per hour; they are a starting point for
exploration, not a validated design.

The parameters map onto the physical components the project cares about:

  * fan / compressor .......... ``fan_isentropic_efficiency``, ``fan_motor_efficiency``,
                                ``duct_pressure_drop_pa``
  * evaporation surface ....... ``boiling_htc``, ``fouling_resistance``
  * condensation surface ...... ``condensing_htc``
  * heat-exchange wall ........ ``hx_area``, ``wall_thickness``, ``wall_conductivity``
  * feed heat exchanger ....... ``feed_hx_effectiveness``
  * insulation ................ ``insulation_ua`` (or geometry helpers)
  * cold-side operating point . ``evaporator_temp_C``, ``temp_lift``
  * ambient / feed ............ ``ambient_temp_C``, ``feed_temp_C``,
                                ``recovery_ratio``, ``boiling_point_elevation``
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DesignParameters:
    """Complete parameter set for one MVR operating point.

    All attributes are SI (meters, kelvin/Celsius, watts, pascal, seconds)
    unless the name indicates otherwise.
    """

    # --- Operating point ------------------------------------------------------
    #: Cold-side nominal (evaporator) boiling temperature, C.  Sets the
    #: operating pressure; running under partial vacuum lowers this.
    evaporator_temp_C: float = 100.0
    #: Temperature lift the compressor sustains across the HX wall, K.  This is
    #: the heat-transfer driving temperature difference and a prime optimization
    #: variable: larger lift -> more throughput but more fan power per kg.
    temp_lift: float = 5.0

    # --- Fan / vapor compressor ----------------------------------------------
    #: Isentropic (aerodynamic) efficiency of the vapor mover, 0-1.
    fan_isentropic_efficiency: float = 0.55
    #: Electrical-to-shaft motor+drive efficiency of the fan, 0-1.
    fan_motor_efficiency: float = 0.85
    #: Extra pressure drop the fan must overcome (ducting, demister), Pa.
    duct_pressure_drop_pa: float = 300.0

    # --- Heat-exchange wall (evaporation <-> condensation surface) -----------
    #: Active area of the plate separating the two chambers, m^2.
    hx_area: float = 0.30
    #: Boiling-side film heat-transfer coefficient, W/(m^2 K).
    boiling_htc: float = 5000.0
    #: Condensing-side film heat-transfer coefficient, W/(m^2 K).
    condensing_htc: float = 8000.0
    #: Wall thickness, m.
    wall_thickness: float = 0.0008
    #: Wall thermal conductivity, W/(m K) (e.g. ~16 for stainless, ~200 for Al).
    wall_conductivity: float = 16.0
    #: Additional fouling/scale resistance on the boiling side, m^2 K/W.
    fouling_resistance: float = 0.0001

    # --- Feed heat exchanger (economizer) ------------------------------------
    #: Effectiveness of the feed pre-heater recovering heat from the hot
    #: distillate and concentrate streams, 0-1.
    feed_hx_effectiveness: float = 0.80

    # --- Insulation -----------------------------------------------------------
    #: Overall heat-loss conductance of the insulated shell, W/K
    #: (= exposed area / thermal resistance).  See :meth:`insulation_ua_from_geometry`.
    insulation_ua: float = 0.25

    # --- Ambient, feed, water chemistry --------------------------------------
    #: Temperature of surroundings for insulation loss, C.
    ambient_temp_C: float = 20.0
    #: Incoming greywater temperature, C (defaults to ambient if left None-like).
    feed_temp_C: float = 20.0
    #: Fraction of feed that leaves as distillate (rest is concentrate), 0-1.
    recovery_ratio: float = 0.80
    #: Boiling-point elevation from dissolved solids, K.  A pure thermodynamic
    #: penalty the compressor must overcome that does no useful heat transfer.
    boiling_point_elevation: float = 0.3

    # -------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` on physically impossible inputs."""
        checks = {
            "temp_lift": self.temp_lift > 0,
            "hx_area": self.hx_area > 0,
            "boiling_htc": self.boiling_htc > 0,
            "condensing_htc": self.condensing_htc > 0,
            "wall_thickness": self.wall_thickness > 0,
            "wall_conductivity": self.wall_conductivity > 0,
            "fouling_resistance": self.fouling_resistance >= 0,
            "insulation_ua": self.insulation_ua >= 0,
            "boiling_point_elevation": self.boiling_point_elevation >= 0,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"invalid (non-positive) parameters: {', '.join(bad)}")
        for name in ("fan_isentropic_efficiency", "fan_motor_efficiency",
                     "feed_hx_effectiveness", "recovery_ratio"):
            val = getattr(self, name)
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]; got {val}")

    @staticmethod
    def insulation_ua_from_geometry(surface_area_m2: float,
                                    insulation_thickness_m: float,
                                    insulation_conductivity: float = 0.035) -> float:
        """Convenience: conductance (W/K) for a uniform insulation blanket.

        ``insulation_conductivity`` defaults to ~0.035 W/(m K), typical of rigid
        foam or mineral wool.  Convective/radiative outer film is neglected
        (conservative -> slightly overestimates loss).
        """
        return surface_area_m2 * insulation_conductivity / insulation_thickness_m
