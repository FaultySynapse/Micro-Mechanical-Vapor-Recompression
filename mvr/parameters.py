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

    # --- Evaporation / condensation surfaces (mass-transfer model) -----------
    # Used only by the evaporative (sub-boiling, fan-swept) model. In that
    # regime water evaporates from a free surface and condenses through the
    # vapor space, so the phase-change *areas* are distinct design variables and
    # transport can be limited by mass transfer rather than wall heat flow.
    #: Free liquid-surface area for evaporation, m^2 (defaults to the wall area).
    evap_area: float = 0.30
    #: Condensation surface area, m^2 (defaults to the wall area).
    condenser_area: float = 0.30
    #: Gas-side mass-transfer coefficient over the evaporator surface, m/s.
    #: Used only as a *fallback* when ``transfer_from_flow`` is False; otherwise
    #: it is computed from the fan-driven surface velocity and channel geometry.
    evap_mass_transfer_coeff: float = 0.020
    #: Gas-side mass-transfer coefficient over the condensation surface, m/s
    #: (fallback; see ``transfer_from_flow``).
    condenser_mass_transfer_coeff: float = 0.020

    # --- Fan-driven transport (evaporative model) ----------------------------
    # When enabled, the fan circulates gas over the surfaces; that sweep velocity
    # sets the boundary-layer transfer coefficients (via flat-plate Sh/Nu
    # correlations) AND the fan power scales with the circulated flow -- so more
    # sweep buys better transfer/more production but costs more fan work.
    #: If True, compute the mass/heat-transfer coefficients from the fan flow and
    #: channel geometry instead of using the fixed coefficients above.
    transfer_from_flow: bool = True
    #: Volumetric gas flow the fan circulates over the surfaces, m^3/s
    #: (at evaporator inlet conditions). The key fan-sizing design variable.
    fan_volumetric_flow: float = 0.010
    #: Flow-direction length of each surface / channel, m (the boundary-layer
    #: development length). Surface width is implied as area / length.
    channel_length: float = 0.5
    #: Vapor-space channel gap above each surface, m (sets the flow cross-section
    #: width = area/length, height = gap, hence the sweep velocity).
    channel_gap: float = 0.02
    #: Partial pressure of non-condensable gas (dissolved air/CO2 flashed from
    #: the greywater), Pa, referenced to the phase-change *surface* (where the
    #: water vapor is saturated): the chamber total pressure is
    #: ``P_sat(T_surface) + noncondensable_pressure``. By Dalton's law the total
    #: is uniform but the composition is not -- in the bulk, mass transfer has
    #: depleted the water vapor, so the NCG partial there is *higher* than this
    #: surface value. The NCG blankets the condenser and throttles both surfaces;
    #: a vent/purge keeps it low, and 0 recovers the pure-vapor, heat-transfer-
    #: limited (boiling) result.
    noncondensable_pressure: float = 500.0

    # --- Gas-phase sensible heat transfer (evaporative model) ----------------
    # The fan delivers vapor *superheated* (compression heats it, more so at low
    # fan efficiency); shedding that superheat at the condenser needs gas-phase
    # heat transfer, which -- like the mass transfer -- is not instantaneous.
    #: Gas-phase sensible heat-transfer coefficient at the condenser, W/(m^2 K).
    #: 0 => derive it from ``condenser_mass_transfer_coeff`` via the Chilton-
    #: Colburn (Lewis) analogy, so the same gas film governs heat and mass.
    condenser_gas_htc: float = 0.0
    #: Vapor-space gas density used for the heat<->mass analogy, kg/m^3
    #: (low because the space runs at reduced pressure).
    gas_density: float = 0.1
    #: Vapor-space gas specific heat used for the analogy and for desuperheating,
    #: J/(kg K) (~superheated steam).
    gas_specific_heat: float = 1900.0
    #: Lewis number (Sc/Pr) of the water-vapor/air gas mixture, for the analogy.
    lewis_number: float = 0.85

    # --- Non-condensable bleed valve -----------------------------------------
    # A bleed valve holds ``noncondensable_pressure`` by venting gas; the vented
    # gas is mostly steam, so it costs product and pump work unless recovered.
    #: Dissolved non-condensable gas in the feed, mol/L (air-saturated ~0.8 mmol/L;
    #: CO2-rich greywater several times that).
    dissolved_gas_mol_per_l: float = 0.00082
    #: Fraction of dissolved gas that flashes off at operating conditions.
    gas_release_fraction: float = 1.0
    #: Route the bleed through a feed-cooled condenser to recover most of its
    #: steam (back to product) and latent heat (to the feed), venting only the
    #: residual non-condensables.
    bleed_to_feed_condenser: bool = False
    #: Temperature approach of the feed-cooled bleed condenser, K.
    feed_condenser_approach_C: float = 5.0
    #: Efficiency of the vacuum pump that vents the bleed (when below ambient).
    purge_pump_efficiency: float = 0.30

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
            "evap_area": self.evap_area > 0,
            "condenser_area": self.condenser_area > 0,
            "evap_mass_transfer_coeff": self.evap_mass_transfer_coeff > 0,
            "condenser_mass_transfer_coeff": self.condenser_mass_transfer_coeff > 0,
            "noncondensable_pressure": self.noncondensable_pressure >= 0,
            "condenser_gas_htc": self.condenser_gas_htc >= 0,
            "gas_density": self.gas_density > 0,
            "gas_specific_heat": self.gas_specific_heat > 0,
            "lewis_number": self.lewis_number > 0,
            "fan_volumetric_flow": self.fan_volumetric_flow > 0,
            "channel_length": self.channel_length > 0,
            "channel_gap": self.channel_gap > 0,
            "dissolved_gas_mol_per_l": self.dissolved_gas_mol_per_l >= 0,
            "gas_release_fraction": self.gas_release_fraction >= 0,
            "feed_condenser_approach_C": self.feed_condenser_approach_C >= 0,
            "purge_pump_efficiency": self.purge_pump_efficiency > 0,
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
