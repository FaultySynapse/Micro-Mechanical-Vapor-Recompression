# Micro Mechanical Vapor Recompression (MVR) — greywater still

A modeling, simulation, and optimization toolkit for a **very small scale**
mechanical-vapor-recompression still that purifies greywater. The goal of the
project is to evaluate candidate designs against physically-grounded performance
metrics (production rate, energy per liter, gained-output-ratio) and eventually
to optimize the design parameters.

> **Status:** v0.1 — a first, deliberately transparent steady-state model. The
> physics is first-principles but uses engineering correlations and a handful of
> lumped assumptions (documented below). It is a foundation to iterate on, not a
> validated design tool. Numbers should be read as *relative* guidance for
> comparing designs, not absolute guarantees.

## The device

Two chambers, each partially filled with water, share a **heat-exchange wall**:

```
        greywater feed
             │
        ┌────▼─────────────────────── feed heat exchanger (economizer) ───┐
        │  preheated by outgoing hot distillate + concentrate             │
        └────┬───────────────────────────────────────────────────────────┘
             │  preheated feed
   ┌─────────▼──────────┐        fan / vapor        ┌────────────────────┐
   │   EVAPORATOR       │  ──── compressor ────►    │   CONDENSER        │
   │   (cold side)      │   raises vapor pressure   │   (hot side)       │
   │   boils at T_evap  │                           │   condenses T_cond │
   │                    │◄═══ heat-exchange wall ═══►│                    │
   │  concentrate out ──┼──► (to economizer)        │── distillate out ──┼──► (to economizer)
   └────────────────────┘                           └────────────────────┘
        insulated shell (heat loss to ambient)
```

The trick that makes MVR efficient: water boils in the evaporator, the fan pulls
that vapor across to the condenser and slightly **raises its pressure**, which
raises its **condensing temperature** above the boiling temperature. The vapor
then condenses on the far side of the shared wall, and the **latent heat it
releases conducts straight back** through the wall to keep the evaporator
boiling. The latent heat is recycled; the fan only pays for the *compression*
work plus whatever the unit leaks. Contaminants stay behind in the concentrate;
the condensate is purified product.

## What the model computes

Given a `DesignParameters` set, `solve()` returns a steady-state operating point:

| Result | Meaning |
| --- | --- |
| `distillate_rate` / `distillate_lph` | purified water production (kg/s and L/h) |
| `overall_U`, `heat_duty` | wall heat-transfer coefficient and duty |
| `fan_power` | electrical power drawn by the vapor mover |
| `makeup_heat` | external heat to close the energy balance (negative ⇒ self-heats / needs cooling) |
| `specific_energy_kwh_per_l` | total electrical-equivalent input per liter of product |
| `gain_output_ratio` (GOR) | latent heat produced ÷ energy input |

## The physics, briefly

**Temperature stack.** With boiling-point elevation (BPE) from dissolved solids
and a heat-transfer lift across the wall:

```
T_evap                     cold-side nominal boiling temp     (design input)
T_boil = T_evap + BPE      actual boiling-liquid temp
T_cond = T_boil + lift     condensing-vapor temp              (drives the wall)
```

Only `lift` does useful heat transfer; the BPE part is a pure thermodynamic
penalty the compressor must still overcome.

**Heat transfer.** Series resistances set the overall coefficient of the shared
wall, and the duty follows the useful lift:

```
1/U   = 1/h_boil + R_fouling + t_wall/k_wall + 1/h_cond
Q     = U · A · lift
ṁ     = Q / h_fg(T_boil)          (distillate rate)
```

**Compression.** The fan raises the vapor from `P_sat(T_evap)` to
`P_sat(T_cond)` (plus duct losses). Work is modeled as ideal-gas isentropic
compression divided by fan efficiency — for these small pressure ratios it is
within a few percent of the incompressible estimate `ΔP / ρ_vapor`.

**Energy balance (whole insulated unit, steady state).** All fan shaft work
dissipates into the vapor loop and helps close the balance:

```
W_shaft + Q_makeup = Q_insulation_loss + Q_unrecovered_streams + Q_feed_heating
```

`Q_makeup` is the external heat still required (or, if negative, surplus the unit
must reject). The **economizer** recovers a fraction `ε` of the outgoing hot
streams' sensible heat into the incoming feed.

**Figures of merit.** `specific_energy = (fan_power + max(makeup,0)) / production`
and `GOR = ṁ·h_fg / total_input`.

See `mvr/model.py` for the fully-commented equations and `mvr/properties.py` for
the water/steam correlations (Antoine saturation pressure, Watson latent heat,
ideal-gas vapor density) with their accuracy ranges.

## Design parameters

Every knob lives in `mvr/parameters.py` (`DesignParameters`), grouped by the
component it belongs to: fan/compressor, evaporation & condensation surfaces, the
heat-exchange wall, the feed economizer, the insulation, the cold-side operating
point, and feed/water chemistry. All have documented defaults describing a
plausible bench-top unit.

## Install & run

The **core model has no dependencies** — just Python ≥ 3.10.

```bash
# optional: editable install so `mvr` is importable everywhere and the CLI works
pip install -e .

# baseline report
python -m mvr.cli

# override any parameter (flag = field name with dashes)
python -m mvr.cli --temp-lift 3 --hx-area 0.5 --feed-hx-effectiveness 0.9

# scan one parameter and print a table
python -m mvr.cli --sweep temp_lift 1 12 12

# worked example: baseline + optimized temperature lift
python examples/baseline.py
```

Optional plotting (needs `matplotlib`):

```bash
pip install -e ".[plots]"
python scripts/sweep_lift.py      # writes scripts/sweep_lift.png
```

## Optimization

`mvr/optimize.py` provides gradient-free search over the cheap model:

```python
from mvr import DesignParameters
from mvr.optimize import minimize_scalar, grid_search, specific_energy_objective

base = DesignParameters()
objective = specific_energy_objective(min_distillate_lph=4.0)

# best temperature lift for lowest kWh/L above a production floor
best_lift, value = minimize_scalar(base, "temp_lift", 0.5, 15.0, objective)

# brute-force over a few parameters
best, value = grid_search(base, {
    "feed_hx_effectiveness": [0.7, 0.85, 0.95],
    "insulation_ua": [0.05, 0.25, 1.0],
}, objective)
```

Objectives are plain `(DesignParameters) -> float` callables (smaller is better),
so you can encode any trade-off you like — energy, production, a capital proxy.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers property correlations against reference steam-table values, and
model invariants: mass balance, energy-balance closure, `Q = ṁ·h_fg`,
series-resistance bounds on `U`, and the expected monotonic responses (more lift
→ more production but more fan power per kg; better insulation/economizer → less
energy).

## Modeling assumptions & limitations (v0.1)

- **Steady state only** — no transient start-up, thermal mass, or level control.
- **Lumped temperatures** — each chamber is a single temperature; no spatial
  gradients along the plate, no boiling/condensing regime maps. `U` is a design
  input built from constant film coefficients, not computed from flow/geometry.
- **Ideal-gas vapor**, single isentropic-exponent compression model.
- **Economizer** modeled by a single effectiveness `ε`; the hot side is a
  mass-weighted blend of distillate and concentrate.
- **BPE** is a fixed input, not computed from greywater composition/recovery.
- **Makeup heat** is counted as full electrical-equivalent input; a real unit
  might source it from low-grade/waste heat, which would lower reported kWh/L.

These are the natural places to deepen the model next (e.g. compute `U` from
plate geometry and boiling/condensation correlations, add concentration-
dependent BPE, add a transient tank model, add capital-cost objectives).

## Project layout

```
mvr/
  properties.py   water/steam thermophysical correlations (pure stdlib)
  parameters.py   DesignParameters dataclass + validation
  model.py        steady-state solver, Results, text report
  optimize.py     golden-section + grid-search optimizers
  cli.py          `python -m mvr.cli` entry point + parameter sweep
examples/
  baseline.py     baseline report + optimized lift
scripts/
  sweep_lift.py   matplotlib trade-off plot (optional dep)
tests/            pytest suite
```

## License

MIT — see `LICENSE`.
