# Micro Mechanical Vapor Recompression (MVR) — greywater still

A modeling, simulation, and optimization toolkit for a **very small scale**
mechanical-vapor-recompression still that purifies greywater. The goal of the
project is to evaluate candidate designs against physically-grounded performance
metrics (production rate, energy per liter, gained-output-ratio) and eventually
to optimize the design parameters.

> **Status:** v0.2 — deliberately transparent steady-state models. The physics is
> first-principles but uses engineering correlations and a handful of lumped
> assumptions (documented below). It is a foundation to iterate on, not a
> validated design tool. Numbers should be read as *relative* guidance for
> comparing designs, not absolute guarantees.

## Two operating regimes, two models

The still can run in either of two physically distinct regimes, and the toolkit
provides a model for each:

| Regime | Model | Rate set by | Use when |
| --- | --- | --- | --- |
| **Boiling** | `mvr.solve` | wall heat flow, `ṁ = Q/h_fg` | vigorous boiling at a wetted wall |
| **Evaporative** (sub-boiling, fan-swept) | `mvr.solve_evaporative` | **mass transfer** across the vapor-space gas films, throttled by non-condensable gas | low-temperature unit where water evaporates from a warm free surface and the fan sweeps the vapor across |

The two share the same `DesignParameters`, the same feed-economizer/energy-balance
bookkeeping, and the same figures of merit. As non-condensable gas → 0 and the
mass-transfer coefficients get large, the evaporative model **collapses onto the
boiling result** — they are two ends of one physical picture. For a small,
low-temperature greywater unit the **evaporative model is the relevant one**.

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

### The evaporative mass-transfer model

For a sub-boiling, fan-swept unit, production can be limited by how fast vapor
crosses the gas films at the surfaces rather than by wall heat flow. A single
mass flow `ṁ` threads a chain of resistances, and the fan supplies the
compression in the middle:

```
evaporator liquid surface   p_v = P_sat(T_evap)
    │  evaporation mass transfer     (gas film, throttled by NCG)
evaporator bulk vapor       p_v = P_v_evap
    │  FAN — compresses total pressure by ratio r ⇒ water p_v scales by r
condenser bulk vapor        p_v = P_v_cond = r · P_v_evap
    │  condensation mass transfer   (gas film, throttled by NCG)
condensation surface        p_v = P_sat(T_surf_cond)
    │  WALL — latent heat conducts back:  ṁ·h_fg = U·A·(T_surf_cond − T_evap)
evaporator liquid           T_evap
```

Each mass-transfer step uses the **stagnant-film (Stefan-flow)** law

```
ṁ = h_m · A · (P_tot / (R_v·T)) · ln[(P_tot − p_v,sink) / (P_tot − p_v,source)]
```

which stiffens as non-condensable gas (NCG) → 0 (the log term diverges, the film
resistance vanishes) so the model reduces to the heat-transfer-limited boiling
case. The coupled steady state is found by a single robust root-find on `ṁ`.

**Non-condensable gas** — dissolved air and CO₂ flashed out of the greywater —
is a first-class effect here (`noncondensable_pressure`). It blankets the
condenser and throttles both surfaces, which is why real units need a vent/purge.

**Lift budget.** The result partitions the nominal compression lift into the
parts spent on NCG dilution, evaporation mass transfer, condensation mass
transfer, and the *useful* wall ΔT — so you can see exactly which resistance is
eating your compression work. Example (50 °C, 6 K lift, 1 kPa NCG): only ~1.4 K
of the 6 K lift reaches the wall; condensation across the NCG-blanketed film eats
~2.7 K, and production is ~24 % of the naive heat-transfer limit.

**Gas-phase sensible heat.** Working in partial pressures alone would assume the
vapor is always at its local saturation temperature — i.e. that gas-phase
*sensible* heat transfer is instantaneous. It isn't: the fan delivers the vapor
**superheated** (compression heats it, and a low fan efficiency heats it more),
and that superheat has to be shed through the same poorly-conducting gas film
before the vapor can condense. The model makes this explicit via a lumped
interface energy balance,

```
U·A·(T_i − T_evap) = ṁ·h_fg + ṁ·cp·ε_ds·(T_discharge − T_i)
```

where `T_discharge` is the actual (efficiency-adjusted) fan discharge
temperature and `ε_ds` is an NTU desuperheating effectiveness built from a
gas-phase heat-transfer coefficient (tied to the mass-transfer coefficient by
the **Lewis analogy**, so the same film governs both). The report shows the
discharge temperature, superheat, and the sensible duty.

In this device the sensible load turns out to be a *small* fraction of the
condenser duty (~1–4 % across 3–15 K lift) and changes production by well under
1 %, because latent heat dwarfs sensible heat (`ṁ·h_fg ≫ ṁ·cp·ΔT`). That is a
result of the model, not an assumption baked into it — the mechanism is present
and would bite at high pressure ratio / low fan efficiency, and it also correctly
books the superheat as fan work that ends up needing rejection.

### Transfer coefficients from the fan flow

The mass- and heat-transfer coefficients aren't free inputs — they're set by the
gas boundary layer, which the **fan-driven sweep velocity** over the surfaces
creates. With `transfer_from_flow` (on by default) the model computes them from
the fan flow and channel geometry instead of taking them as given:

```
sweep velocity  u   = fan_volumetric_flow / (channel cross-section)
Reynolds        Re  = u · L / ν            (ν, D_AB rise as pressure drops)
Sherwood        Sh  = 0.664 Re^0.5 Sc^(1/3)   (laminar flat plate)
h_m = Sh · D_AB / L ,   h_g = Nu · k_gas / L   (Lewis analogy holds automatically)
```

Because the fan *pressurizes everything it circulates*, the fan power scales
with that circulated flow — so this closes the loop between the fan, the
geometry, and the transport rates, and surfaces the core **fan-sizing
trade-off**: more sweep gives a thinner film (`h_m ∝ √u`) and more production,
but fan power grows ~linearly with flow. Production therefore rises only as
~`√(flow)` while cost rises as `flow`, so specific energy has a genuine optimum
rather than always favoring more fan. Example (50 °C, 6 K lift, 1 kPa NCG):

| fan flow (L/s) | sweep (m/s) | h_m (mm/s) | L/h | fan (W) | kWh/m³ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2  | 0.17 | 6.3  | 0.43 | 18  | 43  |
| 10 | 0.83 | 14.2 | 0.91 | 91  | 101 |
| 40 | 3.33 | 28.4 | 1.65 | 365 | 220 |

The report also shows the sweep velocity, Reynolds number, and the resulting
coefficients on each surface, and lower-pressure operation helps twice over
(higher diffusivity **and**, via lower density, higher sweep velocity). Set
`transfer_from_flow=False` to fall back to fixed `*_mass_transfer_coeff` inputs.

See `mvr/masstransfer.py` and `mvr/transport.py` for the fully-commented
derivations.

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

# EVAPORATIVE model: low-temperature, fan-swept, mass-transfer-limited unit
python -m mvr.cli --model evaporative --evaporator-temp-C 50 --temp-lift 6
python -m mvr.cli --model evaporative --sweep noncondensable_pressure 100 5000 8

# worked examples
python examples/baseline.py        # boiling: baseline + optimized lift
python examples/evaporative.py     # evaporative: lift budget + NCG purge sweep
```

Flags mirror the `DesignParameters` field names (underscores → dashes, keeping
case, e.g. `--evaporator-temp-C`, `--noncondensable-pressure`).

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

### Maximize flow for a power budget (the headline design problem)

The general design goal is **maximum distillate flow for a fixed power budget
(fan + auxiliary heating), within size and material limits.** `maximize_flow`
solves exactly that:

```python
from mvr import DesignParameters, maximize_flow

base = DesignParameters(noncondensable_pressure=500.0, feed_hx_effectiveness=0.85)
params, result, info = maximize_flow(base, budget_w=600.0)   # ~15 s
print(info["material"], round(result.distillate_lph, 1), "L/h")
```

It searches the design variables (plate area, temperature lift, operating
temperature, channel gap/length, insulation, and wall material) with a
derivative-free pattern search; at every trial the fan flow is set — by
false-position on the near-linear power curve — so the design spends *exactly*
the budget. Bounds and the material list are overridable
(`DEFAULT_BOUNDS`, `WALL_MATERIALS`). Run `python examples/optimize_600W.py`.

The default bounds keep the **cold-side temperature below 80 °C**; at **600 W**
the optimizer lands on ~**12.4 L/h**, and the design tells a clear story:

- **Spend the budget on the fan, not the heater.** The optimum runs the fan hard
  enough that its dissipated work covers the losses (makeup heat goes *negative*,
  a surplus), so auxiliary heating adds nothing to flow — it only ever covers
  unavoidable losses. Keep losses low and drive the fan.
- **Size limits bind.** Plate area maxes out, the channel gap goes to its minimum
  (tighter gap → faster sweep → thinner film), and the operating temperature
  runs to its ceiling. If you can build it bigger/hotter, you get more flow.
- **Material barely matters.** Stainless trails copper by ~2 % despite 24× lower
  conductivity — the wall resistance is tiny next to the gas/film resistances —
  so the durable, non-corroding choice is essentially free.
- **Flow scales sub-linearly with power** (~300 W → 9.5, 600 → 12.4, 900 →
  ~14 L/h): doubling the budget buys only ~35 % more flow, because production
  rises roughly as `√(fan power)`.

**Cold-side temperature trade-off.** Capping `T_cold` at 80 °C is a *soft*
compromise — it keeps ~95 % of the flow/efficiency of an 85 °C design. Below
that the cost is gradual and roughly linear (~0.13 L/h and ~0.9 kWh/m³ per °C),
and the bottleneck flips from condensation-limited (below ~65 °C) to
wall-heat-limited above:

| T_cold (°C) | 50 | 60 | 70 | **80** | 85 |
| --- | ---: | ---: | ---: | ---: | ---: |
| flow (L/h) | 7.8 | 9.3 | 10.7 | **12.2** | 12.9 |
| kWh/m³ | 77 | 65 | 56 | **49** | 47 |

### What binds — sensitivity analysis

`flow_sensitivity` returns the elasticity of budget-constrained flow to each
parameter (`d ln flow / d ln param`), so you can see where extra hardware pays
off. At the 80 °C / 600 W optimum the design is **balanced**: every surface has a
similar, moderate elasticity (~0.16) and none dominates, while **wall
conductivity is ~0** (material choice is irrelevant for a flat divider).
Increasing all surfaces *together* (a bigger/denser plate) is the real lever
(~0.5 combined); a tighter channel gap also helps (higher sweep velocity).

### Non-condensable gas: load and purge (`mvr.ncg`)

Air-saturated greywater carries ~0.8 mmol/L of dissolved gas (CO₂-rich water
several times that); it flashes out under vacuum and must be purged. `mvr.ncg`
estimates the **load** from the feed and the **purge cost** — the water vapor
lost to hold a target NCG partial pressure (the bled gas is mostly steam), plus
a rough purge-pump power. `python scripts/ncg_sensitivity.py` plots it.

The purge target should track the operating regime. At 80 °C (wall-heat limited)
net flow peaks near **P_ncg ≈ 200 Pa**; purging cleaner barely lifts gross flow
(condensation isn't the bottleneck) while water loss and pump power climb. At
lower temperature (condensation limited) deeper purging genuinely buys flow. A
vent placed at the coldest, NCG-richest corner loses far less water than the
bulk-composition estimate.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite (62 tests) covers property correlations against reference steam-table
values and model invariants for **both** models: mass balance, energy-balance
closure, `Q = ṁ·h_fg`, series-resistance bounds on `U`, the lift-budget
partition, that the evaporative model never beats the heat limit and **reduces
to the boiling model as NCG → 0**, and expected monotonic responses (more lift →
more production but more fan power per kg; better insulation/economizer → less
energy; more NCG / slower mass transfer → less production).

## Modeling assumptions & limitations (v0.2)

- **Steady state only** — no transient start-up, thermal mass, or level control.
- **Lumped temperatures** — each chamber is a single temperature; no spatial
  gradients along the plate, no boiling/condensing regime maps. `U` is a design
  input built from constant film coefficients, not computed from flow/geometry.
- **Ideal-gas vapor**, single isentropic-exponent compression model.
- **Mass transfer** uses a stagnant-film (Stefan-flow) law with a linear-in-flux
  form (no high-flux/interfacial-kinetic corrections); NCG is a single specified
  partial pressure, uniform per chamber.
- **Fan-driven transport** uses flat-plate boundary-layer correlations and a
  single specified fan circulation flow (a point on the fan curve, not a
  pressure-vs-flow characteristic); the compression work is applied to the whole
  circulated flow each pass (free-throttle return, no pressure recovery).
- **Gas-phase sensible heat** is modeled at the condenser only, as a single-node
  (inlet-superheat) interface energy balance with an NTU desuperheating
  effectiveness — not a zonal desuperheat→condense integration. The evaporator
  free surface is taken at the pool temperature (liquid-side heat delivery
  assumed fast, i.e. no evaporative-cooling depression).
- **Economizer** modeled by a single effectiveness `ε`; the hot side is a
  mass-weighted blend of distillate and concentrate.
- **BPE** is a fixed input, not computed from greywater composition/recovery.
- **Makeup heat** is counted as full electrical-equivalent input; a real unit
  might source it from low-grade/waste heat, which would lower reported kWh/L.

Natural places to deepen the model next: compute `U` and the mass-transfer
coefficients from real plate geometry and flow (boiling/condensation and
Sherwood correlations), model the NCG vent/purge flow and its parasitic load,
add concentration-dependent BPE, add a transient tank model, and add
capital-cost objectives.

## Project layout

```
mvr/
  properties.py   water/steam thermophysical correlations (pure stdlib)
  transport.py    gas transport properties + flat-plate transfer correlations
  ncg.py          non-condensable-gas load from feed + purge cost model
  parameters.py   DesignParameters dataclass + validation
  model.py        boiling (heat-limited) solver + shared stream/energy helper
  masstransfer.py evaporative (mass-transfer-limited) coupled solver
  optimize.py     golden-section + grid-search optimizers
  cli.py          `python -m mvr.cli` entry point (--model, --sweep)
examples/
  baseline.py     boiling: baseline report + optimized lift
  evaporative.py  evaporative: lift budget + NCG purge + fan-flow sweep
  optimize_600W.py  maximize flow for a 600 W power budget
scripts/
  sweep_lift.py   matplotlib trade-off plot (optional dep)
  ncg_sensitivity.py  NCG purge trade-off plot (optional dep)
tests/            pytest suite (62 tests)
```

## License

MIT — see `LICENSE`.
