"""Command-line entry point for the MVR model.

Examples::

    python -m mvr.cli                       # baseline report
    python -m mvr.cli --temp-lift 3         # override one parameter
    python -m mvr.cli --sweep temp_lift 1 12 12   # scan a parameter, print table
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from .parameters import DesignParameters
from .model import solve, report
from .masstransfer import solve_evaporative, report_evaporative


def _add_param_args(parser: argparse.ArgumentParser) -> None:
    """Expose every numeric DesignParameters field as ``--field-name``."""
    for f in fields(DesignParameters):
        flag = "--" + f.name.replace("_", "-")
        parser.add_argument(flag, type=float, default=None,
                            help=f"override {f.name} (default {f.default})")


def _params_from_args(args: argparse.Namespace) -> DesignParameters:
    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(DesignParameters)
        if getattr(args, f.name) is not None
    }
    return DesignParameters(**overrides)


def _run_sweep(base: DesignParameters, name: str, lo: float, hi: float, n: int,
               evaporative: bool) -> None:
    if not any(f.name == name for f in fields(DesignParameters)):
        raise SystemExit(f"unknown parameter to sweep: {name}")
    n = max(int(n), 2)
    solver = solve_evaporative if evaporative else solve
    print(f"# sweeping {name} from {lo} to {hi} ({n} points)"
          f"  [{'evaporative' if evaporative else 'boiling'} model]\n")
    extra = f" | {'limit':>12} | {'MT eff %':>8}" if evaporative else ""
    header = (f"{name:>22} | {'distillate L/h':>14} | {'fan W':>8} | "
              f"{'makeup W':>9} | {'kWh/m3':>8} | {'GOR':>6}{extra}")
    print(header)
    print("-" * len(header))
    for i in range(n):
        value = lo + (hi - lo) * i / (n - 1)
        kwargs = {**{f.name: getattr(base, f.name) for f in fields(DesignParameters)}, name: value}
        try:
            r = solver(DesignParameters(**kwargs))
        except ValueError as exc:
            print(f"{value:22.4g} | (invalid: {exc})")
            continue
        tail = (f" | {r.limiting_mechanism:>12} | {r.mass_transfer_effectiveness*100:8.1f}"
                if evaporative else "")
        print(f"{value:22.4g} | {r.distillate_lph:14.2f} | {r.fan_power:8.1f} | "
              f"{r.makeup_heat:9.1f} | {r.specific_energy_kwh_per_l*1000:8.2f} | "
              f"{r.gain_output_ratio:6.2f}{tail}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_param_args(parser)
    parser.add_argument("--model", choices=("boiling", "evaporative"), default="boiling",
                        help="which regime to solve (default: boiling). Use "
                             "'evaporative' for a sub-boiling, fan-swept, "
                             "mass-transfer-limited unit.")
    parser.add_argument("--sweep", nargs=4, metavar=("PARAM", "LO", "HI", "N"),
                        help="scan PARAM over [LO, HI] in N points and print a table")
    args = parser.parse_args(argv)

    base = _params_from_args(args)
    evaporative = args.model == "evaporative"

    if args.sweep:
        name, lo, hi, n = args.sweep
        _run_sweep(base, name, float(lo), float(hi), float(n), evaporative)
    elif evaporative:
        print(report_evaporative(base, solve_evaporative(base)))
    else:
        print(report(base, solve(base)))


if __name__ == "__main__":
    main()
