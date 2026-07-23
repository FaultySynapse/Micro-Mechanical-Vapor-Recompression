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


def _run_sweep(base: DesignParameters, name: str, lo: float, hi: float, n: int) -> None:
    if not any(f.name == name for f in fields(DesignParameters)):
        raise SystemExit(f"unknown parameter to sweep: {name}")
    n = max(int(n), 2)
    print(f"# sweeping {name} from {lo} to {hi} ({n} points)\n")
    header = f"{name:>22} | {'distillate L/h':>14} | {'fan W':>8} | {'makeup W':>9} | {'kWh/m3':>8} | {'GOR':>6}"
    print(header)
    print("-" * len(header))
    for i in range(n):
        value = lo + (hi - lo) * i / (n - 1)
        kwargs = {**{f.name: getattr(base, f.name) for f in fields(DesignParameters)}, name: value}
        try:
            r = solve(DesignParameters(**kwargs))
        except ValueError as exc:
            print(f"{value:22.4g} | (invalid: {exc})")
            continue
        print(f"{value:22.4g} | {r.distillate_lph:14.2f} | {r.fan_power:8.1f} | "
              f"{r.makeup_heat:9.1f} | {r.specific_energy_kwh_per_l*1000:8.2f} | "
              f"{r.gain_output_ratio:6.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_param_args(parser)
    parser.add_argument("--sweep", nargs=4, metavar=("PARAM", "LO", "HI", "N"),
                        help="scan PARAM over [LO, HI] in N points and print a table")
    args = parser.parse_args(argv)

    base = _params_from_args(args)

    if args.sweep:
        name, lo, hi, n = args.sweep
        _run_sweep(base, name, float(lo), float(hi), float(n))
    else:
        print(report(base, solve(base)))


if __name__ == "__main__":
    main()
