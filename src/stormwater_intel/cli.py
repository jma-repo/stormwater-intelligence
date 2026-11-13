"""Command line entry point: `python -m stormwater_intel <command>`."""
from __future__ import annotations

import argparse
import logging

from . import fetch as F


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stormwater-intel", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download raw inputs into data/raw")
    p_fetch.add_argument("--only", nargs="*", choices=list(F.STEPS), help="subset of inputs")
    p_fetch.add_argument("--force", action="store_true", help="re-download existing files")

    p_run = sub.add_parser("run", help="build panel, backtest, write outputs/ (metrics, tables, figures, map)")
    p_run.add_argument("--out", default=None, help="output directory (default: outputs/)")
    p_run.add_argument("--skip-ablation", action="store_true", help="skip the slower ablation/sensitivity runs")

    p_maps = sub.add_parser("maps", help="re-render the interactive map and static figure from outputs/ (no model run)")
    p_maps.add_argument("--out", default=None, help="output directory (default: outputs/)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "fetch":
        F.fetch_all_inputs(args.only, force=args.force)
    elif args.cmd == "run":
        from .pipeline import run_poc  # heavy imports only when needed
        run_poc(args.out, skip_ablation=args.skip_ablation)
    elif args.cmd == "maps":
        from . import config as C
        from . import data as D
        from .pipeline import render_maps
        comm = D.load_communities()
        universe = comm[comm["class"].isin(C.UNIVERSE_CLASSES)].reset_index(drop=True)
        render_maps(args.out or C.OUTPUT_DIR, universe, D.load_gauges())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
