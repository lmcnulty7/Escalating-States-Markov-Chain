"""
CLI entry point.

Usage:
    python run_pipeline.py              # use cached GDELT data if available
    python run_pipeline.py --refresh    # force re-download from GDELT
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the MENA escalation pipeline")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download from GDELT (ignores cache)",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Subset of countries to run (default: all MENA)",
    )
    parser.add_argument(
        "--backfill",
        metavar="START_DATE",
        default=None,
        help="One-time historical backfill from this date (e.g. 2024-01-01) before running",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Proceed even if some countries have no data (default: hard fail)",
    )
    args = parser.parse_args()

    try:
        if args.backfill:
            from src.data_fetcher import backfill_gdelt
            backfill_gdelt(countries=args.countries, start_date=args.backfill)
        results = run_pipeline(
            countries=args.countries,
            force_refresh=args.refresh,
            allow_missing=args.allow_missing,
        )
        print(f"\nPipeline complete. {len(results)} rows, {results['country'].nunique()} countries.")
        print("\nLatest state per country:")
        latest = (
            results.sort_values("date")
            .groupby("country")
            .last()[["date", "intensity", "hmm_state_label"]]
        )
        print(latest.to_string())
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
