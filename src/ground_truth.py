"""
External ground truth: UCDP GED conflict events (battle fatalities).

Purpose: anchor the system to OBSERVED conflict, not media coverage.
The supervised target becomes "did fatalities escalate in the next 4 weeks"
instead of "what will our own media-derived index say next week".

Requires a (free) UCDP API token — https://ucdp.uu.se/apidocs/ — exported as:
    export UCDP_API_TOKEN=<token>
Without a token this module logs a warning and the pipeline falls back to the
HMM-state target.

Notes:
  - UCDP codes Gaza/West Bank events under Israel (GW 666); we map those to
    "Palestine" — imperfect (includes Israel-side events) but the least-bad
    option for this country list.
  - GED final releases lag ~6-12 months; Candidate datasets cover recent
    months. Both are tried, newest first.
"""

import logging
import os

import pandas as pd
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

logger = logging.getLogger(__name__)

UCDP_API = "https://ucdpapi.pcr.uu.se/api"
GED_CACHE = RAW_DIR / "ucdp_ged.parquet"

# Final releases, newest first — extend as UCDP publishes new versions
GED_VERSIONS = ["26.1", "25.1", "24.1"]

# Gleditsch-Ward country codes for our MENA list
GW_TO_COUNTRY = {
    678: "Yemen", 652: "Syria", 645: "Iraq", 660: "Lebanon", 620: "Libya",
    625: "Sudan", 651: "Egypt", 600: "Morocco", 615: "Algeria", 616: "Tunisia",
    663: "Jordan", 670: "Saudi Arabia", 630: "Iran", 666: "Palestine",
}

ESCALATION_HORIZON_WEEKS = 4
ESCALATION_MULTIPLIER = 2.0     # next-4w fatalities > 2x trailing rate
ESCALATION_MIN_FATALITIES = 10  # ...and at least this many deaths


def _token() -> str | None:
    return os.environ.get("UCDP_API_TOKEN")


def _fetch_pages(dataset: str, start_date: str) -> list[dict]:
    """Page through one UCDP dataset, filtered to our countries."""
    headers = {"x-ucdp-access-token": _token()}
    country_filter = ",".join(str(c) for c in GW_TO_COUNTRY)
    url = (f"{UCDP_API}/{dataset}?pagesize=1000&page=0"
           f"&Country={country_filter}&StartDate={start_date}")
    events = []
    while url:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        events.extend(payload.get("Result", []))
        url = payload.get("NextPageUrl") or None
    return events


def fetch_ucdp_fatalities(start_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Weekly fatalities (UCDP 'best' estimate) per country.
    Returns empty DataFrame (with a warning) if no token or the API fails.
    Result cached to data/raw/ucdp_ged.parquet.
    """
    if not _token():
        logger.warning(
            "UCDP_API_TOKEN not set — skipping ground truth. "
            "Register at https://ucdp.uu.se/apidocs/ to enable it."
        )
        if GED_CACHE.exists():
            logger.info("Using cached UCDP data from %s", GED_CACHE)
            return pd.read_parquet(GED_CACHE)
        return pd.DataFrame()

    events = []
    for version in GED_VERSIONS:
        try:
            events = _fetch_pages(f"gedevents/{version}", start_date)
            logger.info("UCDP GED %s: %d events", version, len(events))
            break
        except requests.HTTPError as exc:
            logger.info("GED %s unavailable (%s) — trying older version", version, exc)
    if not events:
        logger.warning("No UCDP events fetched")
        return pd.read_parquet(GED_CACHE) if GED_CACHE.exists() else pd.DataFrame()

    df = pd.DataFrame(events)
    df["country"] = df["country_id"].map(GW_TO_COUNTRY)
    df = df.dropna(subset=["country"])
    df["date"] = pd.to_datetime(df["date_start"]).dt.to_period("W").dt.start_time
    weekly = (
        df.groupby(["country", "date"])["best"].sum()
        .reset_index()
        .rename(columns={"best": "ged_fatalities"})
    )
    weekly.to_parquet(GED_CACHE, index=False)
    return weekly


def add_escalation_target_from_column(features_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Binary target from a weekly fatality column already on the frame:
      escalated_next4w = next-4-week fatalities exceed BOTH 2x the trailing
      12-week rate AND 10 deaths. NaN where the future window or trailing
      baseline is unknown (embargoed weeks stay NaN, never zero).
    Uses groupby().transform (not .apply) — apply drops the grouping column
    in pandas >= 2.2.
    """
    import numpy as np

    if col not in features_df.columns:
        return features_df
    df = features_df.sort_values(["country", "date"]).copy()
    h = ESCALATION_HORIZON_WEEKS
    g = df.groupby("country")[col]
    fut = g.transform(lambda s: s.shift(-h).rolling(h, min_periods=h).sum())
    trail = g.transform(lambda s: s.rolling(12, min_periods=8).sum() * (h / 12))
    df["fatalities_next4w"] = fut
    df["escalated_next4w"] = (
        (fut > ESCALATION_MULTIPLIER * trail) & (fut >= ESCALATION_MIN_FATALITIES)
    ).astype(float)
    df.loc[fut.isna() | trail.isna(), "escalated_next4w"] = np.nan
    return df


def add_escalation_target(features_df: pd.DataFrame, fatalities: pd.DataFrame) -> pd.DataFrame:
    """UCDP variant: merge weekly GED fatalities, then build the target."""
    if fatalities.empty:
        return features_df
    df = features_df.merge(fatalities, on=["country", "date"], how="left")
    df["ged_fatalities"] = df["ged_fatalities"].fillna(0)
    return add_escalation_target_from_column(df, "ged_fatalities")
