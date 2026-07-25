"""
GDELT raw daily event files — historical media-event archive.

Why this exists: the DOC API enforces a small per-IP request quota that made
multi-year backfills impossible (see DEVLOG 2026-07-17). The raw daily files
on data.gdeltproject.org are static downloads with no rate limiting. The DOC
API remains the source for the light weekly live refresh; this module supplies
HISTORY, as additive gdelt_ev_* feature columns (never mixed into DOC columns).

Design:
  - Stream-and-discard: download one day's zip (~8 MB), filter to MENA rows,
    append to a single parquet store, delete the zip. Peak disk ≈ one file.
  - Resumable: days already in the store are skipped.
  - Availability-correct: aggregation keys on the FILE date (publication day),
    not the event date — what the media said *that day* (charter §5.1).
  - Palestine = FIPS WE (West Bank) + GZ (Gaza); Israel-side events excluded
    (same convention as the UCDP mapping).
"""

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

logger = logging.getLogger(__name__)

FILES_BASE = "http://data.gdeltproject.org/events"
STORE_PATH = RAW_DIR / "gdelt_events_mena.parquet"

# GDELT 1.0 daily export: fixed 0-based column positions
_COLS = {1: "event_day", 29: "quad_class", 30: "goldstein", 31: "num_mentions",
         33: "num_articles", 34: "avg_tone", 51: "geo_country"}

FIPS_TO_COUNTRY = {
    "IZ": "Iraq", "IR": "Iran", "SY": "Syria", "YM": "Yemen", "LE": "Lebanon",
    "LY": "Libya", "SU": "Sudan", "EG": "Egypt", "MO": "Morocco", "AG": "Algeria",
    "TS": "Tunisia", "JO": "Jordan", "SA": "Saudi Arabia",
    "WE": "Palestine", "GZ": "Palestine",
}


def fetch_day(date: pd.Timestamp) -> pd.DataFrame | None:
    """One day's global file → MENA-filtered events. None on download failure."""
    url = f"{FILES_BASE}/{date.strftime('%Y%m%d')}.export.CSV.zip"
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            logger.warning("GDELT file missing for %s", date.date())
            return None
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as fh:
                df = pd.read_csv(fh, sep="\t", header=None,
                                 usecols=list(_COLS), low_memory=False)
    except Exception as exc:
        logger.warning("GDELT file fetch failed for %s: %s", date.date(), exc)
        return None

    df = df.rename(columns=_COLS)
    df = df[df["geo_country"].isin(FIPS_TO_COUNTRY)].copy()
    df["country"] = df["geo_country"].map(FIPS_TO_COUNTRY)
    df["file_date"] = date
    return df.drop(columns=["geo_country"])


def build_archive(
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    pause_s: float = 0.5,
) -> None:
    """
    Download all missing days into the parquet store. Resumable — days already
    stored are skipped. ~910 days ≈ 7 GB transfer, ~2-3 h, ~150 MB retained.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

    store = pd.read_parquet(STORE_PATH) if STORE_PATH.exists() else pd.DataFrame()
    have = set(pd.to_datetime(store["file_date"]).dt.normalize()) if not store.empty else set()

    days = [d for d in pd.date_range(start, end, freq="D") if d not in have]
    logger.info("GDELT files: %d days to fetch (%d already stored)", len(days), len(have))

    batch, since_flush = [], 0
    for i, day in enumerate(days):
        df = fetch_day(day)
        if df is not None and not df.empty:
            batch.append(df)
        since_flush += 1
        if since_flush >= 30 or i == len(days) - 1:     # flush every ~month
            if batch:
                store = pd.concat([store, *batch], ignore_index=True)
                store.to_parquet(STORE_PATH, index=False)
                logger.info("  stored through %s (%d rows total)", day.date(), len(store))
            batch, since_flush = [], 0
        time.sleep(pause_s)
    logger.info("GDELT file archive complete: %d rows", len(store))


def compute_event_features(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Weekly (country, date) aggregates on the shared Monday-start spine:
      gdelt_ev_total       — all events located in country that week
      gdelt_ev_conflict    — QuadClass 4 (material conflict)
      gdelt_ev_verbal      — QuadClass 3 (verbal conflict)
      gdelt_ev_coop        — QuadClass 1+2 (cooperation)
      gdelt_ev_goldstein   — mean Goldstein scale (negative = conflictual)
      gdelt_ev_tone        — mean AvgTone of event coverage
      gdelt_ev_mentions    — total mentions (media weight)
    """
    if events is None:
        if not STORE_PATH.exists():
            return pd.DataFrame()
        events = pd.read_parquet(STORE_PATH)
    if events.empty:
        return pd.DataFrame()

    df = events.copy()
    df["date"] = pd.to_datetime(df["file_date"]).dt.to_period("W").dt.start_time
    weekly = (
        df.groupby(["country", "date"])
        .agg(
            gdelt_ev_total=("quad_class", "count"),
            gdelt_ev_conflict=("quad_class", lambda s: (s == 4).sum()),
            gdelt_ev_verbal=("quad_class", lambda s: (s == 3).sum()),
            gdelt_ev_coop=("quad_class", lambda s: s.isin([1, 2]).sum()),
            gdelt_ev_goldstein=("goldstein", "mean"),
            gdelt_ev_tone=("avg_tone", "mean"),
            gdelt_ev_mentions=("num_mentions", "sum"),
        )
        .reset_index()
    )
    return weekly.sort_values(["country", "date"]).reset_index(drop=True)
