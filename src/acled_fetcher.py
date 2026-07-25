"""
ACLED data layer — charter Phase 1, Step 2.

Call-budget design (requirement: never trip request/over-call limits):
  1. OAuth token cached on disk and reused until expiry → ~1 auth call per day,
     not one per request.
  2. Per-country parquet cache; only the gap between the cache's newest event
     and the embargo boundary is ever fetched. Historical data is never
     re-downloaded (a small overlap window catches late revisions).
  3. Large pages (5000 events/request) — fewest possible calls. A full country
     backfill is typically 2-4 requests.
  4. Politeness delay between pages, exponential backoff on 429/5xx.
  5. Hard per-run call budget (MAX_CALLS_PER_RUN) — raises before hammering.
  6. Embargo-aware: the account's date_recency restriction (free tier: events
     must be ≥12 months old) is read from every response. Weeks beyond the
     boundary are never queried — they'd silently return empty AND waste calls.
     Callers must treat post-embargo weeks as MISSING, never as calm.
"""

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR, RAW_DIR

logger = logging.getLogger(__name__)

ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
ACLED_READ_URL  = "https://acleddata.com/api/acled/read"
TOKEN_CACHE     = BASE_DIR / ".acled_token.json"     # gitignored
ACLED_CACHE_DIR = RAW_DIR / "acled"
ACLED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAGE_SIZE            = 5000
PAGE_DELAY_S         = 1.5
MAX_CALLS_PER_RUN    = 40
REVISION_OVERLAP_DAYS = 14   # refetch this much before cache tail to catch late edits
CACHE_FRESH_DAYS      = 3    # skip tail-refresh entirely if cache updated this recently

# Module state: per-run call counter + embargo boundary learned from responses
_calls_this_run = 0
_embargo_end: pd.Timestamp | None = None

# Columns kept (full raw record — charter §5.5 needs raw fields + stable IDs)
_NUMERIC = {"fatalities": "int64", "latitude": "float64", "longitude": "float64",
            "geo_precision": "int64", "time_precision": "int64", "timestamp": "int64"}


class AcledBudgetExceeded(RuntimeError):
    """Raised before the run can exceed its API call budget."""


# ── Auth ──────────────────────────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    env = {}
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k:
                env[k] = v
    return env


def _get_token() -> str:
    """Disk-cached OAuth token; re-authenticates only when <30 min validity left."""
    if TOKEN_CACHE.exists():
        try:
            tok = json.loads(TOKEN_CACHE.read_text())
            if tok.get("expires_at", 0) - time.time() > 1800:
                return tok["access_token"]
        except (json.JSONDecodeError, KeyError):
            pass

    env = _load_env()
    email, password = env.get("ACLED_EMAIL"), env.get("ACLED_PASSWORD")
    if not email or not password:
        raise RuntimeError("ACLED_EMAIL / ACLED_PASSWORD not found in .env")

    resp = requests.post(ACLED_TOKEN_URL, data={
        "username": email, "password": password,
        "grant_type": "password", "client_id": "acled", "scope": "authenticated",
    }, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    TOKEN_CACHE.write_text(json.dumps({
        "access_token": payload["access_token"],
        "expires_at": time.time() + payload.get("expires_in", 86400),
    }))
    TOKEN_CACHE.chmod(0o600)
    logger.info("ACLED: new token obtained (cached to disk)")
    return payload["access_token"]


# ── Low-level request with budget + backoff ──────────────────────────────────

def _api_get(params: dict, max_retries: int = 4) -> dict:
    global _calls_this_run, _embargo_end

    if _calls_this_run >= MAX_CALLS_PER_RUN:
        raise AcledBudgetExceeded(
            f"ACLED call budget ({MAX_CALLS_PER_RUN}/run) reached — refusing to "
            "continue. Cached data is intact; re-run later or raise the budget deliberately."
        )

    headers = {"Authorization": f"Bearer {_get_token()}", "Accept": "application/json"}
    for attempt in range(max_retries):
        _calls_this_run += 1
        resp = requests.get(ACLED_READ_URL, headers=headers, params=params, timeout=90)
        if resp.status_code in (429, 500, 502, 503):
            wait = 10 * (2 ** attempt)
            logger.warning("ACLED HTTP %d — backing off %ds", resp.status_code, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        payload = resp.json()

        # Learn the embargo boundary from the server itself
        rec = (payload.get("data_query_restrictions") or {}).get("date_recency") or {}
        if rec.get("date"):
            _embargo_end = pd.Timestamp(rec["date"])
        return payload
    raise RuntimeError(f"ACLED gave up after {max_retries} retries (params={params})")


def calls_used() -> int:
    return _calls_this_run


def accessible_end() -> pd.Timestamp:
    """Newest event date this account can see. Conservative estimate until the
    first response teaches us the server's actual boundary."""
    return _embargo_end or (pd.Timestamp.now().normalize() - pd.DateOffset(months=12))


def access_date(country: str | None = None) -> pd.Timestamp | None:
    """
    When this machine last downloaded ACLED data — required by ACLED's
    attribution policy ("include when you accessed the data").
    Per-country cache mtime, or the newest across all countries.
    """
    paths = [_cache_path(country)] if country else list(ACLED_CACHE_DIR.glob("*.parquet"))
    stamps = [pd.Timestamp(p.stat().st_mtime, unit="s") for p in paths if p.exists()]
    return max(stamps).normalize() if stamps else None


# ACLED attribution policy: cite source, access date, filters, and manipulation
ATTRIBUTION_SHORT = "Data: ACLED (acleddata.com)"


def attribution(country: str | None = None) -> str:
    """One-line ACLED citation for display on visuals and in reports."""
    acc = access_date(country)
    when = f", accessed {acc:%Y-%m-%d}" if acc is not None else ""
    scope = f" — {country}" if country else " — 14 MENA countries"
    return (f"{ATTRIBUTION_SHORT}{when}{scope}, all event types; "
            "aggregated by author to Monday-start weekly totals")


# ── Public fetch ──────────────────────────────────────────────────────────────

def fetch_acled(country: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    All ACLED events for one country in [start_date, end_date], typed DataFrame.
    The window is clamped to the account's embargo boundary; if the entire
    window is embargoed, returns empty WITH a loud log line.
    """
    start = pd.Timestamp(start_date)
    end   = min(pd.Timestamp(end_date), accessible_end())
    if start > end:
        logger.warning("ACLED %s: window %s→%s entirely embargoed (accessible ≤ %s) — 0 calls made",
                       country, start_date, end_date, accessible_end().date())
        return pd.DataFrame()

    frames, page = [], 1
    while True:
        payload = _api_get({
            "country": country,
            "event_date": f"{start.date()}|{end.date()}",
            "event_date_where": "BETWEEN",
            "limit": PAGE_SIZE,
            "page": page,
        })
        rows = payload.get("data", [])
        frames.extend(rows)
        logger.info("  ACLED %s page %d: %d events (calls used: %d)",
                    country, page, len(rows), _calls_this_run)
        if len(rows) < PAGE_SIZE:
            break
        page += 1
        time.sleep(PAGE_DELAY_S)

    if not frames:
        logger.info("ACLED %s: 0 events in %s→%s (window IS accessible — genuinely quiet or no coverage)",
                    country, start.date(), end.date())
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df["event_date"] = pd.to_datetime(df["event_date"])
    for col, dtype in _NUMERIC.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
    return df.sort_values("event_date").reset_index(drop=True)


# ── Cached fetch (the function everything else should use) ────────────────────

def _slug(country: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", country.lower())


def _cache_path(country: str) -> Path:
    return ACLED_CACHE_DIR / f"{_slug(country)}.parquet"


def fetch_acled_cached(country: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Cache-first fetch: only the gap between the cache's newest event (minus a
    small revision-overlap window) and the requested/embargo end is downloaded.
    Repeat calls with a warm cache cost ZERO API requests.
    Dedupe key: event_id_cnty, keeping the latest revision (max timestamp).
    """
    path = _cache_path(country)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()

    want_start = pd.Timestamp(start_date)
    want_end   = min(pd.Timestamp(end_date), accessible_end())

    fetch_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if cached.empty:
        fetch_ranges.append((want_start, want_end))
    else:
        have_min, have_max = cached["event_date"].min(), cached["event_date"].max()
        if want_start < have_min:                      # missing history before cache
            fetch_ranges.append((want_start, have_min))
        cache_age_days = (time.time() - path.stat().st_mtime) / 86400
        refresh_from = have_max - pd.Timedelta(days=REVISION_OVERLAP_DAYS)
        if want_end > refresh_from and cache_age_days >= CACHE_FRESH_DAYS:
            fetch_ranges.append((refresh_from, want_end))  # new data / revisions past tail

    for f_start, f_end in fetch_ranges:
        if f_start >= f_end:
            continue
        new = fetch_acled(country, str(f_start.date()), str(f_end.date()))
        if not new.empty:
            cached = pd.concat([cached, new], ignore_index=True)

    if cached.empty:
        return cached
    if not fetch_ranges:
        logger.info("ACLED %s: cache hit, 0 API calls", country)

    cached = (
        cached.sort_values("timestamp")
        .drop_duplicates(subset="event_id_cnty", keep="last")
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    cached.to_parquet(path, index=False)

    mask = cached["event_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return cached[mask].reset_index(drop=True)


def fetch_all_acled_weekly(
    countries: list[str],
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Weekly ACLED features for all countries (cache-first; a warm run costs 0
    API calls). Countries with no ACLED data are logged loudly and absent from
    the result — callers must treat them as missing, not calm.
    """
    frames = []
    for country in countries:
        ev = fetch_acled_cached(country, start_date, str(accessible_end().date()))
        if ev.empty:
            logger.warning("ACLED: no events for %s — country absent from ACLED output", country)
            continue
        frames.append(compute_acled_features(ev, end=accessible_end()))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info("ACLED weekly: %d rows, %d countries, through %s (calls used: %d)",
                len(out), out["country"].nunique(), accessible_end().date(), calls_used())
    return out


# ── Weekly features (charter Phase 1, Step 3) ─────────────────────────────────

EVENT_TYPE_COLS = {
    "Battles":                     "acled_battles",
    "Explosions/Remote violence":  "acled_explosions",
    "Violence against civilians":  "acled_civilian_violence",
    "Protests":                    "acled_protests",
    "Riots":                       "acled_riots",
    "Strategic developments":      "acled_strategic",
}

COUNT_COLS = ["acled_fatalities", "acled_events", *EVENT_TYPE_COLS.values(),
              "acled_civilian_targeting", "acled_geo_spread", "acled_new_actors"]


def compute_acled_features(acled_df: pd.DataFrame, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Event-level ACLED records → weekly (country, date) aggregates.

    Weeks are Monday-start periods — the same spine features.py uses for GDELT,
    so fusion is a plain (country, date) join.

    Zero-filling: within the accessible window, a week with no events is a real
    zero (nothing recorded). The grid extends only to `end` (pass the embargo
    boundary / accessible_end()); beyond it weeks are ABSENT — missing, not calm.

    Caveats:
      - acled_new_actors is left-censored: every actor is "new" in the first
        accessible weeks. Ignore the first ~4 weeks of that column.
      - Lags/rolling stats are deliberately NOT computed here — features.py owns
        that machinery (single lag convention across all sources).
    """
    if acled_df.empty:
        return pd.DataFrame()

    df = acled_df.copy()
    df["date"] = df["event_date"].dt.to_period("W").dt.start_time

    # First-appearance flag for actor1 within each country (computed on events
    # sorted by date so "new" means "never seen before this week")
    df = df.sort_values("event_date")
    df["_new_actor"] = ~df.duplicated(subset=["country", "actor1"]) & df["actor1"].ne("")

    weekly = (
        df.groupby(["country", "date"])
        .agg(
            acled_fatalities=("fatalities", "sum"),
            acled_events=("event_id_cnty", "count"),
            acled_civilian_targeting=("civilian_targeting", lambda s: (s != "").sum()),
            acled_geo_spread=("admin1", "nunique"),
            acled_new_actors=("_new_actor", "sum"),
            acled_last_update=("timestamp", "max"),
        )
        .reset_index()
    )

    # Per-event-type counts
    type_counts = (
        df.pivot_table(index=["country", "date"], columns="event_type",
                       values="event_id_cnty", aggfunc="count", fill_value=0)
        .rename(columns=EVENT_TYPE_COLS)
        .reset_index()
    )
    type_counts = type_counts[["country", "date"] +
                              [c for c in EVENT_TYPE_COLS.values() if c in type_counts.columns]]
    weekly = weekly.merge(type_counts, on=["country", "date"], how="left")
    for col in EVENT_TYPE_COLS.values():
        if col not in weekly.columns:
            weekly[col] = 0

    # Complete weekly grid per country, honest zeros up to `end` only
    end_ts = pd.Timestamp(end) if end is not None else df["date"].max()
    end_week = end_ts.to_period("W").start_time
    filled = []
    for country, grp in weekly.groupby("country"):
        grid = pd.date_range(grp["date"].min(), end_week, freq="7D")
        grp = grp.set_index("date").reindex(grid).rename_axis("date").reset_index()
        grp["country"] = country
        for col in COUNT_COLS:
            grp[col] = grp[col].fillna(0).astype(int)
        filled.append(grp)

    return pd.concat(filled, ignore_index=True).sort_values(["country", "date"]).reset_index(drop=True)
