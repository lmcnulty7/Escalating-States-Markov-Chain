"""
Data ingestion layer.

Sources:
  1. GDELT DOC API  — per-country weekly: conflict_tone, conflict_volume,
                      total_volume (for real conflict_ratio), peace_signal
  2. RSS feeds      — live MENA headlines with VADER sentiment

Design:
  - Per-country parquet cache in data/raw/<slug>.parquet
  - Only re-fetches countries whose cache is older than CACHE_MAX_AGE_DAYS
  - Exponential backoff on 429 rate-limit errors
  - RSS fetched with browser User-Agent (feeds block Python's default UA)
"""

import logging
import re
import time
from pathlib import Path

import feedparser
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GDELT_CONFLICT_KEYWORDS,
    GDELT_DOC_API,
    GDELT_TIMESPAN,
    MENA_COUNTRIES,
    NEWS_FEEDS,
    RAW_DIR,
)

logger = logging.getLogger(__name__)
_vader = SentimentIntensityAnalyzer()

CACHE_MAX_AGE_DAYS = 6

# Browser UA — feeds return 0 entries without it
_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Country name aliases for news headline matching
COUNTRY_ALIASES: dict[str, list[str]] = {
    "Yemen":        ["Yemen", "Yemeni", "Houthi", "Houthis", "Sanaa", "Aden"],
    "Syria":        ["Syria", "Syrian", "Assad", "Damascus", "Aleppo", "Idlib"],
    "Iraq":         ["Iraq", "Iraqi", "Baghdad", "Mosul", "Kurdistan", "Erbil"],
    "Lebanon":      ["Lebanon", "Lebanese", "Hezbollah", "Beirut"],
    "Libya":        ["Libya", "Libyan", "Tripoli", "Benghazi"],
    "Sudan":        ["Sudan", "Sudanese", "Khartoum", "Darfur", "RSF"],
    "Egypt":        ["Egypt", "Egyptian", "Cairo", "Sinai", "Sisi"],
    "Morocco":      ["Morocco", "Moroccan", "Rabat", "Casablanca"],
    "Algeria":      ["Algeria", "Algerian", "Algiers"],
    "Tunisia":      ["Tunisia", "Tunisian", "Tunis"],
    "Jordan":       ["Jordan", "Jordanian", "Amman"],
    "Saudi Arabia": ["Saudi", "Saudi Arabia", "Riyadh", "MBS", "Aramco"],
    "Iran":         ["Iran", "Iranian", "Tehran", "IRGC", "Khamenei", "Rouhani", "Raisi"],
    "Palestine":    ["Palestine", "Palestinian", "Gaza", "West Bank", "Hamas", "Ramallah", "Rafah"],
}

GDELT_PEACE_KEYWORDS = "ceasefire peace negotiations diplomacy agreement truce"


# ── GDELT helpers ─────────────────────────────────────────────────────────────

def _gdelt_request(
    query: str,
    mode: str,
    timespan: str | None = GDELT_TIMESPAN,
    start: str | None = None,
    end: str | None = None,
    max_retries: int = 4,
) -> dict | None:
    """start/end are YYYYMMDDHHMMSS strings; if given they override timespan."""
    params = {"query": query, "mode": mode, "format": "json"}
    if start and end:
        params["startdatetime"] = start
        params["enddatetime"] = end
    else:
        params["timespan"] = timespan
    for attempt in range(max_retries):
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 15 * (2 ** attempt)
                logger.info("GDELT rate-limited — waiting %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            if not resp.text.strip():
                return None
            return resp.json()
        except requests.HTTPError as exc:
            logger.warning("GDELT HTTP error (query=%s): %s", query[:60], exc)
            return None
        except Exception as exc:
            logger.warning("GDELT error (query=%s): %s", query[:60], exc)
            return None
    logger.warning("GDELT gave up after %d retries (query=%s)", max_retries, query[:60])
    return None


def _parse_timeline(payload: dict | None, value_col: str) -> pd.DataFrame:
    """Parse GDELT timeline response → daily DataFrame."""
    if not payload or "timeline" not in payload:
        return pd.DataFrame(columns=["date", value_col])
    records = []
    for series in payload["timeline"]:
        for point in series.get("data", []):
            if not isinstance(point, dict):
                continue
            raw_date = str(point.get("date", ""))
            if len(raw_date) >= 8:
                try:
                    records.append({
                        "date": pd.to_datetime(raw_date[:8], format="%Y%m%d"),
                        value_col: float(point.get("value", 0.0)),
                    })
                except ValueError:
                    continue
    if not records:
        return pd.DataFrame(columns=["date", value_col])
    df = pd.DataFrame(records)
    return df.groupby("date")[value_col].mean().reset_index()


def _slug(country: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", country.lower())


def _country_cache_path(country: str) -> Path:
    return RAW_DIR / f"{_slug(country)}.parquet"


def _cache_is_fresh(country: str) -> bool:
    path = _country_cache_path(country)
    if not path.exists():
        return False
    age_days = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
    return age_days < CACHE_MAX_AGE_DAYS


# ── Per-country GDELT fetch ───────────────────────────────────────────────────

def fetch_country_gdelt(
    country: str,
    timespan: str = GDELT_TIMESPAN,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch four GDELT signals for one country:
      conflict_tone    — avg sentiment of conflict articles (negative = severe)
      conflict_volume  — volume intensity of conflict articles
      total_volume     — volume intensity of ALL articles about country
      peace_signal     — volume of ceasefire/peace/negotiation articles

    start/end (YYYYMMDDHHMMSS) fetch a historical window instead of timespan.
    """
    conflict_q = f'"{country}" {GDELT_CONFLICT_KEYWORDS}'
    total_q    = f'"{country}"'
    peace_q    = f'"{country}" {GDELT_PEACE_KEYWORDS}'

    # GDELT nominally allows 1 req/5s but clamps sustained sequences from one
    # IP (observed 2026-07-12/16: isolated probes pass while batches 429).
    # Spacing is env-tunable so backfills can trickle under the quota:
    #   GDELT_DELAY=45 python ... (default 15s for light refreshes)
    import os
    def get(q, mode, delay=int(os.environ.get("GDELT_DELAY", 15))):
        result = _gdelt_request(q, mode, timespan, start=start, end=end)
        time.sleep(delay)
        return result

    cv  = _parse_timeline(get(conflict_q, "timelinevol"),  "conflict_volume")
    ct  = _parse_timeline(get(conflict_q, "timelinetone"), "conflict_tone")
    tv  = _parse_timeline(get(total_q,    "timelinevol"),  "total_volume")
    pv  = _parse_timeline(get(peace_q,    "timelinevol"),  "peace_signal")

    frames = [df for df in [cv, ct, tv, pv] if not df.empty]
    if not frames:
        logger.warning("No GDELT data returned for %s", country)
        return pd.DataFrame()

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged["country"] = country
    return merged.sort_values("date").reset_index(drop=True)


def _merge_into_cache(country: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append new data to the per-country cache instead of overwriting it.
    On overlapping dates the newer fetch wins. History is never destroyed.
    """
    path = _country_cache_path(country)
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = (
            combined.sort_values("date")
            .drop_duplicates(subset="date", keep="last")
            .reset_index(drop=True)
        )
    else:
        combined = new_df.reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


def backfill_gdelt(
    countries: list[str] | None = None,
    start_date: str = "2024-01-01",
    chunk_days: int = 180,
    abort_after_empty: int = 2,
) -> None:
    """
    One-time historical backfill: fetch GDELT in ~6-month chunks from
    start_date to today and merge each chunk into the per-country cache.
    Resumable — chunks already covered by the cache are skipped.
    """
    countries = countries or list(MENA_COUNTRIES.keys())
    start = pd.Timestamp(start_date)
    today = pd.Timestamp.now().normalize()

    chunks = []
    cur = start
    while cur < today:
        nxt = min(cur + pd.Timedelta(days=chunk_days), today)
        chunks.append((cur, nxt))
        cur = nxt

    empty_streak = 0   # consecutive attempted chunks yielding nothing → dead zone
    for country in countries:
        path = _country_cache_path(country)
        if path.exists():
            cached = pd.read_parquet(path)
            # A chunk only counts as covered where the PRIMARY signal exists —
            # otherwise a partially-failed fetch (e.g. conflict query 429'd but
            # total_volume succeeded) permanently blocks the retry
            if "conflict_volume" in cached.columns:
                have = cached.loc[cached["conflict_volume"].notna(), "date"]
            else:
                have = pd.Series(dtype="datetime64[ns]")
        else:
            have = pd.Series(dtype="datetime64[ns]")
        have = pd.to_datetime(have)
        for c_start, c_end in chunks:
            # Skip chunk if cache already covers >90% of its days
            if len(have):
                covered = have.between(c_start, c_end).sum()
                if covered >= 0.9 * (c_end - c_start).days:
                    logger.info("  %s %s→%s already cached — skip", country, c_start.date(), c_end.date())
                    continue
            logger.info("  Backfill %s %s→%s", country, c_start.date(), c_end.date())
            df = fetch_country_gdelt(
                country,
                start=c_start.strftime("%Y%m%d%H%M%S"),
                end=c_end.strftime("%Y%m%d%H%M%S"),
            )
            if df.empty:
                logger.warning("  Backfill returned no data: %s %s→%s", country, c_start.date(), c_end.date())
                empty_streak += 1
                if empty_streak >= abort_after_empty:
                    logger.error("  %d consecutive empty chunks — GDELT dead zone, aborting pass "
                                 "(resumable; nothing lost)", empty_streak)
                    return
                continue
            empty_streak = 0
            merged = _merge_into_cache(country, df)
            if "conflict_volume" in merged.columns:
                have = pd.to_datetime(merged.loc[merged["conflict_volume"].notna(), "date"])
            else:
                have = pd.to_datetime(merged["date"])
            logger.info("  %s cache now %d rows (%s → %s)", country, len(merged),
                        merged["date"].min().date(), merged["date"].max().date())


# ── Batch fetch with per-country caching ─────────────────────────────────────

def fetch_all_gdelt(
    countries: list[str] | None = None,
    timespan: str = GDELT_TIMESPAN,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch GDELT data for all countries.
    Uses per-country parquet cache; only re-downloads stale/missing countries.
    """
    countries = countries or list(MENA_COUNTRIES.keys())
    frames = []
    stale = [c for c in countries if force_refresh or not _cache_is_fresh(c)]
    fresh = [c for c in countries if c not in stale]

    # Load fresh from cache
    for country in fresh:
        path = _country_cache_path(country)
        df = pd.read_parquet(path)
        frames.append(df)
        logger.info("  Cache hit: %s (%d rows)", country, len(df))

    if stale:
        logger.info("Fetching %d countries from GDELT: %s", len(stale), stale)
        for i, country in enumerate(stale):
            logger.info("  [%d/%d] %s", i + 1, len(stale), country)
            df = fetch_country_gdelt(country, timespan)
            if not df.empty:
                merged = _merge_into_cache(country, df)   # append, never overwrite
                frames.append(merged)
            else:
                # Fall back to stale cache rather than dropping the country
                path = _country_cache_path(country)
                if path.exists():
                    logger.warning("  Fetch failed for %s — using stale cache", country)
                    frames.append(pd.read_parquet(path))
                else:
                    logger.warning("  No data for %s — skipping", country)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined.sort_values(["country", "date"]).reset_index(drop=True)


# ── RSS news ──────────────────────────────────────────────────────────────────

def fetch_rss_headlines(
    feeds: list[str] | None = None,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse RSS feeds and match headlines to MENA countries via alias lookup.
    Fetches with a browser User-Agent — feeds block Python's default agent.
    """
    feeds = feeds or NEWS_FEEDS
    countries = countries or list(MENA_COUNTRIES.keys())

    # Build reverse alias map: lowercase alias → canonical country name
    alias_map: dict[str, str] = {}
    for country in countries:
        for alias in COUNTRY_ALIASES.get(country, [country]):
            alias_map[alias.lower()] = country

    records = []
    for url in feeds:
        try:
            resp = requests.get(url, headers=_RSS_HEADERS, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            source = feed.feed.get("title", url)

            for entry in feed.entries:
                title   = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                link    = getattr(entry, "link", "")

                # Skip non-news sections (Al Jazeera all.xml includes sport etc.)
                if any(seg in link for seg in ("/sports/", "/sport/", "/economy/", "/features/")):
                    continue

                text    = f"{title} {summary}"
                sentiment = _vader.polarity_scores(text)["compound"]

                pub = getattr(entry, "published_parsed", None)
                date = pd.Timestamp(*pub[:3]) if pub else pd.Timestamp.now().normalize()

                # Assign the headline to EVERY matched country (deterministic),
                # not just the first alias hit from an unordered set.
                words = {w.lower() for w in re.findall(r"[A-Za-z']+", text)}
                matched = sorted({alias_map[w] for w in words if w in alias_map})

                for country_match in matched:
                    records.append({
                        "date":      date,
                        "country":   country_match,
                        "title":     title,
                        "sentiment": sentiment,
                        "source":    source,
                        "url":       link,
                    })
        except Exception as exc:
            logger.warning("RSS parse error (%s): %s", url, exc)

    if not records:
        return pd.DataFrame(columns=["date", "country", "title", "sentiment", "source", "url"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def aggregate_news_weekly(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    df = news_df.copy()
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    return (
        df.groupby(["country", "week"])
        .agg(news_sentiment=("sentiment", "mean"), news_volume=("title", "count"))
        .reset_index()
        .rename(columns={"week": "date"})
    )
