"""
Feature engineering: GDELT + news signals → model-ready feature matrix.

Key fixes vs v1:
  - conflict_ratio uses real total_volume (no more ×20 hack)
  - conflict_tone nulls are forward-filled then backward-filled (not zeroed)
  - peace_signal added as a feature
  - conflict_acceleration (2nd-order momentum) added
  - Per-country percentile rank added for each signal
  - Intensity score uses 3 reliable signals; news_sentiment only included if present
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LAG_WEEKS, ROLLING_WINDOWS


# ── Weekly aggregation ────────────────────────────────────────────────────────

def resample_weekly(gdelt_df: pd.DataFrame) -> pd.DataFrame:
    gdelt_df = gdelt_df.copy()
    gdelt_df["date"] = pd.to_datetime(gdelt_df["date"])
    gdelt_df["week"] = gdelt_df["date"].dt.to_period("W").dt.start_time

    agg_map = {
        "conflict_volume": ("conflict_volume", "sum"),
        "conflict_tone":   ("conflict_tone",   "mean"),
    }
    if "total_volume" in gdelt_df.columns:
        agg_map["total_volume"] = ("total_volume", "sum")
    if "peace_signal" in gdelt_df.columns:
        agg_map["peace_signal"] = ("peace_signal", "sum")

    agg = (
        gdelt_df.groupby(["country", "week"])
        .agg(**agg_map)
        .reset_index()
        .rename(columns={"week": "date"})
    )

    # Drop the in-progress week: summing a partial week biases volume low,
    # which reads as false de-escalation exactly where users look ("now")
    current_week_start = pd.Timestamp.now().to_period("W").start_time
    agg = agg[agg["date"] < current_week_start]

    # Reindex each country to a complete weekly grid so missing weeks are
    # explicit NaN gaps instead of silently absent rows (the HMM otherwise
    # treats irregular gaps as uniform time steps)
    filled = []
    for country, grp in agg.groupby("country"):
        grid = pd.date_range(grp["date"].min(), grp["date"].max(), freq="7D")
        grp = grp.set_index("date").reindex(grid).rename_axis("date").reset_index()
        grp["country"] = country
        filled.append(grp)
    return pd.concat(filled, ignore_index=True)


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country", "date"]).copy()

    # Real conflict_ratio (needs total_volume). Weeks with NO DOC data at all
    # (extended-spine rows) stay NaN — zero would fake "no conflict coverage"
    if "total_volume" in df.columns:
        ratio = (
            df["conflict_volume"] / df["total_volume"].replace(0, np.nan)
        ).clip(0, 1)
        has_doc = df["conflict_volume"].notna()
        df["conflict_ratio"] = ratio.where(has_doc, np.nan)
        df.loc[has_doc & df["conflict_ratio"].isna(), "conflict_ratio"] = 0.0
    else:
        df["conflict_ratio"] = np.nan

    # Forward-fill then backward-fill tone so nulls don't become false stability
    df["conflict_tone"] = (
        df.groupby("country")["conflict_tone"]
        .transform(lambda x: x.ffill().bfill())
        .infer_objects(copy=False)
    )

    # Week-over-week changes (within country)
    df["conflict_tone_change"] = df.groupby("country")["conflict_tone"].diff()
    df["conflict_volume_pct"]  = (
        df.groupby("country")["conflict_volume"].pct_change().clip(-5, 5)
    )
    # Acceleration: 2nd-order momentum (is escalation speeding up?)
    df["conflict_acceleration"] = df.groupby("country")["conflict_volume_pct"].diff()

    if "peace_signal" in df.columns:
        df["peace_change"] = df.groupby("country")["peace_signal"].diff()

    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = LAG_WEEKS) -> pd.DataFrame:
    base_cols = [
        c for c in [
            "conflict_volume", "conflict_tone", "conflict_ratio",
            "conflict_tone_change", "peace_signal",
        ] if c in df.columns
    ]
    for lag in lags:
        for col in base_cols:
            df[f"{col}_lag{lag}w"] = df.groupby("country")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    for w in windows:
        df[f"conflict_volume_roll{w}w_mean"] = (
            df.groupby("country")["conflict_volume"]
            .transform(lambda x: x.rolling(w, min_periods=2).mean())
        )
        df[f"conflict_tone_roll{w}w_mean"] = (
            df.groupby("country")["conflict_tone"]
            .transform(lambda x: x.rolling(w, min_periods=2).mean())
        )
        df[f"conflict_volume_roll{w}w_std"] = (
            df.groupby("country")["conflict_volume"]
            .transform(lambda x: x.rolling(w, min_periods=2).std())
        )
        if "peace_signal" in df.columns:
            df[f"peace_roll{w}w_mean"] = (
                df.groupby("country")["peace_signal"]
                .transform(lambda x: x.rolling(w, min_periods=2).mean())
            )
    return df


def add_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-country percentile rank for each base signal.
    E.g., conflict_tone_pctile = 85 means this week's tone is
    worse than 85% of all weeks for that country.
    For tone: higher percentile = more negative = worse (inverted).
    """
    signals = {
        "conflict_tone":   True,   # invert: more negative → higher percentile
        "conflict_volume": False,
        "conflict_ratio":  False,
        "peace_signal":    True,   # invert: less peace coverage → higher conflict percentile
    }
    for col, invert in signals.items():
        if col not in df.columns:
            continue
        series = -df[col] if invert else df[col]
        df[f"{col}_pctile"] = (
            series.groupby(df["country"])
            .transform(lambda x: x.rank(pct=True, na_option="keep") * 100)
            .round(1)
        )
    return df


def extend_weekly_spine(weekly: pd.DataFrame, other: pd.DataFrame | None) -> pd.DataFrame:
    """
    Extend each country's weekly grid to cover the union of the DOC-API range
    and another source's range (e.g. the GDELT event-file archive), and add
    countries the DOC API has no rows for at all (Iran/Iraq). DOC columns are
    NaN on extended rows — missing, never zero. Both sources use Monday-start
    weeks, so the grids align by construction.
    """
    if other is None or other.empty:
        return weekly
    frames = []
    for country in sorted(set(weekly["country"]) | set(other["country"])):
        w = weekly[weekly["country"] == country]
        o = other[other["country"] == country]
        bounds = pd.concat([w["date"], o["date"]])
        grid = pd.date_range(bounds.min(), bounds.max(), freq="7D")
        w = w.set_index("date").reindex(grid).rename_axis("date").reset_index()
        w["country"] = country
        frames.append(w)
    return pd.concat(frames, ignore_index=True)


def merge_gdelt_events(df: pd.DataFrame, ev_weekly: pd.DataFrame | None) -> pd.DataFrame:
    """Merge weekly GDELT event-file aggregates (gdelt_ev_*). These are media
    signals published same-week, so lag-0 use in models is availability-safe."""
    if ev_weekly is None or ev_weekly.empty:
        return df
    return df.merge(ev_weekly, on=["country", "date"], how="left")


def merge_acled(df: pd.DataFrame, acled_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weekly ACLED aggregates on (country, date). Weeks outside ACLED's
    accessible window stay NaN — missing, never zero (the embargo trap).
    """
    if acled_weekly is None or acled_weekly.empty:
        return df
    return df.merge(acled_weekly, on=["country", "date"], how="left")


# ACLED signals that get availability-safe model features
_ACLED_LAG_COLS  = ["acled_fatalities", "acled_events", "acled_battles",
                    "acled_explosions", "acled_civilian_violence", "acled_protests",
                    "acled_geo_spread", "acled_new_actors"]
_ACLED_ROLL_COLS = ["acled_fatalities", "acled_battles", "acled_civilian_violence"]


def add_acled_features(df: pd.DataFrame, lags: list[int] = LAG_WEEKS) -> pd.DataFrame:
    """
    Availability rule (charter §5.1): ACLED publishes on a weekly cycle, so
    week t's events are NOT knowable during week t. Model features are
    therefore lags (>=1w) only; the raw lag-0 columns exist for reports and
    are excluded from modeling by get_feature_cols.
    Rolling means are computed THEN shifted 1w for the same reason.
    """
    if "acled_fatalities" not in df.columns:
        return df
    df = df.sort_values(["country", "date"])
    g = df.groupby("country")
    for col in _ACLED_LAG_COLS:
        for lag in lags:
            df[f"{col}_lag{lag}w"] = g[col].shift(lag)
    for col in _ACLED_ROLL_COLS:
        df[f"{col}_roll4w_lag1w"] = (
            g[col].transform(lambda x: x.rolling(4, min_periods=2).mean().shift(1))
        )
    return df


def merge_news(df: pd.DataFrame, news_weekly: pd.DataFrame) -> pd.DataFrame:
    if news_weekly.empty:
        df["news_sentiment"] = np.nan
        df["news_volume"] = 0
        return df
    merged = df.merge(news_weekly, on=["country", "date"], how="left")
    # Don't fill news_sentiment with 0 — keep NaN when no news, so model knows data is absent
    merged["news_volume"] = merged["news_volume"].fillna(0).astype(int)
    return merged


# ── Intensity score ───────────────────────────────────────────────────────────

def compute_intensity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite escalation intensity ∈ [0, 1].

    Weights (updated to use only reliable signals):
      40%  conflict_tone (negative)   — most reliable GDELT signal
      35%  conflict_volume            — article volume (can be sparse)
      25%  conflict_ratio             — conflict share of total coverage
      (news_sentiment added at 10% weight only when present)

    Normalization: RobustScaler per column (robust to outlier spikes),
    then min-max to [0, 1] across the full dataset.
    """
    df = df.copy()
    scaler = RobustScaler()

    ct_norm = scaler.fit_transform(-df[["conflict_tone"]].fillna(df["conflict_tone"].median())).flatten()
    cv_norm = scaler.fit_transform(df[["conflict_volume"]].fillna(0)).flatten()
    cr_norm = scaler.fit_transform(df[["conflict_ratio"]].fillna(0)).flatten()

    raw = 0.40 * ct_norm + 0.35 * cv_norm + 0.25 * cr_norm

    # Add news_sentiment if it has real data
    if "news_sentiment" in df.columns and df["news_sentiment"].notna().sum() > 10:
        ns_filled = df["news_sentiment"].fillna(0)
        ns_norm = scaler.fit_transform(-ns_filled.values.reshape(-1, 1)).flatten()
        raw = 0.35 * ct_norm + 0.30 * cv_norm + 0.25 * cr_norm + 0.10 * ns_norm

    lo, hi = raw.min(), raw.max()
    df["intensity"] = ((raw - lo) / (hi - lo)) if hi > lo else 0.5

    # Weeks with no data at all must be NaN (ffilled downstream), not
    # "low intensity" — zero-filled volume would read as calm
    df.loc[df["conflict_volume"].isna(), "intensity"] = np.nan

    return df


# ── Main entrypoint ───────────────────────────────────────────────────────────

def build_feature_matrix(
    gdelt_df: pd.DataFrame,
    news_df: pd.DataFrame,
    acled_weekly: pd.DataFrame | None = None,
    ev_weekly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from src.data_fetcher import aggregate_news_weekly

    weekly = resample_weekly(gdelt_df)
    weekly = extend_weekly_spine(weekly, ev_weekly)
    weekly = compute_base_features(weekly)
    news_weekly = aggregate_news_weekly(news_df) if not news_df.empty else pd.DataFrame()
    weekly = merge_news(weekly, news_weekly)
    weekly = merge_gdelt_events(weekly, ev_weekly)
    weekly = merge_acled(weekly, acled_weekly)
    weekly = add_lag_features(weekly)
    weekly = add_rolling_features(weekly)
    weekly = add_acled_features(weekly)
    weekly = add_percentile_features(weekly)
    weekly = compute_intensity_score(weekly)
    return weekly.sort_values(["country", "date"]).reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "country", "date", "intensity", "hmm_state", "hmm_state_label",
        "next_state", "next_intensity", "predicted_state", "risk_score",
        "hmm_prob_stable", "hmm_prob_rising_tension", "hmm_prob_active_conflict",
        "risk_in_sample",
        # UCDP ground-truth columns: targets/outcomes, never features
        # (fatality reporting lags ~1 month, so they aren't known at predict time)
        "ged_fatalities", "fatalities_next4w", "escalated_next4w",
    }
    # Also exclude percentile columns from model features (use raw signals instead)
    # ACLED availability rule: only lagged/shifted ACLED columns are knowable at
    # prediction time — raw lag-0 acled_* columns are report-only.
    return [
        c for c in df.columns
        if c not in exclude
        and not c.endswith("_pctile")
        and not (c.startswith("acled_") and "_lag" not in c)
        and df[c].dtype in (np.float64, np.int64, float, int)
    ]
