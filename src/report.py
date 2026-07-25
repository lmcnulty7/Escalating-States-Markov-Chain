"""
Report engine — charter Phase 1 (§2 editorial contract).

Generates the weekly country assessment as typed sentences. Every sentence is
one of exactly five kinds:
  measure   — a quantity with source + timestamp
  change    — a claim of change, backed by a formal test (Mann-Whitney U,
              last 4 weeks vs prior 12; claimed only at p < 0.05)
  constant  — an explicit "no significant change" (same test, not significant)
  forecast  — model output; ships ONLY with validation caveats (uncalibrated
              scores are presented as rankings, never probabilities — §5.2)
  gap       — declared missing data (embargo, absent sources)

Sections carry an as-of date (staleness is declared, not hidden — §5.6) and
citations resolving to source records (§5.5): ACLED event IDs from the local
raw cache, GDELT archive file dates, DOC API query windows.

Pure logic — no Streamlit. The dashboard renders the structure this returns.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

P_THRESHOLD = 0.05
RECENT_W, BASELINE_W = 4, 12


def _change_test(series: pd.Series) -> tuple[str, float | None]:
    """Mann-Whitney U: last RECENT_W weeks vs the BASELINE_W before them.
    Returns (direction, p) where direction ∈ {'higher','lower','flat','na'}."""
    s = series.dropna()
    if len(s) < RECENT_W + 8:
        return "na", None
    recent, base = s.iloc[-RECENT_W:], s.iloc[-(RECENT_W + BASELINE_W):-RECENT_W]
    if base.nunique() <= 1 and recent.nunique() <= 1 and recent.iloc[0] == base.iloc[0]:
        return "flat", 1.0
    try:
        _, p = mannwhitneyu(recent, base, alternative="two-sided")
    except ValueError:
        return "flat", 1.0
    if p >= P_THRESHOLD:
        return "flat", float(p)
    return ("higher" if recent.mean() > base.mean() else "lower"), float(p)


def _pctile(series: pd.Series, value: float) -> float | None:
    s = series.dropna()
    if len(s) < 8 or pd.isna(value):
        return None
    return float((s <= value).mean() * 100)


def _s(stype: str, text: str, cite: str | None = None) -> dict:
    return {"type": stype, "text": text, "cite": cite}


# ── Sections ──────────────────────────────────────────────────────────────────

def _violence_section(cdf: pd.DataFrame, country: str, acled_events: pd.DataFrame | None) -> dict:
    out: list[dict] = []
    avail = cdf[cdf["acled_fatalities"].notna()] if "acled_fatalities" in cdf.columns else pd.DataFrame()
    if avail.empty:
        return {"title": "Violence (ACLED)", "as_of": None,
                "sentences": [_s("gap", f"No ACLED event data available for {country}.")]}

    cur = avail.iloc[-1]
    as_of = cur["date"]
    n_ev = int(cur.get("acled_events", 0))
    cite_base = f"ACLED, week of {as_of:%Y-%m-%d}, {n_ev} event records"

    out.append(_s("measure",
                  f"ACLED recorded **{int(cur['acled_fatalities'])} fatalities** across "
                  f"{n_ev} events in the week of {as_of:%b %d, %Y}.", cite_base))

    if n_ev > 0:
        comp = {"battles": cur.get("acled_battles", 0),
                "explosions/remote violence": cur.get("acled_explosions", 0),
                "violence against civilians": cur.get("acled_civilian_violence", 0),
                "protests": cur.get("acled_protests", 0)}
        top = sorted(comp.items(), key=lambda kv: -kv[1])[:2]
        out.append(_s("measure",
                      "Composition: " + ", ".join(f"{int(v)} {k}" for k, v in comp.items()) +
                      f" — dominated by {top[0][0]}.", cite_base))

    direction, p = _change_test(avail["acled_fatalities"])
    pct = _pctile(avail["acled_fatalities"], cur["acled_fatalities"])
    pct_txt = f" This week sits at the {pct:.0f}th percentile of {country}'s own history." if pct is not None else ""
    if direction in ("higher", "lower"):
        out.append(_s("change",
                      f"Fatalities over the last {RECENT_W} weeks are **significantly "
                      f"{'elevated' if direction == 'higher' else 'reduced'}** vs the prior "
                      f"{BASELINE_W}-week baseline (Mann-Whitney p={p:.3f}).{pct_txt}",
                      f"test: last {RECENT_W}w vs prior {BASELINE_W}w of weekly fatality counts"))
    elif direction == "flat":
        out.append(_s("constant",
                      f"No statistically significant change in fatalities vs the prior "
                      f"{BASELINE_W}-week baseline (p={p:.2f}).{pct_txt}",
                      f"test: last {RECENT_W}w vs prior {BASELINE_W}w of weekly fatality counts"))

    g_dir, g_p = _change_test(avail["acled_geo_spread"]) if "acled_geo_spread" in avail.columns else ("na", None)
    if g_dir == "higher":
        out.append(_s("change",
                      f"Violence is **spreading geographically**: active admin-1 regions "
                      f"significantly up vs baseline (p={g_p:.3f}).",
                      "test: distinct admin1 regions with ≥1 event per week"))
    elif g_dir in ("flat", "lower"):
        out.append(_s("constant",
                      f"Geographic spread {'contracting' if g_dir == 'lower' else 'stable'}: "
                      f"{int(cur.get('acled_geo_spread', 0))} admin-1 regions active this week.",
                      "distinct admin1 regions with ≥1 event"))

    # Embargo gap — the single most important honesty line in the report
    last_all = cdf["date"].max()
    if last_all > as_of:
        out.append(_s("gap",
                      f"**No event data after {as_of:%b %d, %Y}** (ACLED account tier embargoes "
                      f"the most recent 12 months). Absence of data is not absence of violence."))

    # Record-level citations: largest events that week from the raw cache
    records = []
    if acled_events is not None and not acled_events.empty:
        wk = acled_events[(acled_events["event_date"] >= as_of) &
                          (acled_events["event_date"] < as_of + pd.Timedelta(days=7))]
        for _, ev in wk.nlargest(3, "fatalities").iterrows():
            if ev["fatalities"] > 0:
                records.append(f"{ev['event_id_cnty']} — {ev['event_type']}, "
                               f"{ev['admin1']}, {int(ev['fatalities'])} deaths (source: {ev['source']})")

    return {"title": "Violence (ACLED)", "as_of": as_of, "sentences": out, "records": records}


def _media_section(cdf: pd.DataFrame, country: str) -> dict:
    out: list[dict] = []
    cur = cdf.iloc[-1]
    as_of = cur["date"]

    if "gdelt_ev_conflict" in cdf.columns and pd.notna(cur.get("gdelt_ev_conflict")):
        cite = f"GDELT event archive (daily files), week of {as_of:%Y-%m-%d}"
        out.append(_s("measure",
                      f"Global media coverage located **{int(cur['gdelt_ev_conflict'])} material-conflict "
                      f"events** in {country} this week (of {int(cur['gdelt_ev_total'])} total events; "
                      f"mean Goldstein {cur['gdelt_ev_goldstein']:+.1f}, tone {cur['gdelt_ev_tone']:+.1f}).",
                      cite))
        direction, p = _change_test(cdf["gdelt_ev_conflict"])
        if direction in ("higher", "lower"):
            out.append(_s("change",
                          f"Conflict-event coverage is **significantly "
                          f"{'elevated' if direction == 'higher' else 'reduced'}** vs the prior "
                          f"{BASELINE_W}-week baseline (p={p:.3f}).",
                          f"test: last {RECENT_W}w vs prior {BASELINE_W}w, gdelt_ev_conflict"))
        elif direction == "flat":
            out.append(_s("constant",
                          f"Conflict-event coverage shows no significant change vs baseline (p={p:.2f}).",
                          f"test: last {RECENT_W}w vs prior {BASELINE_W}w, gdelt_ev_conflict"))

    if pd.notna(cur.get("conflict_tone")):
        pct = _pctile(cdf["conflict_tone"].dropna() * -1, -cur["conflict_tone"])
        if pct is not None:
            out.append(_s("measure",
                          f"DOC-API conflict coverage tone is {cur['conflict_tone']:+.2f} — "
                          f"{pct:.0f}th percentile of {country}'s history (higher = more negative).",
                          f"GDELT DOC API, week of {as_of:%Y-%m-%d}"))
    else:
        out.append(_s("gap", "No DOC-API media signal for this week (fetch gap or country not yet covered)."))

    return {"title": "Media Signal (GDELT)", "as_of": as_of, "sentences": out}


def _outlook_section(cdf: pd.DataFrame, all_latest: pd.DataFrame, country: str,
                     metrics: dict | None) -> dict:
    out: list[dict] = []
    cur = cdf.iloc[-1]
    if "risk_score" not in cdf.columns or pd.isna(cur.get("risk_score")):
        return {"title": "Outlook", "as_of": cur["date"],
                "sentences": [_s("gap", "No model output available for this week.")]}

    rank = int((all_latest["risk_score"] > cur["risk_score"]).sum()) + 1
    n = int(all_latest["risk_score"].notna().sum())
    m = metrics or {}
    wf = m.get("walk_forward", {})
    pr, base = wf.get("pr_auc"), wf.get("persistence_pr_auc")
    skill = (f"purged walk-forward PR-AUC {pr:.2f} vs {base:.2f} persistence, "
             f"{wf.get('oos_rows', 0)} out-of-sample weeks"
             if pr is not None else "validation metrics unavailable")

    calibrated_ok = bool(m.get("calibration_ok")) and pd.notna(cur.get("risk_calibrated", np.nan))
    if calibrated_ok:
        out.append(_s("forecast",
                      f"Calibrated escalation probability, next 4 weeks: "
                      f"**{cur['risk_calibrated'] * 100:.0f}%** (regional base rate "
                      f"{wf.get('oos_base_rate', 0) * 100:.0f}%). Calibrated probabilities sit close "
                      f"to the base rate — the model's information is mostly in the ranking: "
                      f"{country} is **#{rank} of {n}** this week.",
                      f"Platt-calibrated on out-of-sample folds; Brier {m.get('brier_calibrated', 0):.3f} "
                      f"vs {wf.get('brier_climatology', 0):.3f} climatology; {skill}"))
    else:
        out.append(_s("forecast",
                      f"The escalation model ranks {country} **#{rank} of {n}** countries by "
                      f"4-week escalation risk (score {cur['risk_score']:.2f}). "
                      f"Scores are **uncalibrated** — treat as a ranking, not a probability ({skill}).",
                      f"XGBoost on media+ACLED features; target: ACLED fatalities next 4w; {skill}"))
    if bool(cur.get("risk_in_sample", False)):
        out.append(_s("gap", "This week was inside the model's training window — score is in-sample."))
    return {"title": "Outlook", "as_of": cur["date"], "sentences": out}


# ── Entry point ───────────────────────────────────────────────────────────────

def build_report(results: pd.DataFrame, country: str,
                 acled_events: pd.DataFrame | None = None,
                 metrics: dict | None = None) -> dict:
    """Full typed report for one country's latest week."""
    cdf = results[results["country"] == country].sort_values("date")
    if cdf.empty:
        return {"country": country, "week": None, "sections": [],
                "completeness": [], "error": "no data"}

    week = cdf["date"].max()
    latest_all = results.sort_values("date").groupby("country").last().reset_index()

    completeness = []
    for label, col in [("GDELT DOC API", "conflict_volume"),
                       ("GDELT event archive", "gdelt_ev_total"),
                       ("ACLED events", "acled_fatalities")]:
        if col in cdf.columns and cdf[col].notna().any():
            last = cdf.loc[cdf[col].notna(), "date"].max()
            completeness.append({"source": label, "through": last,
                                 "current": bool(last >= week)})
        else:
            completeness.append({"source": label, "through": None, "current": False})

    sections = [
        _violence_section(cdf, country, acled_events),
        _media_section(cdf, country),
        _outlook_section(cdf, latest_all, country, metrics),
    ]
    return {"country": country, "week": week, "sections": sections,
            "completeness": completeness}
