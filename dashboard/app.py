"""
MENA Conflict Escalation Monitor — dashboard

Tabs:
  1. MENA Overview   — map + signal table
  2. Country Analysis — state confidence, signal breakdown with percentiles,
                        auto-generated text summary, SHAP explanation
  3. Forecast        — HMM forward simulation + transition matrix
  4. Live News       — RSS headlines with VADER sentiment
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FORECAST_WEEKS, MENA_COUNTRIES, NAME_TO_ISO3,
    STATE_COLORS, STATE_INT_COLORS, STATE_LABELS,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MENA Conflict Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_results() -> pd.DataFrame:
    from src.pipeline import load_results as _load
    df = _load()
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def load_fi() -> pd.DataFrame:
    from src.pipeline import load_feature_importance
    return load_feature_importance()

@st.cache_data(ttl=900)
def load_news(countries: list[str]) -> pd.DataFrame:
    from src.data_fetcher import fetch_rss_headlines
    return fetch_rss_headlines(countries=countries)

@st.cache_resource
def load_hmm():
    try:
        from src.hmm_model import load_hmm as _lh
        return _lh()
    except Exception:
        return None

@st.cache_resource
def load_xgb():
    """Returns bundle dict (model, le, feature_cols, conflict_component) or None."""
    try:
        from src.ml_model import load_model
        return load_model()
    except Exception:
        return None

@st.cache_resource
def load_escalation_bundle():
    try:
        import joblib
        from config import MODELS_DIR
        return joblib.load(MODELS_DIR / "escalation_model.joblib")
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    import json
    from config import PROCESSED_DIR
    p = PROCESSED_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

# ── Bootstrap ─────────────────────────────────────────────────────────────────

try:
    df = load_results()
except FileNotFoundError:
    st.error("No data found. Run `python run_pipeline.py` then refresh.")
    st.stop()

# Data-completeness banner: never present a partial region as a full picture
_missing = sorted(set(MENA_COUNTRIES) - set(df["country"].unique()))
if _missing:
    st.error(
        f"⚠️ **No data for {len(_missing)} of {len(MENA_COUNTRIES)} countries: "
        f"{', '.join(_missing)}.** Regional counts and the map exclude them — "
        "this is a data gap, not stability."
    )

RISK_IS_UCDP = "risk_target" in df.columns and df["risk_target"].astype(str).str.contains("escalation").any()
RISK_LABEL = "P(Escalation next 4w)" if RISK_IS_UCDP else "XGB Conflict Risk"

EMOJI = {"Active Conflict": "🔴", "Rising Tension": "🟡", "Stable": "🟢"}
PROB_COLS = {
    "Stable":          "hmm_prob_stable",
    "Rising Tension":  "hmm_prob_rising_tension",
    "Active Conflict": "hmm_prob_active_conflict",
}

def latest_all() -> pd.DataFrame:
    return df.sort_values("date").groupby("country").last().reset_index()

def state_of(country: str) -> str:
    row = latest_all()[latest_all()["country"] == country]
    return row["hmm_state_label"].values[0] if len(row) else "Unknown"

def _sort_key(c: str) -> tuple:
    s = state_of(c)
    order = {"Active Conflict": 0, "Rising Tension": 1, "Stable": 2}
    row = latest_all()[latest_all()["country"] == c]
    intensity = float(row["intensity"].values[0]) if len(row) else 0
    return (order.get(s, 9), -intensity)

all_countries = sorted(df["country"].unique(), key=_sort_key)
_latest = latest_all()
data_as_of = df["date"].max().strftime("%B %d, %Y")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌍 MENA Escalation")
    st.caption(f"Data as of **{data_as_of}**")
    st.divider()

    selected = st.selectbox(
        "Country (sorted by severity)",
        all_countries,
        format_func=lambda c: f"{EMOJI.get(state_of(c), '⚪')}  {c}",
        label_visibility="collapsed",
    )

    # Quick summary card for selected country
    sel_row = _latest[_latest["country"] == selected]
    if not sel_row.empty:
        r = sel_row.iloc[0]
        st.markdown(f"### {EMOJI.get(r.get('hmm_state_label',''), '⚪')} {selected}")
        st.markdown(f"**{r.get('hmm_state_label', 'Unknown')}**")
        intensity_val = float(r.get("intensity", 0))
        st.progress(intensity_val, text=f"Intensity: {intensity_val*100:.1f}%")

        # Show posteriors if available
        for label, col in PROB_COLS.items():
            if col in r and pd.notna(r[col]):
                prob = float(r[col]) * 100
                emoji = EMOJI.get(label, "⚪")
                color = STATE_COLORS.get(label, "#888")
                st.markdown(
                    f"<small>{emoji} {label}: <b>{prob:.0f}%</b></small>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.caption(
        f"**Countries:** {len(all_countries)}  \n"
        f"**Weeks of data:** {df['date'].nunique()}  \n"
        f"**Model:** GaussianHMM + XGBoost  \n"
        f"**Signals:** GDELT DOC API + RSS"
        + ("  \n**Target:** UCDP escalation (next 4w)" if RISK_IS_UCDP else "")
    )
    _m = load_metrics()
    if _m:
        lines = []
        wf = _m.get("walk_forward", {})
        if wf:
            lines.append(f"Walk-forward PR-AUC {wf['pr_auc']:.2f} vs {wf.get('persistence_pr_auc', 0):.2f} persistence")
            lines.append(f"Brier (calibrated) {_m.get('brier_calibrated', 0):.3f} vs {wf.get('brier_climatology', 0):.3f} climatology")
            lines.append("Calibration: " + ("✅ passes gate" if _m.get("calibration_ok") else "❌ ranking only"))
        elif "pr_auc" in _m:
            lines.append(f"PR-AUC {_m['pr_auc']:.2f} vs baseline {_m.get('persistence_pr_auc', 0):.2f}")
            lines.append(f"Brier {_m['brier']:.3f} · base rate {_m.get('test_base_rate', 0):.2f}")
        elif "model_acc" in _m:
            lines.append(f"Test acc {_m['model_acc']:.2f} vs persistence {_m['persistence_acc']:.2f}")
            if "model_acc_transitions" in _m:
                lines.append(f"Transition weeks (n={_m['n_transition_weeks']}): "
                             f"{_m['model_acc_transitions']:.0%} correct (baseline 0%)")
        if _m.get("hmm_transmat_problems"):
            lines.append("⚠️ HMM: " + "; ".join(_m["hmm_transmat_problems"]))
        if lines:
            st.caption("**Held-out evaluation:**  \n" + "  \n".join(lines))
    if st.button("🔄 Refresh cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ MENA Overview", "📊 Country Analysis", "🔮 Forecast", "📰 Live News", "📋 Report"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  —  MENA Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header(f"MENA Region · Conflict States  —  {data_as_of}")

    la = latest_all()
    n_c = (la["hmm_state_label"] == "Active Conflict").sum()
    n_r = (la["hmm_state_label"] == "Rising Tension").sum()
    n_s = (la["hmm_state_label"] == "Stable").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Active Conflict", n_c)
    c2.metric("🟡 Rising Tension",  n_r)
    c3.metric("🟢 Stable",          n_s)

    # Map
    map_df = la.copy()
    map_df["iso3"]         = map_df["country"].map(NAME_TO_ISO3)
    map_df["intensity_pct"]= (map_df["intensity"].fillna(0) * 100).round(1)
    map_df["state_label"]  = map_df["hmm_state_label"].fillna("Unknown")

    fig_map = px.choropleth(
        map_df,
        locations="iso3",
        color="state_label",
        hover_name="country",
        hover_data={"iso3": False, "state_label": True, "intensity_pct": True},
        color_discrete_map={**{v: STATE_COLORS[v] for v in STATE_COLORS}, "Unknown": "#555"},
        labels={"state_label": "State", "intensity_pct": "Intensity (%)"},
    )
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True,
                 projection_type="natural earth",
                 center=dict(lon=38, lat=25), projection_scale=2.4),
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
        height=460,
        legend_title_text="Conflict State",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Signal table
    st.subheader("Signal Summary — All Countries")
    st.caption(
        "**Conflict Tone**: GDELT avg sentiment of conflict articles (more negative = more severe).  "
        "**Conflict Volume**: share of global news devoted to this country's conflict keywords.  "
        "**Conflict Ratio**: conflict articles / total country coverage.  "
        "**Peace Signal**: ceasefire/diplomacy article volume (higher = active negotiations).  "
        "**Intensity**: composite score [0–100%] relative to worst week in dataset."
    )
    display_cols = ["country", "hmm_state_label", "intensity",
                    "conflict_tone", "conflict_volume", "conflict_ratio"]
    if "peace_signal" in la.columns:
        display_cols.append("peace_signal")
    if "news_sentiment" in la.columns and la["news_sentiment"].notna().any():
        display_cols.append("news_sentiment")

    tbl = la[display_cols].copy()
    tbl.columns = [c.replace("_", " ").title() for c in display_cols]
    # Sort numerically BEFORE formatting as strings ("9.5%" > "85.0%" lexically)
    tbl = tbl.sort_values("Intensity", ascending=False).reset_index(drop=True)
    tbl["Intensity"] = (tbl["Intensity"] * 100).round(1).astype(str) + "%"
    tbl["Hmm State Label"] = tbl["Hmm State Label"].apply(
        lambda s: f"{EMOJI.get(s,'⚪')} {s}"
    )
    for col in ["Conflict Tone", "Conflict Volume", "Conflict Ratio",
                "Peace Signal", "News Sentiment"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].round(4)
    tbl = tbl.rename(columns={"Hmm State Label": "State"})
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  —  Country Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    cdf = df[df["country"] == selected].sort_values("date").copy()

    if cdf.empty:
        st.warning(f"No data for {selected}.")
        st.stop()

    cur = cdf.iloc[-1]
    prv = cdf.iloc[-2] if len(cdf) >= 2 else cur
    avg4 = cdf.tail(4)

    state_label    = str(cur.get("hmm_state_label", "Unknown"))
    intensity_now  = float(cur.get("intensity", 0)) * 100
    intensity_prev = float(prv.get("intensity", 0)) * 100
    intensity_4wk  = float(avg4["intensity"].mean()) * 100
    delta_1w       = intensity_now - intensity_prev
    delta_4w       = intensity_now - intensity_4wk

    risk_now  = float(cur.get("risk_score", 0)) * 100 if "risk_score" in cur.index else None
    risk_prev = float(prv.get("risk_score", 0)) * 100 if "risk_score" in prv.index else None

    st.header(f"{EMOJI.get(state_label,'⚪')} {selected} — {state_label}")

    # ── Metric row ─────────────────────────────────────────────────────────

    cols = st.columns(4)
    cols[0].metric("HMM State", state_label)
    cols[1].metric(
        "Intensity Score",
        f"{intensity_now:.1f}%",
        delta=f"{delta_1w:+.1f}% vs last week",
        delta_color="inverse",
    )
    cols[2].metric(
        "4-Week Average",
        f"{intensity_4wk:.1f}%",
        delta=f"{delta_4w:+.1f}% vs now",
        delta_color="inverse",
    )
    if risk_now is not None:
        cols[3].metric(
            RISK_LABEL,
            f"{risk_now:.1f}%",
            delta=f"{(risk_now - risk_prev):+.1f}%" if risk_prev is not None else None,
            delta_color="inverse",
        )
        if bool(cur.get("risk_in_sample", False)):
            cols[3].caption("⚠️ in-sample (model saw this week in training)")

    st.divider()

    # ── State confidence (HMM posteriors) ──────────────────────────────────

    has_posteriors = all(col in cdf.columns for col in PROB_COLS.values())
    if has_posteriors:
        st.subheader("State Confidence")
        st.caption(
            "The **HMM forward-backward algorithm** (not just the Viterbi path) gives a "
            "probability distribution over all three states. This shows how *certain* the model is."
        )
        prob_values = {
            label: float(cur.get(col, 0) or 0)
            for label, col in PROB_COLS.items()
        }
        # Horizontal stacked bar
        fig_conf = go.Figure()
        for label in ["Stable", "Rising Tension", "Active Conflict"]:
            p = prob_values.get(label, 0)
            fig_conf.add_trace(go.Bar(
                name=f"{EMOJI.get(label,'')} {label} ({p*100:.0f}%)",
                x=[p],
                y=["State Confidence"],
                orientation="h",
                marker_color=STATE_COLORS.get(label, "#888"),
                text=f"{p*100:.0f}%",
                textposition="inside",
                insidetextanchor="middle",
            ))
        fig_conf.update_layout(
            barmode="stack",
            height=90,
            xaxis=dict(range=[0, 1], visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.1, x=0),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        dominant = max(prob_values, key=prob_values.get)
        dominant_p = prob_values[dominant] * 100
        if dominant_p < 55:
            st.caption(
                f"⚠️ Low confidence — the model is uncertain between states "
                f"({dominant_p:.0f}% for {dominant}). Treat the classification cautiously."
            )

    st.divider()

    # ── Auto-generated text summary ─────────────────────────────────────────

    def _trend_word(delta):
        if delta > 3:   return "**intensifying**"
        if delta > 0.5: return "slightly worsening"
        if delta < -3:  return "**improving**"
        if delta < -0.5:return "slightly improving"
        return "stable"

    ct_now  = float(cur.get("conflict_tone",   np.nan) or np.nan)
    cv_now  = float(cur.get("conflict_volume", 0) or 0)
    cr_now  = float(cur.get("conflict_ratio",  0) or 0)
    ps_now  = float(cur.get("peace_signal",    0) or 0)

    ct_pctile = float(cur.get("conflict_tone_pctile",   np.nan) or np.nan)
    cv_pctile = float(cur.get("conflict_volume_pctile", np.nan) or np.nan)
    ps_pctile = float(cur.get("peace_signal_pctile",    np.nan) or np.nan)

    summary_lines = []
    summary_lines.append(
        f"Conflict intensity is {_trend_word(delta_1w)} "
        f"({intensity_now:.1f}% this week vs {intensity_4wk:.1f}% 4-week average)."
    )
    if not np.isnan(ct_pctile):
        severity = "its worst" if ct_pctile > 85 else "above average" if ct_pctile > 60 else "below average"
        summary_lines.append(
            f"Conflict tone is at {ct_pctile:.0f}th percentile for {selected} — {severity} historically."
        )
    if not np.isnan(cv_pctile) and cv_pctile > 60:
        summary_lines.append(
            f"Conflict article volume is elevated ({cv_pctile:.0f}th percentile)."
        )
    if ps_now > 0 and not np.isnan(ps_pctile):
        if ps_pctile < 30:
            summary_lines.append("Peace/ceasefire coverage is low — no active negotiations visible in news.")
        elif ps_pctile > 70:
            summary_lines.append("Peace/diplomacy coverage is elevated — active negotiations may be underway.")
    if risk_now is not None:
        risk_what = ("observed escalation (recorded conflict fatalities) in the next 4 weeks"
                     if RISK_IS_UCDP else "the media-derived state being Active Conflict next week")
        if risk_now > 60:
            summary_lines.append(f"Model gives {risk_now:.0f}% probability of {risk_what}.")
        elif risk_now < 30:
            summary_lines.append(f"Model gives only {risk_now:.0f}% probability of {risk_what}.")

    st.info("  ".join(summary_lines) if summary_lines else "Insufficient data for automated summary.")

    st.divider()

    # ── Signal breakdown ────────────────────────────────────────────────────

    st.subheader("Signal Breakdown — This Week vs Historical Range")
    st.caption(
        "Each bar shows where this week's signal sits **relative to this country's own history**. "
        "100 = worst week ever recorded for this country; 0 = calmest. "
        "Percentile ranks are computed within-country, so a 90th percentile in Tunisia ≠ 90th in Syria."
    )

    signal_defs = [
        {
            "name":    "Conflict Tone",
            "raw_col": "conflict_tone",
            "pct_col": "conflict_tone_pctile",
            "weight":  "40%",
            "note":    "GDELT avg article sentiment. Negative = more alarming conflict coverage.",
            "color":   "#e74c3c",
            "invert":  True,
        },
        {
            "name":    "Conflict Volume",
            "raw_col": "conflict_volume",
            "pct_col": "conflict_volume_pctile",
            "weight":  "35%",
            "note":    "Volume intensity of conflict-keyword articles (GDELT global share).",
            "color":   "#e67e22",
            "invert":  False,
        },
        {
            "name":    "Conflict Ratio",
            "raw_col": "conflict_ratio",
            "pct_col": "conflict_ratio_pctile",
            "weight":  "25%",
            "note":    "Conflict articles / total country coverage. Higher = conflict dominates news.",
            "color":   "#f39c12",
            "invert":  False,
        },
        {
            "name":    "Peace Signal",
            "raw_col": "peace_signal",
            "pct_col": "peace_signal_pctile",
            "weight":  "—",
            "note":    "Ceasefire/peace/diplomacy article volume. High = active negotiations. Not in intensity formula.",
            "color":   "#2ecc71",
            "invert":  True,   # more peace = lower conflict percentile
        },
    ]
    if "news_sentiment" in cdf.columns and cdf["news_sentiment"].notna().any():
        signal_defs.append({
            "name":    "News Sentiment",
            "raw_col": "news_sentiment",
            "pct_col": None,
            "weight":  "10%",
            "note":    "VADER sentiment from live RSS headlines (−1 = very negative, +1 = positive).",
            "color":   "#9b59b6",
            "invert":  True,
        })

    rows_bd = []
    for sig in signal_defs:
        raw_val = cur.get(sig["raw_col"])
        pct_val = cur.get(sig["pct_col"]) if sig["pct_col"] else None
        raw_str = f"{float(raw_val):.4f}" if pd.notna(raw_val) else "N/A"
        pct_str = f"{float(pct_val):.0f}th pctile" if pct_val is not None and pd.notna(pct_val) else "N/A"
        bar_val = float(pct_val) if pct_val is not None and pd.notna(pct_val) else None
        rows_bd.append({
            "Signal": sig["name"],
            "Weight": sig["weight"],
            "Raw value": raw_str,
            "Severity (pctile)": bar_val,
            "color": sig["color"],
            "note": sig["note"],
        })

    # Horizontal percentile bars
    valid_rows = [r for r in rows_bd if r["Severity (pctile)"] is not None]
    if valid_rows:
        fig_bd = go.Figure()
        for r in valid_rows:
            fig_bd.add_trace(go.Bar(
                y=[r["Signal"]],
                x=[r["Severity (pctile)"]],
                orientation="h",
                marker_color=r["color"],
                marker_opacity=0.85,
                text=f"{r['Severity (pctile)']:.0f}th percentile  ·  raw: {r['Raw value']}",
                textposition="outside",
                hovertemplate=(
                    f"<b>{r['Signal']}</b> (weight: {r['Weight']})<br>"
                    f"This week: {r['Raw value']}<br>"
                    f"Percentile: {r['Severity (pctile)']:.0f}th<br>"
                    f"{r['note']}<extra></extra>"
                ),
            ))
        fig_bd.add_vline(x=50, line_dash="dot", line_color="#555", annotation_text="Historical median")
        fig_bd.update_layout(
            xaxis=dict(title="Percentile within country history", range=[0, 130]),
            yaxis=dict(autorange="reversed"),
            height=220,
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=40),
        )
        st.plotly_chart(fig_bd, use_container_width=True)
    else:
        st.info("Percentile data not available — run pipeline to regenerate.")

    # Signal grid with raw values
    sig_cols_grid = st.columns(len(signal_defs))
    for i, sig in enumerate(signal_defs):
        with sig_cols_grid[i]:
            raw_val = cur.get(sig["raw_col"])
            pct_val = cur.get(sig["pct_col"]) if sig["pct_col"] else None
            raw_str = f"{float(raw_val):.4f}" if pd.notna(raw_val) else "—"
            pct_str = f"{float(pct_val):.0f}th pctile" if pct_val is not None and pd.notna(pct_val) else ""
            # Trend arrow
            prev_val = prv.get(sig["raw_col"])
            arrow = ""
            if pd.notna(raw_val) and pd.notna(prev_val):
                diff = float(raw_val) - float(prev_val)
                if sig["invert"]:
                    diff = -diff
                arrow = " ↑" if diff > 0.001 else " ↓" if diff < -0.001 else " →"
            st.markdown(
                f"**{sig['name']}**  \n"
                f"`{raw_str}`{arrow}  \n"
                f"<small style='color:#888'>{pct_str}</small>",
                unsafe_allow_html=True,
            )
            st.caption(sig["note"])

    st.divider()

    # ── SHAP explanation ────────────────────────────────────────────────────

    with st.expander("🔍 What drove this week's XGBoost prediction?"):
        _esc = load_escalation_bundle()
        _bundle = load_xgb()
        if _esc is None and _bundle is None:
            st.info("XGBoost model not found.")
        else:
            try:
                import shap as _shap
                if _esc is not None:
                    # Binary escalation model: SHAP toward P(escalation)
                    feat_names = _esc["feature_cols"]
                    X_last = cdf.sort_values("date").tail(1)[feat_names]
                    sv = _shap.TreeExplainer(_esc["model"]).shap_values(X_last)
                    sv = sv[0] if getattr(sv, "ndim", 2) == 2 else sv
                    shap_title = "contribution to P(escalation next 4 weeks)"
                else:
                    from src.ml_model import compute_shap_for_country
                    sv, feat_names = compute_shap_for_country(
                        _bundle["model"], _bundle["le"], _bundle["feature_cols"],
                        cdf, conflict_component=_bundle["conflict_component"], n_rows=1,
                    )
                    shap_title = "contribution to 'Active Conflict' prediction"

                top_n = 12
                idx_sorted = np.argsort(np.abs(sv))[::-1][:top_n]
                names  = [feat_names[i] for i in idx_sorted]
                values = [sv[i] for i in idx_sorted]
                colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

                fig_shap = go.Figure(go.Bar(
                    y=names,
                    x=values,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in values],
                    textposition="outside",
                ))
                fig_shap.add_vline(x=0, line_color="#555")
                fig_shap.update_layout(
                    title=f"SHAP values — {shap_title}",
                    xaxis_title="SHAP value (red = pushes toward conflict, green = away)",
                    yaxis=dict(autorange="reversed"),
                    height=350,
                    margin=dict(l=10, r=60, t=40, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption(
                    "SHAP (SHapley Additive exPlanations) shows which features most influenced "
                    "this week's prediction. Red = pushed toward Active Conflict; Green = pushed away."
                )
            except Exception as exc:
                st.info(f"SHAP could not be computed: {exc}")

    with st.expander("📐 How is the Intensity Score calculated?"):
        st.markdown(f"""
**Intensity = weighted sum of three RobustScaler-normalized signals:**

| Signal | Weight | Raw value this week |
|---|---|---|
| Conflict Tone (negative) | **40%** | `{ct_now:.4f}` |
| Conflict Volume | **35%** | `{cv_now:.5f}` |
| Conflict Ratio | **25%** | `{cr_now:.4f}` |

Each signal is normalized using RobustScaler (median=0, IQR=1), so outlier conflict spikes
don't compress all other countries. The result is then min-max scaled to [0%, 100%] across
the entire dataset — meaning **100% = the single most conflict-intensive country-week observed**.

{selected} this week: **{intensity_now:.1f}%**  ·  4-week average: **{intensity_4wk:.1f}%**

*Note: news_sentiment is included in the score only when RSS data is available.*
        """)

    st.divider()

    # ── Time series ─────────────────────────────────────────────────────────

    st.subheader("Intensity Over Time")
    fig_ts = go.Figure()

    # HMM state background bands
    if "hmm_state" in cdf.columns:
        sc = cdf["hmm_state"].fillna(-1).astype(int).values
        dates = cdf["date"].values
        seg_s, seg_st = dates[0], sc[0]
        for i in range(1, len(dates)):
            if sc[i] != seg_st or i == len(dates) - 1:
                color = STATE_INT_COLORS.get(int(seg_st), "#888")
                fig_ts.add_vrect(
                    x0=str(seg_s)[:10], x1=str(dates[i])[:10],
                    fillcolor=color, opacity=0.12, layer="below", line_width=0,
                )
                seg_st = sc[i]
                seg_s  = dates[i]

    fig_ts.add_trace(go.Scatter(
        x=cdf["date"], y=cdf["intensity"],
        mode="lines+markers", name="Intensity",
        line=dict(color="#4fc3f7", width=2.5), marker=dict(size=5),
    ))
    if "risk_score" in cdf.columns:
        fig_ts.add_trace(go.Scatter(
            x=cdf["date"], y=cdf["risk_score"],
            mode="lines", name=RISK_LABEL,
            line=dict(color="#ff7043", width=1.5, dash="dot"),
        ))
    if "acled_fatalities" in cdf.columns and cdf["acled_fatalities"].fillna(0).sum() > 0:
        fig_ts.add_trace(go.Bar(
            x=cdf["date"], y=cdf["acled_fatalities"],
            name="ACLED fatalities (right axis)", yaxis="y2",
            marker_color="#8d6e63", opacity=0.45,
        ))
        fig_ts.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Fatalities/week",
                        showgrid=False, rangemode="tozero"),
        )
    # Posterior confidence bands if available
    if "hmm_prob_active_conflict" in cdf.columns:
        fig_ts.add_trace(go.Scatter(
            x=cdf["date"], y=cdf["hmm_prob_active_conflict"],
            mode="lines", name="P(Active Conflict)",
            line=dict(color="#e74c3c", width=1, dash="dashdot"), opacity=0.7,
        ))

    fig_ts.update_layout(
        xaxis_title="Week", yaxis_title="Score [0–1]",
        yaxis=dict(range=[0, 1.05]),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Raw signals
    st.subheader("Raw GDELT Signals Over Time")
    raw_sig_cols = [c for c in ["conflict_tone", "conflict_volume", "conflict_ratio", "peace_signal"]
                    if c in cdf.columns]
    if raw_sig_cols:
        fig_raw = go.Figure()
        colors_raw = {"conflict_tone": "#e74c3c", "conflict_volume": "#e67e22",
                      "conflict_ratio": "#f39c12", "peace_signal": "#2ecc71"}
        for col in raw_sig_cols:
            series = cdf[col].ffill()
            fig_raw.add_trace(go.Scatter(
                x=cdf["date"], y=series,
                mode="lines", name=col.replace("_", " ").title(),
                line=dict(color=colors_raw.get(col, "#888"), width=1.5),
            ))
        fig_raw.update_layout(
            height=260, yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # ── Observed violence — ACLED (draft Step 6: two charts, no redesign) ───
    if "acled_fatalities" in cdf.columns and cdf["acled_fatalities"].notna().any():
        st.subheader("Observed Violence — ACLED")
        acled_last = cdf.loc[cdf["acled_fatalities"].notna(), "date"].max()
        st.caption(
            f"Recorded conflict events (ACLED). Coverage ends **{acled_last:%b %d, %Y}** — "
            "the account tier embargoes the most recent 12 months. "
            "Shaded region = **no data**, not calm."
        )
        emb0 = acled_last + pd.Timedelta(days=7)
        emb1 = cdf["date"].max()

        fig_fat = go.Figure(go.Bar(
            x=cdf["date"], y=cdf["acled_fatalities"],
            marker_color="#e74c3c", name="Fatalities",
            hovertemplate="Week of %{x|%b %d, %Y}<br>Fatalities: %{y:.0f}<extra></extra>",
        ))
        if emb1 > emb0:
            fig_fat.add_vrect(x0=emb0, x1=emb1, fillcolor="#888888", opacity=0.15,
                              line_width=0, annotation_text="embargo — no data",
                              annotation_position="top left")
        fig_fat.update_layout(
            title="Weekly fatalities", height=240, showlegend=False, bargap=0.25,
            yaxis_title="Deaths / week", margin=dict(t=40, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_fat, use_container_width=True)

        # Stack order + hues chosen for CVD-safe adjacency (validated palette)
        type_cols = [
            ("acled_battles",           "Battles",                      "#e74c3c"),
            ("acled_civilian_violence", "Violence against civilians",   "#9b59b6"),
            ("acled_explosions",        "Explosions / remote violence", "#d35400"),
            ("acled_protests",          "Protests",                     "#3498db"),
        ]
        fig_types = go.Figure()
        for col, label, color in type_cols:
            if col in cdf.columns:
                fig_types.add_trace(go.Bar(
                    x=cdf["date"], y=cdf[col], name=label, marker_color=color,
                    marker_line=dict(width=1, color="#0e1117"),
                    hovertemplate=(f"{label}<br>Week of %{{x|%b %d, %Y}}"
                                   "<br>Events: %{y:.0f}<extra></extra>"),
                ))
        if emb1 > emb0:
            fig_types.add_vrect(x0=emb0, x1=emb1, fillcolor="#888888",
                                opacity=0.15, line_width=0)
        fig_types.update_layout(
            barmode="stack", title="Weekly events by type", height=300,
            bargap=0.25, yaxis_title="Events / week",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=70, b=30, l=10, r=10),
        )
        st.plotly_chart(fig_types, use_container_width=True)

    # Feature importance
    fi_df = load_fi()
    if not fi_df.empty:
        st.subheader("Top XGBoost Features (Global)")
        fig_fi = px.bar(
            fi_df.head(12), x="importance", y="feature",
            orientation="h", color="importance", color_continuous_scale="Reds",
        )
        fig_fi.update_layout(
            height=320, yaxis=dict(autorange="reversed"),
            showlegend=False, coloraxis_showscale=False,
            margin=dict(l=10, r=20, t=10, b=40),
            xaxis_title="Importance",
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  —  Forecast
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header(f"Forecast · {selected}")
    st.caption(
        "Forward simulation using the HMM transition matrix. "
        "Each bar is the **probability distribution** over all three states at that future week. "
        "Uncertainty grows with horizon — treat week +8 as much less reliable than week +1."
    )

    hmm_model = load_hmm()
    cdf_fc = df[df["country"] == selected].sort_values("date")

    if hmm_model is None:
        st.warning("HMM model not found. Run the pipeline first.")
    else:
        from src.hmm_model import _assign_state_labels, forecast_state_probs
        valid = cdf_fc[cdf_fc["hmm_state"] >= 0] if "hmm_state" in cdf_fc.columns else pd.DataFrame()

        if valid.empty:
            st.warning("No valid HMM states to forecast from.")
        else:
            state_map = _assign_state_labels(hmm_model)
            cur_state_int = int(valid.iloc[-1]["hmm_state"])
            cur_state_lbl = valid.iloc[-1].get("hmm_state_label", "Unknown")

            c1, c2 = st.columns([3, 1])
            n_weeks = c1.slider("Forecast horizon (weeks)", 2, FORECAST_WEEKS, 6)
            c2.metric("Current state", f"{EMOJI.get(cur_state_lbl,'⚪')} {cur_state_lbl}")

            # Start from the posterior distribution (propagates uncertainty),
            # falling back to a point mass if posteriors are unavailable
            last = valid.iloc[-1]
            init = cur_state_int
            post = []
            for comp in range(hmm_model.n_components):
                col = "hmm_prob_" + state_map[comp].lower().replace(" ", "_")
                post.append(last.get(col, np.nan))
            if all(pd.notna(p) for p in post):
                init = np.array(post, dtype=float)

            fdf = forecast_state_probs(hmm_model, init, n_weeks)
            state_order = [s for s in ["Stable", "Rising Tension", "Active Conflict"] if s in fdf.columns]

            fig_fc = go.Figure()
            for s in state_order:
                fig_fc.add_trace(go.Bar(
                    name=s, x=[f"+{w}w" for w in fdf["week"]], y=fdf[s],
                    marker_color=STATE_COLORS.get(s, "#888"),
                ))
            fig_fc.update_layout(
                barmode="stack",
                yaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
                height=360, legend_title_text="State",
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Transition matrix
            st.subheader("HMM Transition Matrix")
            st.caption(
                "P(next state | current state). Each row sums to 1. "
                "High diagonal = strong state persistence (conflicts tend to stay in their current phase)."
            )
            trans_labels = [state_map.get(i, str(i)) for i in range(hmm_model.n_components)]
            trans_df = pd.DataFrame(hmm_model.transmat_, index=trans_labels, columns=trans_labels)
            fig_tm = px.imshow(
                trans_df, text_auto=".2f",
                color_continuous_scale="RdYlGn_r", zmin=0, zmax=1, aspect="auto",
            )
            fig_tm.update_layout(height=280, coloraxis_showscale=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig_tm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  —  Live News
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header(f"Live News · {selected}")
    st.caption(
        "Headlines from BBC Middle East, Al Jazeera, NYT. "
        "Matched by country name aliases (e.g., 'Syrian', 'Houthi' → Yemen). "
        "VADER sentiment: −1 very negative, 0 neutral, +1 very positive."
    )

    with st.spinner("Fetching headlines..."):
        news_df = load_news(all_countries)

    if news_df.empty:
        st.warning("No headlines fetched.")
    else:
        cnews = news_df[news_df["country"] == selected].copy()

        if cnews.empty:
            st.info(
                f"No headlines matched '{selected}' or its known aliases. "
                "This can happen if major outlets are focused elsewhere this week."
            )
        else:
            cnews["date"] = pd.to_datetime(cnews["date"])
            cnews = cnews.sort_values("date", ascending=False)

            avg_s  = cnews["sentiment"].mean()
            neg_n  = (cnews["sentiment"] < -0.05).sum()
            pos_n  = (cnews["sentiment"] > 0.05).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Headlines", len(cnews))
            c2.metric("Avg Sentiment", f"{avg_s:+.3f}")
            c3.metric("Negative", f"{neg_n}")
            c4.metric("Positive", f"{pos_n}")

            if len(cnews) >= 3:
                dly = (
                    cnews.groupby(cnews["date"].dt.date)["sentiment"].mean()
                    .reset_index()
                )
                dly.columns = ["date", "avg_sentiment"]
                fig_s = px.bar(
                    dly, x="date", y="avg_sentiment",
                    color="avg_sentiment", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, range_color=[-1, 1],
                    title="Daily Avg Headline Sentiment",
                    labels={"avg_sentiment": "Sentiment", "date": ""},
                )
                fig_s.update_layout(height=200, coloraxis_showscale=False,
                                    margin=dict(t=40, b=20))
                st.plotly_chart(fig_s, use_container_width=True)

            st.divider()
            for _, row in cnews.head(25).iterrows():
                score  = float(row.get("sentiment", 0))
                color  = "#2ecc71" if score > 0.05 else "#e74c3c" if score < -0.05 else "#888"
                label  = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
                url    = row.get("url", "")
                title  = row.get("title", "")
                source = row.get("source", "")
                date_s = str(row.get("date", ""))[:10]
                link   = (f'<a href="{url}" target="_blank" style="color:#90caf9;">{title}</a>'
                          if url else title)
                st.markdown(
                    f"""<div style="border-left:4px solid {color};padding:8px 14px;
margin-bottom:6px;background:#1a1a2e;border-radius:4px;">
{link}<br>
<small style="color:#aaa;">{source} &nbsp;·&nbsp; {date_s} &nbsp;·&nbsp;
<span style="color:{color};">{label} ({score:+.2f})</span></small>
</div>""",
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  —  Report (charter Phase 1: typed empirical sentences)
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    from src.report import build_report

    @st.cache_data(ttl=3600)
    def load_acled_events(country: str):
        p = Path(__file__).parent.parent / "data" / "raw" / "acled" / f"{country.lower().replace(' ', '_')}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    rep = build_report(df, selected, acled_events=load_acled_events(selected),
                       metrics=load_metrics())

    st.header(f"Assessment Report · {selected}")
    st.caption(f"Week of **{rep['week']:%B %d, %Y}**")

    # Completeness strip — a partial picture must announce itself first
    comp_bits = []
    for c in rep["completeness"]:
        if c["current"]:
            comp_bits.append(f"🟢 {c['source']}: current")
        elif c["through"] is not None:
            comp_bits.append(f"🟡 {c['source']}: through {c['through']:%b %d, %Y}")
        else:
            comp_bits.append(f"🔴 {c['source']}: absent")
    st.markdown(" &nbsp;·&nbsp; ".join(comp_bits))

    BADGES = {
        "measure":  ("MEASURED",  "#3498db"),
        "change":   ("CHANGE",    "#e67e22"),
        "constant": ("CONSTANT",  "#2ecc71"),
        "forecast": ("FORECAST",  "#9b59b6"),
        "gap":      ("DATA GAP",  "#e74c3c"),
    }
    with st.expander("ℹ️ How to read this report"):
        st.markdown(
            "Every sentence is typed under the project's empirical standard: "
            "**MEASURED** — a quantity with a named source and timestamp. "
            "**CHANGE / CONSTANT** — a claim backed by a Mann-Whitney test "
            "(last 4 weeks vs the prior 12; change is claimed only at p < 0.05). "
            "**FORECAST** — model output, shipped with its validation caveats "
            "(uncalibrated scores are rankings, not probabilities). "
            "**DATA GAP** — declared missing data; absence of data is never shown as calm. "
            "Citations under each sentence resolve to source records."
        )

    for sec in rep["sections"]:
        as_of = f" — as of {sec['as_of']:%b %d, %Y}" if sec.get("as_of") is not None else ""
        stale = (sec.get("as_of") is not None and rep["week"] is not None
                 and sec["as_of"] < rep["week"])
        st.subheader(sec["title"] + as_of)
        if stale:
            st.caption("⚠️ This section is older than the report week — staleness declared, not hidden.")
        for s in sec["sentences"]:
            label, color = BADGES[s["type"]]
            st.markdown(
                f"<span style='background:{color};color:#fff;padding:1px 7px;"
                f"border-radius:3px;font-size:0.72em;font-weight:600'>{label}</span>&nbsp; "
                f"{s['text']}",
                unsafe_allow_html=True,
            )
            if s.get("cite"):
                st.caption(f"↳ {s['cite']}")
        if sec.get("records"):
            with st.expander(f"📎 Source records ({len(sec['records'])})"):
                for rec in sec["records"]:
                    st.markdown(f"- `{rec}`")
        st.divider()

    # Reliability curve — a published probability ships with its calibration
    _rel_path = Path(__file__).parent.parent / "data" / "processed" / "reliability.parquet"
    if _rel_path.exists():
        with st.expander("📐 Calibration — reliability curve (out-of-sample folds)"):
            rel = pd.read_parquet(_rel_path)
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(
                x=[0, rel[["mean_raw", "observed"]].max().max() * 1.05] * 1,
                y=[0, rel[["mean_raw", "observed"]].max().max() * 1.05],
                mode="lines", name="perfect calibration",
                line=dict(color="#888888", width=1, dash="dot"), hoverinfo="skip",
            ))
            fig_rel.add_trace(go.Scatter(
                x=rel["mean_raw"], y=rel["observed"], mode="lines+markers",
                name="raw score", marker=dict(size=9), line=dict(color="#d35400", width=2),
                hovertemplate="raw %{x:.3f} → observed %{y:.3f}<extra></extra>",
            ))
            fig_rel.add_trace(go.Scatter(
                x=rel["mean_cal"], y=rel["observed"], mode="lines+markers",
                name="calibrated", marker=dict(size=9), line=dict(color="#3498db", width=2),
                hovertemplate="calibrated %{x:.3f} → observed %{y:.3f}<extra></extra>",
            ))
            fig_rel.update_layout(
                height=320, xaxis_title="Mean predicted probability (per quintile bin)",
                yaxis_title="Observed escalation rate",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=40, b=40, l=10, r=10),
            )
            st.plotly_chart(fig_rel, use_container_width=True)
            st.caption(
                "Each point is a quintile of out-of-sample predictions (purged walk-forward). "
                "Raw scores are overconfident (far right of the diagonal); Platt scaling pulls "
                "them onto it, at the cost of compressing them toward the base rate."
            )


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Data: GDELT Project · RSS (BBC, Al Jazeera, NYT) · as of {data_as_of}.  "
    "Model: GaussianHMM (forward-backward posteriors) + XGBoost (SHAP explanations).  "
    "States are inferred latent variables, not ground-truth conflict classifications."
)
