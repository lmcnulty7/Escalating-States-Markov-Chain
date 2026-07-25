"""Orchestrates: data → quality gate → features → ground truth → HMM → XGBoost → results."""

import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MENA_COUNTRIES, PROCESSED_DIR
from src.data_fetcher import fetch_all_gdelt, fetch_rss_headlines
from src.features import build_feature_matrix
from src.ground_truth import add_escalation_target, fetch_ucdp_fatalities
from src.hmm_model import (
    conflict_component, decode_states, fit_hmm, is_degenerate, save_hmm,
)
from src.ml_model import (
    get_feature_importance, predict_risk, save_metrics, save_model,
    train_escalation_model, train_model,
)

logger = logging.getLogger(__name__)

RESULTS_PATH     = PROCESSED_DIR / "results.parquet"
FEATURE_IMP_PATH = PROCESSED_DIR / "feature_importance.parquet"


def run_pipeline(
    countries: list[str] | None = None,
    force_refresh: bool = False,
    allow_missing: bool = False,
) -> pd.DataFrame:
    countries = countries or list(MENA_COUNTRIES.keys())
    run_metrics: dict = {}

    logger.info("=== Step 1: Data collection ===")
    gdelt_df = fetch_all_gdelt(countries, force_refresh=force_refresh)
    if gdelt_df.empty:
        raise RuntimeError("GDELT returned no data.")

    # Quality gate: a regional monitor silently missing countries is worse
    # than one that refuses to run. Yemen/Iran/Palestine going missing must
    # be loud, not a log line.
    missing = sorted(set(countries) - set(gdelt_df["country"].unique()))
    run_metrics["missing_countries"] = missing
    if missing:
        msg = f"No data for {len(missing)} countries: {', '.join(missing)}"
        if allow_missing:
            logger.error("%s — continuing because allow_missing=True", msg)
        else:
            raise RuntimeError(f"{msg}. Re-run fetch, or pass --allow-missing to proceed.")

    news_df = fetch_rss_headlines(countries=countries)
    logger.info("  RSS headlines: %d", len(news_df))

    logger.info("=== Step 1b: ACLED events (embargo-aware, cache-first) ===")
    from src.acled_fetcher import accessible_end, calls_used, fetch_all_acled_weekly
    try:
        acled_weekly = fetch_all_acled_weekly(countries)
        run_metrics["acled_countries"] = (
            sorted(acled_weekly["country"].unique()) if not acled_weekly.empty else []
        )
        run_metrics["acled_coverage_end"] = str(accessible_end().date())
        run_metrics["acled_api_calls"] = calls_used()
    except Exception as exc:
        logger.error("ACLED fetch failed (%s) — continuing without ACLED", exc)
        acled_weekly = pd.DataFrame()

    logger.info("=== Step 1c: GDELT event-file archive ===")
    from src.gdelt_files import compute_event_features
    ev_weekly = compute_event_features()
    if not ev_weekly.empty:
        run_metrics["gdelt_ev_rows"] = len(ev_weekly)
        run_metrics["gdelt_ev_range"] = f"{ev_weekly['date'].min().date()} → {ev_weekly['date'].max().date()}"
        logger.info("  Event archive: %d country-weeks (%s)",
                    len(ev_weekly), run_metrics["gdelt_ev_range"])
    else:
        logger.warning("  No event archive yet (run src.gdelt_files.build_archive)")

    logger.info("=== Step 2: Feature engineering ===")
    features_df = build_feature_matrix(
        gdelt_df, news_df, acled_weekly=acled_weekly, ev_weekly=ev_weekly,
    )
    logger.info("  Feature matrix: %d rows × %d cols", *features_df.shape)

    logger.info("=== Step 2b: Escalation target (ACLED → UCDP → none) ===")
    from src.ground_truth import add_escalation_target_from_column
    if "acled_fatalities" in features_df.columns and \
            features_df["acled_fatalities"].notna().sum() >= 50:
        features_df = add_escalation_target_from_column(features_df, "acled_fatalities")
        run_metrics["target_source"] = "acled_fatalities"
    else:
        fatalities = fetch_ucdp_fatalities()
        if not fatalities.empty:
            features_df = add_escalation_target(features_df, fatalities)
            run_metrics["target_source"] = "ucdp_ged"
        else:
            run_metrics["target_source"] = "hmm_fallback"
            logger.warning("  No ground-truth fatalities — falling back to HMM-state target")
    if "escalated_next4w" in features_df.columns:
        n_lbl = features_df["escalated_next4w"].notna().sum()
        logger.info("  Target (%s): %d labelled rows, %.0f escalations",
                    run_metrics["target_source"], n_lbl,
                    features_df["escalated_next4w"].sum())

    logger.info("=== Step 3: HMM ===")
    hmm_model  = fit_hmm(features_df)
    features_df = decode_states(hmm_model, features_df)   # adds hmm_state + posteriors
    save_hmm(hmm_model)
    run_metrics["hmm_transmat_problems"] = is_degenerate(hmm_model)
    conflict_comp = conflict_component(hmm_model)

    logger.info("=== Step 4: XGBoost ===")
    esc = train_escalation_model(features_df)
    if esc is not None:
        # Primary path: risk = P(observed escalation next 4 weeks)
        model, feature_cols, cutoff, m = esc
        run_metrics["single_split"] = m

        # Purged walk-forward: the honest skill estimate + the OOS predictions
        # calibration is fit on (charter §5.1/§5.2)
        from sklearn.metrics import brier_score_loss
        from src.ml_model import (
            _fit_escalation_xgb, fit_calibrator, walk_forward_escalation,
        )
        calibrator = None
        wf = walk_forward_escalation(features_df)
        if wf is not None:
            oos, folds, pooled = wf
            run_metrics["walk_forward"] = {"folds": folds, **pooled}
            calibrator, reliability = fit_calibrator(oos)
            y = oos["escalated_next4w"].astype(int)
            p_cal = calibrator.predict_proba(oos["score"].to_numpy().reshape(-1, 1))[:, 1]
            run_metrics["brier_calibrated"] = float(brier_score_loss(y, p_cal))
            # Gate: calibrated probabilities are printable ONLY if they beat
            # always-predict-the-base-rate on held-out data
            run_metrics["calibration_ok"] = bool(
                run_metrics["brier_calibrated"] <= pooled["brier_climatology"]
            )
            reliability.to_parquet(PROCESSED_DIR / "reliability.parquet", index=False)
            logger.info("Walk-forward: PR-AUC %.3f vs %.3f persistence | Brier cal %.4f vs %.4f climatology | calibration_ok=%s",
                        pooled["pr_auc"], pooled["persistence_pr_auc"],
                        run_metrics["brier_calibrated"], pooled["brier_climatology"],
                        run_metrics["calibration_ok"])

        # Deployed model: refit on ALL labelled rows (eval is done; don't
        # ship a model that ignores the newest 30% of history)
        labelled = features_df.dropna(subset=["escalated_next4w"])
        model = _fit_escalation_xgb(labelled[feature_cols],
                                    labelled["escalated_next4w"].astype(int))
        features_df["risk_score"] = model.predict_proba(
            features_df[feature_cols])[:, 1]
        features_df["risk_in_sample"] = features_df["escalated_next4w"].notna()
        if calibrator is not None:
            features_df["risk_calibrated"] = calibrator.predict_proba(
                features_df["risk_score"].to_numpy().reshape(-1, 1))[:, 1]
        features_df["risk_target"] = run_metrics.get("target_source", "escalation") + "_escalation"
        import joblib
        from config import MODELS_DIR
        joblib.dump({"model": model, "feature_cols": feature_cols,
                     "target": "escalated_next4w", "calibrator": calibrator},
                    MODELS_DIR / "escalation_model.joblib")
        fi = get_feature_importance(model, feature_cols)
        fi.to_parquet(FEATURE_IMP_PATH, index=False)
    else:
        # Fallback: next-week HMM state (media-derived, weaker claim).
        # Train ONLY on weeks with real DOC data — on extended-spine rows the
        # intensity (and thus the HMM label) is backfilled filler, not signal.
        labelled = features_df[
            (features_df["hmm_state"] >= 0) & features_df["conflict_volume"].notna()
        ]
        if len(labelled) >= 20:
            xgb_model, le, feature_cols, cutoff, m = train_model(labelled)
            run_metrics.update(m)
            features_df = predict_risk(
                xgb_model, le, features_df, feature_cols,
                conflict_component=conflict_comp, train_cutoff=cutoff,
            )
            features_df["risk_target"] = "next_hmm_state"
            save_model(xgb_model, le, feature_cols, conflict_component=conflict_comp)
            fi = get_feature_importance(xgb_model, feature_cols)
            fi.to_parquet(FEATURE_IMP_PATH, index=False)
            logger.info("  Top feature: %s (%.3f)", fi.iloc[0]["feature"], fi.iloc[0]["importance"])
        else:
            logger.warning("  Too few rows (%d) to train XGBoost reliably", len(labelled))

    save_metrics(run_metrics)
    features_df.to_parquet(RESULTS_PATH, index=False)
    logger.info("Results saved → %s", RESULTS_PATH)
    return features_df


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"No results at {RESULTS_PATH}. Run `python run_pipeline.py` first."
        )
    return pd.read_parquet(RESULTS_PATH)


def load_feature_importance() -> pd.DataFrame:
    return pd.read_parquet(FEATURE_IMP_PATH) if FEATURE_IMP_PATH.exists() else pd.DataFrame()
