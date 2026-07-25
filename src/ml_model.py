"""
XGBoost model: predict next week's HMM conflict state from current features.

Fixes vs v1:
  - risk_score now uses the HMM component actually labelled "Active Conflict"
    (component indices are arbitrary; the old code assumed the last one)
  - Evaluated against the persistence baseline ("next state = current state")
    overall AND on transition weeks — the only weeks that matter
  - Metrics persisted to data/processed/metrics.json
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, PROCESSED_DIR, XGB_PARAMS
from src.features import get_feature_cols

logger = logging.getLogger(__name__)

METRICS_PATH = PROCESSED_DIR / "metrics.json"


def _prepare_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.sort_values(["country", "date"]).copy()
    df["next_state"] = df.groupby("country")["hmm_state"].shift(-1)
    df = df.dropna(subset=["next_state", "hmm_state"])
    df = df[df["hmm_state"] >= 0]

    feature_cols = [c for c in get_feature_cols(df) if c != "hmm_state"]
    # NaN stays NaN: XGBoost handles missing natively. Zero-filling would turn
    # "ACLED embargoed / no data" into "zero fatalities" — fake calm.
    X = df[feature_cols]
    y = df["next_state"].astype(int)
    return X, y, feature_cols


def train_model(
    df: pd.DataFrame,
) -> tuple[XGBClassifier, LabelEncoder, list[str], pd.Timestamp, dict]:
    """
    Temporal 70/30 split. Evaluates model AND the persistence baseline
    ("next state = current state"), overall and on transition weeks.
    Returns (model, label_encoder, feature_cols, train_cutoff, metrics).
    """
    df = df.sort_values("date")
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    train_cutoff = train_df["date"].max()

    X_train, y_train, feature_cols = _prepare_Xy(train_df)
    X_test,  y_test,  _            = _prepare_Xy(test_df)

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)

    model = XGBClassifier(
        num_class=len(le.classes_),
        objective="multi:softprob",
        **XGB_PARAMS,
    )
    model.fit(X_train, y_train_enc, verbose=False)

    metrics: dict = {"train_rows": len(X_train), "test_rows": len(X_test),
                     "train_cutoff": str(train_cutoff.date())}

    if len(X_test) > 0 and len(y_test) > 0:
        valid_mask = y_test.isin(le.classes_)
        if valid_mask.sum() > 0:
            yt = y_test[valid_mask].to_numpy()
            y_pred = le.inverse_transform(model.predict(X_test[valid_mask]))

            # Persistence baseline: predict next state = current state
            test_prepared = test_df.sort_values(["country", "date"]).copy()
            test_prepared["next_state"] = test_prepared.groupby("country")["hmm_state"].shift(-1)
            test_prepared = test_prepared.dropna(subset=["next_state", "hmm_state"])
            test_prepared = test_prepared[test_prepared["hmm_state"] >= 0]
            y_persist = test_prepared["hmm_state"].to_numpy()[valid_mask.to_numpy()]

            transitions = yt != y_persist   # weeks where the state actually changed

            metrics["model_acc"]          = float((y_pred == yt).mean())
            metrics["persistence_acc"]    = float((y_persist == yt).mean())
            metrics["n_transition_weeks"] = int(transitions.sum())
            if transitions.sum() > 0:
                metrics["model_acc_transitions"] = float((y_pred[transitions] == yt[transitions]).mean())
                # persistence is 0% on transitions by definition
                metrics["persistence_acc_transitions"] = 0.0

            logger.info(
                "XGBoost test acc: %.3f | persistence baseline: %.3f | "
                "transition weeks: %d (model correct on %.0f%%)",
                metrics["model_acc"], metrics["persistence_acc"],
                metrics["n_transition_weeks"],
                100 * metrics.get("model_acc_transitions", 0),
            )
            logger.info(
                "XGBoost test report:\n%s",
                classification_report(yt, y_pred, zero_division=0),
            )

    logger.info("XGBoost trained on %d rows, %d features", len(X_train), len(feature_cols))
    return model, le, feature_cols, train_cutoff, metrics


def train_escalation_model(df: pd.DataFrame):
    """
    Binary model on the UCDP ground-truth target (escalated_next4w).
    This is the honest task: media signals -> future OBSERVED escalation.
    Returns (model, feature_cols, train_cutoff, metrics) or None if the
    target is missing / has too few positive examples.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss

    if "escalated_next4w" not in df.columns:
        return None
    data = df.dropna(subset=["escalated_next4w"]).sort_values("date")
    if len(data) < 50 or data["escalated_next4w"].sum() < 10:
        logger.warning("Too little UCDP-labelled data for escalation model "
                       "(%d rows, %.0f positives)", len(data),
                       data["escalated_next4w"].sum() if len(data) else 0)
        return None

    feature_cols = get_feature_cols(data)
    split_idx = int(len(data) * 0.70)
    train_df, test_df = data.iloc[:split_idx], data.iloc[split_idx:]
    train_cutoff = train_df["date"].max()

    X_tr, y_tr = train_df[feature_cols], train_df["escalated_next4w"].astype(int)
    X_te, y_te = test_df[feature_cols],  test_df["escalated_next4w"].astype(int)

    pos_weight = max((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), 1.0)
    params = {k: v for k, v in XGB_PARAMS.items() if k != "eval_metric"}
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          scale_pos_weight=pos_weight, **params)
    model.fit(X_tr, y_tr, verbose=False)

    metrics = {
        "target": "escalated_next4w",
        "train_rows": len(X_tr), "test_rows": len(X_te),
        "train_cutoff": str(train_cutoff.date()),
        "test_base_rate": float(y_te.mean()) if len(y_te) else None,
    }
    if len(y_te) and y_te.nunique() > 1:
        p = model.predict_proba(X_te)[:, 1]
        metrics["brier"] = float(brier_score_loss(y_te, p))
        metrics["pr_auc"] = float(average_precision_score(y_te, p))
        # Persistence baseline: "escalating in the past 4 weeks" as the score
        persist = (
            test_df.groupby("country")["escalated_next4w"]
            .shift(4).fillna(0).to_numpy()
        )
        metrics["persistence_pr_auc"] = float(average_precision_score(y_te, persist))
        logger.info("Escalation model: Brier %.3f | PR-AUC %.3f (baseline %.3f, base rate %.3f)",
                    metrics["brier"], metrics["pr_auc"],
                    metrics["persistence_pr_auc"], metrics["test_base_rate"])

    return model, feature_cols, train_cutoff, metrics


PURGE_WEEKS = 4   # target at week t uses fatalities t+1..t+4 — purge that
                  # window before each test block or training labels leak
                  # test-period outcomes (the old single split had this flaw)


def _fit_escalation_xgb(X, y):
    pos_weight = max((len(y) - y.sum()) / max(y.sum(), 1), 1.0)
    params = {k: v for k, v in XGB_PARAMS.items() if k != "eval_metric"}
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          scale_pos_weight=pos_weight, **params)
    model.fit(X, y, verbose=False)
    return model


def walk_forward_escalation(df: pd.DataFrame, n_folds: int = 4,
                            min_train_frac: float = 0.4):
    """
    Purged expanding-window evaluation of the escalation model.
    Returns (oos, fold_metrics, pooled) where oos has out-of-sample
    predictions for every test-block row, or None if data is insufficient.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss

    if "escalated_next4w" not in df.columns:
        return None
    data = df.dropna(subset=["escalated_next4w"]).sort_values("date")
    if len(data) < 200 or data["escalated_next4w"].sum() < 20:
        logger.warning("Too little labelled data for walk-forward")
        return None

    feature_cols = get_feature_cols(data)
    dates = sorted(data["date"].unique())
    blocks = np.array_split(dates[int(len(dates) * min_train_frac):], n_folds)

    oos_parts, fold_metrics = [], []
    for i, block in enumerate(blocks):
        block = list(block)
        purge_cutoff = block[0] - pd.Timedelta(weeks=PURGE_WEEKS)
        train = data[data["date"] <= purge_cutoff]
        test = data[data["date"].isin(block)]
        if train["escalated_next4w"].sum() < 8 or len(test) == 0:
            continue
        model = _fit_escalation_xgb(train[feature_cols],
                                    train["escalated_next4w"].astype(int))
        part = test[["country", "date", "escalated_next4w"]].copy()
        part["score"] = model.predict_proba(test[feature_cols])[:, 1]
        oos_parts.append(part)

        y, p = part["escalated_next4w"].astype(int), part["score"]
        fm = {"fold": i + 1, "train_rows": len(train), "test_rows": len(test),
              "test_start": str(block[0].date()), "test_end": str(block[-1].date()),
              "base_rate": float(y.mean())}
        if y.nunique() > 1:
            fm["pr_auc"] = float(average_precision_score(y, p))
            fm["brier"] = float(brier_score_loss(y, p))
        fold_metrics.append(fm)

    if not oos_parts:
        return None
    oos = pd.concat(oos_parts, ignore_index=True)
    y, p = oos["escalated_next4w"].astype(int), oos["score"]

    persist = (
        oos.sort_values(["country", "date"])
        .groupby("country")["escalated_next4w"].shift(4).fillna(0)
    )
    pooled = {
        "oos_rows": len(oos), "oos_base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_raw": float(brier_score_loss(y, p)),
        "brier_climatology": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "persistence_pr_auc": float(average_precision_score(y, persist)),
    }
    return oos, fold_metrics, pooled


def fit_calibrator(oos: pd.DataFrame):
    """Platt scaling (sigmoid) on pooled out-of-sample scores — stable with few
    positives, unlike isotonic. Returns (calibrator, reliability_table)."""
    from sklearn.linear_model import LogisticRegression

    y = oos["escalated_next4w"].astype(int).to_numpy()
    s = oos["score"].to_numpy().reshape(-1, 1)
    cal = LogisticRegression(C=1e6)
    cal.fit(s, y)

    calibrated = cal.predict_proba(s)[:, 1]
    bins = pd.qcut(oos["score"], q=5, duplicates="drop")
    rel = (
        pd.DataFrame({"bin": bins, "raw": oos["score"], "cal": calibrated, "y": y})
        .groupby("bin", observed=True)
        .agg(mean_raw=("raw", "mean"), mean_cal=("cal", "mean"),
             observed=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    return cal, rel


def save_metrics(metrics: dict) -> None:
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", METRICS_PATH)


def predict_risk(
    model: XGBClassifier,
    le: LabelEncoder,
    df: pd.DataFrame,
    feature_cols: list[str],
    conflict_component: int,
    train_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    risk_score = P(next state is the component labelled "Active Conflict").
    HMM component indices are arbitrary — the caller must pass the component
    whose learned mean intensity is highest (see hmm_model._assign_state_labels).
    """
    df = df.copy()
    X = df[feature_cols]          # NaN = missing, handled natively by XGBoost
    proba = model.predict_proba(X)
    pred_enc = model.predict(X)

    classes = list(le.classes_)
    if conflict_component in classes:
        df["risk_score"] = proba[:, classes.index(conflict_component)]
    else:
        # Training data never reached Active Conflict — probability is unknowable
        logger.warning("Conflict component %d absent from training labels; risk_score set to NaN",
                       conflict_component)
        df["risk_score"] = np.nan

    df["predicted_state"] = le.inverse_transform(pred_enc)
    # Flag rows the model was trained on — their risk_score is in-sample
    if train_cutoff is not None:
        df["risk_in_sample"] = df["date"] <= train_cutoff
    return df


def get_feature_importance(
    model: XGBClassifier,
    feature_cols: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_shap_for_country(
    model: XGBClassifier,
    le: LabelEncoder,
    feature_cols: list[str],
    country_df: pd.DataFrame,
    conflict_component: int,
    n_rows: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """
    SHAP values for the last n_rows of a country's data, for the class that is
    actually "Active Conflict" (mapped via conflict_component, not position).
    Returns (values of shape (n_features,), feature_names).
    """
    import shap

    X = country_df.sort_values("date").tail(n_rows)[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    classes = list(le.classes_)
    cls = classes.index(conflict_component) if conflict_component in classes else len(classes) - 1

    if isinstance(shap_vals, list):                      # older shap: list of (n_rows, n_features)
        sv = shap_vals[cls][0]
    elif getattr(shap_vals, "ndim", 2) == 3:             # newer shap: (n_rows, n_features, n_classes)
        sv = shap_vals[0, :, cls]
    else:                                                # binary: (n_rows, n_features)
        sv = shap_vals[0]
    return sv, feature_cols


def save_model(
    model: XGBClassifier,
    le: LabelEncoder,
    feature_cols: list[str],
    conflict_component: int,
) -> Path:
    path = MODELS_DIR / "xgb_model.joblib"
    joblib.dump({
        "model": model, "le": le, "feature_cols": feature_cols,
        "conflict_component": conflict_component,
    }, path)
    logger.info("XGBoost saved to %s", path)
    return path


def load_model(path: Path | None = None) -> dict:
    """Returns bundle dict: model, le, feature_cols, conflict_component."""
    path = path or (MODELS_DIR / "xgb_model.joblib")
    bundle = joblib.load(path)
    bundle.setdefault("conflict_component", len(bundle["le"].classes_) - 1)
    return bundle
