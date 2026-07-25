"""
Hidden Markov Model layer.

Fits a single GaussianHMM across all country sequences, then:
  - Decodes Viterbi state sequence (hmm_state, hmm_state_label)
  - Computes posterior state probabilities via forward-backward
    (hmm_prob_stable, hmm_prob_rising_tension, hmm_prob_active_conflict)
  - Forecasts forward state probability distributions
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FORECAST_WEEKS, MODELS_DIR, N_STATES, STATE_LABELS

logger = logging.getLogger(__name__)


def _seq_intensity(grp: pd.DataFrame) -> np.ndarray:
    """Gap weeks are forward-filled, NOT zero-filled — zero means 'perfectly
    calm', which is the wrong imputation for missing data."""
    vals = grp.sort_values("date")["intensity"].ffill().bfill()
    return vals.dropna().values.reshape(-1, 1)


def _sequences_from_df(df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    seqs, lengths = [], []
    for _, grp in df.groupby("country"):
        seq = _seq_intensity(grp)
        if len(seq) >= 3:
            seqs.append(seq)
            lengths.append(len(seq))
    if not seqs:
        return np.empty((0, 1)), []
    return np.vstack(seqs), lengths


def is_degenerate(model: hmm.GaussianHMM) -> list[str]:
    """
    Sanity-check the transition matrix. Returns a list of problems (empty = OK).
    Catches absorbing states (can never leave) and unreachable states
    (can never enter) — both make forecasts meaningless.
    """
    problems = []
    T = model.transmat_
    for i in range(model.n_components):
        if T[i, i] > 0.999:
            problems.append(f"state {i} is absorbing (self-transition {T[i, i]:.3f})")
        inbound = T[:, i].sum() - T[i, i]
        if inbound < 1e-3:
            problems.append(f"state {i} is unreachable from other states")
    return problems


def fit_hmm(df: pd.DataFrame, n_states: int = N_STATES) -> hmm.GaussianHMM:
    """
    Fit with several random seeds; prefer the best-likelihood NON-degenerate
    solution. EM on small pooled data can carve out absorbing 'boring country'
    states — those fits are rejected if any healthy alternative exists.
    """
    X, lengths = _sequences_from_df(df)
    if len(X) == 0:
        raise ValueError("No sequences with enough data to fit HMM")

    candidates = []
    for seed in (42, 7, 123, 2024, 99):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=300,
            tol=1e-5,
            random_state=seed,
        )
        try:
            model.fit(X, lengths)
            ll = model.score(X, lengths)
            candidates.append((ll, seed, model, is_degenerate(model)))
        except Exception as exc:
            logger.warning("HMM fit failed for seed %d: %s", seed, exc)

    if not candidates:
        raise ValueError("HMM failed to fit for all seeds")

    healthy = [c for c in candidates if not c[3]]
    pool = healthy or candidates
    ll, seed, model, problems = max(pool, key=lambda c: c[0])

    if problems:
        logger.warning("All HMM fits degenerate; best (seed %d) has: %s", seed, "; ".join(problems))
    logger.info("HMM fitted (seed %d, loglik %.1f). Transition matrix:\n%s",
                seed, ll, np.round(model.transmat_, 3))
    return model


def _assign_state_labels(model: hmm.GaussianHMM) -> dict[int, str]:
    """Map component indices to labels ordered by mean intensity (low→Stable)."""
    means = model.means_.flatten()
    order = np.argsort(means)
    label_list = [STATE_LABELS[i] for i in range(len(order))]
    return {int(order[i]): label_list[i] for i in range(len(order))}


def _col_name(label: str) -> str:
    return "hmm_prob_" + label.lower().replace(" ", "_")


def decode_states(model: hmm.GaussianHMM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add to df:
      hmm_state         — Viterbi integer state
      hmm_state_label   — semantic label (Stable / Rising Tension / Active Conflict)
      hmm_prob_*        — posterior probability of each state (forward-backward)
    """
    df = df.copy().sort_values(["country", "date"]).reset_index(drop=True)
    state_map = _assign_state_labels(model)

    state_col = np.full(len(df), -1, dtype=int)
    prob_cols: dict[str, np.ndarray] = {
        _col_name(lbl): np.full(len(df), np.nan) for lbl in STATE_LABELS.values()
    }

    for country, grp in df.groupby("country"):
        seq = _seq_intensity(grp)
        if len(seq) < 2 or len(seq) != len(grp):
            if len(seq) != len(grp):
                logger.warning("Skipping %s decode: %d NaN intensity rows", country, len(grp) - len(seq))
            continue
        try:
            # Viterbi states
            states = model.predict(seq)
            state_col[grp.index] = states

            # Posterior probabilities (forward-backward smoothing)
            posteriors = model.predict_proba(seq)   # (n_timesteps, n_components)
            for comp_idx, label in state_map.items():
                col = _col_name(label)
                prob_cols[col][grp.index] = posteriors[:, comp_idx]

        except Exception as exc:
            logger.warning("HMM decode failed for %s: %s", country, exc)

    df["hmm_state"] = state_col
    df["hmm_state_label"] = df["hmm_state"].map(state_map).fillna("Unknown")
    for col, vals in prob_cols.items():
        df[col] = vals

    return df


def conflict_component(model: hmm.GaussianHMM) -> int:
    """Component index whose label is 'Active Conflict' (highest mean intensity)."""
    state_map = _assign_state_labels(model)
    for comp, label in state_map.items():
        if label == STATE_LABELS[len(STATE_LABELS) - 1]:
            return comp
    return model.n_components - 1


def forecast_state_probs(
    model: hmm.GaussianHMM,
    current: int | np.ndarray,
    n_weeks: int = FORECAST_WEEKS,
) -> pd.DataFrame:
    """
    Forward-simulate state probability distributions.
    `current` is either a state index (point mass) or, preferably, the
    posterior probability vector — starting from the posterior propagates
    the model's uncertainty instead of pretending the state is known.
    """
    state_map = _assign_state_labels(model)

    if np.isscalar(current):
        probs = np.zeros(model.n_components)
        probs[int(current)] = 1.0
    else:
        probs = np.asarray(current, dtype=float)
        probs = probs / probs.sum()

    rows = []
    for w in range(1, n_weeks + 1):
        probs = probs @ model.transmat_
        row = {"week": w}
        for idx, p in enumerate(probs):
            row[state_map[idx]] = float(p)
        rows.append(row)

    return pd.DataFrame(rows)


def save_hmm(model: hmm.GaussianHMM, path: Path | None = None) -> Path:
    path = path or (MODELS_DIR / "hmm_model.joblib")
    joblib.dump(model, path)
    logger.info("HMM saved to %s", path)
    return path


def load_hmm(path: Path | None = None) -> hmm.GaussianHMM:
    path = path or (MODELS_DIR / "hmm_model.joblib")
    return joblib.load(path)
