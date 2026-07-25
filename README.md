# MENA Country Assessment System

Weekly, empirically grounded conflict assessment reports for 14 MENA countries —
built from media signals (GDELT) anchored to observed violence (ACLED), with a
calibrated escalation model whose forecasts ship with their validation.

The project deliberately is **not** a single "risk score." It is a measurement
system that generates typed, cited report sentences; forecasting is one section,
and it only claims what held-out evaluation supports.

## The empirical standard

Every sentence in a generated report is one of five kinds, and each carries its
own backing:

| Type | Example | Backing |
|---|---|---|
| **MEASURED** | "ACLED recorded 663 fatalities across 175 events" | Named source + timestamp |
| **CHANGE** | "Fatalities significantly elevated vs baseline (p=0.03)" | Mann-Whitney, last 4w vs prior 12w, claimed only at p<0.05 |
| **CONSTANT** | "No significant change vs the prior 12-week baseline" | Same test, not significant |
| **FORECAST** | "Calibrated probability 6% (base rate 7%) — #12 of 14" | Purged walk-forward + calibration gate |
| **DATA GAP** | "No event data after Jul 14, 2025 — absence of data is not absence of violence" | Declared, never rendered as calm |

Citations resolve to source records — a Syria report cites the actual ACLED
event IDs and news sources behind that week's deadliest incidents.

## Honest results

Measured by **purged** walk-forward evaluation (expanding window, 4 folds, with a
4-week purge before each test block — the target at week *t* encodes fatalities
through *t+4*, so unpurged splits leak):

| Metric | Model | Baseline |
|---|---|---|
| PR-AUC (pooled out-of-sample) | **0.093** | 0.067 (persistence) |
| Brier (Platt-calibrated) | **0.0623** | 0.0624 (climatology) |
| Out-of-sample base rate | 6.7% | — |

A real but modest edge. Skill concentrates on genuine onsets — the fold covering
the June 2025 Israel–Iran escalation scores PR-AUC 0.22. Calibrated probabilities
sit close to the base rate, so the model's usable information is in the *ranking*,
and the reports say exactly that.

A **calibration gate** enforces this automatically: calibrated probabilities are
printed only while out-of-sample Brier beats climatology. Otherwise reports fall
back to ranking language with no human in the loop.

## Data sources

| Source | Role | Cadence | Notes |
|---|---|---|---|
| **GDELT raw event files** | Historical media signal | Daily files → weekly | 5.95M MENA events, 2024→present. No rate limits. |
| **GDELT DOC API** | Live media refresh | Weekly | Tone/volume timelines. Small per-IP quota — unusable for bulk backfill. |
| **ACLED** | Ground truth: observed violence | Weekly | Events, fatalities, actors, admin1. Free tier embargoes the last 12 months. |
| **UCDP GED** | Fallback ground truth | — | Token-gated; used if ACLED is unavailable. |
| **RSS (BBC/AJ/NYT)** | Live headline sentiment | Continuous | VADER sentiment. |

Sources are fused on a shared `(country, ISO-week)` spine. Each carries its own
publication lag, and **ACLED features enter models lagged ≥1 week** because week
*t*'s events are not knowable during week *t*.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add your ACLED credentials
```

Build the historical GDELT archive (~35 min, ~7 GB transferred, 41 MB retained —
daily files are filtered in memory and discarded):

```bash
python -c "from src.gdelt_files import build_archive; build_archive('2024-01-01')"
```

Run the pipeline, then the dashboard:

```bash
python run_pipeline.py            # add --allow-missing to tolerate source gaps
streamlit run dashboard/app.py
```

The pipeline **fails loudly** on data-quality violations (missing countries,
partial weeks, degenerate model fits) rather than silently producing a partial
regional picture.

## Layout

```
src/
  data_fetcher.py    GDELT DOC API + RSS ingestion (merge-on-fetch caching)
  gdelt_files.py     GDELT raw daily event files → weekly event features
  acled_fetcher.py   ACLED events (call-budgeted, embargo-aware) + weekly features
  ground_truth.py    Escalation target: fatalities next 4w vs trailing baseline
  features.py        Weekly spine, lags/rolls, intensity score
  hmm_model.py       GaussianHMM regimes (multi-seed, degeneracy-checked)
  ml_model.py        XGBoost + purged walk-forward + Platt calibration
  report.py          Typed-sentence report engine
  pipeline.py        Orchestration + data-quality gates
dashboard/app.py     Streamlit: overview, country analysis, forecast, news, report
```

`PROJECT_CHARTER.md` defines scope, phases, and the six binding technical
foundations (leakage-resistant validation, calibration, change-point detection,
geospatial representation, retrieval/citations, multi-source fusion).
`DEVLOG.md` is the running history, newest first.

## Status

Phases 1–2 complete: violence + media report sections, calibrated outlook with
purged evaluation. Next: humanitarian indicators (UNHCR/IPC), admin1 geographic
detail, and economic consequences.

## Data attribution & licensing

**No raw data is redistributed in this repository.** Rebuild it locally with the
setup commands above, using your own credentials.

**ACLED** — Armed Conflict Location & Event Data ([acleddata.com](https://acleddata.com)).
Used under [ACLED's Terms of Use](https://acleddata.com/terms-of-use/) and
[Attribution Policy](https://acleddata.com/attributionpolicy). Data are filtered to
14 MENA countries, all event types, and aggregated by the author into Monday-start
weekly totals (fatalities, event counts by type, distinct admin-1 regions, distinct
first-seen actors). The access date is recorded from the local cache and displayed
on every visualization and report citation produced by this project.
*ACLED bears no responsibility for the analysis or conclusions presented here.*

**The GDELT Project** ([gdeltproject.org](https://www.gdeltproject.org/)) — daily
event files and the DOC API, used under GDELT's open terms (unrestricted use with
citation and a link to the project).

**UCDP GED** ([ucdp.uu.se](https://ucdp.uu.se/)) — optional fallback ground truth.
**VADER** sentiment (MIT) for RSS headline scoring.

### Code license

None. All rights reserved by the author — this repository is published for reading
and review, not for reuse.
