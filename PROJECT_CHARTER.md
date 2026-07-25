# Project Charter — MENA Country Assessment System

**Working name:** MENA Conflict Monitor (evolving from "escalation score" to "assessment reports")
**Owner:** Lucien McNulty · **Charter date:** 2026-07-12 · **Status:** Phase 1 pending green light

---

## 1. Purpose

Produce **weekly, empirically grounded assessment reports** for 14 MENA countries,
describing all relevant categories of **change and constants** — from type and
magnitude of violence through humanitarian and (in future phases) economic
consequences — with every claim traceable to a source.

This is a **measurement-first** system. Forecasting exists as one report section
and appears only where it has demonstrated, validated skill. There is no single
master "risk score" that the project funnels through.

## 2. The empirical standard (editorial contract)

Every sentence in a generated report must be one of exactly four things:

| # | Type | Example | Backing required |
|---|------|---------|------------------|
| 1 | **Measured quantity** | "ACLED recorded 340 fatalities this week" | Named source + timestamp |
| 2 | **Change vs baseline** | "protest events at 92nd pctile of own 2-yr history" | Defined baseline + change-point test |
| 3 | **Declared constant** | "no anti-civilian violence for 11 consecutive weeks" | Same baseline machinery |
| 4 | **Validated forecast** | "battle escalation p=0.61 (PR-AUC 0.44 vs 0.19 baseline)" | Leakage-safe validation + calibration |

Data gaps are declared in the report, never hidden (generalization of the
missing-country banner).

## 3. Scope

**In scope**
- 14 MENA countries, weekly cadence (sections may update slower; staleness is declared per section)
- Violence measurement (ACLED: types, fatalities, geography, actors)
- Media signal (GDELT tone/volume/ratio + RSS) **and media–reality divergence** (underreported-crisis flag)
- Humanitarian indicators (UNHCR displacement, IPC food-security phases)
- Per-type escalation outlook with validated skill
- Report rendered as a **dashboard tab** (one report per country per week)

**Future work — essential, not optional**
- **Economic consequences** (exchange rates, inflation, trade disruption).
  Required for report relevancy; deferred only because timely economic data for
  conflict-affected MENA states needs its own sourcing investigation. The report
  schema and fusion layer must be designed so an economic section slots in
  without redesign.
- Admin1-level risk modeling (measurement at admin1 arrives earlier, in Phase 1–2)

**Out of scope**
- Real-time (sub-weekly) alerting · social-media ingestion · non-MENA regions

## 4. Phases

| Phase | Ships | Definition of done |
|-------|-------|--------------------|
| **1. Violence + Media sections** | ACLED fetcher & weekly features; change/constant detection; report generator + dashboard tab | Syria's report matches news memory; every line carries a citation; no silent nulls |
| **2. Outlook section** | Per-event-type escalation probabilities | Beats persistence baseline on transition weeks; calibration curve published in report |
| **3. Humanitarian section** | UNHCR displacement + IPC phases | Sections render with declared staleness; fusion layer handles monthly cadence |
| **4. Geo detail** | Admin1 maps & spread metrics in reports | Choropleth per country; spread metrics change-point tested |
| **5. Economic section** (future-essential) | TBD after sourcing investigation | Same empirical standard as all sections |

Each phase is independently complete — the project is shippable after any phase.

## 5. Technical foundations (binding design commitments)

These six components are foundational. New code must not violate them; existing
code is refactored toward them as phases touch it.

### 5.1 Leakage-resistant temporal validation
- All model evaluation uses **walk-forward (expanding-window) splits**; no random splits, ever.
- **Publication-lag alignment:** features are timestamped by *availability*, not
  occurrence. ACLED publishes weekly → week *t* events enter features at *t+1*
  or later. GDELT is near-real-time → lag 0 acceptable.
- Labels and normalizations are computed **within training windows only**
  (no full-dataset scalers feeding model inputs).
- Historical report content is **frozen at generation time**; refits never
  silently rewrite past claims (versioned artifacts).

### 5.2 Calibration and uncertainty estimation
- Every published probability ships with a **reliability (calibration) curve**
  and Brier score vs baseline; recalibrate (isotonic/Platt) when miscalibrated.
- Magnitude estimates use **quantile regression / prediction intervals**
  ("20–80 deaths likely, tail risk 300+"), never point estimates alone.
- Report language maps to calibrated bins (e.g., "likely" only if p∈[0.6,0.8]
  *and* the bin is calibrated).

### 5.3 Change-point detection
- "Change" and "constant" claims (sentence types 2–3) are backed by **formal
  change-point tests** (e.g., PELT / Bayesian online CPD) on each indicator
  series — not eyeballed percentile jumps.
- The HMM regime layer is refit on **observed violence** (not the media index)
  and reconciled with CPD output; disagreements are surfaced, not averaged away.

### 5.4 Geospatial representations
- Events carry admin1 + coordinates from ingestion onward; country aggregates
  are derived views, never the storage format.
- **Spread metrics** as first-class indicators: № active admin1 regions,
  spatial concentration (Gini/HHI), new-region onset flags.
- Reports render admin1 choropleths; media features remain country-level and
  are fused at the country grain (declared in metadata).

### 5.5 Retrieval and citation systems
- Raw source records (ACLED event IDs, GDELT query windows, RSS URLs, UNHCR
  dataset versions) are **stored with stable IDs**; report sentences carry
  citation keys resolving to those records.
- A retrieval layer answers "show me the events behind this sentence" from the
  dashboard.
- Data snapshots are **versioned per report week** so any historical report can
  be reproduced exactly.

### 5.6 Multi-source data fusion
- All sources join on a shared **(country, ISO-week) spine**; each carries
  metadata: cadence, publication lag, last-updated, coverage.
- Per-section **staleness is computed and displayed**, not assumed uniform.
- **Divergence indicators** are first-class outputs (media attention vs recorded
  fatalities → underreporting flag).
- Conflict rules are explicit (e.g., ACLED authoritative for events; UCDP as
  cross-check if both present); missingness is a declared state, never imputed
  as calm.

## 6. Operating principles (carried from the 2026-07-12 review)

1. Models are always compared to the **persistence baseline**; transition/onset
   weeks are the primary evaluation slice.
2. Pipelines **fail loudly** on data-quality violations (missing countries,
   partial weeks, degenerate model fits).
3. History **accumulates**; fetches merge, never overwrite.
4. In-sample predictions are flagged wherever displayed.
5. Every run persists metrics (`data/processed/metrics.json`) and is logged in `DEVLOG.md`.

## 7. Current status & gating items

- Review fixes complete (see DEVLOG 2026-07-12): honest baselines live, HMM
  healthy, 12/14 countries with data (Iran, Iraq pending GDELT rate-limit cooldown).
- Current honest result: media-only model **loses** to persistence overall
  (0.67 vs 0.86) but catches 27% of transitions — the gap Phase 1–2 exists to close.
- **Gating item (owner):** ACLED registration → `ACLED_API_KEY` env var.
- **Gating item (agreed):** explicit green light before Phase 1 execution begins.

## 8. Risks

| Risk | Mitigation |
|------|------------|
| ACLED/UCDP API access terms change | Fusion layer is source-agnostic; UCDP module already exists as fallback |
| GDELT rate limiting (observed today) | Merge-on-fetch caching; resumable backfill; cool-down discipline |
| Humanitarian/economic data too slow or sparse | Sections declare staleness; economic section gated on a sourcing investigation before commitment |
| Scope creep toward "everything dashboard" | Phases are independently shippable; charter §3 boundaries |
