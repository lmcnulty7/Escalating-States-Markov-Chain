# DEVLOG

## 2026-07-18 — Published to GitHub

- Repo: https://github.com/lmcnulty7/Escalating-States-Markov-Chain
- **Made this a standalone git repo.** The directory was previously owned by
  the home-directory repo (`/Users/lucienmcnulty`) — committing from there
  would have pushed the entire home folder (SSH keys, .claude.json, etc.).
  Same pattern as NBA Comp Viz V3.
- Excluded and verified absent from the remote: `.env` (ACLED password),
  `.acled_token.json`, `data/` (41 MB GDELT archive + 12 MB ACLED raw records
  — also an ACLED redistribution-terms issue), `models/`, and `.claude/`
  (its worktrees held copies of unrelated projects).
- Added README (architecture, honest metrics, setup, data licensing) and
  `.env.example`; added scipy to requirements (report.py uses it).
- Removed stale `src/data_fetcher 2.py` — a July 16 macOS duplicate predating
  the circuit breaker and env-tunable delay; unreferenced.
- Local data and credentials untouched — nothing needs re-fetching.

## 2026-07-18 — Phase 2: purged walk-forward + calibration

- **Purged walk-forward evaluation** (`walk_forward_escalation`): expanding
  window, 4 test blocks, and a 4-week purge before each block — the target at
  week t contains fatalities through t+4, so unpurged splits (including our
  old 70/30) leaked test outcomes into training labels.
- Honest downgrade: pooled OOS **PR-AUC 0.093 vs 0.067 persistence** (the
  single split's 0.129 was flattered by the leak). Fold 4 (Apr–Jun 2025,
  Israel–Iran run-up): PR-AUC 0.22 — the model earns its keep on real onsets.
- **Platt calibration** fit on pooled OOS predictions: Brier 0.078 → 0.0623,
  just beating climatology (0.0624). Raw scores were badly overconfident
  (top quintile predicted 0.26, observed 0.10).
- **Calibration gate**: calibrated probabilities print in reports ONLY while
  OOS Brier ≤ climatology; otherwise reports fall back to ranking language.
  Currently PASSES (marginally). Honest framing everywhere: calibrated
  probabilities hug the base rate (~6-7%) — the model's information is in the
  ranking, and the report says so.
- Deployed model now refit on all labelled rows after evaluation;
  `risk_calibrated` column added; calibrator saved in the model bundle;
  reliability table → `data/processed/reliability.parquet`.
- Dashboard: reliability curve (raw vs calibrated vs diagonal) in the Report
  tab; sidebar shows walk-forward metrics + gate status.
- `risk_in_sample` now means "row was in the deployed model's training data"
  (all labelled rows); current weeks are genuinely out-of-sample.

## 2026-07-17 — Report tab (charter Phase 1 deliverable)

- New `src/report.py`: pure-logic report engine enforcing the charter §2
  editorial contract. Every sentence is typed — MEASURED (source+timestamp),
  CHANGE/CONSTANT (Mann-Whitney, last 4w vs prior 12w, change claimed only at
  p<0.05), FORECAST (uncalibrated scores shipped as rankings with the honest
  PR-AUC quoted), DATA GAP (embargo/missing declared, never shown as calm).
- Citations resolve to real records: ACLED event IDs + sources for the week's
  largest events (Syria's report cites the actual July 2025 Sweida clashes).
- Sections carry as-of dates; stale sections (ACLED behind embargo) are
  flagged, not hidden. Completeness strip up top (🟢 current / 🟡 lagged / 🔴 absent).
- New 📋 Report dashboard tab renders the typed sentences with badges,
  citations, and a source-records expander. Isolation-tested on Syria (rich)
  and Iran (no DOC data — all gap paths exercised).
- Notes: change-point test is Mann-Whitney for now (charter §5.3 names
  PELT/BOCPD as the eventual formal CPD); divergence flag (media vs recorded
  violence) deferred until it can be computed on overlapping windows.

## 2026-07-17 — Step 6: dashboard violence charts

- Country Analysis tab gains an "Observed Violence — ACLED" section:
  weekly fatality bars + stacked event-type breakdown (battles / civilian
  violence / explosions / protests). No tab redesign.
- Event-type palette validated for colorblind safety + dark-surface contrast
  (validator: all checks pass; stack ordered for CVD adjacency).
- The 12-month ACLED embargo is drawn as a shaded "no data" region with a
  label — missing is never rendered as calm.
- Fixed: the intensity-chart fatality overlay still referenced the removed
  ged_fatalities column (silently absent since the ACLED switch).
- Verified: figures build against real Syria data (peak week 2025-03-03,
  1,714 deaths — the coastal massacres); dashboard serves on :8502.

## 2026-07-17 — Event-file migration + first full-picture run

- **Migrated historical media data to GDELT raw daily event files**
  (data.gdeltproject.org): 914 days downloaded in ~35 min with zero rate
  limiting (the DOC API couldn't deliver 2 chunks in 2 days). 5.95M MENA
  events retained in a single 41 MB parquet — nothing else stored locally;
  raw zips are filtered in memory and discarded.
- New `src/gdelt_files.py`: fetch + weekly `gdelt_ev_*` features (event counts
  by quad class, Goldstein, tone, mentions). Additive columns; DOC API keeps
  the weekly live refresh.
- Feature spine now extends to wherever ANY source has data: all 14 countries
  get complete Jan-2024→now weekly rows. **Iran and Iraq enter the dataset for
  the first time** (event-archive features; DOC still missing for them and
  still flagged).
- **Escalation target now built from ACLED fatalities** (charter: ACLED anchors
  ground truth). Priority: ACLED → UCDP → HMM fallback.
- Fixed: pandas ≥2.2 groupby.apply drops the grouping column (target builder
  rewritten with transform); conflict_ratio no longer fakes 0 on no-DOC weeks;
  fallback model only trains on real-DOC rows.
- **First honest full run** (1,862 rows, 14 countries; was 234 rows/9 countries
  a week ago): escalation model PR-AUC **0.129 vs 0.086 persistence baseline**
  (base rate 0.085), Brier 0.090. Real but modest ranking signal; probabilities
  are over-scaled (scale_pos_weight) — calibration is exactly the Phase 2 work.
- ACLED features now load-bearing on fair footing: 6 of the top 12
  importances (civilian-violence and battles lags/rolls, fatalities rolls).
  Draft Step 5 check: passed.
- Test window includes the June-2025 Israel–Iran war onset — a real
  out-of-sample escalation the model was scored on.

## 2026-07-17 (overnight) — GDELT backfill: quota diagnosis

- Diagnostic (4 controlled requests): GDELT DOC API enforces a small
  **token-bucket quota per IP** — short bursts succeed when the bucket refills,
  then everything 429s for hours regardless of spacing or query type.
- Landed so far: Yemen COMPLETE (2024-01→now), Syria complete except one
  2025-06→12 hole. Remaining 12 countries: pre-Nov-2025 history still missing.
- Loop hardened for unattended running: probe-gated passes (dead zones cost
  seconds), circuit breaker (2 empty chunks → abort), permission allowlist so
  nothing prompts overnight, auto-relaunch on kills.
- **Decision needed (morning):** if passes stay dry, migrate the historical
  backfill to GDELT's raw daily event files (data.gdeltproject.org — static
  files, no rate limits, ~1 day of work). DOC API stays for weekly live
  refreshes (56 req/week fits any quota). File-derived features (event counts,
  Goldstein, tone from CAMEO events) would be ADDITIVE columns, not
  replacements — cleaner than mixing sources in one column.

## 2026-07-12 — ACLED Step 4: pipeline integration

- ACLED is now a parallel source in the pipeline (Step 1b), merged on
  (country, date). 39 ACLED columns in results; **27 enter the model — all
  lagged ≥1 week** (availability rule; lag-0 columns are report-only).
- Removed `fillna(0)` from all model paths: NaN is now native XGBoost missing.
  Zero-filling would have turned embargoed ACLED weeks into "zero fatalities".
- Subset test (3 countries): embargo semantics exact (pre-embargo 0 nulls,
  post-embargo all-NaN), lag alignment verified, no lag-0 leaks, 9 API calls.
- Full run: all 14 countries fetched from ACLED (27 calls, well under budget).
- **Coverage interaction discovered:** ACLED (accessible ≤ 2025-07-13) only
  overlaps GDELT history for backfilled countries. Syria + Yemen have 80
  ACLED weeks each; the other 10 have ZERO because their GDELT caches start
  2025-11-19 — after the embargo boundary. Fix = finish the GDELT backfill
  (`--backfill 2024-01-01`, resumable) once GDELT's IP cooldown lifts
  (still throttled as of tonight; Iran/Iraq fetch failed again).
- Caution: ACLED features rank high in importance (geo_spread_lag1w #2) but
  with only 2 countries non-null, part of that is country identification via
  missingness, not signal. Don't read importance until backfill completes.
- Honest metrics unchanged in character: model 0.70 vs persistence 0.86.

## 2026-07-12 — ACLED Step 3: weekly features

- `compute_acled_features()` added to `src/acled_fetcher.py` (still isolated;
  nothing else touches it yet).
- 11 weekly columns: fatalities, events, 6 per-type counts, civilian_targeting,
  geo_spread (distinct admin1), new_actors, last_update (availability timestamp).
- Same Monday-start weekly spine as GDELT → Step 4 fusion is a plain join.
- Zeros are honest (no events recorded) but the grid stops at the embargo
  boundary — embargoed weeks are absent/missing, never zero.
- Lags/rolls deliberately left to features.py (one lag convention, one place).
- Caveat: new_actors is left-censored (first ~4 weeks meaningless).
- Isolation test (Syria): week of 2025-03-03 shows 1,714 fatalities with
  civilian-violence composition — the March 2025 coastal massacres, exactly
  as news memory expects. No nulls; per-type counts sum to event totals;
  0 API calls (warm cache).

## 2026-07-12 — ACLED Step 2: fetcher module (call-budget design)

- New `src/acled_fetcher.py`. Public API: `fetch_acled_cached(country, start, end)`.
- Over-request protection (6 layers):
  1. OAuth token cached to disk (`.acled_token.json`, gitignored) — ~1 auth/day
  2. Per-country parquet cache — only the gap since last fetch is downloaded
  3. 5000 events/page — full country backfill ≈ 2-4 requests
  4. Backoff on 429/5xx + politeness delay between pages
  5. Hard budget: 40 calls/run, raises `AcledBudgetExceeded` before hammering
  6. Embargo-aware clamping — never spends calls on windows the tier can't see
- Tail of cache refreshed at most every 3 days (14-day overlap catches ACLED revisions).
- Isolation test (Syria 2025-01-01→embargo): 6,248 events, 31 columns, unique
  event IDs, only nulls in sparse `tags`, fatalities median 0 / max 108,
  **2 API calls cold, 0 calls warm** (asserted).
- Not yet integrated into pipeline (Step 4 does that).

## 2026-07-12 — ACLED Step 1: exploration (schema + access tier)

- OAuth works (token endpoint, Bearer, 24h expiry). Credentials in gitignored
  `.env` (never in code or repo).
- **Key finding: our tier has a 12-month embargo** — only events older than
  12 months are accessible (verified: Apr/Jun 2025 return ~790 events for
  Syria, Aug 2025 returns 0, silently). Live weekly violence reporting is NOT
  possible on this tier; historical training/baselines are fully covered.
- Schema is rich and fits the charter: 6 event types (incl. Protests, Riots),
  sub_event_type, fatalities, actors + civilian_targeting, admin1/2/3 +
  lat/lon + geo_precision, `event_id_cnty` (stable ID → citation system §5.5),
  `timestamp` (record publication time → leakage-safe availability dating §5.1),
  `source` + `notes` (per-event citations).
- Sanity: Syria Apr 2025 = 792 events, 490 fatalities, 14 admin1 regions — sensible.
- One sentence, per Step 1 success criterion: ACLED gives dated, located,
  typed conflict events with fatality counts and named actors; GDELT only
  measures how much the media talks about a country and in what tone.
- Fetcher requirement discovered: embargo-empty responses look identical to
  genuinely-quiet weeks → fetcher must check the account's `date_recency`
  restriction and declare embargoed weeks as MISSING, not calm.
- Open decision: how the live violence section gets data (tier upgrade /
  UCDP candidate / media-only current section with ACLED historical baselines).

## 2026-07-12 — Project charter + vision pivot

- Vision settled after scoping discussion: **weekly empirical country assessment
  reports** (dashboard tab), not a single escalation score. Measurement first;
  forecasts are one section and only appear with validated skill.
- Wrote `PROJECT_CHARTER.md`: 4-type empirical sentence standard, 5 phases
  (violence+media → outlook → humanitarian → geo → economic), and 6 binding
  technical foundations (leakage-resistant temporal validation, calibration &
  uncertainty, change-point detection, geospatial representations, retrieval &
  citations, multi-source fusion).
- Economic consequences confirmed **future-essential** (Phase 5, after sourcing
  investigation).
- First full pipeline run on fixed code: 569 rows, 12/14 countries (Iran/Iraq
  blocked by GDELT rate-limit cooldown), HMM healthy (no absorbing states),
  honest metrics: model 0.67 vs persistence baseline 0.86, catches 27% of
  transition weeks — the gap the new target exists to close.
- Waiting on: ACLED registration (`ACLED_API_KEY`) + green light → Phase 1.

## 2026-07-12 — Fixes from architecture review

An end-to-end review found one critical bug and several structural problems.
All fixed in this pass, in priority order.

### 1. Critical bug: risk score was the probability of the WRONG state
- HMM component numbers are arbitrary (0/1/2 don't mean anything by themselves).
  The old code assumed component 2 = "Active Conflict"; in the trained model
  component 2 was actually **Stable** — so the dashboard's "XGB Conflict Risk"
  was showing the probability of *stability*.
- Fix: the pipeline now looks up which component is actually labelled
  "Active Conflict" (highest mean intensity) and uses that everywhere:
  risk score, SHAP chart, saved model bundle.
- Files: `src/ml_model.py`, `src/hmm_model.py` (`conflict_component()`), `src/pipeline.py`, dashboard SHAP.

### 2. History is no longer destroyed
- Before: every fetch overwrote the cache with a rolling 6-month window →
  the dataset could never grow past ~26 weeks.
- Now: fetches **merge** into the cache (newest wins on overlap), and a new
  `--backfill 2024-01-01` flag pulls historical GDELT in 6-month chunks.
  Backfill is resumable — already-covered chunks are skipped, so re-run it
  freely if GDELT rate-limits kill a pass.
- Files: `src/data_fetcher.py` (`_merge_into_cache`, `backfill_gdelt`), `run_pipeline.py`.

### 3. Real ground truth: UCDP fatalities (needs a free API token)
- New `src/ground_truth.py` pulls UCDP GED conflict events and builds an
  honest target: **escalated_next4w** = fatalities in the next 4 weeks are
  both >2× the country's trailing 12-week rate and ≥10 deaths.
- When this target is available the model predicts *observed escalation*
  instead of its own media-derived index (which was circular).
- **ACTION NEEDED:** register at https://ucdp.uu.se/apidocs/ and
  `export UCDP_API_TOKEN=<token>`. Without it the pipeline falls back to the
  old HMM-state target (now with the class bug fixed).
- Note: UCDP codes Gaza/West Bank under Israel; we map those to "Palestine" (imperfect).

### 4. Honest evaluation: persistence baseline + transition weeks
- 91% of weeks have the same state as last week, so "predict no change" gets
  91% accuracy. The model is now scored **against that baseline**, and
  separately on transition weeks (the only weeks that matter).
- UCDP path reports Brier + PR-AUC vs baseline. All metrics saved to
  `data/processed/metrics.json` and shown in the dashboard sidebar.
- Dashboard also flags risk scores on weeks the model trained on (in-sample).

### 5. Data-quality gates
- Missing countries now **fail the pipeline loudly** (5 of 14 were silently
  absent before — including Yemen, Iran, Palestine). Override with `--allow-missing`;
  the dashboard shows a red banner listing any gaps either way.
- The partial current week is dropped (summing half a week looked like
  false de-escalation right at "now").
- Each country is reindexed to a complete weekly grid: missing weeks are
  explicit NaNs (forward-filled for the HMM), not "perfectly calm" zeros.

### 6. HMM fixes
- Degenerate-fit detection: absorbing/unreachable states are now caught.
  (The previous saved model literally said "no country can ever become
  Stable" — P(→Stable) = 0 from everywhere.)
- Fit tries 5 random seeds and keeps the best non-degenerate solution.
- Forecasts start from the posterior probability vector, not a "we know the
  state exactly" point mass.

### 7. Small bugs
- Overview table sorted intensity as strings ("9.5%" ranked above "85.0%") — fixed.
- A headline mentioning two countries was assigned to one **at random**
  (Python set iteration) — now assigned to all matched countries, deterministically.
- Al Jazeera sports/features articles no longer pollute sentiment
  (the "Jordan" alias was matching basketball news).

### Known limitations (deliberate, documented)
- Intensity score normalization still uses the full dataset (future leaks
  into past *display* values only — model features are unaffected).
- Historical HMM states use smoothing (hindsight); fine for display, but
  past states can shift on refit. Filtered (forward-only) states are future work.
- GDELT backfill of 2024–2026 is slow (rate limits); re-run
  `python run_pipeline.py --backfill 2024-01-01` until all chunks are cached.
