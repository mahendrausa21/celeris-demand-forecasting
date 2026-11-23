# Celeris Automobili Industrial - Demand Forecasting & Planning Control Tower

Individual submission, MKS AI & Data Hackathon, Phase 2.
Cleans the raw ERP extract, finds and explains anomalies, forecasts `Units_Sold` for every row of
`test.csv` (Nov 2026 - May 2028, 19 months) and serves a Streamlit dashboard for planners.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# copy the 7 provided CSV files into data/raw/
python run_pipeline.py            # cleaning -> anomalies -> backtest -> forecast (a few minutes)
pytest -q                         # 38 automated checks
streamlit run dashboard/app.py    # the dashboard
```

`python run_pipeline.py` writes everything into `outputs/`, including `outputs/submission.csv`.
The dashboard reads only from `outputs/`; it contains no hand-typed data.

## Project structure

```
data/raw/            the 7 original CSVs (never modified)
pipeline/
  clean.py           step 1: cleaning of all files, row-level fix flags, data-quality log
  anomalies.py       step 2: unconstrained demand, anomaly register, regimes (switches, cycles, lost accounts)
  forecast.py        step 3: baselines, final model, rolling-origin backtest, submission
  ml.py              global gradient-boosting model + Croston/SBA benchmark
  features.py        demand drivers: macro lead/lag with stability test, what-if sensitivities, EV shift, intermittency classes
  ranges.py          P10/P50/P90 ranges calibrated on backtest errors and checked on held-out origins
  planner.py         exception list for planners
  tests/             38 pytest checks on every output
dashboard/app.py     Streamlit app (5 views, shared filters)
run_pipeline.py      one command that rebuilds everything
outputs/             generated files (see below)
```

## What the pipeline does

**1. Cleaning (`clean.py`)** - one row per Date x SKU x Region inside each series' active window.
- Labels: 2,659 SKU spellings -> 598 SKUs; 29 region and 17 channel spellings -> 6 and 4; 220 customer
  spellings/typos -> 12 customers; 7 date formats parsed; `Date` is authoritative.
- Duplicates: repeated `Record_ID` -> latest `Load_Timestamp` kept (732 of the 3,682 duplicates carried
  different quantities, i.e. restatements).
- Numbers are decided by triangulating units x price = revenue and checking which field disagrees
  with the SKU's history: price x100 (614), unit-of-measure slip - units counted in packs of 10 and priced per
  pack, all India 2021 (385), units 10x/100x inflated (368), revenue errors, negatives (491, sign slip).
- Missing row is not a zero: every cell has `row_status` = observed / true_zero / repaired /
  missing_row / units_unknown. Gaps stay empty.
- Reference data is checked against transactions (BSS-018 launch date, PMP-004 business unit).
- Macro file: duplicated month, PMI decimal slip (4.79 -> 47.9), 2019 percentages stored as fractions.
- Every fix is logged in `outputs/dq_report.json` and flagged per row in `dq_flags`.

**2. Anomalies (`anomalies.py`)** - separates errors (fixed in step 1) from real events.
- Unconstrained demand: the stockout file's `Requested_Units` where listed; backorders carried into
  unlisted months removed; unlisted months of the chip-shortage family (Electrical Assemblies, 2021-03..2022-09)
  scaled by the quarter's measured demand/shipment ratio (the file is incomplete: those months are as depressed as listed ones).
  Result: +105,334 units of demand vs shipments.
- Register (`anomaly_register.csv`): errors fixed, data gaps, stockouts (listed and estimated), one-off orders
  (capped for training, kept in history), COVID and customer disruptions, holidays, customer switches,
  ordering-cycle changes, lost accounts, master-data contradictions - each with reason, evidence, action.
- Lost accounts: 16 series with 10-19 trailing zero months (normal zero runs never exceed 4) -> forecast 0.
- Regimes: after a customer switch, ordering-cycle change or permanent capacity event only the new regime drives the forecast.

**3. Forecast (`forecast.py`, `ml.py`)**
- Statistical model `series_seasonal_avg3_trend`: each series' own same-month average over the last 3 years
  (keeps its ordering cycle), lifted at half strength where a family-region has grown >6% in both of the last two years.
- Global ML model `global_gbm`: one gradient-boosting model for all series, trained on rolling origins. Features: the
  seasonal averages of 1/2/3 years, recent levels, zero shares, ordering-cycle and regime information, growth, horizon,
  calendar month, family, region. Every feature uses only data before its forecast origin.
- Final model `blend_gbm_seasonal`: 50/50 blend of the two. Chosen by `choose_final()`: lowest backtest WAPE, preferring the
  blend when it is within 0.005 of the best because it is more robust.
- Rules on top of every model: zero after EOL and for lost accounts; new launches start from the predecessor's level times a
  ramp measured on 21 past successions (42% / 90% / 121% / 108%); predecessors fade to 61% during the overlap;
  planned customer shutdown scaled by the measured Nexon-strike effect.
- Rolling-origin backtest (origins 2023-03, 2024-03, 2025-03; 19-month horizon; every rule and the ML training data
  restricted to the period before the origin), scored against unconstrained demand:

| Model | WAPE | Bias |
|---|---|---|
| Croston/SBA (intermittent-demand benchmark) | 0.501 | -10.3% |
| Moving average (12 m) | 0.502 | -3.0% |
| Seasonal naive | 0.369 | -2.9% |
| Statistical model (final of the earlier version) | 0.339 | -2.9% |
| Global gradient boosting | 0.300 | -4.3% |
| **Final: 50/50 blend** | **0.305** | **-3.6%** |

The blend improves every region, every family and every horizon bucket over the statistical model alone
(e.g. Middle East & Africa 0.434 -> 0.378, Electrical Assemblies 0.378 -> 0.335).

**4. Drivers (`features.py`)**
- Macro lead/lag: 8 indicators x 7 families x lags 0-12 months on year-over-year growth, with a permutation test (circular
  shifts, accounting for choosing the best of 13 lags) and a stability test (same sign and similar lag in both halves of the
  data, |r| >= 0.4 in both). Result: 10 stable relationships, all leading indicators (new-vehicle orders and PMI about 1 month,
  auto production 1-3 months, r = 0.68-0.91); 17 significant but unstable (e.g. industrial utilisation at lag 12 flips sign);
  29 not significant. Stable sensitivities are consistent between the two halves, so they are not a COVID artefact.
- Macro features at the forecast origin were tested in the ML model and did not help (WAPE 0.2968 vs 0.2946): the indicators lead by
  1-3 months, the horizon is 19. They are used for what-if scenarios instead (`macro_sensitivity.csv`).
- EV shift: EV share of units 0% (2019) -> 9.9% (2025) -> 9.8% (2026): the shift has flattened.
- Intermittency classes (Syntetos-Boylan ADI / CV2, last 36 months): 2,179 smooth, 761 intermittent, 88 erratic, 12 lumpy.

**5. Ranges and planner support (`ranges.py`, `planner.py`)**
- Series-level P10/P50/P90: normalised residual (actual - forecast) / level, quantiles per intermittency class x horizon bucket.
  Checked by calibrating on two backtest origins and scoring the third: the P10-P90 band contained 78.9% of actuals (nominal 80%),
  78-79% in every class and horizon bucket. The calibrated median is not better than the point forecast (WAPE 0.307 vs 0.305), so the
  point forecast is what is submitted.
- Hierarchy: point forecasts are bottom-up, so SKU -> family -> region -> total add up exactly (tested). Quantiles do not add up, so
  ranges for total, family, region and family x region are calibrated on the errors of those aggregates. Their raw held-out coverage was
  only 65-72% (errors are dominated by origin-wide shifts), so each band is widened by the smallest factor reaching 80%
  (about x1.2 for family x region up to x1.55 for the total). Do not sum quantiles across series or across months.
- Planner exceptions (`planner_exceptions.csv`): lost accounts, new launches, EOL phase-outs, planned customer shutdowns, forecast jumps
  (>30% vs last 12 months), recent stockouts, recent data gaps, highest-uncertainty series, each with a reason and a suggested action.

## Outputs (`outputs/`)

`clean_history.parquet`, `clean_test.parquet`, `clean_prices.parquet`, `clean_stockouts.parquet`,
`clean_sku_master.parquet`, `clean_macro.csv`, `clean_events.csv`, `dq_report.json`,
`unconstrained_history.parquet`, `anomaly_register.csv`, `series_regimes.csv`,
`macro_leadlag.csv`, `macro_leadlag_grid.csv`, `macro_sensitivity.csv`, `ev_shift.csv`, `series_profile.csv`,
`backtest_metrics.csv`, `forecast.csv`, `forecast_ranges.csv`, `forecast_ranges_aggregate.csv`, `range_calibration.csv`,
`range_evaluation.csv`, `planner_exceptions.csv`, `submission.csv`.

## Assumptions and limitations

- Negative units are treated as sign slips (size matches normal demand, neighbouring months are normal).
- The new-launch ramp comes from only 21 past successor pairs and is the least certain part of the forecast.
- Early buying before price rises was tested: correlation with the size of the rise is 0.56 on 6 years; only the 9% rise
  announced in Nov 2022 stands out, so it is reported but not modelled separately.
- The planned Orion shutdown (Jul-Aug 2027) uses the Nexon strike as an analogue: an assumption.
- Aggregate ranges rest on only three backtest origins, hence the widening; a 19-month total interval cannot be calibrated from three observations.
- ML results can differ in the last digits between machines (floating-point order in multi-threaded training).
- Middle East & Africa (opened 2022) is the weakest region (WAPE 0.43, bias -12%).