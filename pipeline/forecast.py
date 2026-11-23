"""Step 4 - forecasting: 2 baselines + 1 better model, time-based backtest, submission file.
Run with:  python -m pipeline.forecast"""
import numpy as np
import pandas as pd
from pipeline.config import OUT
from pipeline.anomalies import detect_regimes
from pipeline import ml

HORIZON = 19                                        # months to forecast (Nov 2026 -> May 2028)
ORIGINS = ["2023-03-01", "2024-03-01", "2025-03-01"]   # backtest forecast origins, 19 months each
# launch ramp factors are measured from the history by features.py (outputs/launch_ramp.csv), not typed in


# ----------------------------------------------------------------------------- data
def load_history():
    """P_obs: shipped units. P_model: unconstrained demand with one-off orders capped (training data).
    P_true: unconstrained demand (what the forecast is scored against)."""
    u = pd.read_parquet(OUT / "unconstrained_history.parquet")
    piv = lambda col: u.pivot_table(index=["SKU_ID", "Region"], columns="Date", values=col, aggfunc="first")
    P_obs, P_model, P_true = piv("Units_Observed"), piv("Units_Model"), piv("Units_Unconstrained")
    meta = u.drop_duplicates(["SKU_ID", "Region"]).set_index(["SKU_ID", "Region"])[["Product_Family"]].loc[P_model.index]
    cust = u[["SKU_ID", "Region", "Date", "Customer_Name"]]
    events = pd.read_csv(OUT / "clean_events.csv", parse_dates=["Start_Date", "End_Date"])
    return P_obs, P_model, P_true, meta, cust, events


def mask_regimes(train, reg, min_months=12):
    """Forget history from before a customer switch / ordering-cycle change / permanent capacity change,
    if at least `min_months` of the new regime exist."""
    t = train.copy()
    for k, start in reg["Regime_Start"].dropna().items():
        if k in t.index:
            row = t.loc[k]
            if row[row.index >= start].notna().sum() >= min_months:
                row[row.index < start] = np.nan
                t.loc[k] = row
    return t


def months_after(origin, n):
    return pd.date_range(pd.Timestamp(origin) + pd.offsets.MonthBegin(1), periods=n, freq="MS")


# --------------------------------------------------------------------------- models
def level12(train):
    """Average units over the last 12 months (NaN months skipped, zeros kept)."""
    lv = train.iloc[:, -12:].mean(axis=1, skipna=True)
    return lv.fillna(train.mean(axis=1, skipna=True)).fillna(0.0)


def seasonal_index(train, meta, years=3):
    """Family x region x calendar-month index from the last `years` years (1.0 = average month)."""
    recent = train.iloc[:, -12 * years:]
    long = recent.stack(future_stack=True).rename("u").reset_index()
    long.columns = ["SKU_ID", "Region", "Date", "u"]
    long["Product_Family"] = long.set_index(["SKU_ID", "Region"]).index.map(meta["Product_Family"])
    long["Month"] = long["Date"].dt.month
    cell = long.groupby(["Product_Family", "Region", "Month"])["u"].mean()
    base = long.groupby(["Product_Family", "Region"])["u"].mean()
    idx = (cell / base.reindex(cell.index.droplevel("Month")).values).unstack("Month")
    return idx.reindex(columns=range(1, 13)).fillna(1.0)


def same_month_avg(train, horizon_dates, n_years):
    """Average of the same calendar month over the latest `n_years` years before the origin
    (NaN months skipped). Keeps each series' own pattern, e.g. its quarterly ordering cycle."""
    origin = train.columns[-1]
    res = pd.DataFrame(index=train.index, columns=horizon_dates, dtype=float)
    for d in horizon_dates:
        k0 = int(np.ceil(((d.year - origin.year) * 12 + d.month - origin.month) / 12))
        cols = [d - pd.DateOffset(years=k0 + j) for j in range(n_years)]
        have = [c for c in cols if c in train.columns]
        res[d] = train[have].mean(axis=1, skipna=True) if have else np.nan
    return res


def persistent_growth(train, meta, thr=1.06, cap=1.25):
    """Family x region growth factor, applied only where volume grew > thr in BOTH of the last two
    years (EV-driven Electrical Assemblies, the new MEA region). Half-strength (sqrt of last year's
    growth) because a 3-year average already carries part of it."""
    key = pd.Series(list(zip(meta.loc[train.index, "Product_Family"].values,
                             train.index.get_level_values("Region").values)), index=train.index)

    def yoy(a0, a1, b0, b1):
        a = train.iloc[:, a0:a1].sum(axis=1, min_count=1).groupby(key).sum()
        b = train.iloc[:, b0:b1].sum(axis=1, min_count=1).groupby(key).sum()
        return a / b
    y1, y2 = yoy(-12, None, -24, -12), yoy(-24, -12, -36, -24)
    g = pd.Series(np.where((y1 > thr) & (y2 > thr), np.sqrt(y1.clip(upper=cap ** 2)), 1.0), index=y1.index)
    return key.map(g.clip(1.0, cap)).astype(float)


def forecast_all(train, meta, horizon_dates, train_full=None):
    """Returns {model_name: DataFrame(series x horizon months)} - same training data for every model."""
    lv = level12(train)
    months = np.array([d.month for d in horizon_dates])
    out = {}
    # baseline 1: seasonal naive (same calendar month, latest year)
    out["seasonal_naive"] = same_month_avg(train, horizon_dates, 1).apply(lambda c: c.fillna(lv))
    # baseline 2: 12-month moving average (flat)
    out["moving_avg_12"] = pd.DataFrame({d: lv for d in horizon_dates})
    # tested and rejected: 12-month level x family-region seasonal index (smooths away ordering cycles)
    idx = seasonal_index(train, meta)
    key = pd.MultiIndex.from_arrays([meta["Product_Family"].values, train.index.get_level_values("Region")])
    fac = idx.reindex(key).fillna(1.0).values[:, months - 1]
    out["level_x_family_seasonal"] = pd.DataFrame(lv.values[:, None] * fac, index=train.index,
                                                  columns=horizon_dates)
    # better model: each series' own same-month average over the last 3 years
    out["series_seasonal_avg3"] = same_month_avg(train, horizon_dates, 3).apply(lambda c: c.fillna(lv))
    # final model: the same, lifted where a family-region has grown persistently (lower bias, same WAPE)
    out["series_seasonal_avg3_trend"] = out["series_seasonal_avg3"].mul(persistent_growth(train if train_full is None else train_full, meta), axis=0)
    return out


# ------------------------------------------------------------------------- backtest
def score(actual, fc):
    a, f = actual.values.ravel(), fc.values.ravel()
    ok = ~np.isnan(a)
    a, f = a[ok], f[ok]
    return dict(WAPE=np.abs(a - f).sum() / max(a.sum(), 1e-9), Bias=(f.sum() - a.sum()) / max(a.sum(), 1e-9),
                Rows=int(ok.sum()))


def backtest(P_obs, P_model, P_true, meta, cust, events, ml_cache=None):
    """Rolling-origin backtest. Every rule is detected from data up to the origin only.
    Scored against true (unconstrained) demand - what the forecast has to predict."""
    rows, records = [], {}
    for o in ORIGINS:
        o = pd.Timestamp(o)
        keep = P_model.loc[:, P_model.columns <= o].notna().sum(axis=1) >= 6
        tr_model = P_model.loc[keep, P_model.columns <= o]
        tr_obs = P_obs.loc[tr_model.index, P_obs.columns <= o]
        hd = months_after(o, HORIZON)
        actual = P_true.reindex(index=tr_model.index, columns=hd)
        m = meta.loc[tr_model.index]
        reg = detect_regimes(P_model.loc[tr_model.index], cust, events, o)
        lost = reg.index[reg["Lost_Account"]]
        fam = m["Product_Family"].values
        regn = tr_model.index.get_level_values("Region").values

        fcs = dict(forecast_all(tr_model, m, hd))                       # baselines + models on model data
        base = fcs["series_seasonal_avg3_trend"]
        fcs["ablation 1: shipped units, no rules"] = forecast_all(tr_obs, m, hd)["series_seasonal_avg3_trend"]
        fcs["ablation 2: + unconstrained demand"] = base
        with_lost = base.copy(); with_lost.loc[with_lost.index.isin(lost)] = 0.0
        fcs["ablation 3: + lost accounts"] = with_lost
        masked = mask_regimes(tr_model, reg)
        reg_fc = forecast_all(masked, m, hd, train_full=tr_model)["series_seasonal_avg3_trend"]
        reg_fc.loc[reg_fc.index.isin(lost)] = 0.0
        fcs["series_seasonal_avg3_trend"] = reg_fc                      # statistical model: + customer/cycle regimes
        fcs["croston_sba"] = pd.DataFrame({d: ml.croston_sba(tr_model) for d in hd})   # intermittent-demand benchmark
        if ml_cache is not None:
            tr_ml = pd.concat([df[df.index.get_level_values("Date") <= o] for k, df in ml_cache.items() if k < o])
            pred = ml.fit_predict(tr_ml, ml_cache[o]).unstack("Date")
            pred.index = pd.MultiIndex.from_tuples(pred.index, names=["SKU_ID", "Region"])
            gbm = pred.reindex(index=tr_model.index, columns=hd)
            gbm_full = gbm.where(gbm.notna(), reg_fc)                    # series with <12 months: statistical fallback
            gbm_full.loc[gbm_full.index.isin(lost)] = 0.0
            fcs["global_gbm"] = gbm_full
            fcs["blend_gbm_seasonal"] = 0.5 * gbm_full + 0.5 * reg_fc

        scale = level12(tr_model).clip(lower=1.0)
        for name in (STAT_MODEL, "global_gbm", "blend_gbm_seasonal"):
            if name in fcs:                                           # keep forecast vs actual for the range calibration
                rec = pd.DataFrame({"forecast": fcs[name].stack(future_stack=True),
                                    "actual": actual.stack(future_stack=True)})
                rec.index.names = ["SKU_ID", "Region", "Date"]
                rec = rec.reset_index()
                rec["origin"] = o
                rec["h"] = ((rec["Date"].dt.year - o.year) * 12 + rec["Date"].dt.month - o.month)
                rec["scale"] = rec.set_index(["SKU_ID", "Region"]).index.map(scale).values
                rec["Family"] = rec.set_index(["SKU_ID", "Region"]).index.map(m["Product_Family"]).values
                records.setdefault(name, []).append(rec)
        for name, fc in fcs.items():
            r = score(actual, fc.round()); r.update(origin=o.date().isoformat(), model=name, variant="rounded",
                                                    group_type="overall", group="all"); rows.append(r)
            for gt, labels in (("family", fam), ("region", regn)):
                for g_ in np.unique(labels):
                    mk = labels == g_
                    r = score(actual[mk], fc[mk].round()); r.update(origin=o.date().isoformat(), model=name,
                                                                    variant="rounded", group_type=gt, group=g_)
                    rows.append(r)
            for lo, hi in ((1, 6), (7, 12), (13, 19)):
                cols = hd[lo - 1:hi]
                r = score(actual[cols], fc[cols].round()); r.update(origin=o.date().isoformat(), model=name,
                                                                    variant="rounded", group_type="horizon",
                                                                    group=f"months {lo}-{hi}"); rows.append(r)
    out = pd.DataFrame(rows)[["model", "variant", "origin", "group_type", "group", "WAPE", "Bias", "Rows"]]
    return out, {k: pd.concat(v, ignore_index=True) for k, v in records.items()}


# ------------------------------------------------------------------- final forecast
def load_ramp():
    """Ramp of new launches vs their predecessor's last-12-month level, and the predecessor's fade during the overlap."""
    r = pd.read_csv(OUT / "launch_ramp.csv").set_index("Phase")["Ratio_Median"].round(2)
    ramp = [(1, 3, r["successor months 1-3"]), (4, 6, r["successor months 4-6"]),
            (7, 12, r["successor months 7-12"]), (13, 99, r["successor months 13+"])]
    return ramp, float(r["predecessor months 1-3 after successor launch"])


def apply_lifecycle(fc_long, P, test, master, ramp, overlap_factor):
    """Rule 5 + new launches. fc_long: Date, SKU_ID, Region, Units_Forecast (before rules)."""
    f = fc_long.merge(test[["Date", "SKU_ID", "Region", "After_EOL", "New_Series"]],
                      on=["Date", "SKU_ID", "Region"], how="right")
    lv = level12(P)
    # new launches: predecessor's last-12-month level x ramp learned from past successions
    mm = master.set_index("SKU_ID")
    for (sku, reg), g in f[f["New_Series"]].groupby(["SKU_ID", "Region"]):
        pred = mm.loc[sku, "Replaces_SKU"]
        launch = mm.loc[sku, "Launch_Date"]
        base = lv.get((pred, reg), np.nan)
        if np.isnan(base):                                   # predecessor not sold in that region
            fam = mm.loc[sku, "Product_Family"]
            base = lv[[i for i in lv.index if i[1] == reg and mm.loc[i[0], "Product_Family"] == fam]].median()
        age = (g["Date"].dt.year - launch.year) * 12 + g["Date"].dt.month - launch.month + 1
        ramp_f = age.map(lambda a: next(r for lo, hi, r in ramp if lo <= a <= hi))
        f.loc[g.index, "Units_Forecast"] = (base * ramp_f).values
    # predecessor (EOL announced): volume drops while the successor is ramping up
    for sku in master.loc[master["Replaces_SKU"].notna() & (master["Lifecycle_Status"] == "Planned Launch"), "Replaces_SKU"]:
        succ = master[master["Replaces_SKU"] == sku].iloc[0]
        m = (f["SKU_ID"] == sku) & (f["Date"] >= succ["Launch_Date"])
        f.loc[m, "Units_Forecast"] *= overlap_factor
    f.loc[f["After_EOL"], "Units_Forecast"] = 0.0
    return f


def customer_event_factor():
    """Measured effect of a comparable past customer shutdown (the Nexon strike) from the anomaly register."""
    ar = pd.read_csv(OUT / "anomaly_register.csv")
    r = ar[(ar["Type"] == "customer_event") & ar["Reason"].str.contains("strike", na=False)].iloc[0]
    return float(r["Units_Observed"] / r["Units_Expected"])


STAT_MODEL = "series_seasonal_avg3_trend"


def choose_final(table):
    """Lowest backtest WAPE; the 50/50 blend is preferred when within 0.005 of the best (more robust)."""
    w = table["WAPE"]
    cands = {m: w[m] for m in (STAT_MODEL, "global_gbm", "blend_gbm_seasonal") if m in w.index}
    best = min(cands, key=cands.get)
    if "blend_gbm_seasonal" in cands and cands["blend_gbm_seasonal"] <= cands[best] + 0.005:
        best = "blend_gbm_seasonal"
    return best


def run_forecast(use_ml=True):
    P_obs, P_model, P_true, meta, cust, events = load_history()
    use_ml = use_ml and ml.HistGradientBoostingRegressor is not None
    if use_ml:
        cache = {o: ml.make_features(P_model, P_true, meta, cust, events, o, mask_regimes, persistent_growth)
                 for o in ml.TRAIN_ORIGINS}
    else:
        cache = None
        print("(scikit-learn not installed: ML model skipped, statistical model used)")
    bt, records = backtest(P_obs, P_model, P_true, meta, cust, events, cache)
    bt.to_csv(OUT / "backtest_metrics.csv", index=False)
    overall = bt[(bt.group_type == "overall") & (bt.variant == "rounded")]
    table = overall.groupby("model")[["WAPE", "Bias"]].mean().sort_values("WAPE")
    print("Backtest vs true demand (mean of 3 origins x 19-month horizon, rounded):")
    print(table.round(4).to_string())
    final_model = choose_final(table)
    print("Final model:", final_model)

    test = pd.read_parquet(OUT / "clean_test.parquet")
    master = pd.read_parquet(OUT / "clean_sku_master.parquet")
    hd = pd.DatetimeIndex(sorted(test["Date"].unique()))
    origin = P_model.columns[-1]
    reg = detect_regimes(P_model, cust, events, origin)
    masked = mask_regimes(P_model, reg)
    fc = forecast_all(masked, meta, hd, train_full=P_model)[STAT_MODEL]
    lost = reg.index[reg["Lost_Account"]]
    if final_model != STAT_MODEL:
        tr_all = pd.concat(cache.values())
        te_final = ml.make_features(P_model, P_true, meta, cust, events, origin, mask_regimes,
                                    persistent_growth, with_target=False)
        pred = ml.fit_predict(tr_all, te_final).unstack("Date")
        pred.index = pd.MultiIndex.from_tuples(pred.index, names=["SKU_ID", "Region"])
        gbm = pred.reindex(index=fc.index, columns=hd)
        gbm = gbm.where(gbm.notna(), fc)
        fc = gbm if final_model == "global_gbm" else 0.5 * gbm + 0.5 * fc
    fc.loc[fc.index.isin(lost)] = 0.0
    regime_keys = {k for k in masked.index if masked.loc[k].isna().sum() > P_model.loc[k].isna().sum()}

    long = fc.stack().rename("Units_Forecast").reset_index()
    long.columns = ["SKU_ID", "Region", "Date", "Units_Forecast"]
    ramp, overlap_factor = load_ramp()
    final = apply_lifecycle(long, P_model, test, master, ramp, overlap_factor)
    final = final.merge(test[["Date", "SKU_ID", "Region", "Customer_Name"]], on=["Date", "SKU_ID", "Region"], how="left")
    final["Adjustment"] = ""
    key = list(zip(final["SKU_ID"], final["Region"]))
    final.loc[[k in regime_keys for k in key], "Adjustment"] = "regime_history_only"
    final.loc[[k in set(lost) for k in key], "Adjustment"] = "lost_account"
    final.loc[final["New_Series"], "Adjustment"] = "new_series"
    # planned customer shutdowns inside the forecast window: scale by the measured effect of the Nexon strike
    factor = customer_event_factor()
    for _, e in events[(events["Scope_Type"] == "Customer") & events["In_Forecast_Window"]].iterrows():
        mk = ((final["Customer_Name"] == e["Scope_Customer"]) & (final["Region"] == e["Scope_Region"]) &
              (final["Date"] >= e["Start_Date"]) & (final["Date"] <= e["End_Date"]))
        final.loc[mk, "Units_Forecast"] *= factor
        final.loc[mk, "Adjustment"] = f"customer_shutdown {e['Event_ID']} (x{factor:.2f})"
    final.loc[final["After_EOL"], "Adjustment"] = "after_eol"
    final["Units_Forecast"] = final["Units_Forecast"].fillna(0.0).clip(lower=0)
    final["Model"] = final_model
    final[["Date", "SKU_ID", "Region", "Units_Forecast", "Model", "Adjustment"]].to_csv(OUT / "forecast.csv", index=False)
    from pipeline import ranges as rng                       # P10/P50/P90, calibrated on this backtest's errors
    profile = pd.read_csv(OUT / "series_profile.csv")
    rng.build_ranges(records[final_model], final, P_model, profile, test)

    sub = test[["Test_Row", "Date", "SKU_ID", "Region"]].merge(
        final[["Date", "SKU_ID", "Region", "Units_Forecast"]], on=["Date", "SKU_ID", "Region"], how="left")
    sub = sub.sort_values("Test_Row")
    sub["Units_Sold_Forecast"] = np.rint(sub["Units_Forecast"].fillna(0)).astype(int)
    sub["Date"] = sub["Date"].dt.strftime("%Y-%m")
    sub[["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]].to_csv(OUT / "submission.csv", index=False)
    return bt, final, sub


if __name__ == "__main__":
    bt, final, sub = run_forecast()
    print("\nSanity checks")
    print("  submission rows:", f"{len(sub):,}", "| total forecast units:", f"{sub['Units_Sold_Forecast'].sum():,}")
    print("  negative:", (sub["Units_Sold_Forecast"] < 0).sum(), "| blanks:", sub["Units_Sold_Forecast"].isna().sum())
    print("  adjustments:", final["Adjustment"].replace("", "none").str.split(" ").str[0].value_counts().to_dict())
    print("  history units 2025 (shipped):", int(pd.read_parquet(OUT / "clean_history.parquet")
          .query("Year==2025")["Units_Sold"].astype("float").sum()))
    print("  forecast units 2027:", int(sub[sub["Date"].str[:4] == "2027"]["Units_Sold_Forecast"].sum()))