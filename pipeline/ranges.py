"""P10 / P50 / P90 forecast ranges, calibrated on backtest errors and checked on held-out origins.
Series level: normalised residual (actual - forecast) / level, quantiles per intermittency class x horizon bucket.
Aggregate levels (total, family, region, family x region): quantiles of the relative error of the summed forecast
(quantiles of series do NOT add up, so each aggregate level is calibrated on its own errors).
Outputs: forecast_ranges.csv, forecast_ranges_aggregate.csv, range_calibration.csv, range_evaluation.csv"""
import numpy as np
import pandas as pd
from pipeline.config import OUT

Q = [0.1, 0.5, 0.9]
H_BINS, H_LABELS = [0, 6, 12, 19], ["1-6", "7-12", "13-19"]
CLASS_GROUP = {"smooth": "smooth", "intermittent": "intermittent", "erratic": "erratic_lumpy", "lumpy": "erratic_lumpy"}
NEW_SERIES_CLASS = "erratic_lumpy"          # no history: use the widest (most conservative) calibration


def class_group(cls):
    return cls.map(CLASS_GROUP).fillna(NEW_SERIES_CLASS)


def prepare_records(rec, profile_by_origin):
    r = rec[rec["actual"].notna() & rec["forecast"].notna()].copy()
    r = r.merge(profile_by_origin, on=["SKU_ID", "Region", "origin"], how="left")
    r["cg"] = class_group(r["Class"]).where(r["Class"].notna(), "smooth")
    r["hb"] = pd.cut(r["h"], H_BINS, labels=H_LABELS).astype(str)
    r["resid"] = (r["actual"] - r["forecast"]) / r["scale"]
    return r


def calibration(r):
    c = r.groupby(["cg", "hb"])["resid"].quantile(Q).unstack()
    c.columns = ["q10", "q50", "q90"]
    c["n"] = r.groupby(["cg", "hb"]).size()
    return c.reset_index()


def apply_series(df, cal):
    m = df.merge(cal, on=["cg", "hb"], how="left")
    for q, name in (("q10", "P10"), ("q50", "P50"), ("q90", "P90")):
        m[name] = np.maximum(0.0, m["forecast"] + m[q] * m["scale"])
    m[["P10", "P50", "P90"]] = np.sort(m[["P10", "P50", "P90"]].values, axis=1)      # keep P10 <= P50 <= P90
    return m


def _pinball(a, q, t):
    d = a - q
    return float(np.mean(np.maximum(t * d, (t - 1) * d)))


def leave_one_origin_out(r):
    """Calibrate on two origins, score on the third. Returns per-row held-out intervals."""
    parts = []
    for o in r["origin"].unique():
        te = apply_series(r[r["origin"] == o], calibration(r[r["origin"] != o]))
        parts.append(te)
    return pd.concat(parts)


def evaluation(ho):
    rows = []
    def add(name, g):
        a = g["actual"]
        rows.append(dict(Group=name, Rows=len(g), Coverage_P10_P90=float(((a >= g.P10) & (a <= g.P90)).mean()),
                         Below_P10=float((a < g.P10).mean()), Above_P90=float((a > g.P90).mean()),
                         WAPE_point=float(np.abs(a - g.forecast.round()).sum() / a.sum()),
                         WAPE_P50=float(np.abs(a - g.P50.round()).sum() / a.sum()),
                         Pinball_P10=_pinball(a, g.P10, 0.1), Pinball_P90=_pinball(a, g.P90, 0.9)))
    add("overall", ho)
    for o, g in ho.groupby("origin"):
        add(f"held-out origin {pd.Timestamp(o).date()}", g)
    for c, g in ho.groupby("cg"):
        add(f"class: {c}", g)
    for h, g in ho.groupby("hb"):
        add(f"horizon: {h}", g)
    return pd.DataFrame(rows)


def _held_out_coverage(g, k):
    cov = []
    for o in g["origin"].unique():
        q = g.loc[g["origin"] != o, "rel"].quantile(Q).values
        lo, hi = q[1] + k * (q[0] - q[1]), q[1] + k * (q[2] - q[1])
        te = g[g["origin"] == o]
        cov.append(((te["rel"] >= lo) & (te["rel"] <= hi)).mean())
    return float(np.mean(cov))


def aggregate_calibration(rec, target=0.80):
    """Relative error of summed forecasts at each aggregation level (per origin and month). Aggregate errors are
    dominated by origin-wide shifts, so raw quantiles under-cover on a held-out origin; the band is widened by the
    smallest factor k (searched on held-out origins) that reaches the target coverage."""
    r = rec[rec["actual"].notna() & rec["forecast"].notna()]
    out = []
    for level, keys in (("total", []), ("family", ["Family"]), ("region", ["Region"]), ("family_region", ["Family", "Region"])):
        g = r.groupby(["origin", "Date"] + keys)[["forecast", "actual"]].sum().reset_index()
        g["rel"] = g["actual"] / g["forecast"] - 1
        raw = _held_out_coverage(g, 1.0)
        k = next((x for x in np.arange(1.0, 3.01, 0.05) if _held_out_coverage(g, x) >= target), 3.0)
        q = g["rel"].quantile(Q).values
        out.append(dict(Level=level, q10=q[1] + k * (q[0] - q[1]), q50=q[1], q90=q[1] + k * (q[2] - q[1]), n=len(g),
                        Raw_Held_Out_Coverage=raw, Widening_Factor=float(k), Coverage_Held_Out=_held_out_coverage(g, k)))
    return pd.DataFrame(out)


def build_ranges(rec, final, P_model, profile, test_meta):
    """rec: backtest records of the final model. final: series-month forecast (Date, SKU_ID, Region,
    Units_Forecast, Adjustment, New_Series). profile: series_profile.csv."""
    from pipeline.forecast import level12
    from pipeline.features import series_profile
    from pipeline.forecast import ORIGINS
    prof_o = []
    u = pd.read_parquet(OUT / "unconstrained_history.parquet")
    P_true = u.pivot_table(index=["SKU_ID", "Region"], columns="Date", values="Units_Unconstrained", aggfunc="first")
    for o in ORIGINS:
        o = pd.Timestamp(o)
        p = series_profile(P_true.loc[:, P_true.columns <= o])[["SKU_ID", "Region", "Class"]]
        p["origin"] = o
        prof_o.append(p)
    r = prepare_records(rec, pd.concat(prof_o))
    cal = calibration(r)
    ev = evaluation(leave_one_origin_out(r))
    agg_cal = aggregate_calibration(rec)

    # ---- series-level ranges for the final forecast
    f = final[["Date", "SKU_ID", "Region", "Units_Forecast", "Adjustment", "New_Series"]].copy()
    origin = P_model.columns[-1]
    f["h"] = (f["Date"].dt.year - origin.year) * 12 + f["Date"].dt.month - origin.month
    lv = level12(P_model).clip(lower=1.0)
    f["scale"] = f.set_index(["SKU_ID", "Region"]).index.map(lv).values
    new_scale = f.groupby(["SKU_ID", "Region"])["Units_Forecast"].transform("mean").clip(lower=1.0)
    f["scale"] = f["scale"].fillna(new_scale)
    prof = profile.set_index(["SKU_ID", "Region"])["Class"]
    f["Class"] = f.set_index(["SKU_ID", "Region"]).index.map(prof).values
    f["cg"] = class_group(f["Class"])
    f.loc[f["New_Series"], "cg"] = NEW_SERIES_CLASS
    f["hb"] = pd.cut(f["h"], H_BINS, labels=H_LABELS).astype(str)
    f = apply_series(f.rename(columns={"Units_Forecast": "forecast"}), cal)
    zero = f["Adjustment"].isin(["lost_account", "after_eol"])
    f.loc[zero, ["P10", "P50", "P90"]] = 0.0
    f["Units_Forecast"] = f["forecast"]
    ranges = f[["Date", "SKU_ID", "Region", "Units_Forecast", "P10", "P50", "P90", "cg", "Adjustment"]].rename(
        columns={"cg": "Uncertainty_Class"})
    for c in ("P10", "P50", "P90"):
        ranges[c] = ranges[c].round(1)

    # ---- aggregate ranges (monthly): total, family, region, family x region
    fam = test_meta.set_index(["Date", "SKU_ID", "Region"])["Product_Family"]
    fm = final.merge(test_meta[["Date", "SKU_ID", "Region", "Product_Family"]], on=["Date", "SKU_ID", "Region"], how="left")
    agg = []
    for _, a in agg_cal.iterrows():
        keys = {"total": [], "family": ["Product_Family"], "region": ["Region"], "family_region": ["Product_Family", "Region"]}[a["Level"]]
        g = fm.groupby(["Date"] + keys)["Units_Forecast"].sum().reset_index()
        g["Level"] = a["Level"]
        g["Group"] = g[keys].astype(str).agg(" | ".join, axis=1) if keys else "All"
        for q, name in (("q10", "P10"), ("q50", "P50"), ("q90", "P90")):
            g[name] = np.maximum(0.0, g["Units_Forecast"] * (1 + a[q])).round(1)
        agg.append(g[["Level", "Group", "Date", "Units_Forecast", "P10", "P50", "P90"]])
    agg = pd.concat(agg, ignore_index=True)

    ranges.to_csv(OUT / "forecast_ranges.csv", index=False)
    agg.to_csv(OUT / "forecast_ranges_aggregate.csv", index=False)
    cal.to_csv(OUT / "range_calibration.csv", index=False)
    ev_out = pd.concat([ev.assign(Level="series"),
                        agg_cal.assign(Group=lambda d: "aggregate: " + d["Level"], Rows=lambda d: d["n"],
                                       Coverage_P10_P90=lambda d: d["Coverage_Held_Out"])
                        [["Group", "Rows", "Coverage_P10_P90", "Raw_Held_Out_Coverage", "Widening_Factor"]].assign(Level="aggregate")],
                       ignore_index=True)
    ev_out.to_csv(OUT / "range_evaluation.csv", index=False)
    return ranges, agg, ev, agg_cal
