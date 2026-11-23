"""Step 3 - demand drivers: macro lead/lag (with stability test), what-if sensitivities, EV shift,
intermittency classes.   Run with:  python -m pipeline.features
Outputs: macro_leadlag.csv, macro_leadlag_grid.csv, macro_sensitivity.csv, ev_shift.csv, series_profile.csv,
launch_ramp.csv"""
import numpy as np
import pandas as pd
from pipeline.config import OUT

LEVEL_INDICATORS = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days", "Industrial_Util_Rate_Pct"]
LAGS = list(range(0, 13))
N_PERM, SEED = 300, 0
MIN_ABS_CORR = 0.4                 # a relationship must reach this in BOTH halves of the data to count as stable
ADI_CUT, CV2_CUT = 1.32, 0.49      # Syntetos-Boylan thresholds


def indicator_frame(mac):
    """Level for ratio/diffusion indicators, 12-month % change for index-type indicators (removes trends)."""
    cols = [c for c in mac.columns if c not in ("Year", "Month", "Is_Forecast", "dq_flags")]
    X = pd.DataFrame(index=mac.index)
    for c in cols:
        X[c] = mac[c] if c in LEVEL_INDICATORS else mac[c].pct_change(12) * 100
    return X


def corr_by_lag(y, x):
    out = []
    for k in LAGS:
        xs = x.shift(k).reindex(y.index)
        ok = xs.notna() & y.notna()
        out.append(np.corrcoef(y[ok], xs[ok])[0, 1] if ok.sum() > 20 else np.nan)
    return np.array(out)


def lead_lag(fam_growth, X):
    """Correlation of each family's YoY demand growth with each indicator at lags 0-12 months.
    p-value: circular-shift permutation test of the BEST |correlation| over the 13 lags
    (keeps autocorrelation and accounts for choosing the best of 13)."""
    rng = np.random.default_rng(SEED)
    rows, grid = [], []
    for f in fam_growth.columns:
        y = fam_growth[f]
        h = len(y) // 2
        for ind in X.columns:
            x = X[ind]
            c = corr_by_lag(y, x)
            kb = int(np.nanargmax(np.abs(c)))
            xv = x.dropna()
            null = []
            for _ in range(N_PERM):
                s = rng.integers(14, len(xv) - 14)
                null.append(np.nanmax(np.abs(corr_by_lag(y, pd.Series(np.roll(xv.values, s), index=xv.index)))))
            p = (np.sum(np.array(null) >= abs(c[kb])) + 1) / (N_PERM + 1)
            c1, c2 = corr_by_lag(y.iloc[:h], x), corr_by_lag(y.iloc[h:], x)
            k1, k2 = int(np.nanargmax(np.abs(c1))), int(np.nanargmax(np.abs(c2)))
            xs = x.shift(kb).reindex(y.index)
            ok = xs.notna() & y.notna()
            slope = np.polyfit(xs[ok], y[ok], 1)[0]
            half = lambda sl: np.polyfit(xs[ok].iloc[sl], y[ok].iloc[sl], 1)[0]
            slope_h1, slope_h2 = half(slice(0, h)), half(slice(h, None))
            rows.append(dict(Family=f, Indicator=ind, Best_Lag=kb, Corr=c[kb], Corr_Lag0=c[0], P_Value=p,
                             Slope_LogGrowth_per_Unit=slope, Slope_H1=slope_h1, Slope_H2=slope_h2, Lag_H1=k1, Corr_H1=c1[k1], Lag_H2=k2, Corr_H2=c2[k2]))
            grid += [(f, ind, k, v) for k, v in zip(LAGS, c)]
    t = pd.DataFrame(rows)
    t["Stable"] = ((np.sign(t["Corr_H1"]) == np.sign(t["Corr_H2"])) & (t["Corr_H1"].abs() >= MIN_ABS_CORR) &
                   (t["Corr_H2"].abs() >= MIN_ABS_CORR) & ((t["Lag_H1"] - t["Lag_H2"]).abs() <= 3))
    t["Signal"] = np.where((t["P_Value"] < 0.05) & t["Stable"], "stable",
                           np.where(t["P_Value"] < 0.05, "significant but unstable", "not significant"))
    return t, pd.DataFrame(grid, columns=["Family", "Indicator", "Lag", "Corr"])


def ev_shift(u, sm, mac):
    u = u.merge(sm[["SKU_ID", "Powertrain_Application"]], on="SKU_ID")
    pt = u.groupby(["Date", "Powertrain_Application"])["Units_Unconstrained"].sum().unstack(fill_value=0)
    pt["Total"] = pt.sum(axis=1)
    pt["EV_Share"] = pt["EV"] / pt["Total"]
    pt = pt.join(mac["EV_Adoption_Index"])
    return pt.reset_index()


def series_profile(P_true):
    """Intermittency class from the last 36 months (Syntetos-Boylan: ADI and CV^2 of non-zero demand)."""
    rows = []
    for idx, row in zip(P_true.index, P_true.iloc[:, -36:].values):
        v = row[~np.isnan(row)]
        nz = v[v > 0]
        if len(v) < 12 or len(nz) < 2:
            rows.append((*idx, np.nan, np.nan, np.nan, "insufficient history"))
            continue
        adi, cv2 = len(v) / len(nz), (nz.std() / nz.mean()) ** 2
        cls = ("smooth" if cv2 < CV2_CUT else "erratic") if adi < ADI_CUT else ("intermittent" if cv2 < CV2_CUT else "lumpy")
        rows.append((*idx, adi, cv2, float((v == 0).mean()), cls))
    return pd.DataFrame(rows, columns=["SKU_ID", "Region", "ADI", "CV2", "Zero_Share_36m", "Class"])


RAMP_PHASES = [("months 1-3", 0, 3), ("months 4-6", 3, 6), ("months 7-12", 6, 12), ("months 13+", 12, 24)]


def launch_ramp(P_obs, sm):
    """How past successors ramped up, relative to their predecessor's last-12-month level, and how the predecessor
    behaved while the successor launched. Pairs = predecessor -> successor SKU pairs, per region, in the history.
    The forecast of new launches reads this table (no hand-typed constants)."""
    pairs = sm[sm["Replaces_SKU"].notna() & (sm["Lifecycle_Status"] == "Active")]
    skus = set(P_obs.index.get_level_values(0))
    succ_rows, pred_rows = [], []
    for succ, pred, launch in zip(pairs["SKU_ID"], pairs["Replaces_SKU"], pairs["Launch_Date"]):
        if succ not in skus or pred not in skus:
            continue
        for reg in P_obs.loc[succ].index.intersection(P_obs.loc[pred].index):      # successor ramp: same region
            ps, pp = P_obs.loc[(succ, reg)].dropna(), P_obs.loc[(pred, reg)].dropna()
            pre = pp[pp.index < launch].tail(12)
            post_s = ps[ps.index >= launch]
            if len(pre) >= 6 and pre.mean() > 0 and len(post_s) >= 12:
                succ_rows.append(dict(Successor=succ, Predecessor=pred, Region=reg,
                                      **{ph: post_s.iloc[a:b].mean() / pre.mean() for ph, a, b in RAMP_PHASES}))
        for reg in P_obs.loc[pred].index:             # predecessor fade: every region of the predecessor (the forecast applies it to all)
            pp = P_obs.loc[(pred, reg)].dropna()
            pre, post_p = pp[pp.index < launch].tail(12), pp[pp.index >= launch]
            if len(pre) >= 6 and pre.mean() > 0 and len(post_p) >= 3:
                pred_rows.append(dict(Predecessor=pred, Region=reg, Ratio=post_p.iloc[:3].mean() / pre.mean()))
    sr, pr = pd.DataFrame(succ_rows), pd.DataFrame(pred_rows)
    out = [dict(Phase=f"successor {ph}", Ratio_Median=sr[ph].median(), Ratio_P25=sr[ph].quantile(0.25),
                Ratio_P75=sr[ph].quantile(0.75), Pairs=int(sr[ph].notna().sum()),
                SKU_Pairs=int(sr[["Successor", "Predecessor"]].drop_duplicates().shape[0]))
           for ph, _, _ in RAMP_PHASES]
    out.append(dict(Phase="predecessor months 1-3 after successor launch", Ratio_Median=pr["Ratio"].median(),
                    Ratio_P25=pr["Ratio"].quantile(0.25), Ratio_P75=pr["Ratio"].quantile(0.75), Pairs=len(pr),
                    SKU_Pairs=int(pr["Predecessor"].nunique())))
    return pd.DataFrame(out)


def run_features():
    mac = pd.read_csv(OUT / "clean_macro.csv", parse_dates=["Date"]).set_index("Date")
    u = pd.read_parquet(OUT / "unconstrained_history.parquet")
    sm = pd.read_parquet(OUT / "clean_sku_master.parquet")
    fam = u.groupby(["Date", "Product_Family"])["Units_Unconstrained"].sum().unstack()
    growth = np.log(fam / fam.shift(12)).dropna()
    table, grid = lead_lag(growth, indicator_frame(mac))
    table.to_csv(OUT / "macro_leadlag.csv", index=False)
    grid.to_csv(OUT / "macro_leadlag_grid.csv", index=False)
    sens = table[table["Signal"] == "stable"][["Family", "Indicator", "Best_Lag", "Corr", "Slope_LogGrowth_per_Unit", "Slope_H1", "Slope_H2"]]
    sens.to_csv(OUT / "macro_sensitivity.csv", index=False)
    ev = ev_shift(u, sm, mac)
    ev.to_csv(OUT / "ev_shift.csv", index=False)
    P_true = u.pivot_table(index=["SKU_ID", "Region"], columns="Date", values="Units_Unconstrained", aggfunc="first")
    prof = series_profile(P_true)
    prof.to_csv(OUT / "series_profile.csv", index=False)
    P_obs = u.pivot_table(index=["SKU_ID", "Region"], columns="Date", values="Units_Observed", aggfunc="first")
    ramp = launch_ramp(P_obs, sm)
    ramp.to_csv(OUT / "launch_ramp.csv", index=False)
    return table, sens, ev, prof


if __name__ == "__main__":
    t, s, ev, prof = run_features()
    print("Lead/lag signals:", t["Signal"].value_counts().to_dict())
    print("Stable relationships (family | indicator | lead in months | corr | change in YoY growth per unit):")
    for _, r in s.sort_values("Corr", key=abs, ascending=False).iterrows():
        print(f"  {r['Family']:<26}{r['Indicator']:<26}lag {int(r['Best_Lag'])}  r={r['Corr']:+.2f}  slope={r['Slope_LogGrowth_per_Unit']:+.4f}")
    y = ev.set_index("Date")["EV_Share"]
    print("EV share of units: 2019 %.1f%% -> 2025 %.1f%% -> 2026 %.1f%%" % (
        100 * y["2019"].mean(), 100 * y["2025"].mean(), 100 * y["2026"].mean()))
    print("Intermittency classes:", prof["Class"].value_counts().to_dict())
    print(pd.read_csv(OUT / "launch_ramp.csv").round(3).to_string(index=False))