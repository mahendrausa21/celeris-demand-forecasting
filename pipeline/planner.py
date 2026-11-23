"""Exception list for planners: the series that need a human look, with a reason and a suggested action.
Run with:  python -m pipeline.planner      Output: planner_exceptions.csv"""
import numpy as np
import pandas as pd
from pipeline.config import OUT

LAST_MONTHS = 3                    # "recent" = last 3 months of history
JUMP_MIN_UNITS, JUMP_PCT_MED, JUMP_PCT_HIGH = 120, 0.30, 0.50
N_UNCERTAIN = 40
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
ACTIONS = {
    "lost_account": "Confirm with the account manager before cutting procurement; forecast is 0",
    "new_launch": "No history: forecast comes from the predecessor's ramp. Review against the launch plan",
    "eol_phase_out": "Run down stock; no procurement after the EOL month",
    "forecast_jump": "Check the driver (customer or cycle change, growth) before committing orders",
    "high_uncertainty": "Plan with the P90 buffer and review monthly",
    "recent_stockout": "Check supply and backorders; demand was higher than shipments",
    "recent_data_gap": "Missing ERP data in the last months: confirm the actuals",
    "customer_shutdown": "Planned customer shutdown is in the forecast: confirm dates with the customer",
}


def build_exceptions():
    r = pd.read_csv(OUT / "forecast_ranges.csv", parse_dates=["Date"])
    test = pd.read_parquet(OUT / "clean_test.parquet")
    u = pd.read_parquet(OUT / "unconstrained_history.parquet")
    hist_end = u["Date"].max()
    st = pd.read_parquet(OUT / "clean_stockouts.parquet")
    g = pd.read_parquet(OUT / "clean_history.parquet")
    reg = pd.read_csv(OUT / "series_regimes.csv")
    meta = test.drop_duplicates(["SKU_ID", "Region"]).set_index(["SKU_ID", "Region"])[
        ["Product_Family", "Customer_Name", "Replaces_SKU", "EOL_Date", "Launch_Date"]]
    first12 = r[r["Date"] <= r["Date"].min() + pd.DateOffset(months=11)]
    ser = pd.DataFrame({
        "Forecast_Next12": first12.groupby(["SKU_ID", "Region"])["Units_Forecast"].sum(),
        "P50_Next12": first12.groupby(["SKU_ID", "Region"])["P50"].sum(),
        "P90_Next12": first12.groupby(["SKU_ID", "Region"])["P90"].sum()})
    last12 = u[u["Date"] > hist_end - pd.DateOffset(months=12)].groupby(["SKU_ID", "Region"])["Units_Unconstrained"].sum()
    ser["History_Last12"] = last12.reindex(ser.index)
    ser["Change_Pct"] = (ser["Forecast_Next12"] / ser["History_Last12"].where(ser["History_Last12"] > 0) - 1) * 100
    ser = ser.join(meta)
    adj = r.groupby(["SKU_ID", "Region"])["Adjustment"].agg(lambda s: next((a for a in s.dropna() if a), ""))
    ser["Adjustment"] = adj.reindex(ser.index).fillna("")
    last_sale = u[u["Units_Unconstrained"] > 0].groupby(["SKU_ID", "Region"])["Date"].max()
    rows = []

    def add(idx, typ, sev, detail):
        for k, d in zip(idx, detail):
            rows.append(dict(Exception_Type=typ, Severity=sev if isinstance(sev, str) else sev[k], SKU_ID=k[0], Region=k[1],
                             Product_Family=ser.at[k, "Product_Family"], Customer_Name=ser.at[k, "Customer_Name"], Detail=d,
                             Forecast_Next12=ser.at[k, "Forecast_Next12"], History_Last12=ser.at[k, "History_Last12"],
                             Change_Pct=ser.at[k, "Change_Pct"], Suggested_Action=ACTIONS[typ]))

    lost = reg[reg["Lost_Account"]].set_index(["SKU_ID", "Region"])
    idx = [k for k in lost.index if k in ser.index]
    add(idx, "lost_account", "high", [f"No orders for {int(lost.at[k, 'Trailing_Zeros'])} months (last sale "
                                      f"{last_sale.get(k, pd.NaT):%Y-%m}); forecast set to 0" for k in idx])
    new = [k for k in ser.index if ser.at[k, "Adjustment"] == "new_series"]
    add(new, "new_launch", "medium", [f"Launches {ser.at[k, 'Launch_Date']:%Y-%m}, replaces {ser.at[k, 'Replaces_SKU']}" for k in new])
    eol_keys = test[test["After_EOL"]].drop_duplicates(["SKU_ID", "Region"])[["SKU_ID", "Region"]].itertuples(index=False)
    eol = [k for k in map(tuple, eol_keys) if k in ser.index]
    add(eol, "eol_phase_out", "medium", [f"End of life {ser.at[k, 'EOL_Date']:%Y-%m}; forecast 0 afterwards" for k in eol])
    sd = [k for k in ser.index if str(ser.at[k, "Adjustment"]).startswith("customer_shutdown")]
    add(sd, "customer_shutdown", "medium", [ser.at[k, "Adjustment"] for k in sd])
    skip = set(lost.index) | set(new) | set(eol)
    jump = ser[(ser["History_Last12"] >= JUMP_MIN_UNITS) & (ser["Change_Pct"].abs() > JUMP_PCT_MED * 100)]
    jk = [k for k in jump.index if k not in skip]
    sev = {k: ("high" if abs(ser.at[k, "Change_Pct"]) > JUMP_PCT_HIGH * 100 else "medium") for k in jk}
    add(jk, "forecast_jump", sev, [f"Next 12 months {ser.at[k, 'Change_Pct']:+.0f}% vs last 12 ("
                                  f"{ser.at[k, 'History_Last12']:,.0f} -> {ser.at[k, 'Forecast_Next12']:,.0f} units)" for k in jk])
    buf = (ser["P90_Next12"] - ser["P50_Next12"]).drop(index=list(skip), errors="ignore").nlargest(N_UNCERTAIN)
    add(list(buf.index), "high_uncertainty", "low", [f"P90 is {buf[k]:,.0f} units above P50 over the next 12 months" for k in buf.index])
    recent = st[(st["Date"] > hist_end - pd.DateOffset(months=LAST_MONTHS)) & (st["Requested_Units"] > st["Shipped_Units"])]
    rs = recent.groupby(["SKU_ID", "Region"]).agg(lost=("Lost_Units", "sum"), req=("Requested_Units", "sum"), ship=("Shipped_Units", "sum"))
    rs = rs[rs.index.isin(ser.index)]
    rsev = {k: ("high" if rs.at[k, "ship"] < 0.5 * rs.at[k, "req"] else "medium") for k in rs.index}
    add(list(rs.index), "recent_stockout", rsev, [f"Last {LAST_MONTHS} months: shipped {rs.at[k, 'ship']:,.0f} of {rs.at[k, 'req']:,.0f} requested" for k in rs.index])
    gaps = g[(g["Date"] > hist_end - pd.DateOffset(months=LAST_MONTHS)) & g["row_status"].isin(["missing_row", "units_unknown"])]
    gk = gaps.groupby(["SKU_ID", "Region"]).size()
    gk = gk[gk.index.isin(ser.index)]
    add(list(gk.index), "recent_data_gap", "low", [f"{n} month(s) without data in the last {LAST_MONTHS} months" for n in gk.values])

    out = pd.DataFrame(rows)
    out["_s"] = out["Severity"].map(SEV_ORDER)
    out = out.sort_values(["_s", "Forecast_Next12"], ascending=[True, False]).drop(columns="_s").reset_index(drop=True)
    out.insert(0, "Exception_ID", [f"EX-{i:04d}" for i in range(1, len(out) + 1)])
    out.to_csv(OUT / "planner_exceptions.csv", index=False)
    return out


if __name__ == "__main__":
    e = build_exceptions()
    print("Planner exceptions:", len(e))
    print(e.groupby(["Exception_Type", "Severity"]).size().to_string())
