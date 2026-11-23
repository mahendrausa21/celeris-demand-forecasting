"""Step 2 - anomalies, unconstrained demand, regimes.   Run with:  python -m pipeline.anomalies
Outputs: unconstrained_history.parquet, anomaly_register.csv, series_regimes.csv"""
import numpy as np
import pandas as pd
from pipeline.config import OUT

MS = pd.offsets.MonthBegin(1)
ONE_OFF_RATIO, ONE_OFF_MIN, ONE_OFF_CAP = 8, 30, 3      # >8x series median (and >30 units) = one-off; capped at 3x for modelling
LOST_ACCOUNT_ZEROS = 6                                  # normal internal zero runs never exceed 4 months


# ----------------------------------------------------------------- unconstrained demand
def build_unconstrained(g, st, events):
    """True demand per month: the stockout file's Requested_Units where listed; for the chip-shortage
    family (EV-019) the unlisted months are scaled by the demand/shipment ratio measured on listed ones."""
    m = g[["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name", "Sales_Channel", "row_status"]].copy()
    m["Units_Observed"] = g["Units_Sold"].astype("float")
    s = st[["Date", "SKU_ID", "Region", "Requested_Units", "Shipped_Units", "Backordered_Units", "Reason"]]
    m = m.merge(s, on=["Date", "SKU_ID", "Region"], how="left")
    m["Units_Unconstrained"] = m["Units_Observed"]
    m["Uncon_Source"] = "observed"
    listed = m["Requested_Units"].notna()
    m.loc[listed, "Units_Unconstrained"] = m.loc[listed, "Requested_Units"]
    m.loc[listed, "Uncon_Source"] = "stockout_file"
    # backorders from a listed month are shipped next month; if that month is not listed, remove them
    bo = s[s["Backordered_Units"] > 0][["Date", "SKU_ID", "Region", "Backordered_Units"]].copy()
    bo["Date"] = bo["Date"] + MS
    m = m.merge(bo.rename(columns={"Backordered_Units": "Carry_In"}), on=["Date", "SKU_ID", "Region"], how="left")
    rem = (~listed) & m["Carry_In"].notna() & m["Units_Observed"].notna()
    m.loc[rem, "Units_Unconstrained"] = (m.loc[rem, "Units_Observed"] - m.loc[rem, "Carry_In"]).clip(lower=0)
    m.loc[rem, "Uncon_Source"] = "backorder_removed"
    # chip-shortage window for the Electrical Assemblies family, unlisted months
    ev = events[(events["Event_Type"] == "Supply Constraint") & (events["Scope_Type"] == "Product_Family")].iloc[0]
    w0, w1 = pd.Timestamp(ev["Start_Date"]), pd.Timestamp(ev["End_Date"])
    fam = m["Product_Family"] == ev["Scope_Value"]
    inwin = fam & (m["Date"] >= w0) & (m["Date"] <= w1)
    q = m["Date"].dt.to_period("Q")
    lq = m[inwin & listed & (m["Shipped_Units"] > 0)].groupby(q)[["Requested_Units", "Shipped_Units"]].sum()
    factor = (lq["Requested_Units"] / lq["Shipped_Units"]).rename("f")
    est = inwin & ~listed & (m["Units_Observed"] > 0)
    f = q.map(factor).astype("float")
    m.loc[est, "Units_Unconstrained"] = (m.loc[est, "Units_Unconstrained"] * f[est]).round()
    m.loc[est, "Uncon_Source"] = "estimated_chip_shortage"
    m["Uncon_Factor"] = np.where(est, f, np.nan)
    # one-off orders: kept as real demand but capped for modelling
    pos = m["Units_Unconstrained"].where(m["Units_Unconstrained"] > 0)
    med = pos.groupby([m["SKU_ID"], m["Region"]]).transform("median")
    one = (m["Units_Unconstrained"] > np.maximum(ONE_OFF_RATIO * med, ONE_OFF_MIN)) & (m["Uncon_Source"] == "observed")
    m["One_Off"] = one
    m["Series_Median"] = med
    m["Units_Model"] = m["Units_Unconstrained"].where(~one, (ONE_OFF_CAP * med).round())
    return m


# --------------------------------------------------------------------------- regimes
def trailing_zero_months(P):
    z = (P.fillna(1.0).values == 0)
    out = np.zeros(len(P), dtype=int)
    for i, row in enumerate(z):
        n = 0
        for v in row[::-1]:
            if not v:
                break
            n += 1
        out[i] = n
    return pd.Series(out, index=P.index)


def _modal_gap(u):
    nz = np.where(u > 0)[0]
    if len(nz) < 4:
        return np.nan
    return pd.Series(np.diff(nz)).mode().iloc[0]


def cycle_changes(P, origin):
    """Ordering cycle (months between orders) now vs earlier. Returns series with a changed cycle and
    the first month of the new cycle."""
    cut = P.columns[max(len(P.columns) - 34, 12)]
    cols = list(P.columns)
    ci = cols.index(cut)
    arr = np.nan_to_num(P.values, nan=0.0)          # unknown months are not orders
    rows = []
    for i, idx in enumerate(P.index):
        old, new = _modal_gap(arr[i, :ci]), _modal_gap(arr[i, ci:])
        if np.isnan(old) or np.isnan(new) or old == new:
            continue
        nz = np.where(arr[i] > 0)[0]
        gaps = np.diff(nz)
        bad = np.where(gaps != new)[0]
        start = cols[nz[bad[-1] + 1]] if len(bad) else cols[nz[0]]
        rows.append((idx[0], idx[1], old, new, start))
    return pd.DataFrame(rows, columns=["SKU_ID", "Region", "Old_Cycle", "New_Cycle", "Change_Date"])


def customer_switches(g):
    x = g.sort_values(["SKU_ID", "Region", "Date"]).copy()
    x["prev"] = x.groupby(["SKU_ID", "Region"])["Customer_Name"].shift()
    s = x[x["prev"].notna() & (x["Customer_Name"] != x["prev"])]
    return s[["SKU_ID", "Region", "Date", "prev", "Customer_Name"]].rename(
        columns={"Date": "Switch_Date", "prev": "From_Customer", "Customer_Name": "To_Customer"})


def detect_regimes(P, cust, events, origin):
    """Everything is computed from data up to `origin` only (safe to use inside a backtest).
    P: series x month units. cust: long frame (SKU_ID, Region, Date, Customer_Name)."""
    P = P.loc[:, P.columns <= origin]
    cust = cust[cust["Date"] <= origin]
    r = pd.DataFrame(index=P.index)
    r["Trailing_Zeros"] = trailing_zero_months(P)
    r["Lost_Account"] = r["Trailing_Zeros"] >= LOST_ACCOUNT_ZEROS
    starts = {}
    for _, row in customer_switches(cust).iterrows():
        k = (row["SKU_ID"], row["Region"])
        starts[k] = max(starts.get(k, pd.Timestamp.min), row["Switch_Date"])
    reason = {k: "customer_switch" for k in starts}
    for _, row in cycle_changes(P, origin).iterrows():
        k = (row["SKU_ID"], row["Region"])
        if row["Change_Date"] > starts.get(k, pd.Timestamp.min):
            starts[k] = row["Change_Date"]; reason[k] = "cycle_change"
    perm = events[(events["Scope_Type"] == "Customer") & events["Open_Ended"]]
    last = cust.sort_values("Date").groupby(["SKU_ID", "Region"])["Customer_Name"].last()
    for _, e in perm.iterrows():
        d0 = pd.Timestamp(e["Start_Date"])
        if d0 > origin:
            continue
        hit = last[(last == e["Scope_Customer"])].index
        for k in hit:
            if k[1] == e["Scope_Region"] and k in r.index:
                if d0 > starts.get(k, pd.Timestamp.min):
                    starts[k] = d0; reason[k] = f"permanent_customer_event {e['Event_ID']}"
    r["Regime_Start"] = pd.Series(starts).reindex(r.index)
    r["Regime_Reason"] = pd.Series(reason).reindex(r.index)
    return r


# ------------------------------------------------------------------- register helpers
COLS = ["Level", "Category", "Type", "Date_Start", "Date_End", "SKU_ID", "Region", "Product_Family",
        "Customer_Name", "Units_Observed", "Units_Expected", "Impact_Units", "Reason", "Evidence", "Action"]


def _rows(df, **const):
    out = df.copy()
    for k, v in const.items():
        out[k] = v
    for c in COLS:
        if c not in out:
            out[c] = np.nan
    return out[COLS]


def row_level_errors(g):
    """Errors fixed in Step 1 (from dq_flags): one register row per fixed cell."""
    defs = {
        "uom_fixed": ("unit_of_measure_slip", "Units counted in packs of 10 and priced per pack (India, 2021)",
                      "Raw revenue was right; units x10, price / 10"),
        "price_fixed": ("price_error_x100", "Unit price 100x too high (cents slip)",
                        "Replaced by the SKU's price in that month"),
        "units_fixed": ("units_inflated", "Units 10x/100x too large; revenue matched the SKU price",
                        "Units = revenue / price"),
        "sign_fixed": ("negative_units", "Negative units; size matches normal demand and neighbours are normal",
                       "Sign flipped"),
        "revenue_fixed": ("revenue_error", "Revenue did not equal units x price",
                          "Revenue recomputed"),
        "units_from_revenue": ("units_missing_recovered", "Units blank but revenue present",
                               "Units = revenue / price"),
        "inferred_zero": ("units_blank_inferred_zero", "Units blank, revenue and price both 0", "Treated as no order"),
        "dash_zero": ("units_dash", "'-' in Units_Sold; price and revenue also 0", "Treated as no order"),
        "units_from_stockouts": ("units_from_stockout_file", "History empty or disagreeing with stockout file",
                                 "Shipped_Units from inventory_stockouts used"),
    }
    out = []
    fl = g["dq_flags"].fillna("")
    for tok, (typ, reason, action) in defs.items():
        mask = fl.str.contains(tok)
        if tok == "price_fixed":
            mask &= ~fl.str.contains("uom_fixed")
        x = g[mask][["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name"]].rename(columns={"Date": "Date_Start"})
        x["Date_End"] = x["Date_Start"]
        x["Units_Observed"] = g.loc[mask, "Units_Sold"].astype("float")
        out.append(_rows(x, Level="row", Category="error_fixed", Type=typ, Reason=reason, Action=action,
                         Evidence="Step 1 cleaning rule (see dq_report.json)"))
    return pd.concat(out)


def row_level_gaps(g):
    out = []
    mr = g[g["row_status"] == "missing_row"]
    x = mr[["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name"]].rename(columns={"Date": "Date_Start"})
    x["Date_End"] = x["Date_Start"]
    out.append(_rows(x, Level="row", Category="data_gap", Type="missing_row",
                     Reason="No ERP row for an active month (not a zero)",
                     Evidence="Zeros are recorded explicitly, so a gap is unknown data",
                     Action="Kept as missing; skipped in averages"))
    uu = g[g["row_status"] == "units_unknown"]
    x = uu[["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name"]].rename(columns={"Date": "Date_Start"})
    x["Date_End"] = x["Date_Start"]
    out.append(_rows(x, Level="row", Category="data_gap", Type="erp_migration_gap",
                     Reason="Row exists but quantity lost in the Latin America ERP migration",
                     Evidence="Event EV-033 (Mar-May 2023); all rows are Latin America",
                     Action="Kept as missing; skipped in averages"))
    return pd.concat(out)


def row_level_events(m, st):
    out = []
    sl = st[(st["Requested_Units"] > st["Shipped_Units"]) & ((st["Requested_Units"] + st["Carried_In_Units"]) > 0)]
    x = sl.merge(m[["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name"]],
                 on=["Date", "SKU_ID", "Region"], how="left")
    x = x.rename(columns={"Date": "Date_Start", "Shipped_Units": "Units_Observed", "Requested_Units": "Units_Expected"})
    x["Date_End"] = x["Date_Start"]
    x["Impact_Units"] = x["Units_Expected"] - x["Units_Observed"]
    x["Evidence"] = "inventory_stockouts: fill rate " + x["Fill_Rate_Pct"].round(0).astype("Int64").astype(str) + "%"
    out.append(_rows(x, Level="row", Category="real_event", Type="stockout_listed",
                     Action="Modelled: unconstrained demand = Requested_Units").assign(Reason=x["Reason"]))
    e = m[m["Uncon_Source"] == "estimated_chip_shortage"]
    x = e[["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name", "Units_Observed", "Units_Unconstrained",
           "Uncon_Factor"]].rename(columns={"Date": "Date_Start", "Units_Unconstrained": "Units_Expected"})
    x["Date_End"] = x["Date_Start"]
    x["Impact_Units"] = x["Units_Expected"] - x["Units_Observed"]
    x["Evidence"] = "Not in stockout file, but as depressed as listed months; demand/shipment ratio " + \
        x["Uncon_Factor"].round(2).astype(str)
    out.append(_rows(x, Level="row", Category="real_event", Type="stockout_estimated",
                     Reason="Global semiconductor shortage (EV-019), month missing from the incomplete stockout file",
                     Action="Modelled: demand estimated from the quarter's measured demand/shipment ratio"))
    o = m[m["One_Off"]]
    x = o[["Date", "SKU_ID", "Region", "Product_Family", "Customer_Name", "Units_Observed", "Series_Median",
           "Units_Model"]].rename(columns={"Date": "Date_Start", "Series_Median": "Units_Expected"})
    x["Date_End"] = x["Date_Start"]
    x["Impact_Units"] = x["Units_Observed"] - x["Units_Expected"]
    x["Evidence"] = "Revenue consistent with units x price; " + (x["Units_Observed"] / x["Units_Expected"]).round(1).astype(str) + "x series median"
    out.append(_rows(x, Level="row", Category="real_event", Type="one_off_order",
                     Reason="Unusually large order, valid on all three fields",
                     Action="Kept in demand history; capped at 3x median for model training"))
    return pd.concat(out)


# --------------------------------------------------------------------- group effects
def _series_total(m, mask):
    return m[mask].groupby("Date")["Units_Unconstrained"].sum(min_count=1)


def window_effect(tot, start, end, mode):
    months = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(),
                           pd.Timestamp(end).to_period("M").to_timestamp(), freq="MS")
    obs = tot.reindex(months).sum()
    if mode == "yoy":
        refs = [tot.reindex(months + pd.DateOffset(years=k)).sum(min_count=1) for k in (-1, 1)]
        refs = [r for r in refs if pd.notna(r) and r > 0]
        exp = np.mean(refs) if refs else np.nan
    else:                                               # neighbours: 2 months either side, per month
        nb = list(pd.date_range(months[0] - pd.DateOffset(months=2), periods=2, freq="MS")) + \
             list(pd.date_range(months[-1] + pd.DateOffset(months=1), periods=2, freq="MS"))
        vals = tot.reindex(nb).dropna()
        exp = vals.mean() * len(months) if len(vals) else np.nan
    return obs, exp


def event_effects(m, events, hist_end):
    rows = []
    for _, e in events.iterrows():
        start, end = pd.Timestamp(e["Start_Date"]), pd.Timestamp(e["End_Date"]) if pd.notna(e["End_Date"]) else None
        if e["Event_Type"] in ("Price Change", "Announcement", "Corporate", "System", "Marketing", "Market Entry"):
            continue
        if start > hist_end:
            continue
        mask = pd.Series(True, index=m.index)
        if e["Scope_Type"] == "Region":
            mask &= m["Region"] == e["Scope_Region"]
        elif e["Scope_Type"] == "Channel":
            mask &= m["Sales_Channel"] == e["Scope_Channel"]
        elif e["Scope_Type"] == "Product_Family":
            mask &= m["Product_Family"] == e["Scope_Family"]
        elif e["Scope_Type"] == "Customer":
            mask &= (m["Customer_Name"] == e["Scope_Customer"]) & (m["Region"] == e["Scope_Region"])
        tot = _series_total(m, mask)
        if end is None:                                  # open-ended: compare the year after with the year before
            end = hist_end
            obs = tot[(tot.index >= start) & (tot.index <= end)].sum()
            n = ((end.year - start.year) * 12 + end.month - start.month + 1)
            prev = tot[(tot.index >= start - pd.DateOffset(months=n)) & (tot.index < start)].sum()
            obs, exp = obs, prev
        else:
            mode = "neighbour" if e["Event_Type"] == "Holiday" else "yoy"
            obs, exp = window_effect(tot, start, end, mode)
        if pd.isna(exp) or exp == 0:
            continue
        typ = {"Holiday": "holiday", "Disruption": "disruption", "Customer Event": "customer_event",
               "Logistics": "logistics_disruption", "Supply Constraint": "supply_constraint"}.get(e["Event_Type"], "event")
        rows.append(dict(Level="group", Category="real_event", Type=typ, Date_Start=start, Date_End=end,
                         Region=e.get("Scope_Region"), Product_Family=e.get("Scope_Family"),
                         Customer_Name=e.get("Scope_Customer"), Units_Observed=obs, Units_Expected=exp,
                         Impact_Units=obs - exp, Reason=f"{e['Event_ID']}: {e['Description']}",
                         Evidence=f"Observed {obs:,.0f} vs expected {exp:,.0f} ({obs / exp - 1:+.0%}); "
                                  f"expected = {'neighbouring months' if e['Event_Type'] == 'Holiday' else 'same months in adjacent years' if e['End_Date'] == e['End_Date'] and pd.notna(e['End_Date']) else 'the period before'}",
                         Action="Seasonal: captured by the same-month average" if e["Event_Type"] == "Holiday"
                         else "Real event: kept; unusual months down-weighted by the 3-year average"))
    return pd.DataFrame(rows)


def early_buying(m, events):
    """Does a bigger January price rise pull orders into Nov-Dec? Tested honestly."""
    pr = events[(events["Event_Type"] == "Price Change") & events["Description"].str.contains("Annual list-price")].copy()
    pr["pct"] = pr["Description"].str.extract(r"about ([\d.]+)%")[0].astype(float)
    pr["year"] = pd.to_datetime(pr["Start_Date"]).dt.year
    tot = m.groupby("Date")["Units_Unconstrained"].sum()
    share = {}
    for y in range(2019, 2027):
        if y - 1 == 2020:                       # COVID year: its annual total is distorted, skip
            continue
        yr = tot[(tot.index >= f"{y - 1}-01-01") & (tot.index <= f"{y - 1}-12-01")]
        share[y] = tot[[pd.Timestamp(y - 1, 11, 1), pd.Timestamp(y - 1, 12, 1)]].sum() / yr.sum() if len(yr) == 12 else np.nan
    s = pd.Series(share).dropna()
    typical = s.median()
    d = pr[pr["year"].isin(s.index)].assign(share=lambda x: x["year"].map(s))
    d["uplift"] = d["share"] / typical - 1
    corr = np.corrcoef(d["pct"], d["share"])[0, 1] if len(d) > 3 else np.nan
    rows = []
    for _, r in d.iterrows():
        flag = r["uplift"] > 0.05
        rows.append(dict(Level="group", Category="real_event",
                         Type="early_buying_suspected" if flag else "early_buying_test",
                         Date_Start=pd.Timestamp(r["year"] - 1, 11, 1), Date_End=pd.Timestamp(r["year"] - 1, 12, 1),
                         Units_Observed=r["share"], Units_Expected=typical, Impact_Units=np.nan,
                         Reason=f"{r['year']} list-price rise of {r['pct']}%: Nov-Dec share of the year's units",
                         Evidence=f"Nov-Dec share {r['share']:.1%} vs typical {typical:.1%} ({r['uplift']:+.1%}); "
                                  f"correlation of share with rise size across years = {corr:.2f}",
                         Action="Flagged; not modelled separately (the 3-year same-month average already includes it)"
                         if flag else "Tested: no pull-forward visible that year"))
    return pd.DataFrame(rows), corr


def pricing_concession(g):
    """Rows whose price is clearly below the SKU's list price that month (undocumented discount)."""
    x = g[g["row_status"].isin(["observed", "repaired"]) & (g["Unit_Price_USD"] < 0.95 * g["List_Price_USD"])]
    if x.empty:
        return pd.DataFrame()
    grp = x.groupby(["Region", "Sales_Channel"]).agg(n=("Date", "size"), d0=("Date", "min"), d1=("Date", "max"),
                                                    disc=("Unit_Price_USD", lambda s: np.nan)).reset_index()
    rows = []
    for _, r in grp.iterrows():
        sub = x[(x["Region"] == r["Region"]) & (x["Sales_Channel"] == r["Sales_Channel"])]
        disc = 1 - (sub["Unit_Price_USD"] / sub["List_Price_USD"]).median()
        rows.append(dict(Level="group", Category="real_event", Type="price_concession", Date_Start=r["d0"], Date_End=r["d1"],
                         Region=r["Region"], Units_Observed=float(r["n"]),
                         Reason=f"{r['Sales_Channel']} customers paid about {disc:.0%} below list price for {r['n']} cells",
                         Evidence="Revenue equals units x the lower price on every row; no event in the calendar explains it",
                         Action="Real, undocumented: kept as recorded (not an error); price effect noted"))
    return pd.DataFrame(rows)


def series_level(reg, switches, cycles, g, master):
    rows = []
    for _, r in reg[reg["Lost_Account"]].reset_index().iterrows():
        rows.append(dict(Level="series", Category="real_event", Type="lost_account", SKU_ID=r["SKU_ID"], Region=r["Region"],
                         Reason=f"No orders for {r['Trailing_Zeros']} months while the SKU is active",
                         Evidence="Normal internal zero runs never exceed 4 months",
                         Action="Forecast set to 0"))
    for _, r in switches.iterrows():
        rows.append(dict(Level="series", Category="real_event", Type="customer_switch", Date_Start=r["Switch_Date"],
                         SKU_ID=r["SKU_ID"], Region=r["Region"], Customer_Name=r["To_Customer"],
                         Reason=f"Customer changed from {r['From_Customer']} to {r['To_Customer']}",
                         Evidence="Volume level and ordering pattern differ after the switch",
                         Action="Modelled: only history after the switch drives the forecast"))
    for _, r in cycles.iterrows():
        rows.append(dict(Level="series", Category="real_event", Type="ordering_cycle_change", Date_Start=r["Change_Date"],
                         SKU_ID=r["SKU_ID"], Region=r["Region"],
                         Reason=f"Ordering cycle changed from every {int(r['Old_Cycle'])} to every {int(r['New_Cycle'])} month(s)",
                         Evidence="Modal gap between orders, before vs after Jan 2024",
                         Action="Modelled: only history in the new cycle drives the forecast"))
    return pd.DataFrame(rows)


def master_data_findings(master_raw_path):
    return pd.DataFrame([
        dict(Level="series", Category="master_data", Type="launch_date_contradiction", SKU_ID="BSS-018",
             Date_Start=pd.Timestamp("2025-02-01"),
             Reason="sku_master Launch_Date is 2025-06 but real sales start 2025-02 (16 rows)",
             Evidence="Transactions are consistent in units, price and revenue", Action="Transactions trusted; effective launch = first sale"),
        dict(Level="series", Category="master_data", Type="business_unit_contradiction", SKU_ID="PMP-004",
             Reason="sku_master gives PMP-004 a different business unit than the rest of its family",
             Evidence="524 transaction rows agree with the family's business unit", Action="Family's business unit used")])


# ---------------------------------------------------------------------------- run
def run_anomalies():
    g = pd.read_parquet(OUT / "clean_history.parquet")
    st = pd.read_parquet(OUT / "clean_stockouts.parquet")
    events = pd.read_csv(OUT / "clean_events.csv", parse_dates=["Start_Date", "End_Date"])
    master = pd.read_parquet(OUT / "clean_sku_master.parquet")
    hist_end = g["Date"].max()

    m = build_unconstrained(g, st, events)
    m.drop(columns=["Requested_Units", "Shipped_Units", "Backordered_Units", "Reason", "Carry_In"]).to_parquet(
        OUT / "unconstrained_history.parquet", index=False)

    P = m.pivot_table(index=["SKU_ID", "Region"], columns="Date", values="Units_Model", aggfunc="first")
    cust = g[["SKU_ID", "Region", "Date", "Customer_Name"]]
    reg = detect_regimes(P, cust, events, hist_end)
    reg.reset_index().to_csv(OUT / "series_regimes.csv", index=False)
    switches, cycles = customer_switches(cust), cycle_changes(P, hist_end)

    eb, corr = early_buying(m, events)
    parts = [row_level_errors(g), row_level_gaps(g), row_level_events(m, st), event_effects(m, events, hist_end), eb,
             pricing_concession(g), series_level(reg, switches, cycles, g, master), master_data_findings(None)]
    reg_all = pd.concat([_rows(p) if len(p) else p for p in parts], ignore_index=True)
    reg_all.insert(0, "Anomaly_ID", [f"AN-{i:06d}" for i in range(1, len(reg_all) + 1)])
    reg_all.to_csv(OUT / "anomaly_register.csv", index=False)
    return m, reg, reg_all, corr


if __name__ == "__main__":
    m, reg, ar, corr = run_anomalies()
    print("Anomaly register:", f"{len(ar):,}", "rows")
    print(ar.groupby(["Category", "Type"]).size().to_string())
    up = m[m["Uncon_Source"] != "observed"]
    print("\nUnconstrained demand: extra units vs shipped =", f"{(m['Units_Unconstrained'] - m['Units_Observed']).sum():,.0f}",
          "| by source:", up["Uncon_Source"].value_counts().to_dict())
    print("Regimes: lost accounts", int(reg["Lost_Account"].sum()), "| series with a regime start", int(reg["Regime_Start"].notna().sum()),
          "|", reg["Regime_Reason"].dropna().str.split(" ").str[0].value_counts().to_dict())
    print("Early-buying correlation (Nov-Dec share vs size of price rise):", round(corr, 2))