"""Step 1 - cleaning pipeline.  Run with:  python -m pipeline.clean"""
import re
import json
import pandas as pd
from pipeline.config import RAW, OUT

# ---------------------------------------------------------------- issue log
ISSUES = []


def log_issue(file, issue, rows, action):
    """Record every fix so the dashboard's Data Health view can use it."""
    ISSUES.append({"file": file, "issue": issue,
                   "rows_affected": int(rows), "action": action})
    print(f"  [{file}] {issue}: {int(rows):,} rows -> {action}")


def load_raw(name):
    # dtype=str so pandas does not silently change values like "4,400" or "-"
    return pd.read_csv(RAW / f"{name}.csv", dtype=str)


# ------------------------------------------------------- label standardising
def _key(s):
    """lowercase, '-' -> space, '&' -> 'and', no dots, single spaces."""
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace("-", " ", regex=False).str.replace("&", " and ", regex=False)
    s = s.str.replace(".", "", regex=False)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


REGION_MAP = {
    "asia pacific": "Asia Pacific", "apac": "Asia Pacific",
    "europe": "Europe", "eu": "Europe",
    "latin america": "Latin America", "latam": "Latin America",
    "north america": "North America", "n america": "North America",
    "india": "India",
    "middle east and africa": "Middle East & Africa",
    "me and a": "Middle East & Africa", "mea": "Middle East & Africa",
}
CHANNEL_MAP = {
    "direct oem": "Direct OEM", "oem direct": "Direct OEM",
    "tier 1 distributor": "Tier-1 Distributor", "t1 distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket", "after market": "Aftermarket",
    "export": "Export",
}


def map_labels(df, col, mapping, file):
    before = df[col].copy()
    mapped = _key(before.fillna("")).map(mapping)
    mapped[before.isna()] = None            # keep real blanks as blanks
    unmapped = mapped.isna() & before.notna()
    if unmapped.any():
        print("  UNMAPPED", col, before[unmapped].unique()[:10])
    changed = (mapped != before) & mapped.notna()
    log_issue(file, f"{col} inconsistent spelling", changed.sum(), "mapped to standard label")
    log_issue(file, f"{col} missing", before.isna().sum(), "left blank for recovery step")
    df[col] = mapped


def clean_sku(s):
    s = s.astype(str).str.strip().str.upper().str.replace("_", "-", regex=False)
    parts = s.str.extract(r"^([A-Z]{3})-?(\d+)$")
    return parts[0] + "-" + parts[1].astype(float).astype("Int64").astype(str).str.zfill(3)


def clean_customer(s, min_count=500):
    """Group spelling variants (case, 'Inc.', spaces) and typos under 12 real names."""
    import difflib
    raw = s.fillna("").astype(str)
    key = raw.str.lower().str.replace(r"\binc\b\.?", "", regex=True)
    key = key.str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    counts = key[raw != ""].value_counts()
    canon_keys = counts[counts >= min_count].index.tolist()        # the real customers
    # display name = most common raw spelling inside each canonical key
    display = {k: raw[key == k].value_counts().index[0] for k in canon_keys}

    def resolve(k):
        if k in display:
            return display[k]
        m = difflib.get_close_matches(k, canon_keys, n=1, cutoff=0.8)   # typos
        return display[m[0]] if m else None

    lookup = {k: resolve(k) for k in counts.index}
    out = key.map(lookup)
    out[raw == ""] = None
    return out


DATE_FORMATS = ["%Y-%m", "%Y-%m-%d", "%m/%Y", "%Y/%m", "%b %Y", "%b-%Y", "%m-%Y"]


def parse_dates(s):
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = out.isna()
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out.dt.to_period("M")


# ---------------------------------------------------------- stage 1: train
def standardise_train():
    print("Loading train.csv ...")
    tr = load_raw("train")
    print(f"  rows loaded: {len(tr):,}")

    # SKU
    new_sku = clean_sku(tr["SKU_ID"])
    log_issue("train", "SKU_ID inconsistent format", (new_sku != tr["SKU_ID"]).sum(),
              "uppercased / stripped / zero-padded")
    tr["SKU_ID"] = new_sku

    # Region, Channel, Customer
    map_labels(tr, "Region", REGION_MAP, "train")
    map_labels(tr, "Sales_Channel", CHANNEL_MAP, "train")
    new_cust = clean_customer(tr["Customer_Name"])
    log_issue("train", "Customer_Name spelling variants",
              ((new_cust != tr["Customer_Name"]) & new_cust.notna()).sum(),
              "grouped under one canonical name")
    tr["Customer_Name"] = new_cust

    # Dates: Date is authoritative (business rule 1)
    tr["Period"] = parse_dates(tr["Date"])
    log_issue("train", "Date unparseable", tr["Period"].isna().sum(), "kept as missing, review")
    yr = pd.to_numeric(tr["Year"], errors="coerce")
    mo = pd.to_numeric(tr["Month"], errors="coerce")
    mism = (tr["Period"].dt.year != yr) | (tr["Period"].dt.month != mo)
    log_issue("train", "Year/Month disagree with Date", mism.sum(), "Year/Month rebuilt from Date")
    tr["Year"] = tr["Period"].dt.year
    tr["Month"] = tr["Period"].dt.month
    return tr


# ------------------------------------------------- stage 2: repair + dedupe
def repair_dates(tr):
    """Every good row was loaded exactly 1 month after its delivery month.
    Rows breaking that pattern (e.g. year 2062 typo for 2026) are rebuilt from Load_Timestamp."""
    tr["Load_Timestamp"] = pd.to_datetime(tr["Load_Timestamp"])
    expected = tr["Load_Timestamp"].dt.to_period("M") - 1
    bad = tr["Period"] != expected
    log_issue("train", "Date impossible (e.g. year 2062)", bad.sum(),
              "rebuilt from Load_Timestamp (load month minus 1)")
    tr.loc[bad, "Period"] = expected[bad]
    tr["Year"] = tr["Period"].dt.year
    tr["Month"] = tr["Period"].dt.month
    return tr


def drop_duplicates(tr):
    """Rule 2: a repeated Record_ID is a double load; keep the latest Load_Timestamp."""
    n_before = len(tr)
    dup_ids = tr["Record_ID"].duplicated(keep=False)
    tr = tr.sort_values("Load_Timestamp").drop_duplicates("Record_ID", keep="last")
    log_issue("train", "Duplicate Record_ID (double load)", n_before - len(tr),
              "kept latest Load_Timestamp")
    return tr.sort_index()


def recover_master_fields(tr):
    """Fill Family / Description / Business_Unit from sku_master (via the clean SKU_ID)."""
    sm = load_raw("sku_master")
    sm["SKU_ID"] = clean_sku(sm["SKU_ID"])
    for col in ["Product_Family", "SKU_Description"]:
        fill = tr["SKU_ID"].map(sm.set_index("SKU_ID")[col])
        miss = tr[col].isna()
        log_issue("train", f"{col} missing", miss.sum(), "filled from sku_master")
        tr.loc[miss, col] = fill[miss]

    # Rule: each family belongs to exactly ONE business unit.
    # sku_master contradicts this (one family has 2 BUs) -> use the majority BU per family.
    bu_counts = sm.groupby(["Product_Family", "Business_Unit"]).size().reset_index(name="n")
    fam_bu = bu_counts.sort_values("n").drop_duplicates("Product_Family", keep="last") \
                      .set_index("Product_Family")["Business_Unit"]
    bad_master = sm[sm["Business_Unit"] != sm["Product_Family"].map(fam_bu)]
    log_issue("sku_master", "SKU has a Business_Unit that contradicts its family", len(bad_master),
              "SKU(s) " + ", ".join(bad_master["SKU_ID"]) + ": majority BU of the family used")
    new_bu = tr["Product_Family"].map(fam_bu)
    log_issue("train", "train Business_Unit disagrees with its family", (new_bu != tr["Business_Unit"]).sum(),
              "set from family -> BU mapping")
    tr["Business_Unit"] = new_bu
    return tr


def recover_region(tr):
    """Missing Region: candidates = regions this SKU+customer+channel has sold in,
    minus regions already used in the same month for that SKU (one row per Date-SKU-Region).
    Assign only when exactly ONE candidate is left. (Tested: 99.8% recovered, 100% correct.)"""
    known = tr.dropna(subset=["Region"])
    cand = known.groupby(["SKU_ID", "Customer_Name", "Sales_Channel"])["Region"].agg(set).to_dict()
    used = known.groupby(["Period", "SKU_ID"])["Region"].agg(set).to_dict()
    miss = tr[tr["Region"].isna()]
    result = {}
    for idx, sku, cust, ch, per in zip(miss.index, miss["SKU_ID"], miss["Customer_Name"],
                                       miss["Sales_Channel"], miss["Period"]):
        left = cand.get((sku, cust, ch), set()) - used.get((per, sku), set())
        if len(left) == 1:
            result[idx] = next(iter(left))
    log_issue("train", "Region missing", len(miss), f"{len(result):,} recovered, "
              f"{len(miss) - len(result):,} left unknown")
    tr.loc[list(result), "Region"] = pd.Series(result)
    return tr


def stage2_train():
    tr = standardise_train()
    tr = repair_dates(tr)
    tr = drop_duplicates(tr)
    tr = recover_master_fields(tr)
    tr = recover_region(tr)
    return tr


# --------------------------------------------- stage 3: units, price, revenue
def add_flag(tr, mask, name):
    """Record a fix on the row itself (dq_flags column) so nothing changes silently."""
    mask = mask.fillna(False)
    tr.loc[mask, "dq_flags"] = tr.loc[mask, "dq_flags"].where(
        tr.loc[mask, "dq_flags"] == "", tr.loc[mask, "dq_flags"] + ";") + name


def fix_numbers(tr):
    tr["dq_flags"] = ""

    # 1. Units_Sold arrives as text ----------------------------------------
    raw = tr["Units_Sold"]
    log_issue("train", "Units_Sold has thousands separator ('4,400')",
              raw.str.contains(",", na=False).sum(), "commas removed")
    u = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce")
    dash = raw.eq("-")
    u[dash] = 0      # every '-' row has price 0 and revenue 0 -> means "no order"
    log_issue("train", "Units_Sold is '-'", dash.sum(), "treated as 0 (no order): price and revenue are also 0")
    add_flag(tr, dash, "dash_zero")
    tr["Units_Sold"] = u
    tr["Unit_Price_USD"] = pd.to_numeric(tr["Unit_Price_USD"])
    tr["Revenue_USD"] = pd.to_numeric(tr["Revenue_USD"])
    P0 = tr["Unit_Price_USD"].copy()          # original price (0 means 'no order' marker)

    # 2. Reference price per SKU-month (rule 4: same in every region) -------
    yr = tr["Period"].dt.year
    pos = tr["Unit_Price_USD"] > 0
    tr["_base"] = tr.groupby(["SKU_ID", yr])["Unit_Price_USD"].transform(lambda s: s[s > 0].median())
    ratio = tr["Unit_Price_USD"] / tr["_base"]
    wrong10 = pos & (ratio >= 5) & (ratio < 50)                # price x10
    wrong100 = pos & ((ratio >= 50) | (ratio <= 0.2))          # price x100 (cents)
    wrong = wrong10 | wrong100
    ok_price = tr["Unit_Price_USD"].where(pos & ~wrong)
    tr["_pref"] = ok_price.groupby([tr["SKU_ID"], tr["Period"]]).transform("median")
    tr["_pref"] = tr["_pref"].fillna(tr["_base"])

    # x10 price + units ~10x too small + revenue = units x (wrong) price  ->  unit-of-measure slip:
    # units were counted in packs of 10 and priced per pack. Revenue is right, so units x10, price /10.
    u_raw, r_raw, p_raw = tr["Units_Sold"], tr["Revenue_USD"], tr["Unit_Price_USD"]
    uom = wrong10 & (u_raw > 0) & ((r_raw / (u_raw * p_raw) - 1).abs() < 0.02)
    tr.loc[uom, "Units_Sold"] = u_raw[uom] * 10
    log_issue("train", "Unit-of-measure slip: units counted in 10s, price per 10 (all India, 2021)",
              uom.sum(), "units x10, price / 10 (revenue was right)")
    add_flag(tr, uom, "uom_fixed")
    log_issue("train", "Unit_Price off by x100 (cents slip)", wrong100.sum(),
              "replaced by the SKU's price in that month")
    log_issue("train", "Unit_Price off by x10 (price only)", (wrong10 & ~uom).sum(),
              "replaced by the SKU's price in that month")
    tr.loc[wrong, "Unit_Price_USD"] = tr.loc[wrong, "_pref"]
    add_flag(tr, wrong, "price_fixed")

    # 3. Missing units: recover from revenue, or infer 'no order' ------------
    blank = tr["Units_Sold"].isna()
    rev = tr["Revenue_USD"]
    from_rev = blank & (rev > 0)
    est = (rev / tr["_pref"]).round()
    tr.loc[from_rev, "Units_Sold"] = est[from_rev]
    log_issue("train", "Units_Sold blank but revenue present", from_rev.sum(),
              "units = revenue / SKU price")
    add_flag(tr, from_rev, "units_from_revenue")
    zero_inf = blank & ((rev == 0) | (rev.isna() & (P0 == 0)))
    tr.loc[zero_inf, "Units_Sold"] = 0
    log_issue("train", "Units_Sold blank, revenue and price are 0", zero_inf.sum(),
              "treated as 0 (no order)")
    add_flag(tr, zero_inf, "inferred_zero")
    still = tr["Units_Sold"].isna()
    log_issue("train", "Units_Sold unknown (order exists, quantity lost)", still.sum(),
              "left missing for the model step")
    add_flag(tr, still, "units_unknown")

    # 4. Negative units: sign slip (size matches normal demand, neighbours are normal)
    neg = tr["Units_Sold"] < 0
    log_issue("train", "Units_Sold negative", neg.sum(), "sign flipped (abs); flagged")
    tr.loc[neg, "Units_Sold"] = tr.loc[neg, "Units_Sold"].abs()
    add_flag(tr, neg, "sign_fixed")

    # 5. Price still 0 -> use the SKU's price in that month ------------------
    zp = tr["Unit_Price_USD"] == 0
    log_issue("train", "Unit_Price is 0", zp.sum(), "filled with SKU price of that month "
              "(no-order rows get their list price; revenue stays 0)")
    tr.loc[zp, "Unit_Price_USD"] = tr.loc[zp, "_pref"]

    # 6. Units x Price vs Revenue: decide WHICH field is wrong ---------------
    U, P, R = tr["Units_Sold"], tr["Unit_Price_USD"], tr["Revenue_USD"]
    chk = (U > 0) & (R > 0) & ~from_rev
    rr = R / (U * P)
    near = lambda x: (rr / x - 1).abs() < 0.05
    u10, u100 = chk & near(0.1), chk & near(0.01)
    tr.loc[u10 | u100, "Units_Sold"] = (R / P).round()[u10 | u100]
    log_issue("train", "Units 10x too large (revenue is right)", u10.sum(), "units = revenue / price")
    log_issue("train", "Units 100x too large (revenue is right)", u100.sum(), "units = revenue / price")
    add_flag(tr, u10 | u100, "units_fixed")
    r10 = chk & near(10)
    log_issue("train", "Revenue 10x too large (units are right)", r10.sum(), "revenue = units x price")
    add_flag(tr, r10, "revenue_fixed")
    other = chk & ((rr - 1).abs() > 0.02) & ~(u10 | u100 | r10)
    log_issue("train", "Revenue differs from units x price (other)", other.sum(), "revenue recomputed")
    add_flag(tr, other, "revenue_fixed")
    zr = (tr["Units_Sold"] > 0) & ~(R > 0) & ~from_rev
    log_issue("train", "Revenue 0 or missing although units > 0", zr.sum(), "revenue = units x price")
    add_flag(tr, zr, "revenue_fixed")

    # 7. Final: revenue always equals units x price --------------------------
    tr["Revenue_USD"] = (tr["Units_Sold"] * tr["Unit_Price_USD"]).round(2)
    tr["Units_Sold"] = tr["Units_Sold"].round().astype("Int64")
    return tr.drop(columns=["_base", "_pref"])


def stage3_train():
    tr = stage2_train()
    return fix_numbers(tr)


# ------------------------------------------------ stage 4: the full grid
START = pd.Period("2019-01", "M")
END = pd.Period("2026-10", "M")                       # last month of history
REGION_START = {"Middle East & Africa": pd.Period("2022-01", "M")}   # region opened Jan 2022 (EV-024)


def load_sku_master():
    sm = load_raw("sku_master")
    sm["SKU_ID"] = clean_sku(sm["SKU_ID"])
    sm["Launch"] = parse_dates(sm["Launch_Date"])
    sm["EOL"] = parse_dates(sm["EOL_Date"].fillna("2100-01"))
    return sm


def build_grid(tr):
    """One row per Date x SKU x Region inside each series' active window.
    Rule 3: a missing row is NOT a zero -> it stays missing and gets its own status."""
    sm = load_sku_master()

    # Rule 5: master Launch_Date must not be later than the first real sale.
    first_sale = tr[tr["Units_Sold"].fillna(0) > 0].groupby("SKU_ID")["Period"].min()
    sm["first_sale"] = sm["SKU_ID"].map(first_sale)
    contra = sm["first_sale"].notna() & (sm["first_sale"] < sm["Launch"])
    for _, r in sm[contra].iterrows():
        n = ((tr["SKU_ID"] == r["SKU_ID"]) & (tr["Period"] < r["Launch"])).sum()
        log_issue("sku_master", f"{r['SKU_ID']}: sales start {r['first_sale']} but Launch_Date is {r['Launch']}",
                  n, "transactions trusted; effective launch = first sale")
    sm["eff_launch"] = sm["Launch"].where(~contra, sm["first_sale"])

    # series that exist in the history + their active window
    series = tr[["SKU_ID", "Region"]].drop_duplicates().merge(
        sm[["SKU_ID", "eff_launch", "EOL"]], on="SKU_ID", how="left")
    series["start"] = [max(START, l, REGION_START.get(r, START))
                       for l, r in zip(series["eff_launch"], series["Region"])]
    series["end"] = [min(END, e) for e in series["EOL"]]
    parts = []
    for s, r, a, b in zip(series["SKU_ID"], series["Region"], series["start"], series["end"]):
        per = pd.period_range(a, b, freq="M")
        parts.append(pd.DataFrame({"SKU_ID": s, "Region": r, "Period": per}))
    grid = pd.concat(parts, ignore_index=True)

    obs = tr.copy()
    outside = ~obs.set_index(["SKU_ID", "Region", "Period"]).index.isin(
        grid.set_index(["SKU_ID", "Region", "Period"]).index)
    log_issue("train", "Rows outside the SKU/region active window", outside.sum(),
              "none expected (rule 5 / region start)")
    keep = ["SKU_ID", "Region", "Period", "Record_ID", "Customer_Name", "Sales_Channel",
            "Units_Sold", "Unit_Price_USD", "Revenue_USD", "dq_flags"]
    grid = grid.merge(obs[keep], on=["SKU_ID", "Region", "Period"], how="left")

    # descriptive columns come from the master (constant per SKU)
    meta = sm.drop_duplicates("SKU_ID").set_index("SKU_ID")
    grid["Product_Family"] = grid["SKU_ID"].map(meta["Product_Family"])
    grid["SKU_Description"] = grid["SKU_ID"].map(meta["SKU_Description"])
    fam_bu = tr.drop_duplicates("Product_Family").set_index("Product_Family")["Business_Unit"]
    grid["Business_Unit"] = grid["Product_Family"].map(fam_bu)

    # status ----------------------------------------------------------------
    grid["dq_flags"] = grid["dq_flags"].fillna("")
    fl = grid["dq_flags"]
    status = pd.Series("observed", index=grid.index)
    status[fl.str.contains("units_fixed|units_from_revenue|sign_fixed|uom_fixed")] = "repaired"
    status[grid["Units_Sold"] == 0] = "true_zero"
    status[fl.str.contains("units_unknown")] = "units_unknown"
    status[grid["Record_ID"].isna()] = "missing_row"
    grid["row_status"] = status

    # carry customer / channel into missing rows (same series), list price from the SKU-month
    grid = grid.sort_values(["SKU_ID", "Region", "Period"]).reset_index(drop=True)
    g = grid.groupby(["SKU_ID", "Region"])
    for c in ["Customer_Name", "Sales_Channel"]:
        grid[c] = g[c].ffill()
        grid[c] = grid.groupby(["SKU_ID", "Region"])[c].bfill()
    price = tr.groupby(["SKU_ID", "Period"])["Unit_Price_USD"].median().rename("List_Price_USD").reset_index()
    price = price.sort_values(["SKU_ID", "Period"])
    grid = grid.merge(price, on=["SKU_ID", "Period"], how="left")
    grid["List_Price_USD"] = grid.groupby("SKU_ID")["List_Price_USD"].ffill()
    grid["List_Price_USD"] = grid.groupby("SKU_ID")["List_Price_USD"].bfill()
    grid["Unit_Price_USD"] = grid["Unit_Price_USD"].fillna(grid["List_Price_USD"])

    miss = grid["row_status"] == "missing_row"
    log_issue("train", "Months with NO row inside the active window", miss.sum(),
              "kept as missing (NOT zero); status = missing_row")
    grid.loc[miss, "dq_flags"] = "missing_row"
    grid["Date"] = grid["Period"].dt.to_timestamp()
    grid["Year"], grid["Month"] = grid["Period"].dt.year, grid["Period"].dt.month
    cols = ["Date", "Year", "Month", "SKU_ID", "Region", "Product_Family", "Business_Unit",
            "SKU_Description", "Customer_Name", "Sales_Channel", "Units_Sold", "Unit_Price_USD",
            "Revenue_USD", "List_Price_USD", "row_status", "dq_flags", "Record_ID"]
    master = sm.drop_duplicates("SKU_ID").copy()
    master["Business_Unit"] = master["Product_Family"].map(fam_bu)
    return grid[cols], price, master


# ------------------------------------------------ stage 5: the other files
def family_bu_map(sm):
    """Rule: one family = one business unit. sku_master breaks it for PMP-004, so use the majority."""
    cnt = sm.groupby(["Product_Family", "Business_Unit"]).size().reset_index(name="n")
    return cnt.sort_values("n").drop_duplicates("Product_Family", keep="last") \
              .set_index("Product_Family")["Business_Unit"]


def clean_stockouts():
    st = load_raw("inventory_stockouts")
    st["SKU_ID"] = clean_sku(st["SKU_ID"])
    map_labels(st, "Region", REGION_MAP, "inventory_stockouts")
    st["Period"] = parse_dates(st["Date"])
    num = ["Requested_Units", "Carried_In_Units", "Shipped_Units", "Backordered_Units",
           "Lost_Units", "Fill_Rate_Pct", "Stockout_Days"]
    for c in num:
        st[c] = pd.to_numeric(st[c])
    lost = st["Requested_Units"] + st["Carried_In_Units"] - st["Shipped_Units"] - st["Backordered_Units"]
    log_issue("inventory_stockouts", "Lost_Units formula violated", (lost != st["Lost_Units"]).sum(),
              "none expected; kept as is")
    log_issue("inventory_stockouts", "Duplicate Date-SKU-Region keys",
              st.duplicated(["Period", "SKU_ID", "Region"]).sum(), "none expected")
    den = st["Requested_Units"] + st["Carried_In_Units"]
    calc = (100 * st["Shipped_Units"] / den).where(den > 0)
    st["Fill_Rate_Reported"] = st["Fill_Rate_Pct"]
    log_issue("inventory_stockouts", "Fill_Rate_Pct differs by >1 point from Shipped/(Requested+Carried_In)",
              ((calc - st["Fill_Rate_Reported"]).abs() > 1).sum(),
              "file's rates come from unrounded units; recomputed from integer columns, original kept")
    st["Fill_Rate_Pct"] = calc.round(1)
    st["Zero_Demand"] = den == 0
    log_issue("inventory_stockouts", "Row with zero requested and zero carried-in units", st["Zero_Demand"].sum(),
              "kept, flagged Zero_Demand (no demand was lost)")
    st["Date"] = st["Period"].dt.to_timestamp()
    return st.drop(columns="Period")


def reconcile_history_with_stockouts(grid, st):
    """The stockout file says Shipped_Units is what train must show. Where history is empty or
    disagrees, the stockout file wins."""
    key = ["Date", "SKU_ID", "Region"]
    m = grid.merge(st[key + ["Shipped_Units"]], on=key, how="left")
    cur = m["Units_Sold"].astype("float")
    has = m["Shipped_Units"].notna()
    fill = has & cur.isna()
    diff = has & cur.notna() & (cur != m["Shipped_Units"])
    log_issue("train", "History cell empty, stockout file has shipped units", fill.sum(),
              "filled from inventory_stockouts")
    log_issue("train", "History units differ from stockout file Shipped_Units", diff.sum(),
              "stockout file trusted")
    chg = fill | diff
    units = cur.copy()
    units[chg] = m.loc[chg, "Shipped_Units"]
    m["Units_Sold"] = units
    m.loc[chg, "Revenue_USD"] = (units[chg] * m.loc[chg, "Unit_Price_USD"]).round(2)
    m.loc[chg, "dq_flags"] = (m.loc[chg, "dq_flags"].str.replace("units_unknown", "")
                              .str.replace("missing_row", "").str.strip(";"))
    add_flag(m, chg, "units_from_stockouts")
    m.loc[chg, "row_status"] = pd.Series(
        ["true_zero" if v == 0 else "repaired" for v in units[chg]], index=m.index[chg])
    m["Units_Sold"] = m["Units_Sold"].round().astype("Int64")
    return m.drop(columns="Shipped_Units")


MACRO_COLS = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
              "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
              "Fuel_Price_Index", "EV_Adoption_Index"]


def clean_macro():
    m = load_raw("macro_indicators")
    m["Period"] = parse_dates(m["Date"])
    for c in MACRO_COLS:
        m[c] = pd.to_numeric(m[c])
    m["dq_flags"] = ""

    # duplicated month: keep the row closest to the average of the neighbouring months
    sd = m[MACRO_COLS].std()
    n_dropped = 0
    for p in m.loc[m["Period"].duplicated(), "Period"].unique():
        rows = m[m["Period"] == p]
        ref = m[m["Period"].isin([p - 1, p + 1])][MACRO_COLS].mean()
        keep = (((rows[MACRO_COLS] - ref).abs() / sd).sum(axis=1)).idxmin()
        n_dropped += len(rows) - 1
        m = m.drop(index=rows.index.difference([keep]))
    log_issue("macro_indicators", "Duplicated month", n_dropped,
              "kept the row closest to its neighbouring months")

    slip = m["ISM_Manufacturing_PMI"] < 20
    m.loc[slip, "ISM_Manufacturing_PMI"] *= 10
    log_issue("macro_indicators", "PMI decimal slip (4.79 instead of 47.9)", slip.sum(), "multiplied by 10")
    add_flag(m, slip, "pmi_fixed")

    frac = (m["Auto_Production_YoY_Pct"].abs() < 0.1) & (m["Auto_Production_YoY_Pct"] != 0)
    m.loc[frac, "Auto_Production_YoY_Pct"] *= 100
    log_issue("macro_indicators", "Auto_Production_YoY_Pct stored as a fraction, not percent (all of 2019)",
              frac.sum(), "multiplied by 100")
    add_flag(m, frac, "pct_unit_fixed")

    m = m.sort_values("Period").reset_index(drop=True)
    for c in MACRO_COLS:
        gap = m[c].isna()
        if gap.any():
            m[c] = m[c].interpolate(limit_direction="both")
            add_flag(m, gap, f"{c}_interpolated")
    log_issue("macro_indicators", "Single missing indicator values",
              m["dq_flags"].str.contains("interpolated").sum(), "linear interpolation between neighbours")
    expected = pd.period_range(START, "2028-05", freq="M")
    log_issue("macro_indicators", "Months missing from 2019-01..2028-05",
              len(set(expected) - set(m["Period"])), "none expected")
    m["Is_Forecast"] = m["Period"] > END
    m["Date"] = m["Period"].dt.to_timestamp()
    m["Year"], m["Month"] = m["Period"].dt.year, m["Period"].dt.month
    return m[["Date", "Year", "Month"] + MACRO_COLS + ["Is_Forecast", "dq_flags"]]


def clean_events():
    ev = load_raw("events_calendar")
    log_issue("events_calendar", "Dates in mixed format (YYYY-MM and YYYY-MM-DD)",
              (ev["Start_Date"].str.len() == 7).sum() + (ev["End_Date"].str.len() == 7).sum(),
              "parsed to real dates; month-only end dates mean end of that month")
    ev["Start"] = pd.to_datetime(ev["Start_Date"], format="mixed")
    ev["End"] = pd.to_datetime(ev["End_Date"], format="mixed")
    month_only = ev["End_Date"].str.len() == 7
    ev.loc[month_only, "End"] = ev.loc[month_only, "End"] + pd.offsets.MonthEnd(0)
    ev["Open_Ended"] = ev["End"].isna()
    log_issue("events_calendar", "Open-ended event (no End_Date)", ev["Open_Ended"].sum(),
              "flagged Open_Ended, applies from Start onward")
    parts = ev["Scope_Value"].str.split(r"\s*\|\s*", expand=True)
    is_cust = ev["Scope_Type"] == "Customer"
    ev["Scope_Customer"] = parts[0].where(is_cust)
    ev["Scope_Region"] = parts[1].where(is_cust)
    reg = ev["Scope_Type"] == "Region"
    ev.loc[reg, "Scope_Region"] = ev.loc[reg, "Scope_Value"]
    ev["Scope_Channel"] = ev["Scope_Value"].where(ev["Scope_Type"] == "Channel")
    ev["Scope_Family"] = ev["Scope_Value"].where(ev["Scope_Type"] == "Product_Family")
    log_issue("events_calendar", "Customer scope 'Customer | Region' in one field", is_cust.sum(),
              "split into Scope_Customer and Scope_Region")
    ev["In_Forecast_Window"] = ev["Start"] > END.to_timestamp(how="end")
    return ev.drop(columns=["Start_Date", "End_Date"]).rename(columns={"Start": "Start_Date", "End": "End_Date"})


def clean_test(history):
    te = load_raw("test")
    te["Test_Row"] = range(len(te))                 # keeps the original order for the submission
    te["SKU_ID"] = clean_sku(te["SKU_ID"])
    map_labels(te, "Region", REGION_MAP, "test")
    map_labels(te, "Sales_Channel", CHANNEL_MAP, "test")
    te["Customer_Name"] = clean_customer(te["Customer_Name"])
    te["Period"] = parse_dates(te["Date"])
    log_issue("test", "Duplicate Date-SKU-Region keys", te.duplicated(["Period", "SKU_ID", "Region"]).sum(),
              "none expected")
    sm = load_sku_master()
    fam_bu = family_bu_map(sm)
    new_bu = te["Product_Family"].map(fam_bu)
    log_issue("test", "Business_Unit contradicts its family",
              (new_bu != te["Business_Unit"]).sum(), "set from family -> BU mapping")
    te["Business_Unit"] = new_bu
    te = te.merge(sm[["SKU_ID", "Launch", "EOL", "Replaces_SKU", "Lifecycle_Status"]], on="SKU_ID", how="left")
    te["After_EOL"] = te["Period"] > te["EOL"]
    te["Before_Launch"] = te["Period"] < te["Launch"]
    log_issue("test", "Rows after the SKU's EOL month (forecast must be 0, rule 5)", te["After_EOL"].sum(),
              "flagged After_EOL")
    hist_keys = set(zip(history["SKU_ID"], history["Region"]))
    te["New_Series"] = [(s, r) not in hist_keys for s, r in zip(te["SKU_ID"], te["Region"])]
    log_issue("test", "Series with no history (new launches)", te.loc[te["New_Series"]].groupby(
        ["SKU_ID", "Region"]).ngroups, "flagged New_Series (use Replaces_SKU)")
    te["Date"] = te["Period"].dt.to_timestamp()
    te["Year"], te["Month"] = te["Period"].dt.year, te["Period"].dt.month
    te["Launch_Date"] = te["Launch"].dt.to_timestamp()
    te["EOL_Date"] = te["EOL"].where(te["EOL"] < pd.Period("2099-01", "M")).dt.to_timestamp()
    cols = ["Test_Row", "Date", "Year", "Month", "SKU_ID", "Region", "Product_Family", "Business_Unit",
            "SKU_Description", "Customer_Name", "Sales_Channel", "Lifecycle_Status", "Replaces_SKU",
            "Launch_Date", "EOL_Date", "After_EOL", "Before_Launch", "New_Series"]
    return te.sort_values("Test_Row")[cols].reset_index(drop=True)


def save_master(master):
    m = master.copy()
    m["Master_Launch_Date"] = m["Launch"].dt.to_timestamp()
    m["Launch_Date"] = m["eff_launch"].dt.to_timestamp()
    m["EOL_Date"] = m["EOL"].where(m["EOL"] < pd.Period("2099-01", "M")).dt.to_timestamp()
    cols = ["SKU_ID", "Product_Family", "Business_Unit", "SKU_Description", "Size_Class",
            "Powertrain_Application", "Launch_Date", "Master_Launch_Date", "EOL_Date",
            "Replaces_SKU", "Lifecycle_Status"]
    m[cols].to_parquet(OUT / "clean_sku_master.parquet", index=False)


def write_dq_report(files, grid):
    report = {"files": files,
              "history_row_status": grid["row_status"].value_counts().to_dict(),
              "issues": ISSUES}
    (OUT / "dq_report.json").write_text(json.dumps(report, indent=2, default=int))


def run_all():
    ISSUES.clear()
    raw_rows = {n: len(load_raw(n)) for n in
                ["train", "test", "macro_indicators", "sku_master", "events_calendar", "inventory_stockouts"]}
    st = clean_stockouts()
    tr = stage3_train()
    grid, price, master = build_grid(tr)
    grid = reconcile_history_with_stockouts(grid, st)
    macro, events = clean_macro(), clean_events()
    test = clean_test(grid)

    grid.to_parquet(OUT / "clean_history.parquet", index=False)
    price.assign(Date=price["Period"].dt.to_timestamp()).drop(columns="Period") \
         .to_parquet(OUT / "clean_prices.parquet", index=False)
    st.to_parquet(OUT / "clean_stockouts.parquet", index=False)
    macro.to_csv(OUT / "clean_macro.csv", index=False)
    events.to_csv(OUT / "clean_events.csv", index=False)
    test.to_parquet(OUT / "clean_test.parquet", index=False)
    save_master(master)
    files = {"train": [raw_rows["train"], len(grid)], "test": [raw_rows["test"], len(test)],
             "macro_indicators": [raw_rows["macro_indicators"], len(macro)],
             "sku_master": [raw_rows["sku_master"], len(master)],
             "events_calendar": [raw_rows["events_calendar"], len(events)],
             "inventory_stockouts": [raw_rows["inventory_stockouts"], len(st)]}
    write_dq_report(files, grid)
    return grid, test, macro, events, st


if __name__ == "__main__":
    grid, test, macro, events, st = run_all()
    print("\nSanity checks")
    print("  history rows:", f"{len(grid):,}", "| status:", grid["row_status"].value_counts().to_dict())
    print("  test rows:", f"{len(test):,}", "| after-EOL rows:", int(test["After_EOL"].sum()),
          "| new series:", test.loc[test["New_Series"]].groupby(["SKU_ID", "Region"]).ngroups)
    print("  macro months:", len(macro), "| PMI range:", macro["ISM_Manufacturing_PMI"].min().round(1),
          "-", macro["ISM_Manufacturing_PMI"].max().round(1))
    print("  events:", len(events), "| stockout rows:", len(st))
    print("  saved to:", OUT)