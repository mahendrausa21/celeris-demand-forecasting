"""Checks on the cleaning pipeline's outputs.   Run:  pytest -q
(first build the outputs with:  python run_pipeline.py)"""
import json
import pandas as pd
import pytest
from pipeline.config import OUT, RAW


def _load(name):
    path = OUT / name
    if not path.exists():
        pytest.fail(f"{name} not found - run  python run_pipeline.py  first")
    return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)


@pytest.fixture(scope="module")
def hist():
    return _load("clean_history.parquet")


@pytest.fixture(scope="module")
def test_df():
    return _load("clean_test.parquet")


@pytest.fixture(scope="module")
def master():
    return _load("clean_sku_master.parquet")


# ------------------------------------------------------------------ history
def test_one_row_per_date_sku_region(hist):
    assert not hist.duplicated(["Date", "SKU_ID", "Region"]).any()


def test_status_values(hist):
    ok = {"observed", "true_zero", "repaired", "missing_row", "units_unknown"}
    assert set(hist["row_status"]) <= ok


def test_no_negative_units(hist):
    assert (hist["Units_Sold"].dropna() >= 0).all()


def test_revenue_equals_units_times_price(hist):
    h = hist.dropna(subset=["Units_Sold"])
    assert ((h["Units_Sold"] * h["Unit_Price_USD"] - h["Revenue_USD"]).abs() <= 0.011).all()


def test_missing_rows_are_not_zero(hist):
    assert hist.loc[hist["row_status"] == "missing_row", "Units_Sold"].isna().all()
    assert (hist.loc[hist["Units_Sold"] == 0, "row_status"] == "true_zero").all()
    assert hist.loc[hist["row_status"] == "true_zero", "Units_Sold"].eq(0).all()


def test_standard_labels(hist):
    assert hist["Region"].nunique() == 6
    assert hist["Sales_Channel"].nunique() == 4
    assert hist["Product_Family"].nunique() == 7
    assert hist[["Region", "Sales_Channel", "Product_Family", "Customer_Name",
                 "Business_Unit", "SKU_Description"]].notna().all().all()
    assert hist["SKU_ID"].str.match(r"^[A-Z]{3}-\d{3}$").all()


def test_each_family_has_one_business_unit(hist):
    assert (hist.groupby("Product_Family")["Business_Unit"].nunique() == 1).all()


def test_price_sane(hist):
    assert hist["Unit_Price_USD"].min() > 0
    assert hist["Unit_Price_USD"].max() < 10_000


def test_date_range(hist):
    assert hist["Date"].min() == pd.Timestamp("2019-01-01")
    assert hist["Date"].max() == pd.Timestamp("2026-10-01")


def test_region_start_and_lifecycle(hist, master):
    mea = hist[hist["Region"] == "Middle East & Africa"]
    assert mea["Date"].min() >= pd.Timestamp("2022-01-01")
    h = hist.merge(master[["SKU_ID", "Launch_Date", "EOL_Date"]], on="SKU_ID")
    assert (h["Date"] >= h["Launch_Date"]).all()
    assert (h["EOL_Date"].isna() | (h["Date"] <= h["EOL_Date"])).all()


def test_units_match_stockout_file(hist):
    st = _load("clean_stockouts.parquet")
    m = hist.merge(st[["Date", "SKU_ID", "Region", "Shipped_Units"]], on=["Date", "SKU_ID", "Region"])
    assert len(m) == len(st)
    assert (m["Units_Sold"].astype("float") == m["Shipped_Units"]).all()


# --------------------------------------------------------------------- test
def test_test_file_complete(test_df):
    assert len(test_df) == len(pd.read_csv(RAW / "test.csv"))
    assert not test_df.duplicated(["Date", "SKU_ID", "Region"]).any()
    assert test_df["Test_Row"].is_monotonic_increasing
    assert test_df["Date"].min() == pd.Timestamp("2026-11-01")
    assert test_df["Date"].max() == pd.Timestamp("2028-05-01")


def test_lifecycle_flags_in_test(test_df):
    assert test_df["After_EOL"].sum() == 156
    assert set(test_df.loc[test_df["After_EOL"], "SKU_ID"]) == {"PMP-003", "PTC-005", "THM-001"}
    assert not test_df["Before_Launch"].any()
    assert test_df.loc[test_df["New_Series"]].groupby(["SKU_ID", "Region"]).ngroups == 10


# -------------------------------------------------------------------- macro
def test_macro_clean():
    m = _load("clean_macro.csv")
    assert len(m) == 113 and m["Date"].is_unique
    assert m.drop(columns=["dq_flags"]).isna().sum().sum() == 0
    assert m["ISM_Manufacturing_PMI"].between(30, 70).all()
    ap = m["Auto_Production_YoY_Pct"]
    assert ((ap.abs() >= 0.1) | (ap == 0)).all()          # no fraction-style values left


def test_events_clean():
    e = _load("clean_events.csv")
    e = e.assign(Start_Date=pd.to_datetime(e["Start_Date"]), End_Date=pd.to_datetime(e["End_Date"]))
    assert (e["End_Date"].isna() | (e["End_Date"] >= e["Start_Date"])).all()
    assert e["Open_Ended"].sum() == 1


# ------------------------------------------------------------------- report
def test_dq_report_exists():
    rep = json.loads((OUT / "dq_report.json").read_text())
    assert len(rep["issues"]) > 20
    assert {"file", "issue", "rows_affected", "action"} <= set(rep["issues"][0])
    assert rep["files"]["train"][0] == 248248