"""Checks on the anomaly register and unconstrained demand.  Run:  pytest -q"""
import numpy as np
import pandas as pd
import pytest
from pipeline.config import OUT


def _load(name, **kw):
    path = OUT / name
    if not path.exists():
        pytest.fail(f"{name} not found - run  python run_pipeline.py  first")
    return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path, **kw)


@pytest.fixture(scope="module")
def unc():
    return _load("unconstrained_history.parquet")


@pytest.fixture(scope="module")
def reg():
    return _load("anomaly_register.csv")


def test_register_schema(reg):
    need = {"Anomaly_ID", "Level", "Category", "Type", "Date_Start", "SKU_ID", "Region", "Reason", "Evidence", "Action"}
    assert need <= set(reg.columns)
    assert reg["Anomaly_ID"].is_unique
    assert set(reg["Category"]) == {"error_fixed", "real_event", "data_gap", "master_data"}
    assert reg["Action"].notna().all() and reg["Reason"].notna().all()


def test_every_error_and_gap_is_registered(reg):
    hist = _load("clean_history.parquet")
    assert (reg["Type"] == "missing_row").sum() == (hist["row_status"] == "missing_row").sum()
    assert (reg["Type"] == "erp_migration_gap").sum() == (hist["row_status"] == "units_unknown").sum()
    assert (reg["Type"] == "unit_of_measure_slip").sum() == 385


def test_unconstrained_never_below_observed_when_listed(unc):
    listed = unc[unc["Uncon_Source"] == "stockout_file"]
    st = _load("clean_stockouts.parquet").set_index(["Date", "SKU_ID", "Region"])
    j = listed.set_index(["Date", "SKU_ID", "Region"]).join(st["Requested_Units"])
    assert (j["Units_Unconstrained"] == j["Requested_Units"]).all()


def test_unconstrained_adds_demand_only_in_expected_places(unc):
    up = unc[unc["Units_Unconstrained"] > unc["Units_Observed"]]
    assert set(up["Uncon_Source"]) <= {"stockout_file", "estimated_chip_shortage"}
    est = unc[unc["Uncon_Source"] == "estimated_chip_shortage"]
    assert set(est["Product_Family"]) == {"Electrical Assemblies"}
    assert est["Date"].between("2021-03-01", "2022-09-01").all()


def test_missing_stays_missing(unc):
    assert unc.loc[unc["row_status"].isin(["missing_row", "units_unknown"]) &
                   (unc["Uncon_Source"] == "observed"), "Units_Unconstrained"].isna().all()


def test_one_offs_capped_for_model(unc):
    o = unc[unc["One_Off"]]
    assert len(o) > 0 and (o["Units_Model"] < o["Units_Unconstrained"]).all()


def test_regimes_and_lost_accounts():
    r = _load("series_regimes.csv")
    assert r["Lost_Account"].sum() == 16
    assert (r.loc[r["Lost_Account"], "Trailing_Zeros"] >= 6).all()


def test_forecast_rules_applied():
    f = _load("forecast.csv")
    assert (f.loc[f["Adjustment"] == "lost_account", "Units_Forecast"] == 0).all()
    assert (f.loc[f["Adjustment"] == "after_eol", "Units_Forecast"] == 0).all()
    assert (f["Units_Forecast"] >= 0).all()
