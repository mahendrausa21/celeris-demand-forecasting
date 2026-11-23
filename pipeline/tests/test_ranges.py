"""Checks on the P10/P50/P90 ranges, hierarchy consistency and planner exceptions.  Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest
from pipeline.config import OUT


def _csv(name, **kw):
    path = OUT / name
    if not path.exists():
        pytest.fail(f"{name} not found - run  python run_pipeline.py  first")
    return pd.read_csv(path, **kw)


def test_series_ranges_ordered_and_rules_respected():
    r = _csv("forecast_ranges.csv")
    assert len(r) == 58219
    assert ((r["P10"] <= r["P50"] + 1e-9) & (r["P50"] <= r["P90"] + 1e-9)).all()
    assert (r["P10"] >= 0).all()
    z = r[r["Adjustment"].isin(["lost_account", "after_eol"])]
    assert len(z) == 460 and (z[["P10", "P50", "P90"]].sum().sum() == 0)


def test_series_ranges_calibrated_on_held_out_origins():
    ev = _csv("range_evaluation.csv")
    s = ev[(ev["Level"] == "series") & (ev["Group"] == "overall")].iloc[0]
    assert 0.74 <= s["Coverage_P10_P90"] <= 0.86             # nominal 80%
    assert abs(s["Below_P10"] - s["Above_P90"]) < 0.05       # balanced tails
    for g in ("class: smooth", "class: intermittent", "horizon: 1-6", "horizon: 13-19"):
        c = ev[ev["Group"] == g]["Coverage_P10_P90"].iloc[0]
        assert 0.72 <= c <= 0.88


def test_aggregate_ranges_reach_target_coverage():
    ev = _csv("range_evaluation.csv")
    a = ev[ev["Level"] == "aggregate"]
    assert set(a["Group"]) == {"aggregate: total", "aggregate: family", "aggregate: region", "aggregate: family_region"}
    assert (a["Coverage_P10_P90"] >= 0.78).all()
    assert (a["Widening_Factor"] >= 1.0).all()


def test_hierarchy_adds_up():
    """Point forecasts are bottom-up, so every level sums exactly to the level above."""
    f = _csv("forecast.csv", parse_dates=["Date"])
    r = _csv("forecast_ranges.csv", parse_dates=["Date"])
    a = _csv("forecast_ranges_aggregate.csv", parse_dates=["Date"])
    total = a[a["Level"] == "total"]["Units_Forecast"].sum()
    assert abs(total - f["Units_Forecast"].sum()) < 1.0
    assert abs(r["Units_Forecast"].sum() - f["Units_Forecast"].sum()) < 1.0
    for lvl in ("family", "region", "family_region"):
        assert abs(a[a["Level"] == lvl]["Units_Forecast"].sum() - total) < 1.0
    sub = _csv("submission.csv")
    assert abs(sub["Units_Sold_Forecast"].sum() - f["Units_Forecast"].sum()) < 0.5 * len(sub)   # rounding only


def test_planner_exceptions():
    e = _csv("planner_exceptions.csv")
    need = {"Exception_ID", "Exception_Type", "Severity", "SKU_ID", "Region", "Detail", "Suggested_Action"}
    assert need <= set(e.columns) and e["Exception_ID"].is_unique
    assert set(e["Severity"]) <= {"high", "medium", "low"}
    assert (e["Exception_Type"] == "lost_account").sum() == 16
    assert (e["Exception_Type"] == "new_launch").sum() == 10
    assert e["Suggested_Action"].notna().all() and e["Detail"].notna().all()
