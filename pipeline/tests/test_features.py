"""Checks on the driver analysis (macro lead/lag, EV shift, intermittency classes).  Run: pytest -q"""
import pandas as pd
import pytest
from pipeline.config import OUT


def _csv(name):
    path = OUT / name
    if not path.exists():
        pytest.fail(f"{name} not found - run  python run_pipeline.py  first")
    return pd.read_csv(path)


def test_leadlag_grid_is_complete():
    g = _csv("macro_leadlag_grid.csv")
    assert len(g) == 7 * 8 * 13                      # 7 families x 8 indicators x lags 0..12
    assert g["Corr"].between(-1, 1).all()


def test_leadlag_table_and_stability_rule():
    t = _csv("macro_leadlag.csv")
    assert len(t) == 56 and set(t["Signal"]) == {"stable", "significant but unstable", "not significant"}
    st = t[t["Signal"] == "stable"]
    assert len(st) >= 5 and (st["P_Value"] < 0.05).all()
    assert ((st["Corr_H1"] * st["Corr_H2"]) > 0).all()                # same sign in both halves
    assert (st["Corr_H1"].abs() >= 0.4).all() and (st["Corr_H2"].abs() >= 0.4).all()
    assert (st["Best_Lag"] <= 12).all() and (st["Best_Lag"] >= 0).all()


def test_sensitivity_matches_stable_pairs():
    s, t = _csv("macro_sensitivity.csv"), _csv("macro_leadlag.csv")
    assert len(s) == (t["Signal"] == "stable").sum()
    assert {"Slope_LogGrowth_per_Unit", "Slope_H1", "Slope_H2"} <= set(s.columns)
    assert ((s["Slope_H1"] * s["Slope_H2"]) > 0).all()                # direction agrees across halves


def test_ev_shift():
    ev = _csv("ev_shift.csv")
    assert ev["EV_Share"].between(0, 1).all()
    assert ev["EV_Share"].iloc[:12].mean() < 0.01 and ev["EV_Share"].iloc[-12:].mean() > 0.05


def test_series_profile():
    p = _csv("series_profile.csv")
    assert set(p["Class"]) <= {"smooth", "erratic", "intermittent", "lumpy", "insufficient history"}
    assert len(p) == 3083
    assert (p["ADI"].dropna() >= 1).all()


def test_launch_ramp_is_measured_not_typed_in():
    r = _csv("launch_ramp.csv").set_index("Phase")
    assert list(r.index) == ["successor months 1-3", "successor months 4-6", "successor months 7-12",
                             "successor months 13+", "predecessor months 1-3 after successor launch"]
    assert (r["Pairs"] >= 15).all() and (r["SKU_Pairs"] >= 5).all()          # past predecessor -> successor series pairs
    assert (r["Ratio_P25"] <= r["Ratio_Median"]).all() and (r["Ratio_Median"] <= r["Ratio_P75"]).all()
    assert r.loc["successor months 1-3", "Ratio_Median"] < r.loc["successor months 4-6", "Ratio_Median"] < 1.0
    assert r.loc["predecessor months 1-3 after successor launch", "Ratio_Median"] < 1.0   # predecessor fades