"""Checks on the final submission (skipped until outputs/submission.csv exists)."""
import pandas as pd
import pytest
from pipeline.config import OUT, RAW

SUB = OUT / "submission.csv"


@pytest.mark.skipif(not SUB.exists(), reason="no submission.csv yet")
def test_submission_format():
    sub = pd.read_csv(SUB)
    test = pd.read_csv(RAW / "test.csv")
    assert list(sub.columns) == ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    assert len(sub) == len(test)
    assert sub["Units_Sold_Forecast"].notna().all()
    assert (sub["Units_Sold_Forecast"] >= 0).all()
    assert (sub["Units_Sold_Forecast"] % 1 == 0).all()
    assert not sub.duplicated(["Date", "SKU_ID", "Region"]).any()


@pytest.mark.skipif(not SUB.exists(), reason="no submission.csv yet")
def test_zero_after_eol():
    sub = pd.read_csv(SUB)
    t = pd.read_parquet(OUT / "clean_test.parquet")
    t["Date"] = t["Date"].dt.strftime("%Y-%m")
    m = t.merge(sub, on=["Date", "SKU_ID", "Region"])
    assert (m.loc[m["After_EOL"], "Units_Sold_Forecast"] == 0).all()


METRICS = OUT / "backtest_metrics.csv"


def _mean_wape(model):
    bt = pd.read_csv(METRICS)
    x = bt[(bt["model"] == model) & (bt["group_type"] == "overall") & (bt["variant"] == "rounded")]
    return x["WAPE"].mean() if len(x) else None


@pytest.mark.skipif(not METRICS.exists(), reason="no backtest yet")
def test_final_model_beats_baselines():
    f = pd.read_csv(OUT / "forecast.csv")
    final = _mean_wape(f["Model"].iloc[0])
    assert final is not None
    for base in ("seasonal_naive", "moving_avg_12", "croston_sba"):
        assert final < _mean_wape(base)


@pytest.mark.skipif(not METRICS.exists(), reason="no backtest yet")
def test_backtest_has_three_origins_and_horizons():
    bt = pd.read_csv(METRICS)
    assert bt["origin"].nunique() == 3
    assert set(bt.loc[bt["group_type"] == "horizon", "group"]) == {"months 1-6", "months 7-12", "months 13-19"}