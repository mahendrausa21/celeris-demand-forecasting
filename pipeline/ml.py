"""Global ML model (one gradient-boosting model for all series) + Croston/SBA benchmark.
Every feature is computed from data up to its forecast origin only (no leakage)."""
import numpy as np
import pandas as pd
from pipeline.anomalies import detect_regimes
from pipeline.config import OUT

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:                                   # pipeline still runs without scikit-learn
    HistGradientBoostingRegressor = None

HORIZON = 19
TRAIN_ORIGINS = list(pd.date_range("2020-03-01", "2026-09-01", freq="3MS"))   # origins used to build training samples
FEATURES = ["f_sn1", "f_sn2", "f_sn3", "f_base", "f_l3", "f_l6", "f_l24", "zero12", "zero36", "trail", "logscale",
            "growth_fr", "yoy", "nobs", "age", "regime", "regime_age", "h", "month", "fam", "reg"]
CATEGORICAL = ["month", "fam", "reg"]


# ------------------------------------------------------------------------ Croston / SBA
def croston_sba(train, alpha=0.1):
    """Syntetos-Boylan approximation of Croston's method for intermittent demand (flat forecast)."""
    out = pd.Series(0.0, index=train.index)
    arr = train.values
    for i in range(len(arr)):
        u = arr[i][~np.isnan(arr[i])]
        nz = np.where(u > 0)[0]
        if len(nz) == 0:
            continue
        z, p, last = u[nz[0]], float(nz[0] + 1), nz[0]
        for j in nz[1:]:
            z += alpha * (u[j] - z)
            p += alpha * ((j - last) - p)
            last = j
        out.iloc[i] = (1 - alpha / 2) * z / p
    return out


# -------------------------------------------------------------------------- features
def _same_month_avg(train, hd, n_years):
    origin = train.columns[-1]
    res = np.full((len(train), len(hd)), np.nan)
    for j, d in enumerate(hd):
        k0 = int(np.ceil(((d.year - origin.year) * 12 + d.month - origin.month) / 12))
        cols = [c for c in (d - pd.DateOffset(years=k0 + k) for k in range(n_years)) if c in train.columns]
        if cols:
            res[:, j] = np.nanmean(train[cols].values, axis=1) if len(cols) > 1 else train[cols[0]].values
    return res


def make_features(P_model, P_true, meta, cust, events, origin, mask_regimes, persistent_growth, with_target=True):
    """One row per (series, horizon month) at `origin`."""
    origin = pd.Timestamp(origin)
    train_full = P_model.loc[:, P_model.columns <= origin]
    keep = train_full.notna().sum(axis=1) >= 12
    train_full = train_full[keep]
    m = meta.loc[train_full.index]
    reg = detect_regimes(train_full, cust, events, origin)
    train = mask_regimes(train_full, reg)
    hd = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")

    def lvl(k):
        return np.nanmean(train.iloc[:, -k:].values, axis=1)
    with np.errstate(all="ignore"):
        l12, l3, l6, l24 = lvl(12), lvl(3), lvl(6), lvl(24)
        s = np.where(np.isnan(l12), 1.0, np.maximum(l12, 1.0))
        sn1, sn2, sn3 = (_same_month_avg(train, hd, n) for n in (1, 2, 3))
        g = persistent_growth(train_full, m).values
        base = np.where(np.isnan(sn3), l12[:, None], sn3) * g[:, None]
        prev12 = np.nanmean(train_full.iloc[:, -24:-12].values, axis=1)
        z12 = np.nanmean((train_full.iloc[:, -12:].values == 0).astype(float), axis=1)
        z36 = np.nanmean((train_full.iloc[:, -36:].values == 0).astype(float), axis=1)
    trail = reg["Trailing_Zeros"].reindex(train_full.index).values
    rs = pd.to_datetime(reg["Regime_Start"].reindex(train_full.index))
    regime_age = ((origin.year - rs.dt.year) * 12 + origin.month - rs.dt.month).values.astype(float)
    nobs = train_full.notna().sum(axis=1).values
    first = train_full.notna().values.argmax(axis=1)
    age = len(train_full.columns) - first
    # fixed category lists (MEA only exists from 2022: per-origin codes would shift between origins)
    fam_code = pd.Categorical(m["Product_Family"].values, categories=sorted(meta["Product_Family"].unique())).codes
    reg_code = pd.Categorical(train_full.index.get_level_values("Region"),
                              categories=sorted(set(P_model.index.get_level_values("Region")))).codes
    n, H = len(train_full), len(hd)
    rep = lambda a: np.repeat(np.asarray(a, dtype=float)[:, None], H, axis=1)
    sc = rep(s)
    F = {
        "f_sn1": np.nan_to_num(sn1, nan=-1) / sc, "f_sn2": np.nan_to_num(sn2, nan=-1) / sc,
        "f_sn3": np.nan_to_num(sn3, nan=-1) / sc, "f_base": base / sc,
        "f_l3": rep(l3 / s), "f_l6": rep(l6 / s), "f_l24": rep(l24 / s),
        "zero12": rep(z12), "zero36": rep(z36), "trail": rep(np.minimum(trail, 12)), "logscale": rep(np.log1p(s)),
        "growth_fr": rep(g), "yoy": rep(np.clip(l12 / np.where(prev12 > 0, prev12, np.nan), 0, 4)), "nobs": rep(np.minimum(nobs, 60)),
        "age": rep(np.minimum(age, 96)), "regime": rep(np.isfinite(regime_age).astype(float)), "regime_age": rep(regime_age),
        "h": np.tile(np.arange(1, H + 1), (n, 1)).astype(float),
        "month": np.tile([d.month for d in hd], (n, 1)).astype(float),
        "fam": rep(fam_code), "reg": rep(reg_code),
    }
    df = pd.DataFrame({k: v.ravel() for k, v in F.items()})
    idx = pd.MultiIndex.from_product([train_full.index, hd], names=["series", "Date"])
    df.index = idx
    df["scale"] = sc.ravel()
    df["base_fc"] = base.ravel()
    df["lost"] = np.repeat(reg["Lost_Account"].reindex(train_full.index).values, H)
    df["origin"] = origin
    if with_target:
        y = P_true.reindex(index=train_full.index, columns=hd).values.ravel()
        df["y"] = y
    return df


# ----------------------------------------------------------------------------- model
def fit_predict(train_df, test_df, n_iter=250, seed=0):
    """Predict units/scale with weights = scale (so big series count more, like WAPE)."""
    tr = train_df[(~train_df["lost"]) & train_df["y"].notna()]
    X = tr[FEATURES].astype("float32")
    mdl = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=n_iter, max_leaf_nodes=31, min_samples_leaf=300,
        l2_regularization=2.0, categorical_features=[FEATURES.index(c) for c in CATEGORICAL], random_state=seed)
    mdl.fit(X, (tr["y"] / tr["scale"]).clip(upper=8), sample_weight=tr["scale"].clip(upper=300))
    pred = np.clip(mdl.predict(test_df[FEATURES].astype("float32")), 0, None) * test_df["scale"].values
    pred[test_df["lost"].values] = 0.0
    return pd.Series(pred, index=test_df.index)
