"""Feature panel: one row per (forecast origin, step) with features and optional labels.

Feature naming:  <series>__last / __mean4h  state at the origin (newest usable values)
                 <series>__now               known-ahead series at the origin quarter-hour
                 <series>__d                 known-ahead series at the delivery quarter-hour
Labels (y_*) are read as known at the training cut-off, never later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import DIR, IMB, SPOT, Store

# Realised series: describe the system up to the origin (published with a lag).
REALISED = [
    IMB, DIR, "fi_mfrr_up_price", "fi_mfrr_down_price", "fi_mfrr_up_activated", "fi_mfrr_down_activated",
    "fi_afrr_up_price", "fi_afrr_down_price", "fi_consumption", "fi_production",
    "fi_se3_flow", "fi_no4_flow", "fi_se1_flow", "fi_ee_flow",
    "se1_imbalance_price_long", "se3_imbalance_price_long", "ee_imbalance_price_long",
]
# Known-ahead series: published before delivery, read at the delivery quarter-hour.
AHEAD = [
    SPOT, "fi_consumption_fc_da", "fi_wind_fc_da", "fi_solar_fc_da",
    "se1_load_fc_da", "se3_load_fc_da", "ee_load_fc_da", "fi_ee_capacity_da", "ee_fi_capacity_da",
]
CALENDAR = ["cal_weekday_local", "cal_hour_local", "cal_is_holiday_fi", "cal_is_dst"]

META = ["origin", "delivery", "step"]
LABELS = ["y_imb", "y_spot", "y_dir", "y_prem"]
LOOKBACK = 16                      # quarter-hours (4 h) of realised history per origin


def _newest(mat: np.ndarray) -> np.ndarray:
    """Newest non-NaN value per row of a (origins x lags) matrix, lag 1 first."""
    has = ~np.isnan(mat)
    first = has.argmax(1)
    return np.where(has.any(1), mat[np.arange(len(mat)), first], np.nan)


def _nanmean(mat: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(mat, axis=1) if mat.size else np.full(len(mat), np.nan)


def origin_features(store: Store, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """System state at each origin; shared by its 40 steps."""
    f: dict[str, np.ndarray] = {}
    win = {sid: store.window(sid, origins, LOOKBACK) for sid in REALISED + ["fi_consumption_fc_da", SPOT]}
    for sid in REALISED:
        f[f"{sid}__last"] = _newest(win[sid])
        f[f"{sid}__mean4h"] = _nanmean(win[sid])
    prem = win[IMB] - win[SPOT]
    f["prem__last"] = _newest(prem)
    f["prem__mean4h"] = _nanmean(prem)
    with np.errstate(all="ignore"):
        f["prem__std4h"] = np.nanstd(prem, axis=1)
    d = win[DIR]
    f["dir__share_up4h"] = _nanmean(np.where(np.isnan(d), np.nan, d == 1))
    f["dir__share_down4h"] = _nanmean(np.where(np.isnan(d), np.nan, d == -1))
    f["cons_err__mean4h"] = _nanmean(win["fi_consumption"]) - _nanmean(win["fi_consumption_fc_da"])
    f["net_import__last"] = f["fi_consumption__last"] - f["fi_production__last"]
    has = ~np.isnan(win[IMB])
    f["imb__staleness_min"] = np.where(has.any(1), (has.argmax(1) + 1) * 15.0, np.nan)

    # how long the current mFRR direction has lasted (quarter-hours, up to 16 h)
    dlong = store.window(DIR, origins, 64)
    newest = _newest(dlong)
    first = (~np.isnan(dlong)).argmax(1)
    run = np.zeros(len(origins))
    for i in range(len(origins)):
        seq = dlong[i, first[i]:]
        diff = np.flatnonzero(seq != newest[i])
        run[i] = diff[0] if len(diff) else len(seq)
    f["dir__run_length"] = np.where(np.isnan(newest), np.nan, run)
    # aFRR activation frequency: aFRR series have a row only when aFRR was activated. It is
    # published with the same delay as the imbalance price (availability.csv), so among the
    # quarter-hours whose imbalance price is already published, a missing aFRR row means "no activation".
    published = ~np.isnan(win[IMB])
    for sid, name in (("fi_afrr_up_price", "afrr_up"), ("fi_afrr_down_price", "afrr_down")):
        active = ~np.isnan(win[sid]) & published
        with np.errstate(all="ignore"):
            f[f"{name}__share4h"] = active.sum(1) / published.sum(1)
    for sid in ("fi_wind_fc_da", "fi_consumption_fc_da", "fi_solar_fc_da", SPOT):
        f[f"{sid}__now"] = store.asof(sid, origins, origins)
    return pd.DataFrame(f, index=origins)


def build_panel(store: Store, origins, label_cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rows (origin, step 1..40) with features; labels as known at `label_cutoff` if given."""
    origins = pd.DatetimeIndex(origins)
    n = len(origins)
    steps = np.tile(np.arange(1, C.HORIZON + 1), n)
    o = origins.repeat(C.HORIZON)
    d = o + (steps - 1) * C.Q
    p = pd.DataFrame({"origin": o, "delivery": d, "step": steps})
    of = origin_features(store, origins)
    p[list(of.columns)] = of.to_numpy()[np.repeat(np.arange(n), C.HORIZON)]

    for sid in CALENDAR:
        p[sid] = store.asof(sid, d, o)
    p["cal_minute"] = d.minute.to_numpy()           # position in the hour: schedules change on the hour
    if "regime_hourly_price" in store:              # extended-history variants with a regime flag only
        p["regime_hourly_price"] = store.asof("regime_hourly_price", d, o)
    for sid in AHEAD:
        p[f"{sid}__d"] = store.asof(sid, d, o)
    p["netload_fc__d"] = p["fi_consumption_fc_da__d"] - p["fi_wind_fc_da__d"] - p["fi_solar_fc_da__d"].fillna(0)
    p["wind_fc_change__d"] = p["fi_wind_fc_da__d"] - p["fi_wind_fc_da__now"]
    p["spot_change__d"] = p["fi_spot_price__d"] - p["fi_spot_price__now"]
    p["spot_ramp__d"] = p["fi_spot_price__d"] - store.asof(SPOT, d - 4 * C.Q, o)
    p["wind_ramp__d"] = p["fi_wind_fc_da__d"] - store.asof("fi_wind_fc_da", d - 4 * C.Q, o)
    day = pd.Timedelta(hours=24)        # same quarter-hour yesterday: always published by the origin
    p["imb__d_minus_24h"] = store.asof(IMB, d - day, o)
    p["prem__d_minus_24h"] = p["imb__d_minus_24h"] - store.asof(SPOT, d - day, o)
    p["dir__d_minus_24h"] = store.asof(DIR, d - day, o)

    if label_cutoff is not None:
        p["y_imb"] = store.asof(IMB, d, label_cutoff)
        p["y_spot"] = store.asof(SPOT, d, label_cutoff)
        p["y_dir"] = store.asof(DIR, d, label_cutoff)
        p["y_prem"] = p["y_imb"] - p["y_spot"]
        # when the whole label became usable: lets a cross-validation fold keep only
        # labels that were published by its own cut-off
        fu = np.maximum.reduce([store.first_usable(s, d) for s in (IMB, SPOT, DIR)])
        fu[fu == np.iinfo(np.int64).max] = np.iinfo(np.int64).min        # no row -> NaT
        p["label_fu"] = pd.to_datetime(fu.view("datetime64[ns]"), utc=True)
    return p


def feature_cols(p: pd.DataFrame) -> list[str]:
    return [c for c in p.columns if c not in META[:2] and c not in LABELS and c != "label_fu"]


def training_panel(store: Store, train_end: pd.Timestamp) -> pd.DataFrame:
    """All hourly origins before `train_end`, labelled as known at `train_end`, unlabelled rows dropped."""
    origins = pd.date_range(store.start.ceil("h") + pd.Timedelta(days=1), train_end - pd.Timedelta(hours=1), freq="h")
    p = build_panel(store, origins, label_cutoff=train_end)
    p = p.dropna(subset=["y_prem", "y_dir"]).reset_index(drop=True)
    assert p["delivery"].max() < train_end, "training label after the training cut-off"
    return p
