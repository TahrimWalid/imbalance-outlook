"""Extended history (data_ext/): availability rule, leakage, and the price reproduction.

Skipped when data_ext/ has not been built (build_dataset.py --catalogue extended, see REPRODUCE.md).
"""
import numpy as np
import pandas as pd
import pytest

import build_dataset as B
from imbalance import config as C
from imbalance import reprice as R
from imbalance.data import Store, read_long
from imbalance.features import build_panel, feature_cols

pytestmark = pytest.mark.skipif(not (C.EXT_DATA_DIR / "inputs.parquet").is_file(), reason="data_ext/ not built")


@pytest.fixture(scope="module")
def raw():
    return read_long(C.EXT_DATA_DIR / "inputs.parquet")


@pytest.fixture(scope="module")
def wide():
    return R.load_wide(C.EXT_DATA_DIR)


def test_every_row_follows_its_availability_rule(raw):
    """first_usable_origin_utc is never earlier than the availability.csv rule allows (period end + lag,
    rounded up to the hour; hourly source rows count from the end of the hour)."""
    rules = B.load_availability(C.ROOT / "availability.csv", None)
    fg = raw[raw["source"] == "fingrid"]
    assert len(fg)
    for ds, g in fg.groupby("dataset_id"):
        rule = B.find_rule(rules, "fingrid", str(ds))
        if rule.availability_type == "lag_after_period_end":
            earliest = (g["timestamp_utc"] + C.Q + pd.Timedelta(minutes=rule.lag_minutes)).dt.ceil("h")
            assert (g["first_usable_origin_utc"] >= earliest).all(), f"dataset {ds} usable too early"
        else:
            assert g["first_usable_origin_utc"].notna().all()


def test_no_future_data_in_features_extended(raw):
    """Features at origin t are identical when every row published after t is deleted (extended store)."""
    store = Store(raw)
    rng = np.random.default_rng(0)
    hours = pd.date_range("2024-08-15T00:00Z", "2026-10-30T00:00Z", freq="h")
    for o in hours[rng.choice(len(hours), 6, replace=False)]:
        truncated = Store(raw.loc[raw["first_usable_origin_utc"] <= o])
        a, b = build_panel(store, [o]), build_panel(truncated, [o])
        cols = feature_cols(a)
        pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False, obj=f"features at {o}")


def test_fingrid_hourly_example_10_3_2025(wide):
    """Fingrid's Q&A (28.8.2025): 10.3.2025 23:00 EET, direction up, price 233 set by the hourly VWA aFRR."""
    h = R.hourly_old_rule(wide["2025-03-10T21:00Z":"2025-03-10T21:45Z"])
    r = h.iloc[0]
    assert r["published"] == 233.0 and r["price"] == 233.0 and r["direction"] == 1


def test_23_october_2025_afrr_removed(wide):
    """23.10.2025 01:00-11:45 Finnish time: Fingrid removed the aFRR component, 319 = mFRR price in most quarters."""
    x = wide["2025-10-22T22:00Z":"2025-10-23T08:45Z"]
    d = x[R.DIR]
    mp = np.where(d == 1, x[R.MP_UP], x[R.MP_DOWN])
    assert (np.abs(x[R.IMB] - mp)[d != 0] <= 0.01).mean() > 0.7


def test_reproduction_gates(wide):
    new = wide[R.NEW_RULE_START:]
    pf = R.price_frame(new, R.CHOSEN, "new")
    w = pf[pf["case"] == "weighted"]
    assert (w["diff"].abs() <= 0.01 + 1e-6).mean() >= 0.98
    old = wide[R.QH_PRICE_START:R.NEW_RULE_START - C.Q]
    po = R.price_frame(old, None, "old")
    po = po[po["direction"].isin([1, -1])]
    assert (po["diff"].abs() <= 0.01 + 1e-6).mean() >= 0.98
