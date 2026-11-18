"""Rule checks: data availability, training cut-off, submission consistency.

    .venv/bin/python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from imbalance import config as C
from imbalance.data import Store, load_raw
from imbalance.features import build_panel, feature_cols, training_panel
from imbalance.models import OUT_COLS
from imbalance.submission import sanitize


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def store(raw):
    return Store(raw)


def test_no_future_data_in_features(raw, store):
    """Features at origin t are identical when every row published after t is deleted."""
    rng = np.random.default_rng(0)
    hours = pd.date_range(store.start.ceil("h") + pd.Timedelta(days=2), store.end.floor("h") - pd.Timedelta(hours=12), freq="h")
    for o in hours[rng.choice(len(hours), 8, replace=False)]:
        truncated = Store(raw.loc[raw["first_usable_origin_utc"] <= o])
        a, b = build_panel(store, [o]), build_panel(truncated, [o])
        cols = feature_cols(a)
        pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False, obj=f"features at {o}")


@pytest.mark.parametrize("train_end", [C.TRAIN_END_PRE_VALIDATION, C.TRAIN_END_FULL])
def test_training_labels_before_cutoff(store, train_end):
    """No training label is a delivery at or after the cut-off, and none was published after it."""
    tr = training_panel(store, train_end)
    assert tr["delivery"].max() < train_end
    assert tr["origin"].max() < train_end
    assert tr[["y_imb", "y_spot", "y_dir"]].notna().all().all()


def test_validation_experiment_never_sees_validation_month(store):
    exp = C.EXPERIMENTS["validation"]
    tr = training_panel(store, exp.train_end)
    assert tr["delivery"].max() < exp.first_origin
    assert C.EXPERIMENTS["test_pre_validation"].train_end == exp.train_end
    assert C.EDA_END <= exp.first_origin


def test_sanitize_enforces_schema_rules():
    rng = np.random.default_rng(1)
    n = 1000
    raw = pd.DataFrame({c: rng.normal(0, 1, n) for c in OUT_COLS})
    raw[["p_up", "p_down", "p_none", "p_spike_high", "p_spike_low"]] = rng.uniform(-0.2, 1.2, (n, 5))
    p = sanitize(raw)
    assert (p["p10"] <= p["p50"]).all() and (p["p50"] <= p["p90"]).all()
    assert np.allclose(p[["p_up", "p_down", "p_none"]].sum(1), 1, atol=1e-9)
    assert (p[["p_spike_high", "p_spike_low"]].sum(1) <= 1 + 1e-9).all()
    assert ((p[OUT_COLS[3:]] >= 0) & (p[OUT_COLS[3:]] <= 1)).all().all()
