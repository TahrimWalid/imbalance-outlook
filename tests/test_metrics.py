"""imbalance.metrics must reproduce score.py exactly (on the baseline validation file)."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from imbalance import config as C
from imbalance.metrics import all_metrics


def test_metrics_match_score_py(tmp_path):
    sub_path = C.ROOT / "baseline_output" / "validation" / "forecast_backtest.csv"
    if not sub_path.exists():
        pytest.skip("run baseline.ipynb first (scripts/reproduce.sh does)")
    r = subprocess.run([sys.executable, str(C.ROOT / "score.py"), "--submission", str(sub_path), "--actuals",
                        str(C.DATA_DIR / "actuals_validation.csv"), "--spike-margin", "50", "--output-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    official = pd.read_csv(tmp_path / "score_summary.csv").iloc[0]

    sub = pd.read_csv(sub_path)
    act = pd.read_csv(C.DATA_DIR / "actuals_validation.csv")
    df = sub.merge(act, on="delivery_start_utc").dropna(subset=["imbalance_price", "spot_price", "direction"])
    y_dir = df["direction"].map({"up": 1, "down": -1, "none": 0})
    ours = all_metrics(df, df["imbalance_price"], df["spot_price"], y_dir)
    for k in ("mae_p50", "calibration_pinball_mean", "coverage_p10_p90", "brier_spike_mean", "brier_direction"):
        assert np.isclose(ours[k], official[k], rtol=1e-9), (k, ours[k], official[k])
