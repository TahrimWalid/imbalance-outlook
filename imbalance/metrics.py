"""The official metrics (same formulas as score.py), computed in memory for tuning.

tests/test_metrics.py checks that these match score.py exactly on a real file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

WEIGHTS = {"mae_p50": 30, "calibration_pinball_mean": 20, "brier_spike_mean": 20, "brier_direction": 10}


def pinball(y, yhat, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(yhat, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def brier(p, outcome) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(outcome, float)) ** 2))


def brier_direction(p_up, p_down, p_none, y_dir) -> float:
    y = np.asarray(y_dir)
    return float(np.mean((np.asarray(p_up) - (y == 1)) ** 2 + (np.asarray(p_down) - (y == -1)) ** 2
                         + (np.asarray(p_none) - (y == 0)) ** 2))


def spike_outcomes(y_imb, y_spot, margin: float = C.SPIKE_MARGIN):
    y_imb, y_spot = np.asarray(y_imb, float), np.asarray(y_spot, float)
    return y_imb > y_spot + margin, y_imb < y_spot - margin


def all_metrics(pred: pd.DataFrame, y_imb, y_spot, y_dir) -> dict:
    """pred: columns p10..p_spike_low; labels as arrays (actual imbalance, spot, direction -1/0/1)."""
    y = np.asarray(y_imb, float)
    hi, lo = spike_outcomes(y, y_spot)
    m = {"mae_p50": float(np.mean(np.abs(y - pred["p50"].to_numpy()))),
         "pinball_p10": pinball(y, pred["p10"], 0.1), "pinball_p90": pinball(y, pred["p90"], 0.9),
         "coverage_p10_p90": float(np.mean((y >= pred["p10"].to_numpy()) & (y <= pred["p90"].to_numpy()))),
         "brier_spike_high": brier(pred["p_spike_high"], hi), "brier_spike_low": brier(pred["p_spike_low"], lo),
         "brier_direction": brier_direction(pred["p_up"], pred["p_down"], pred["p_none"], y_dir)}
    m["calibration_pinball_mean"] = (m["pinball_p10"] + m["pinball_p90"]) / 2
    m["brier_spike_mean"] = (m["brier_spike_high"] + m["brier_spike_low"]) / 2
    return m


def relative_loss(m: dict, base: dict) -> float:
    """Points-weighted loss relative to the baseline: sum_w w * metric / baseline_metric (lower is better).

    Used to tune single models that produce every column at once.
    """
    return float(sum(w * m[k] / base[k] for k, w in WEIGHTS.items()) / sum(WEIGHTS.values()))
