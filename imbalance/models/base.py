"""Common model interface.

A Forecaster is fitted once on a training panel (labels known at the cut-off) and
can then forecast any set of origins: a backtest period, or a single origin for
the dashboard/API. predict() returns one row per panel row with OUT_COLS and,
optionally, `top_drivers`.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..data import IMB, SPOT, Store

OUT_COLS = ["p10", "p50", "p90", "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low"]


@dataclass
class Context:
    store: Store
    train_end: pd.Timestamp


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame, ctx: Context) -> "Forecaster": ...

    @abstractmethod
    def predict(self, panel: pd.DataFrame, ctx: Context) -> pd.DataFrame: ...

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "Forecaster":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ----------------------------------------------------------------------------- shared helpers
def premium_windows(store: Store, origins: pd.DatetimeIndex, length: int) -> np.ndarray:
    """Premium (imbalance - spot) for the `length` quarter-hours before each origin, as known then.

    Columns run oldest -> newest; the newest ones are NaN (not yet published).
    """
    k = np.arange(length, 0, -1)
    out = np.empty((len(origins), length))
    for i, o in enumerate(origins):
        ts = o - k * C.Q
        out[i] = store.asof(IMB, ts, o) - store.asof(SPOT, ts, o)
    return out


def probs_from_quantiles(levels: np.ndarray, qval: np.ndarray, tau: float = 1.0) -> dict:
    """Spike and direction probabilities from premium quantiles (rows x levels); tails extended linearly."""
    mid = len(levels) // 2
    q_lo, q_mid, q_hi = qval[:, 0], qval[:, mid], qval[:, -1]
    vals = np.column_stack([q_lo - (q_mid - q_lo) - 1.0, qval, q_hi + (q_hi - q_mid) + 1.0])
    vals = np.maximum.accumulate(vals, axis=1) + np.arange(vals.shape[1]) * 1e-6
    lev = np.r_[0.0, levels, 1.0]
    cdf = lambda x: np.array([np.interp(x, v, lev) for v in vals])
    p_up, p_down = 1 - cdf(tau), cdf(-tau)
    return dict(p_spike_high=1 - cdf(C.SPIKE_MARGIN), p_spike_low=cdf(-C.SPIKE_MARGIN),
                p_up=p_up, p_down=p_down, p_none=np.clip(1 - p_up - p_down, 0, 1))
