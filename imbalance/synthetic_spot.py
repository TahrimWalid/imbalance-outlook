"""A spot-like price built only from Fingrid's public data, for the shareable dashboard (no day-ahead price in it)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPOT_HISTORY_DAYS = 14


def synthetic_spot(times: pd.DatetimeIndex, imbalance: pd.Series, direction: pd.Series,
                   days: int = SPOT_HISTORY_DAYS) -> np.ndarray:
    """A spot-like price built only from Fingrid's public data (imbalance price, dataset 319, and mFRR
    direction, dataset 369), never from Nord Pool.

    The real spot lies between the up-regulation price (above it) and the down-regulation price (below it). So
    for each quarter-hour: the midpoint of the median up-regulation imbalance price and the median
    down-regulation one at the same quarter-hour on the previous `days` days (the plain median where one side
    is missing), smoothed over 2 hours. It looks back a day or more only, so like the real spot it is known in
    advance and the live replay never shows future prices through it. On the validation month it has the real
    spot's level and daily shape, but not its day-to-day or quarter-hour values."""
    lag = lambda s: np.column_stack([s.sort_index().reindex(times - pd.Timedelta(days=k)).to_numpy(float)
                                     for k in range(1, days + 1)])
    price, d = lag(imbalance), lag(direction)
    with np.errstate(all="ignore"):
        med = np.nanmedian(price, axis=1)
        up = np.nanmedian(np.where(d == 1, price, np.nan), axis=1)
        down = np.nanmedian(np.where(d == -1, price, np.nan), axis=1)
    mid = np.where(np.isnan(up) | np.isnan(down), med, (up + down) / 2)
    return pd.Series(mid).rolling(8, center=True, min_periods=1).mean().to_numpy()
