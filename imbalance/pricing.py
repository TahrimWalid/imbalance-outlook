"""Fingrid's imbalance price rules as pure, vectorised functions.

Two rules (Fingrid, "Imbalance price in Finland", versions 29.8.2024 and 1.6.2026):

old rule (12 Jun 2024 - 31 May 2026)
    dominating direction up    max(VWA aFRR up,   mFRR price up)
    dominating direction down  min(VWA aFRR down, mFRR price down)
new rule (from 1 Jun 2026)
    if Finland activated mFRR in the dominating direction:
        (VWA aFRR * aFRR satisfied demand + mFRR price * mFRR energy) / (aFRR satisfied demand + mFRR energy)
        all in the dominating direction ("weighted" case)
    otherwise the old rule ("marginal" case)
both rules, no dominating direction: the day-ahead price ("day_ahead" case)

The dominating direction is an input (Fingrid dataset 369), never derived from Finnish volumes: it is
set by mFRR activations in the whole uncongested area. A missing component (no aFRR activation in
that direction, so VWA is empty) is left out of max/min and has zero weight in the weighted price.
Volumes must be energies in the same unit; only their ratio matters.
Prices are rounded to 0.01 EUR/MWh, half away from zero (see `round_cents`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UP, NONE, DOWN = 1, 0, -1
CASES = ("day_ahead", "marginal", "weighted", "missing")


def round_cents(x) -> np.ndarray:
    """Round to 0.01, half away from zero. The small epsilon absorbs binary representation error
    (e.g. 1.005 stored as 1.00499999...)."""
    x = np.asarray(x, float)
    return np.sign(x) * np.floor(np.abs(x) * 100 + 0.5 + 1e-9) / 100


def _arr(x, n: int) -> np.ndarray:
    a = np.asarray(x, float) if x is not None else np.full(n, np.nan)
    return np.broadcast_to(a, (n,)).astype(float)


def _in_direction(direction: np.ndarray, up, down) -> np.ndarray:
    """Pick the up or down variant of a quantity by the dominating direction (NaN when none)."""
    return np.where(direction == UP, up, np.where(direction == DOWN, down, np.nan))


def _marginal(direction: np.ndarray, vwa: np.ndarray, mfrr: np.ndarray) -> np.ndarray:
    """max (up) or min (down) of the components that exist; NaN when neither exists."""
    both = np.column_stack([vwa, mfrr])
    with np.errstate(all="ignore"):
        hi = np.where(np.isnan(both).all(1), np.nan, np.nanmax(np.where(np.isnan(both), -np.inf, both), 1))
        lo = np.where(np.isnan(both).all(1), np.nan, np.nanmin(np.where(np.isnan(both), np.inf, both), 1))
    return np.where(direction == UP, hi, np.where(direction == DOWN, lo, np.nan))


def price(direction, vwa_up, vwa_down, mfrr_price_up, mfrr_price_down, day_ahead,
          afrr_vol_up=None, afrr_vol_down=None, mfrr_vol_up=None, mfrr_vol_down=None,
          rule: str = "new", rounded: bool = True) -> pd.DataFrame:
    """Imbalance price per quarter-hour under `rule` ("old" or "new").

    direction       -1 / 0 / 1 (Fingrid 369)
    vwa_*           VWA aFRR price per direction (Fingrid 347/348), NaN without aFRR activation
    mfrr_price_*    mFRR marginal price per direction (Fingrid 244/106)
    day_ahead       day-ahead price (used when there is no dominating direction)
    afrr_vol_*      Finland's satisfied aFRR demand per direction (energy)        new rule only
    mfrr_vol_*      mFRR energy activated in Finland per direction (energy)      new rule only

    Returns columns: price, case, vwa, mfrr_price, afrr_vol, mfrr_vol (the dominating-direction
    components; afrr_vol/mfrr_vol are NaN under the old rule).
    """
    if rule not in ("old", "new"):
        raise ValueError(f"unknown rule {rule!r}")
    d = np.asarray(direction, float)
    n = len(d)
    vwa = _in_direction(d, _arr(vwa_up, n), _arr(vwa_down, n))
    mp = _in_direction(d, _arr(mfrr_price_up, n), _arr(mfrr_price_down, n))
    da = _arr(day_ahead, n)
    marginal = _marginal(d, vwa, mp)

    case = np.where(d == NONE, "day_ahead", "marginal").astype(object)
    out = np.where(d == NONE, da, marginal)
    av = mv = np.full(n, np.nan)
    if rule == "new":
        av = _in_direction(d, _arr(afrr_vol_up, n), _arr(afrr_vol_down, n))
        mv = _in_direction(d, _arr(mfrr_vol_up, n), _arr(mfrr_vol_down, n))
        weighted = (d != NONE) & (np.nan_to_num(mv) > 0) & ~np.isnan(mp)
        # a missing aFRR component (no activation in that direction) has zero weight
        a_w = np.where(np.isnan(vwa) | np.isnan(av), 0.0, av)
        a_p = np.nan_to_num(vwa)
        with np.errstate(all="ignore"):
            wp = (a_p * a_w + mp * mv) / (a_w + mv)
        out = np.where(weighted, wp, out)
        case = np.where(weighted, "weighted", case)
    case = np.where(np.isnan(out), "missing", case)
    if rounded:
        out = round_cents(out)
    return pd.DataFrame({"price": out, "case": case, "vwa": vwa, "mfrr_price": mp, "afrr_vol": av, "mfrr_vol": mv})


def bounds(direction, vwa_up, vwa_down, mfrr_price_up, mfrr_price_down) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) of the two component prices in the dominating direction. A new-rule weighted price
    always lies in this range, so high - low bounds the error of re-pricing a quarter-hour whose
    volumes are uncertain."""
    d = np.asarray(direction, float)
    n = len(d)
    vwa = _in_direction(d, _arr(vwa_up, n), _arr(vwa_down, n))
    mp = _in_direction(d, _arr(mfrr_price_up, n), _arr(mfrr_price_down, n))
    with np.errstate(all="ignore"):
        lo = np.where(np.isnan(vwa), mp, np.where(np.isnan(mp), vwa, np.minimum(vwa, mp)))
        hi = np.where(np.isnan(vwa), mp, np.where(np.isnan(mp), vwa, np.maximum(vwa, mp)))
    return lo, hi
