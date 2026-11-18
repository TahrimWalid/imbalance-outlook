"""Rolling-origin cross-validation (used for tuning, calibration and ensembling).

Two universes:
  "pre_validation"  labels published by 6 Sep 21:00Z; folds test 12 Jul - 6 Sep.
                    Used for tuning and for everything that is scored on the validation month.
  "full"            labels published by 4 Oct 21:00Z (the official training cut-off); adds one
                    fold that tests the validation month. Used only for the test_full variant.

Fold k: train = rows whose label was published by cutoff_k (exactly what a model fitted at
cutoff_k could know); test = origins in [cutoff_k, cutoff_{k+1}).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as C
from .data import Store
from .features import build_panel

TUNE_END = C.TRAIN_END_PRE_VALIDATION
CUTOFFS = {
    "pre_validation": pd.DatetimeIndex(["2026-07-12T21:00:00Z", "2026-07-26T21:00:00Z", "2026-08-09T21:00:00Z",
                                        "2026-08-23T21:00:00Z", C.TRAIN_END_PRE_VALIDATION.isoformat()]),
}
CUTOFFS["full"] = CUTOFFS["pre_validation"].append(pd.DatetimeIndex([C.TRAIN_END_FULL.isoformat()]))
END = {"pre_validation": C.TRAIN_END_PRE_VALIDATION, "full": C.TRAIN_END_FULL}


@dataclass
class Fold:
    k: int
    cutoff: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame


def universe(store: Store, which: str = "pre_validation") -> pd.DataFrame:
    """Every hourly origin before the end, labelled as known at the end (unlabelled rows dropped)."""
    end = END[which]
    origins = pd.date_range(store.start.ceil("h") + pd.Timedelta(days=1), end - pd.Timedelta(hours=1), freq="h")
    p = build_panel(store, origins, label_cutoff=end)
    p = p.dropna(subset=["y_imb", "y_spot", "y_dir"]).reset_index(drop=True)
    assert p["label_fu"].max() <= end and p["delivery"].max() < end
    return p


def folds(u: pd.DataFrame, which: str = "pre_validation") -> list[Fold]:
    cuts = CUTOFFS[which]
    out = []
    for k in range(len(cuts) - 1):
        cut, nxt = cuts[k], cuts[k + 1]
        train = u.loc[u["label_fu"] <= cut]
        test = u.loc[(u["origin"] >= cut) & (u["origin"] < nxt)]
        assert train["delivery"].max() < cut
        out.append(Fold(k, cut, train, test))
    return out


def describe(fs: list[Fold]) -> str:
    return "\n".join(f"fold {f.k}: train labels <= {f.cutoff:%m-%d %H:%M}Z ({len(f.train):,} rows), "
                     f"test origins {f.test['origin'].min():%m-%d}..{f.test['origin'].max():%m-%d} ({len(f.test):,} rows)"
                     for f in fs)
