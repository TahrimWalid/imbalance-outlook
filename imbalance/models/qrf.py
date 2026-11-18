"""Quantile regression forest on the premium: one model gives the whole predictive distribution.

p10/p50/p90 are read directly; spike and direction probabilities come from the
distribution (99 quantile levels) evaluated at the thresholds.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..features import feature_cols
from .base import Forecaster, probs_from_quantiles

LEVELS = np.round(np.arange(1, 100) / 100, 2)
DEFAULTS = dict(n_estimators=200, min_samples_leaf=20, max_features=0.5, max_samples_leaf=None)


class QRF(Forecaster):
    name = "qrf"

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULTS, **(params or {})}
        # TE_QRF_MAX_SAMPLES_LEAF: keep at most this many training values per leaf (random subsample). The
        # forest stores the leaf values padded to the largest leaf, which ran out of memory on variant F's
        # 730k rows; a few hundred values per leaf still give stable p10/p50/p90.
        cap = os.environ.get("TE_QRF_MAX_SAMPLES_LEAF")
        if cap:
            self.params["max_samples_leaf"] = int(cap)

    def _X(self, panel):
        return panel[self.cols].fillna(self.medians).fillna(0).to_numpy(np.float32)

    def fit(self, train, ctx):
        from quantile_forest import RandomForestQuantileRegressor
        self.cols = feature_cols(train)
        self.medians = train[self.cols].median()
        self.est = RandomForestQuantileRegressor(n_jobs=4, random_state=0, **self.params)
        self.est.fit(self._X(train), train["y_prem"].to_numpy(float))
        return self

    def predict(self, panel, ctx):
        q = np.sort(self.est.predict(self._X(panel), quantiles=list(LEVELS)), axis=1)
        spot = panel["fi_spot_price__d"].to_numpy()
        i10, i50, i90 = (int(np.where(LEVELS == x)[0][0]) for x in (0.1, 0.5, 0.9))
        return pd.DataFrame(dict(p10=spot + q[:, i10], p50=spot + q[:, i50], p90=spot + q[:, i90],
                                 **probs_from_quantiles(LEVELS, q)), index=panel.index)
