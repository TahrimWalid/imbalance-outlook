"""Gradient-boosted trees (XGBoost, LightGBM, CatBoost) with one model per output ("head").

Heads and the metric each one is tuned on:
  p10 / p90    quantile regression of premium = imbalance - spot   pinball loss
  p50          quantile regression (median) of the premium          MAE
  dir          3-class classifier down / none / up                  3-class Brier
  spike_high   binary: premium > +50 EUR/MWh                        Brier
  spike_low    binary: premium < -50 EUR/MWh                        Brier
Each head has its own hyperparameters, so each can be tuned separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..features import feature_cols
from .base import Forecaster

HEADS = {"p10": ("quantile", 0.1), "p50": ("quantile", 0.5), "p90": ("quantile", 0.9),
         "dir": ("multiclass", None), "spike_high": ("binary", 1), "spike_low": ("binary", -1)}
DIR_ORDER = (-1, 0, 1)            # class order of the direction head: down, none, up
THREADS = 4

DEFAULTS = {
    "xgb": dict(n_estimators=500, learning_rate=0.03, max_depth=6, min_child_weight=50,
                subsample=0.8, colsample_bytree=0.7),
    "lgbm": dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                 subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0),
    "catboost": dict(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0),
}


def head_target(train: pd.DataFrame, head: str) -> np.ndarray:
    kind, arg = HEADS[head]
    if kind == "quantile":
        return train["y_prem"].to_numpy(float)
    if kind == "multiclass":
        return train["y_dir"].to_numpy(int)
    prem = train["y_prem"].to_numpy(float)
    return (prem > C.SPIKE_MARGIN if arg == 1 else prem < -C.SPIKE_MARGIN).astype(int)


def make_estimator(backend: str, head: str, params: dict):
    kind, arg = HEADS[head]
    if backend == "xgb":
        import xgboost as xgb
        common = dict(tree_method="hist", n_jobs=THREADS, **params)
        if kind == "quantile":
            return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=arg, **common)
        if kind == "multiclass":
            return xgb.XGBClassifier(objective="multi:softprob", **common)
        return xgb.XGBClassifier(objective="binary:logistic", **common)
    if backend == "lgbm":
        import lightgbm as lgb
        common = dict(subsample_freq=1, verbose=-1, n_jobs=THREADS, **params)
        if kind == "quantile":
            return lgb.LGBMRegressor(objective="quantile", alpha=arg, **common)
        return lgb.LGBMClassifier(objective="multiclass" if kind == "multiclass" else "binary", **common)
    if backend == "catboost":
        import catboost as cb
        common = dict(thread_count=THREADS, verbose=0, allow_writing_files=False, **params)
        if kind == "quantile":
            return cb.CatBoostRegressor(loss_function=f"Quantile:alpha={arg}", **common)
        return cb.CatBoostClassifier(loss_function="MultiClass" if kind == "multiclass" else "Logloss", **common)
    raise ValueError(backend)


class Head:
    """One fitted output of a GBM."""

    def __init__(self, backend: str, head: str, params: dict):
        self.backend, self.head, self.params = backend, head, params

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Head":
        kind = HEADS[self.head][0]
        if kind == "multiclass":                      # rare classes can be absent from a training window
            self.classes = np.array([c for c in DIR_ORDER if (y == c).any()])
            y = np.searchsorted(self.classes, y)
        self.est = make_estimator(self.backend, self.head, self.params).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        kind = HEADS[self.head][0]
        if kind == "quantile":
            return np.asarray(self.est.predict(X), float).ravel()
        proba = self.est.predict_proba(X)
        if kind == "binary":
            return proba[:, 1]
        out = np.zeros((len(X), len(DIR_ORDER)))
        for j, c in enumerate(self.classes):
            out[:, DIR_ORDER.index(c)] = proba[:, j]
        return out


class GBM(Forecaster):
    def __init__(self, backend: str = "xgb", params: dict | None = None):
        """params: {head: hyperparameters}; heads without an entry use DEFAULTS[backend]."""
        self.backend = self.name = backend
        self.params = {h: {**DEFAULTS[backend], **(params or {}).get(h, {})} for h in HEADS}

    def fit(self, train, ctx):
        self.cols = feature_cols(train)
        X = train[self.cols]
        self.heads = {h: Head(self.backend, h, self.params[h]).fit(X, head_target(train, h)) for h in HEADS}
        return self

    def predict(self, panel, ctx):
        X = panel[self.cols]
        spot = panel["fi_spot_price__d"].to_numpy()
        out = {q: spot + self.heads[q].predict(X) for q in ("p10", "p50", "p90")}
        pdir = self.heads["dir"].predict(X)
        out.update(p_down=pdir[:, 0], p_none=pdir[:, 1], p_up=pdir[:, 2],
                   p_spike_high=self.heads["spike_high"].predict(X), p_spike_low=self.heads["spike_low"].predict(X))
        res = pd.DataFrame(out, index=panel.index)
        res["top_drivers"] = self.top_drivers(X)
        return res

    # --- explanations ---------------------------------------------------------
    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Per-row SHAP contributions of each feature to the p50 premium (rows x features)."""
        est = self.heads["p50"].est
        if self.backend == "lgbm":
            return est.predict(X, pred_contrib=True)[:, :-1]
        if self.backend == "xgb":
            import xgboost as xgb
            return est.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        import catboost as cb
        return est.get_feature_importance(type="ShapValues", data=cb.Pool(X))[:, :-1]

    def top_drivers(self, X: pd.DataFrame, k: int = 3) -> list[str]:
        top = np.argsort(-np.abs(self.contributions(X)), axis=1)[:, :k]
        return [";".join(self.cols[j] for j in row) for row in top]

    def importance(self) -> pd.Series:
        est = self.heads["p50"].est
        if self.backend == "lgbm":
            imp = est.booster_.feature_importance("gain")
        elif self.backend == "xgb":
            imp = pd.Series(est.get_booster().get_score(importance_type="gain")).reindex(self.cols).fillna(0).to_numpy()
        else:
            imp = est.get_feature_importance()
        return pd.Series(imp, index=self.cols).sort_values(ascending=False)
