"""Hyperparameter tuning with Optuna on the rolling folds (imbalance.cv).

* Gradient boosting (xgb, lgbm, catboost): one study per output head, each scored on the
  metric that column is graded on (MAE, pinball p10/p90, Brier).
* Single-distribution models (qrf, sarimax, clim, timesfm): one study each, scored on the
  points-weighted loss relative to the baseline (metrics.relative_loss).

The loss is pooled over all fold test rows. The first trial is always the default
configuration, so a tuned model is never worse than its default on the folds.
Studies are stored in runs/tuning/optuna.db, so an interrupted run resumes where it stopped.

    python -m imbalance tune --families clim,sarimax,timesfm,xgb,lgbm,catboost,qrf --trials 50
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import optuna
import pandas as pd

from . import config as C
from . import cv
from .data import Store
from .features import feature_cols
from .metrics import all_metrics, brier, brier_direction, pinball, relative_loss
from .models import Context, make
from .models.gbm import DEFAULTS as GBM_DEFAULTS, HEADS, Head, head_target
from .models.qrf import DEFAULTS as QRF_DEFAULTS
from .submission import sanitize

TUNE_DIR = C.RUNS_DIR / "tuning"
GBM_FAMILIES = ("xgb", "lgbm", "catboost")
HEAD_ORDER = ("p50", "p10", "p90", "spike_high", "spike_low", "dir")


# ----------------------------------------------------------------------------- search spaces
# Memory guard for the quantile forest on the long histories: it stores every training row in its leaves, so
# memory grows with max_samples / min_samples_leaf (a trial with leaf 8 and max_samples 0.97 on 500k rows ran a
# 30 GB machine out of memory). TE_QRF_MIN_LEAF raises the smallest leaf; TE_QRF_SAMPLES_PER_LEAF caps
# max_samples at that many times the leaf size. Unset: the original search space.
QRF_MIN_LEAF = int(os.environ.get("TE_QRF_MIN_LEAF", "1"))
QRF_SAMPLES_PER_LEAF = float(os.environ.get("TE_QRF_SAMPLES_PER_LEAF", "inf"))


def qrf_max_samples(leaf: int) -> float:
    return float(min(1.0, max(0.3, QRF_SAMPLES_PER_LEAF * leaf)))


def space(family: str, trial: optuna.Trial) -> dict:
    s = trial.suggest_int, trial.suggest_float, trial.suggest_categorical
    i, f, c = s
    if family == "xgb":
        return dict(n_estimators=i("n_estimators", 100, 1500, log=True), learning_rate=f("learning_rate", 0.005, 0.2, log=True),
                    max_depth=i("max_depth", 3, 10), min_child_weight=f("min_child_weight", 1, 500, log=True),
                    subsample=f("subsample", 0.5, 1.0), colsample_bytree=f("colsample_bytree", 0.3, 1.0),
                    reg_lambda=f("reg_lambda", 1e-3, 100, log=True), reg_alpha=f("reg_alpha", 1e-3, 100, log=True))
    if family == "lgbm":
        return dict(n_estimators=i("n_estimators", 100, 1500, log=True), learning_rate=f("learning_rate", 0.005, 0.2, log=True),
                    num_leaves=i("num_leaves", 8, 256, log=True), min_child_samples=i("min_child_samples", 10, 2000, log=True),
                    subsample=f("subsample", 0.5, 1.0), colsample_bytree=f("colsample_bytree", 0.3, 1.0),
                    reg_lambda=f("reg_lambda", 1e-3, 100, log=True), reg_alpha=f("reg_alpha", 1e-3, 100, log=True))
    if family == "catboost":
        return dict(iterations=i("iterations", 200, 1500, log=True), learning_rate=f("learning_rate", 0.01, 0.2, log=True),
                    depth=i("depth", 4, 8), l2_leaf_reg=f("l2_leaf_reg", 0.1, 50, log=True),
                    bootstrap_type="Bernoulli", subsample=f("subsample", 0.5, 1.0), rsm=f("rsm", 0.3, 1.0),
                    random_strength=f("random_strength", 0.0, 5.0))
    if family == "qrf":
        leaf = i("min_samples_leaf", QRF_MIN_LEAF, 300, log=True)
        return dict(n_estimators=200, min_samples_leaf=leaf, max_features=f("max_features", 0.1, 1.0),
                    max_samples=f("max_samples", 0.3, qrf_max_samples(leaf)))
    if family == "sarimax":
        return dict(p=i("p", 0, 4), q=i("q", 0, 3), k_fourier=i("k_fourier", 0, 4),
                    clip=c("clip", [200.0, 500.0, 1000.0, 1e9]), window=c("window", [96, 192, 384]))
    if family == "clim":
        edges = {"4/12/24": (4, 12, 24), "8/20": (8, 20), "2/4/8/16/28": (2, 4, 8, 16, 28), "none": (40,)}
        return dict(bucket_edges=edges[c("bucket_edges", list(edges))],
                    state=c("state", ["dir_last", "prem_sign", "share_up", "none"]),
                    hour_group=c("hour_group", [1, 2, 3, 4, 6]), weekend=c("weekend", [False, True]),
                    min_rows=c("min_rows", [30, 100, 300, 1000]))
    raise ValueError(family)


DEFAULT_TRIAL = {
    "xgb": dict(GBM_DEFAULTS["xgb"], reg_lambda=1.0, reg_alpha=1e-3),
    "lgbm": dict(GBM_DEFAULTS["lgbm"], reg_alpha=1e-3),
    "catboost": dict(GBM_DEFAULTS["catboost"], subsample=0.8, rsm=1.0, random_strength=1.0),
    "qrf": dict(min_samples_leaf=20, max_features=0.5, max_samples=1.0),
    "sarimax": dict(p=2, q=1, k_fourier=0, clip=500.0, window=192),
    "clim": dict(bucket_edges="4/12/24", state="dir_last", hour_group=1, weekend=False, min_rows=100),
}
TIMESFM_GRID = ([dict(context=c, target="premium", covariates=False) for c in (256, 512, 1024, 2048)]
                + [dict(context=c, target="premium", covariates=True, xreg_mode=m)
                   for c in (512, 1024, 2048) for m in ("xreg + timesfm", "timesfm + xreg")]
                + [dict(context=1024, target="price", covariates=False)])


# ----------------------------------------------------------------------------- losses
def head_loss(head: str, pred: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    """Per-row loss of one head (its mean over rows is the graded metric)."""
    kind, arg = HEADS[head]
    if kind == "quantile":
        d = test["y_prem"].to_numpy() - pred
        return np.maximum(arg * d, (arg - 1) * d) * (2 if head == "p50" else 1)   # 2*pinball(0.5) = MAE
    if kind == "binary":
        return (pred - head_target(test, head)) ** 2
    y = test["y_dir"].to_numpy()
    return sum((pred[:, j] - (y == c)) ** 2 for j, c in enumerate((-1, 0, 1)))


class Tuner:
    def __init__(self, store: Store, log=print):
        self.store, self.log = store, log
        self.u = cv.universe(store)
        self.folds = cv.folds(self.u)
        self.test_all = pd.concat([f.test for f in self.folds])
        TUNE_DIR.mkdir(parents=True, exist_ok=True)
        self.storage = f"sqlite:///{TUNE_DIR / 'optuna.db'}"
        base = pd.concat([self._predict_full("baseline", None, f) for f in self.folds])
        self.base_metrics = self._metrics(base)
        self.log(cv.describe(self.folds))
        self.log("baseline on folds: " + ", ".join(f"{k} {v:.4f}" for k, v in self.base_metrics.items()))

    # --- helpers -----------------------------------------------------------------------
    def _metrics(self, pred: pd.DataFrame) -> dict:
        t = self.test_all.loc[pred.index]
        return all_metrics(pred, t["y_imb"], t["y_spot"], t["y_dir"])

    def _predict_full(self, family, params, fold) -> pd.DataFrame:
        ctx = Context(self.store, fold.cutoff)
        return sanitize(make(family, params).fit(fold.train, ctx).predict(fold.test, ctx))

    def _study(self, name: str, n_trials: int, objective, enqueue: list[dict]) -> optuna.Study:
        study = optuna.create_study(study_name=name, storage=self.storage, load_if_exists=True, direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
        if len(study.trials) == 0:
            for p in enqueue:
                study.enqueue_trial(p)
        done = len([t for t in study.trials if t.state.is_finished()])
        remaining = max(0, n_trials - done)

        def cb(st, tr):
            v = f"{tr.value:.4f}" if tr.value is not None else tr.state.name
            self.log(f"  {name} trial {tr.number:3d}: {v}  best {st.best_value:.4f}  ({tr.duration.total_seconds():.0f}s)")
        if remaining:
            study.optimize(objective, n_trials=remaining, callbacks=[cb], catch=(Exception,))
        return study

    # --- GBM heads -----------------------------------------------------------------------
    def tune_gbm_head(self, family: str, head: str, n_trials: int) -> optuna.Study:
        def objective(trial):
            params = space(family, trial)
            tot, n = 0.0, 0
            for f in self.folds:
                cols = feature_cols(f.train)
                h = Head(family, head, params).fit(f.train[cols], head_target(f.train, head))
                loss = head_loss(head, h.predict(f.test[cols]), f.test)
                tot, n = tot + loss.sum(), n + len(loss)
                trial.report(tot / n, f.k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return tot / n
        return self._study(f"{family}:{head}", n_trials, objective, [DEFAULT_TRIAL[family]])

    # --- whole-distribution models -------------------------------------------------------
    def tune_family(self, family: str, n_trials: int) -> optuna.Study:
        def objective(trial):
            if family == "timesfm":
                params = TIMESFM_GRID[trial.suggest_categorical("config", list(range(len(TIMESFM_GRID))))]
            else:
                params = space(family, trial)
            preds = []
            for f in self.folds:
                preds.append(self._predict_full(family, params, f))
                m = self._metrics(pd.concat(preds))
                trial.report(relative_loss(m, self.base_metrics), f.k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            for k, v in m.items():
                trial.set_user_attr(k, v)
            return relative_loss(m, self.base_metrics)
        if family == "timesfm":
            enqueue = [{"config": j} for j in range(len(TIMESFM_GRID))]
            n_trials = len(TIMESFM_GRID)
        else:
            enqueue = [DEFAULT_TRIAL[family]]
            if family == "qrf":
                d = enqueue[0]
                enqueue = [dict(d, min_samples_leaf=max(d["min_samples_leaf"], QRF_MIN_LEAF),
                                max_samples=min(d["max_samples"], qrf_max_samples(max(d["min_samples_leaf"], QRF_MIN_LEAF))))]
        return self._study(family, n_trials, objective, enqueue)

    def run(self, families: list[str], n_trials: int) -> dict:
        path = TUNE_DIR / "best_params.json"            # tuning updates the working copy, never config/
        best = json.loads(path.read_text()) if path.exists() else {}
        for fam in families:
            t0 = time.time()
            if fam in GBM_FAMILIES:
                best[fam] = {}
                for head in HEAD_ORDER:
                    st = self.tune_gbm_head(fam, head, n_trials)
                    best[fam][head] = gbm_params(fam, st.best_params)
                    self.log(f"{fam}:{head} best {st.best_value:.4f} (default {st.trials[0].value:.4f})")
            else:
                st = self.tune_family(fam, n_trials)
                best[fam] = family_params(fam, st.best_params)
                self.log(f"{fam} best relative loss {st.best_value:.4f} (default {st.trials[0].value:.4f}) "
                         f"params {best[fam]}")
            save_best(best)
            self.log(f"== {fam} done in {(time.time() - t0) / 60:.0f} min")
        return best


# ----------------------------------------------------------------------------- best params
def gbm_params(family: str, p: dict) -> dict:
    return {**p, "bootstrap_type": "Bernoulli"} if family == "catboost" else dict(p)


def family_params(family: str, p: dict) -> dict:
    if family == "timesfm":
        return TIMESFM_GRID[p["config"]]
    if family == "clim":
        edges = {"4/12/24": (4, 12, 24), "8/20": (8, 20), "2/4/8/16/28": (2, 4, 8, 16, 28), "none": (40,)}
        return {**p, "bucket_edges": edges[p["bucket_edges"]]}
    if family == "qrf":
        return {**QRF_DEFAULTS, **p}
    return dict(p)


# versioned copy used by the final pipeline; an extended-history variant has its own (config/best_params_<v>.json)
FROZEN = C.ROOT / "config" / ("best_params.json" if C.VARIANT is None else f"best_params_{C.VARIANT.name}.json")


def load_best() -> dict:
    """Tuned settings: config/best_params.json (frozen, in git) if present, else the latest tuning run."""
    for path in (FROZEN, TUNE_DIR / "best_params.json"):
        if path.exists():
            return json.loads(path.read_text())
    return {}


def save_best(best: dict) -> None:
    (TUNE_DIR / "best_params.json").write_text(json.dumps(best, indent=2, default=list))
