"""Final blend: per-step-bucket ensemble weights + calibration on the most recent block.

Why (measured on the pre-validation folds, see the evaluation printed by `evaluate`):
* In the first hour the last published price (the baseline) is the best p50, later the
  models are. One weight vector for all 40 steps cannot express that, so weights are
  fitted separately for the step buckets 1-4, 5-12, 13-24 and 25-40.
* Volatility trends over time (coverage of the raw ranges fell 84% -> 66% from July to
  early September while spikes rose 1% -> 27%). A calibration fitted on the whole history
  averages calm and stormy weeks; one fitted on the most recent block follows the regime.
  Quantiles are rescaled around p50 (p10' = p50 - a*(p50 - p10), p90' = p50 + b*(p90 - p50))
  with a, b per step bucket chosen so that 10% of recent outcomes fall below / above.
  (Recalibrating the probabilities on the recent block was tested too and made direction
  and spike scores worse, so it is off by default.)

Honest evaluation: `evaluate` fits everything on earlier data only and scores later data
(forward chaining on the folds, then the validation month with weights and calibration
fitted on pre-validation data only).

    python -m imbalance blend
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config as C
from . import cv
from . import submission as S
from .data import Store
from .metrics import all_metrics
from .models import OUT_COLS
from .stack import STACK_DIR, bucket, col_loss, fit_weights, fold_ids, labels

MEMBERS = ["baseline", "clim", "sarimax", "timesfm", "xgb", "lgbm", "qrf", "catboost"]
if C.VARIANT is not None:          # extended-history experiments: fast members first (config.FAST_MEMBERS)
    MEMBERS = list(C.FAST_MEMBERS)
QCOLS = ["p10", "p50", "p90"]
NB = 4
METRICS = ["mae_p50", "calibration_pinball_mean", "coverage_p10_p90", "brier_spike_mean", "brier_direction"]


# ----------------------------------------------------------------------------- blend
@dataclass
class Blend:
    members: list
    calibrate: bool = True              # rescale p10/p90 on the most recent block
    calibrate_probs: bool = False       # isotonic on recent block: hurt direction/spikes in tests, so off
    weights: dict = field(default_factory=dict)       # (column, bucket) -> weights
    scale: dict = field(default_factory=dict)         # (side, bucket) -> factor
    iso: dict = field(default_factory=dict)           # prob column -> isotonic

    def _combine(self, preds: dict, b: np.ndarray) -> pd.DataFrame:
        idx = preds[self.members[0]].index
        out = pd.DataFrame(index=idx, columns=OUT_COLS, dtype=float)
        for k in range(NB):
            m = b == k
            if not m.any():
                continue
            for col in QCOLS + ["p_spike_high", "p_spike_low"]:
                w = self.weights[(col, k)]
                out.loc[m, col] = sum(wi * preds[n][col].to_numpy()[m] for wi, n in zip(w, self.members))
            w = self.weights[("dir", k)]
            d = sum(wi * preds[n][["p_up", "p_down", "p_none"]].to_numpy()[m] for wi, n in zip(w, self.members))
            out.loc[m, ["p_up", "p_down", "p_none"]] = d
        return out

    def fit_weights(self, preds: dict, y: pd.DataFrame) -> "Blend":
        b = bucket(y["step"])
        tgt = y[["up", "down", "none"]].to_numpy()
        for k in range(NB):
            m = b == k
            yk = y[m]
            for col in QCOLS + ["p_spike_high", "p_spike_low"]:
                self.weights[(col, k)] = fit_weights([preds[n][col].to_numpy()[m] for n in self.members],
                                                     lambda p, c=col, yy=yk: col_loss(c, p, yy))
            self.weights[("dir", k)] = fit_weights([preds[n][["p_up", "p_down", "p_none"]].to_numpy()[m] for n in self.members],
                                                   lambda p, t=tgt[m]: ((p - t) ** 2).sum(1))
        return self

    def fit_calibration(self, preds: dict, y: pd.DataFrame) -> "Blend":
        """Fit on the most recent block only (pass just those rows)."""
        raw = self._combine(preds, bucket(y["step"]))
        b = bucket(y["step"])
        yi = y["y_imb"].to_numpy()
        p10, p50, p90 = (raw[c].to_numpy(float) for c in QCOLS)
        for k in range(NB):
            m = b == k
            lo = (p50[m] - yi[m]) / np.maximum(p50[m] - p10[m], 1e-6)
            hi = (yi[m] - p50[m]) / np.maximum(p90[m] - p50[m], 1e-6)
            self.scale[("lo", k)] = float(np.clip(np.quantile(lo, 0.9), 0.5, 4.0))
            self.scale[("hi", k)] = float(np.clip(np.quantile(hi, 0.9), 0.5, 4.0))
        for col, out in (("p_spike_high", "hi"), ("p_spike_low", "lo"), ("p_up", "up")):
            self.iso[col] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                raw[col].to_numpy(float), y[out].to_numpy())
        return self

    def predict(self, preds: dict, step) -> pd.DataFrame:
        b = bucket(step)
        out = self._combine(preds, b)
        if self.calibrate and self.scale:
            a = np.array([self.scale[("lo", k)] for k in range(NB)])[b]
            c = np.array([self.scale[("hi", k)] for k in range(NB)])[b]
            p50 = out["p50"].to_numpy(float)
            out["p10"] = p50 - a * (p50 - out["p10"].to_numpy(float))
            out["p90"] = p50 + c * (out["p90"].to_numpy(float) - p50)
        if self.calibrate_probs and self.iso:
            for col in ("p_spike_high", "p_spike_low"):
                out[col] = self.iso[col].predict(out[col].to_numpy(float))
            up = self.iso["p_up"].predict(out["p_up"].to_numpy(float))
            none = out["p_none"].to_numpy(float)
            out["p_up"], out["p_down"] = up * (1 - none), (1 - up) * (1 - none)
        return S.sanitize(out.astype(float))


# ----------------------------------------------------------------------------- data
def load_oof(store: Store, which: str = "pre_validation"):
    u = cv.universe(store, which)
    y = labels(pd.concat([f.test for f in cv.folds(u, which)]))
    preds = {m: pd.read_parquet(STACK_DIR / f"oof_{which}_{m}.parquet").loc[y.index][OUT_COLS] for m in MEMBERS}
    return preds, y, fold_ids(y, which)


def load_period(experiment: str, with_labels: bool):
    """Member forecasts written by `combine` for an experiment (runs/<exp>/tuned_<m>/), aligned by row."""
    base = C.RUNS_DIR / experiment
    key = ["forecast_origin_utc", "delivery_start_utc", "step"]
    frames = {m: pd.read_csv(base / f"tuned_{m}" / "forecast_backtest.csv") for m in MEMBERS}
    ref = frames[MEMBERS[-1]][key + ["data_cutoff_utc", "top_drivers"]]
    preds = {m: f.set_index(key).loc[ref.set_index(key).index][OUT_COLS].reset_index(drop=True) for m, f in frames.items()}
    y = None
    if with_labels:
        act = pd.read_csv(C.DATA_DIR / "actuals_validation.csv")
        y = ref[["delivery_start_utc", "step"]].merge(act, on="delivery_start_utc", how="left")
        y = y.rename(columns={"imbalance_price": "y_imb", "spot_price": "y_spot"})
        y["y_dir"] = y["direction"].map({"up": 1, "down": -1, "none": 0})
        y["hi"] = (y["y_imb"] > y["y_spot"] + C.SPIKE_MARGIN).astype(float)
        y["lo"] = (y["y_imb"] < y["y_spot"] - C.SPIKE_MARGIN).astype(float)
        y["up"], y["down"], y["none"] = (y["y_dir"] == 1) * 1.0, (y["y_dir"] == -1) * 1.0, (y["y_dir"] == 0) * 1.0
    return preds, ref, y


def score(pred: pd.DataFrame, y: pd.DataFrame) -> dict:
    ok = y["y_imb"].notna().to_numpy()
    return all_metrics(pred[ok], y["y_imb"][ok], y["y_spot"][ok], y["y_dir"][ok])


# ----------------------------------------------------------------------------- run
def _write(experiment: str, ref: pd.DataFrame, pred: pd.DataFrame, tag: str):
    out = C.RUNS_DIR / experiment / f"combined_{tag}"
    df = pd.concat([ref.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    out.mkdir(parents=True, exist_ok=True)
    cols = pd.read_csv(C.ROOT / "sample_submission.csv", nrows=0).columns.tolist()
    df[cols].to_csv(out / "forecast_backtest.csv", index=False, float_format="%.4f", lineterminator="\n")
    return out / "forecast_backtest.csv"


def run(tag: str = "v3", log=print) -> None:
    """Validation + test_pre_validation: fitted on pre-validation data only.
    test_full: weights also use the validation month, range calibration uses the validation month,
    with only labels published by the official training cut-off (4 Oct 21:00 UTC)."""
    store = Store.load()
    P, y, fold = load_oof(store)
    recent = fold == fold.max()
    pre = Blend(MEMBERS).fit_weights(P, y).fit_calibration({k: v[recent] for k, v in P.items()}, y[recent])

    V, vref, yv = load_period("validation", with_labels=True)
    path = _write("validation", vref, pre.predict(V, vref["step"]), tag)
    ok, rep = S.validate(path, C.RUNS_DIR / "validation" / "templates" / "forecast_origins.csv")
    log(f"validation: {'valid' if ok else rep}")
    s = S.score(path, C.DATA_DIR / "actuals_validation.csv", path.parent / "score")
    log("validation official metrics: " + ", ".join(f"{k} {s[k]:.4f}" for k in METRICS))

    T, tref, _ = load_period("test_pre_validation", with_labels=False)
    path = _write("test_pre_validation", tref, pre.predict(T, tref["step"]), tag)
    log(f"test_pre_validation: {'valid' if S.validate(path, C.ROOT / 'forecast_origins.csv')[0] else 'INVALID'}")

    # test_full: add the validation month (only labels usable at the official cut-off)
    act = pd.read_csv(C.DATA_DIR / "actuals_validation.csv", parse_dates=["first_usable_origin_utc"])
    usable = act.loc[act["first_usable_origin_utc"] <= C.TRAIN_END_FULL, "delivery_start_utc"]
    keep = yv["delivery_start_utc"].isin(usable) & yv["y_imb"].notna()
    keep &= pd.to_datetime(vref["forecast_origin_utc"], utc=True) < C.TRAIN_END_FULL
    Vk = {k: v[keep.to_numpy()].reset_index(drop=True) for k, v in V.items()}
    yk = yv[keep].reset_index(drop=True)
    allP = {k: pd.concat([P[k].reset_index(drop=True), Vk[k]], ignore_index=True) for k in MEMBERS}
    ally = pd.concat([y[["step", "y_imb", "y_spot", "y_dir", "hi", "lo", "up", "down", "none"]].reset_index(drop=True),
                      yk[["step", "y_imb", "y_spot", "y_dir", "hi", "lo", "up", "down", "none"]]], ignore_index=True)
    full = Blend(MEMBERS).fit_weights(allP, ally).fit_calibration(Vk, yk)
    F, fref, _ = load_period("test_full", with_labels=False)
    path = _write("test_full", fref, full.predict(F, fref["step"]), tag)
    log(f"test_full: {'valid' if S.validate(path, C.ROOT / 'forecast_origins.csv')[0] else 'INVALID'} "
        f"(validation-month labels used for calibration: {len(yk):,}, latest delivery {yk['delivery_start_utc'].max()})")
    for name, b in (("pre-validation blend", pre), ("test_full blend", full)):
        log(f"{name} range scale (lo, hi) per bucket: " + str({k: (round(b.scale[('lo', k)], 2), round(b.scale[('hi', k)], 2)) for k in range(NB)}))


# ----------------------------------------------------------------------------- alarm levels
def choose_alarm_levels(min_caught: dict | None = None, log=print) -> dict:
    """Alarm level per spike type, chosen on the pre-validation folds only.

    Each fold's blended probabilities come from weights fitted on the other folds (no row is
    scored by weights that saw it). Rule: the highest level that still catches at least
    `min_caught` of the spikes (our target), which gives the fewest false alarms for that catch rate.
    Writes runs/tuning/alarm_levels.json.
    """
    import json
    min_caught = min_caught or {"high": 0.50, "low": 0.80}
    P, y, fold = load_oof(Store.load())
    parts = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        b = Blend(MEMBERS, calibrate=False).fit_weights({m: v[tr] for m, v in P.items()}, y[tr])
        parts.append(b.predict({m: v[te] for m, v in P.items()}, y["step"][te]))
    pred = pd.concat(parts).loc[y.index]
    levels = {}
    for kind, col, out in (("high", "p_spike_high", "hi"), ("low", "p_spike_low", "lo")):
        p, o = pred[col].to_numpy(), y[out].to_numpy() > 0
        best = None
        for t in np.round(np.arange(0.02, 0.61, 0.01), 2):
            a = p > t
            caught, real = (a & o).sum() / max(o.sum(), 1), (a & o).sum() / max(a.sum(), 1)
            if caught >= min_caught[kind]:
                best = (float(t), float(caught), float(real))
        levels[kind] = best[0]
        log(f"{kind}: alarm level {best[0]:.0%} -> on the folds caught {best[1]:.0%}, alarms real {best[2]:.0%}")
    path = C.RUNS_DIR / "tuning" / "alarm_levels.json"
    path.write_text(json.dumps(levels, indent=2))
    return levels
