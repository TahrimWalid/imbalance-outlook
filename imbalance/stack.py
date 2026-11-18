"""Steps after tuning: out-of-fold predictions, calibration, per-column ensembling.

1. OOF: every tuned model is fitted on each fold's training rows and predicts that fold's
   test rows, so each fold-test row gets an honest prediction from every model.
2. Calibration (fitted on OOF):
     p10 / p50 / p90   additive shift per step bucket = the pinball-optimal shift of the
                       OOF residuals (a split-conformal correction)
     probabilities     isotonic regression of outcome on predicted probability
3. Ensemble (fitted on OOF): for each output column, non-negative weights summing to 1
   across models, minimising that column's graded metric.

Every choice is checked with leave-one-fold-out (LOFO): calibrators and weights are fitted
on the other folds and scored on the held-out fold, so the reported gains are out of sample.
A calibrator is only kept for a column if LOFO says it helps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from . import config as C
from . import cv
from .data import Store
from .metrics import all_metrics
from .models import OUT_COLS, Context, make
from .submission import sanitize

STACK_DIR = C.RUNS_DIR / "stack"
Q_COLS = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
P_COLS = ["p_spike_high", "p_spike_low", "p_up", "p_down", "p_none"]
BUCKETS = (4, 12, 24)                      # step buckets: 1-4, 5-12, 13-24, 25-40


# ----------------------------------------------------------------------------- labels / losses
def labels(test: pd.DataFrame) -> pd.DataFrame:
    y = test[["origin", "step", "y_imb", "y_spot", "y_dir"]].copy()
    y["hi"] = (y["y_imb"] > y["y_spot"] + C.SPIKE_MARGIN).astype(float)
    y["lo"] = (y["y_imb"] < y["y_spot"] - C.SPIKE_MARGIN).astype(float)
    y["up"], y["down"], y["none"] = (y["y_dir"] == 1) * 1.0, (y["y_dir"] == -1) * 1.0, (y["y_dir"] == 0) * 1.0
    return y


OUTCOME = {"p_spike_high": "hi", "p_spike_low": "lo", "p_up": "up", "p_down": "down", "p_none": "none"}


def col_loss(col: str, pred: np.ndarray, y: pd.DataFrame) -> np.ndarray:
    """Per-row loss for one column (mean = the column's contribution to its graded metric)."""
    if col in Q_COLS:
        q, d = Q_COLS[col], y["y_imb"].to_numpy() - pred
        return np.maximum(q * d, (q - 1) * d) * (2 if col == "p50" else 1)
    return (pred - y[OUTCOME[col]].to_numpy()) ** 2


def bucket(step) -> np.ndarray:
    return np.searchsorted(np.asarray(BUCKETS), np.asarray(step), side="left")


# ----------------------------------------------------------------------------- 1. OOF
def oof_predictions(store: Store, models: dict[str, dict | None], which: str = "pre_validation",
                    log=print) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """{model: OOF predictions} and the matching labels, cached in runs/stack/."""
    STACK_DIR.mkdir(parents=True, exist_ok=True)
    u = cv.universe(store, which)
    fs = cv.folds(u, which)
    y = labels(pd.concat([f.test for f in fs]))
    out = {}
    for name, params in models.items():
        path = STACK_DIR / f"oof_{which}_{name}.parquet"
        meta = STACK_DIR / f"oof_{which}_{name}.json"
        if path.exists() and meta.exists() and json.loads(meta.read_text()) == json.loads(json.dumps(params, default=list)):
            out[name] = pd.read_parquet(path)
            continue
        parts = []
        for f in fs:
            ctx = Context(store, f.cutoff)
            p = sanitize(make(name, params).fit(f.train, ctx).predict(f.test, ctx))
            parts.append(p[OUT_COLS])
            log(f"  oof {which} {name} fold {f.k} done")
        out[name] = pd.concat(parts).loc[y.index]
        out[name].to_parquet(path)
        meta.write_text(json.dumps(params, default=list))
    return out, y


# ----------------------------------------------------------------------------- 2. calibration
@dataclass
class Calibrator:
    """Per-column corrections fitted on OOF predictions; `use` marks the columns where LOFO showed a gain."""
    shifts: dict = field(default_factory=dict)       # quantile col -> {bucket: shift}
    iso: dict = field(default_factory=dict)          # prob col -> IsotonicRegression
    use: dict = field(default_factory=dict)          # col -> bool

    def fit(self, pred: pd.DataFrame, y: pd.DataFrame) -> "Calibrator":
        b = bucket(y["step"])
        for col, q in Q_COLS.items():
            r = y["y_imb"].to_numpy() - pred[col].to_numpy()
            self.shifts[col] = {int(k): float(np.quantile(r[b == k], q)) for k in np.unique(b)}
        for col in ("p_spike_high", "p_spike_low", "p_up"):
            self.iso[col] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                pred[col].to_numpy(), y[OUTCOME[col]].to_numpy())
        self.use = self.use or {c: True for c in list(Q_COLS) + ["p_spike_high", "p_spike_low", "p_dir"]}
        return self

    def transform(self, pred: pd.DataFrame, step) -> pd.DataFrame:
        p = pred.copy()
        b = bucket(step)
        for col in Q_COLS:
            if self.use.get(col):
                table = np.array([self.shifts[col].get(k, 0.0) for k in range(len(BUCKETS) + 1)])
                p[col] = p[col].to_numpy() + table[b]
        for col in ("p_spike_high", "p_spike_low"):
            if self.use.get(col):
                p[col] = self.iso[col].predict(p[col].to_numpy())
        if self.use.get("p_dir"):
            # calibrate P(up); keep P(none) and split the rest to down
            up = self.iso["p_up"].predict(p["p_up"].to_numpy())
            none = p["p_none"].to_numpy()
            p["p_up"], p["p_down"] = up * (1 - none), (1 - up) * (1 - none)
        return sanitize(p)


def lofo_calibration(pred: pd.DataFrame, y: pd.DataFrame, fold_of: np.ndarray) -> tuple[Calibrator, pd.DataFrame]:
    """Decide per column whether calibration helps out of sample; return a calibrator fitted on all folds."""
    raw = {c: [] for c in Q_COLS} | {c: [] for c in ("p_spike_high", "p_spike_low", "p_dir")}
    cal = {k: [] for k in raw}
    dir_cols = ("p_up", "p_down", "p_none")
    for k in np.unique(fold_of):
        tr, te = fold_of != k, fold_of == k
        before, yt = pred[te], y[te]
        after = Calibrator().fit(pred[tr], y[tr]).transform(before, yt["step"])
        for col in list(Q_COLS) + ["p_spike_high", "p_spike_low"]:
            raw[col].append(col_loss(col, before[col].to_numpy(), yt))
            cal[col].append(col_loss(col, after[col].to_numpy(), yt))
        raw["p_dir"].append(sum(col_loss(c, before[c].to_numpy(), yt) for c in dir_cols))
        cal["p_dir"].append(sum(col_loss(c, after[c].to_numpy(), yt) for c in dir_cols))
    report = pd.DataFrame({"raw": {k: np.concatenate(v).mean() for k, v in raw.items()},
                           "calibrated": {k: np.concatenate(v).mean() for k, v in cal.items()}})
    report["use"] = report["calibrated"] < report["raw"]
    final = Calibrator(use=report["use"].to_dict()).fit(pred, y)
    return final, report


# ----------------------------------------------------------------------------- 3. ensemble
def fit_weights(preds: list[np.ndarray], loss) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise mean loss(sum_i w_i * pred_i)."""
    n = len(preds)
    raw = lambda w: loss(sum(wi * p for wi, p in zip(w, preds))).mean()
    scale = raw(np.full(n, 1 / n)) or 1.0           # losses like Brier ~0.05 are too flat for SLSQP unscaled
    obj = lambda w: raw(w) / scale
    starts = [np.full(n, 1 / n)] + list(np.eye(n))  # uniform + every single model
    best_w, best_v = None, np.inf
    for w0 in starts:
        res = minimize(obj, w0, method="SLSQP", bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 500, "ftol": 1e-12})
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        for cand in (w, w0):
            v = obj(cand)
            if v < best_v:
                best_w, best_v = cand, v
    return best_w


@dataclass
class Ensemble:
    models: list
    weights: dict = field(default_factory=dict)      # column (or "dir") -> weight vector

    def fit(self, preds: dict[str, pd.DataFrame], y: pd.DataFrame) -> "Ensemble":
        for col in list(Q_COLS) + ["p_spike_high", "p_spike_low"]:
            self.weights[col] = fit_weights([preds[m][col].to_numpy() for m in self.models],
                                            lambda p, c=col: col_loss(c, p, y))
        dir_preds = [preds[m][["p_up", "p_down", "p_none"]].to_numpy() for m in self.models]
        tgt = y[["up", "down", "none"]].to_numpy()
        self.weights["dir"] = fit_weights(dir_preds, lambda p: ((p - tgt) ** 2).sum(1))
        return self

    def predict(self, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
        idx = preds[self.models[0]].index
        out = pd.DataFrame(index=idx)
        for col in list(Q_COLS) + ["p_spike_high", "p_spike_low"]:
            out[col] = sum(w * preds[m][col].to_numpy() for w, m in zip(self.weights[col], self.models))
        d = sum(w * preds[m][["p_up", "p_down", "p_none"]].to_numpy() for w, m in zip(self.weights["dir"], self.models))
        out["p_up"], out["p_down"], out["p_none"] = d[:, 0], d[:, 1], d[:, 2]
        return sanitize(out[OUT_COLS])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.weights, index=self.models).round(3)


# ----------------------------------------------------------------------------- evaluation
def fold_ids(y: pd.DataFrame, which: str) -> np.ndarray:
    cuts = cv.CUTOFFS[which]
    return np.searchsorted(cuts.as_unit("ns").asi8, pd.DatetimeIndex(y["origin"]).as_unit("ns").asi8, side="right") - 1


def evaluate(pred: pd.DataFrame, y: pd.DataFrame) -> dict:
    return all_metrics(pred, y["y_imb"], y["y_spot"], y["y_dir"])


def build(store: Store, models: dict[str, dict | None], which: str = "pre_validation", log=print) -> dict:
    """Run steps 1-3 and report every variant out of sample (LOFO). Returns fitted calibrators + ensemble."""
    preds, y = oof_predictions(store, models, which, log)
    fold = fold_ids(y, which)
    rows, calibrators, cal_preds = {}, {}, {}
    for m, p in preds.items():
        rows[f"{m} raw"] = evaluate(p, y)
        calibrators[m], rep = lofo_calibration(p, y, fold)
        log(f"\n{m}: calibration out of sample (LOFO)\n{rep.round(4).to_string()}")
        # LOFO calibrated predictions, for an honest score of "model + calibration"
        parts = []
        for k in np.unique(fold):
            tr, te = fold != k, fold == k
            parts.append(Calibrator(use=calibrators[m].use).fit(p[tr], y[tr]).transform(p[te], y["step"][te]))
        cal_preds[m] = pd.concat(parts).loc[y.index]
        rows[f"{m} calibrated"] = evaluate(cal_preds[m], y)

    names = list(preds)
    lofo_ens = {}
    for label, src in (("raw", preds), ("calibrated", cal_preds)):
        parts = []
        for k in np.unique(fold):
            tr, te = fold != k, fold == k
            e = Ensemble(names).fit({m: src[m][tr] for m in names}, y[tr])
            parts.append(e.predict({m: src[m][te] for m in names}))
        lofo_ens[label] = pd.concat(parts).loc[y.index]
        rows[f"ENSEMBLE of {label}"] = evaluate(lofo_ens[label], y)

    # per column: keep whichever source (one calibrated model, or the ensemble) is best out of sample
    candidates = {**cal_preds, "ENSEMBLE": lofo_ens["calibrated"]}
    choice, losses = {}, {}
    for grp, cols in GROUPS.items():
        losses[grp] = {s: sum(col_loss(c, candidates[s][c].to_numpy(), y) for c in cols).mean() for s in candidates}
        choice[grp] = min(losses[grp], key=losses[grp].get)
    selected = pd.concat([candidates[choice[g]][cols] for g, cols in GROUPS.items()], axis=1)[OUT_COLS]
    rows["SELECTED per column"] = evaluate(sanitize(selected), y)

    blend = Blend(calibrators=calibrators, ensemble=Ensemble(names).fit(cal_preds, y), choice=choice)
    table = pd.DataFrame(rows).T[["mae_p50", "calibration_pinball_mean", "coverage_p10_p90",
                                  "brier_spike_mean", "brier_direction"]]
    log("\nOut-of-sample (LOFO) scores on the folds:\n" + table.round(4).to_string())
    log("\nPer-column LOFO loss by source:\n" + pd.DataFrame(losses).round(4).to_string())
    log(f"\nChosen source per column: {choice}")
    log("\nEnsemble weights (fitted on all folds, calibrated members):\n" + blend.ensemble.table().to_string())
    table.to_csv(STACK_DIR / f"lofo_{which}.csv")
    return {"blend": blend, "table": table, "losses": pd.DataFrame(losses)}


GROUPS = {"p10": ["p10"], "p50": ["p50"], "p90": ["p90"], "p_spike_high": ["p_spike_high"],
          "p_spike_low": ["p_spike_low"], "dir": ["p_up", "p_down", "p_none"]}


@dataclass
class Blend:
    """Final combiner: calibrate each member, ensemble them, then take each column from its chosen source."""
    calibrators: dict
    ensemble: Ensemble
    choice: dict

    def predict(self, member_preds: dict[str, pd.DataFrame], step) -> pd.DataFrame:
        cal = {m: self.calibrators[m].transform(p[OUT_COLS], step) for m, p in member_preds.items()}
        sources = {**cal, "ENSEMBLE": self.ensemble.predict(cal)}
        out = pd.concat([sources[self.choice[g]][cols] for g, cols in GROUPS.items()], axis=1)[OUT_COLS]
        return sanitize(out)
