"""Why a spike or an extreme low is forecast: per-row drivers of the spike probabilities.

    python -m imbalance explain validation
    python -m imbalance explain test_full

For every forecast row (origin, step) the three inputs that push the high-spike and the low-spike
probability up the most, from the tuned XGBoost spike models (one of the eight ensemble members),
fitted at the experiment's training cut-off. Contributions are SHAP values in log-odds from
XGBoost's own tree explainer. Inputs are read as published at the forecast hour, exactly as for
the forecast itself. Nothing here changes a forecast: the output is a separate file,
runs/<experiment>/explanations/spike_drivers.csv.

These show what the model reacts to, not proven causes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import submission as S
from .data import Store
from .features import build_panel, feature_cols, training_panel
from .models.gbm import Head, head_target
from .tune import load_best

TOP = 3


def _top_positive(contrib: np.ndarray, cols: list[str], k: int = TOP) -> list[str]:
    """Names of the k inputs with the largest positive contribution per row ('' where none is positive)."""
    order = np.argsort(-contrib, axis=1)[:, :k]
    out = []
    for i, row in enumerate(order):
        out.append(";".join(cols[j] for j in row if contrib[i, j] > 0))
    return out


def run(experiment: str, log=print):
    import xgboost as xgb
    exp = C.EXPERIMENTS[experiment]
    store = Store.load()
    tmpl = S.templates(exp.period_days, C.RUNS_DIR / exp.name / "templates")
    origins = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(tmpl / "forecast_origins.csv")["forecast_origin_utc"], utc=True))
    train = training_panel(store, exp.train_end)
    panel = build_panel(store, origins)
    cols = feature_cols(train)
    params = load_best()["xgb"]
    out = panel[["origin", "delivery", "step"]].rename(columns={"origin": "forecast_origin_utc", "delivery": "delivery_start_utc"})
    for head, name in (("spike_high", "high"), ("spike_low", "low")):
        h = Head("xgb", head, params[head]).fit(train[cols], head_target(train, head))
        contrib = h.est.get_booster().predict(xgb.DMatrix(panel[cols]), pred_contribs=True)[:, :-1]
        out[f"{name}_spike_drivers"] = _top_positive(contrib, cols)
        out[f"xgb_p_spike_{name}"] = np.round(h.predict(panel[cols]), 4)
        log(f"  {experiment}: {head} explained ({len(panel):,} rows)")
    for c in ("forecast_origin_utc", "delivery_start_utc"):
        out[c] = pd.to_datetime(out[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path = C.RUNS_DIR / exp.name / "explanations" / "spike_drivers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")
    log(f"  wrote {path}")
    return path
