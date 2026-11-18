"""Experiment runner: fit at the cut-off, forecast the period, write, validate, score.

Output layout:
  runs/<experiment>/templates/                 origins + template (make_templates.py)
  runs/<experiment>/<model>/model.pkl          fitted model
  runs/<experiment>/<model>/forecast_backtest.csv
  runs/<experiment>/<model>/score/             score.py output (validation only)
  runs/<experiment>/summary.csv                one row per model
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from . import submission as S
from .data import Store
from .features import build_panel, feature_cols, training_panel
from .models import Context, make

METRICS = ["mae_p50", "calibration_pinball_mean", "coverage_p10_p90", "brier_spike_mean", "brier_direction"]
WEIGHTS = {"mae_p50": 30, "calibration_pinball_mean": 20, "brier_spike_mean": 20, "brier_direction": 10}


def approx_points(table: pd.DataFrame, baseline: str = "baseline") -> pd.DataFrame:
    """Automatic points as score.py computes them, with the best model in `table` as 'best'."""
    pts = pd.DataFrame(index=table.index)
    for m, w in WEIGHTS.items():
        b, best = table.loc[baseline, m], table[m].min()
        pts[f"pts_{m}"] = 0.0 if b - best <= 1e-6 * abs(b) else w * np.clip((b - table[m]) / (b - best), 0, 1)
    pts["pts_total"] = pts.sum(axis=1)
    return pts


def run(experiment: str, models: list[str], store: Store | None = None, log=print) -> pd.DataFrame:
    exp = C.EXPERIMENTS[experiment]
    assert exp.train_end <= exp.first_origin, "training cut-off must not be after the first forecast origin"
    out = C.RUNS_DIR / exp.name
    t0 = time.time()
    store = store or Store.load()
    tmpl = S.templates(exp.period_days, out / "templates")
    origins = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(tmpl / "forecast_origins.csv")["forecast_origin_utc"], utc=True))

    train = training_panel(store, exp.train_end)
    panel = build_panel(store, origins)           # features only: no labels for the forecast period
    log(f"[{exp.name}] train: {len(train):,} rows, labels up to {train['delivery'].max():%Y-%m-%d %H:%M}Z "
        f"(cut-off {exp.train_end:%Y-%m-%d %H:%M}Z); forecast: {len(origins)} origins; "
        f"{len(feature_cols(train))} features ({time.time() - t0:.0f}s)")
    ctx = Context(store=store, train_end=exp.train_end)

    rows = {}
    for name in models:
        t1, mdir = time.time(), out / name
        try:
            model = make(name).fit(train, ctx)
            model.save(mdir / "model.pkl")
            pred = S.sanitize(model.predict(panel, ctx))
        except Exception as e:                      # one broken model must not stop the run
            log(f"  {name:9s} FAILED: {type(e).__name__}: {e}")
            continue
        S.write(S.to_frame(panel, pred), mdir / "forecast_backtest.csv", tmpl / "sample_submission.csv")
        ok, report = S.validate(mdir / "forecast_backtest.csv", tmpl / "forecast_origins.csv")
        if not ok:
            log(report)
        if hasattr(model, "importance"):
            model.importance().to_csv(mdir / "importance.csv")
        if exp.actuals is not None and ok:
            rows[name] = S.score(mdir / "forecast_backtest.csv", exp.actuals, mdir / "score")
        log(f"  {name:9s} {'valid' if ok else 'INVALID'} ({time.time() - t1:.0f}s)")

    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).T[METRICS].astype(float)
    if "baseline" in table.index:
        table = table.join(approx_points(table))
    table.to_csv(out / "summary.csv")
    return table
