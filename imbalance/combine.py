"""Build a combined forecast from the tuned models and score it on an experiment.

    python -m imbalance combine --tag v1 --models baseline,clim,sarimax,timesfm,xgb,lgbm

1. Tuned settings come from runs/tuning/best_params.json (models without tuned settings use defaults).
2. Steps 1-3 (stack.build) on the pre-validation folds: OOF predictions, calibration, per-column
   ensemble/selection. Nothing from the validation month is used.
3. Every member is fitted at the experiment's training cut-off and forecasts the period; the
   blend from step 2 combines them.
4. Writes runs/<experiment>/combined_<tag>/forecast_backtest.csv (+ each tuned member), validates
   with the official validator, scores with the official score.py (validation only) and measures
   our own targets.
"""
from __future__ import annotations

import time

import pandas as pd

from . import config as C
from . import stack
from . import submission as S
from .data import Store
from .features import build_panel, training_panel
from .models import OUT_COLS, Context, make
from .tune import load_best

METRICS = ["mae_p50", "calibration_pinball_mean", "coverage_p10_p90", "brier_spike_mean", "brier_direction"]


def run(models: list[str], tag: str, experiments=("validation",), log=print) -> None:
    t0 = time.time()
    store = Store.load()
    best = load_best()
    params = {m: best.get(m) for m in models}
    log(f"members: {', '.join(f'{m} ({'tuned' if params[m] else 'default'})' for m in models)}")

    result = stack.build(store, params, "pre_validation", log=log)
    blend = result["blend"]
    log(f"stack built ({(time.time() - t0) / 60:.0f} min)")

    for name in experiments:
        exp = C.EXPERIMENTS[name]
        out = C.RUNS_DIR / exp.name
        tmpl = S.templates(exp.period_days, out / "templates")
        origins = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(tmpl / "forecast_origins.csv")["forecast_origin_utc"], utc=True))
        train = training_panel(store, exp.train_end)
        panel = build_panel(store, origins)
        ctx = Context(store, exp.train_end)

        member_preds, rows = {}, {}
        for m in models:
            pred = S.sanitize(make(m, params[m]).fit(train, ctx).predict(panel, ctx))
            member_preds[m] = pred
            rows[f"{m} (tuned)" if params[m] else m] = _write_and_score(exp, tmpl, panel, pred, out / f"tuned_{m}", log)
            log(f"  {name}: {m} done ({(time.time() - t0) / 60:.0f} min)")
        combined = blend.predict({m: p[OUT_COLS] for m, p in member_preds.items()}, panel["step"])
        if "xgb" in member_preds and "top_drivers" in member_preds["xgb"]:
            combined["top_drivers"] = member_preds["xgb"]["top_drivers"]
        rows[f"COMBINED {tag}"] = _write_and_score(exp, tmpl, panel, combined, out / f"combined_{tag}", log)

        if exp.actuals is not None:
            table = pd.DataFrame({k: v for k, v in rows.items() if v is not None}).T[METRICS]
            table.to_csv(out / f"combined_{tag}_summary.csv")
            log(f"\n{name}: official score.py metrics\n" + table.round(4).to_string())
            from .targets import report
            log(f"\nOur targets, COMBINED {tag}:\n" + report(out / f"combined_{tag}" / "forecast_backtest.csv").to_string())
    log(f"done in {(time.time() - t0) / 60:.0f} min")


def _write_and_score(exp, tmpl, panel, pred, mdir, log):
    S.write(S.to_frame(panel, pred), mdir / "forecast_backtest.csv", tmpl / "sample_submission.csv")
    ok, report = S.validate(mdir / "forecast_backtest.csv", tmpl / "forecast_origins.csv")
    if not ok:
        log(report)
        return None
    if exp.actuals is None:
        return None
    return S.score(mdir / "forecast_backtest.csv", exp.actuals, mdir / "score")
