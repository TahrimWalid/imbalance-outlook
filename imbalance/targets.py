"""Our own minimum targets, measured on a forecast file against actual values.

The organisers' points are relative to the other teams, which we cannot know, so we set
fixed minimum targets of our own. They were chosen on 7 Nov 2026 from the first (untuned)
practice-exam result; they are goals, not a known maximum.

    python -m imbalance targets runs/validation/<model>/forecast_backtest.csv

Alarm levels for spike detection are fixed parameters (ALARM). They must be chosen on the
tuning folds, never on the period being measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# metric -> (target, "min" = at least / "max" = at most, untuned practice-exam value)
TARGETS = {
    "coverage_p10_p90":          ((0.78, 0.82), "range", 0.60),
    "normal_within_25":          (0.80, "min", 0.75),
    "median_abs_error":          (16.0, "max", 19.2),
    "mae_improvement_vs_base":   (0.40, "min", 0.35),
    "low_spikes_caught":         (0.80, "min", 0.78),
    "low_alarms_real":           (0.40, "min", 0.37),
    "high_spikes_caught":        (0.50, "min", 0.39),
    "high_alarms_real":          (0.30, "min", 0.26),
    "direction_right":           (0.72, "min", 0.67),
}
# config/alarm_levels.json (frozen, in git) or the latest `python -m imbalance alarms` run
_ALARM_FILES = [C.ROOT / "config" / ("alarm_levels.json" if C.VARIANT is None else f"alarm_levels_{C.VARIANT.name}.json"),
                C.RUNS_DIR / "tuning" / "alarm_levels.json"]
ALARM = next((__import__("json").loads(f.read_text()) for f in _ALARM_FILES if f.exists()),
             {"high": 0.10, "low": 0.15})


def measure(forecast: pd.DataFrame, actuals: pd.DataFrame, baseline_mae: float, alarm: dict = ALARM) -> dict:
    d = forecast.merge(actuals, on="delivery_start_utc").dropna(subset=["imbalance_price"])
    y, sp = d["imbalance_price"], d["spot_price"]
    err = (y - d["p50"]).abs()
    hi, lo = y > sp + C.SPIKE_MARGIN, y < sp - C.SPIKE_MARGIN
    normal = ~(hi | lo)
    a_hi, a_lo = d["p_spike_high"] > alarm["high"], d["p_spike_low"] > alarm["low"]
    guess = np.select([(d["p_up"] >= d["p_down"]) & (d["p_up"] >= d["p_none"]), d["p_down"] >= d["p_none"]],
                      ["up", "down"], "none")
    return {
        "coverage_p10_p90": float(((y >= d["p10"]) & (y <= d["p90"])).mean()),
        "normal_within_25": float((err[normal] <= 25).mean()),
        "median_abs_error": float(err.median()),
        "mae_improvement_vs_base": float(1 - err.mean() / baseline_mae),
        "low_spikes_caught": float((a_lo & lo).sum() / max(lo.sum(), 1)),
        "low_alarms_real": float((a_lo & lo).sum() / max(a_lo.sum(), 1)),
        "high_spikes_caught": float((a_hi & hi).sum() / max(hi.sum(), 1)),
        "high_alarms_real": float((a_hi & hi).sum() / max(a_hi.sum(), 1)),
        "direction_right": float((guess == d["direction"]).mean()),
    }


def report(path: Path, actuals: Path | None = None, baseline_mae: float | None = None) -> pd.DataFrame:
    actuals = actuals or C.DATA_DIR / "actuals_validation.csv"
    if baseline_mae is None:     # the official baseline on the same period
        base = pd.read_csv(C.ROOT / "baseline_output" / "validation" / "score" / "score_summary.csv").iloc[0]
        baseline_mae = float(base["mae_p50"])
    m = measure(pd.read_csv(path), pd.read_csv(actuals), baseline_mae)
    rows = []
    for k, (target, kind, before) in TARGETS.items():
        v = m[k]
        ok = (target[0] <= v <= target[1]) if kind == "range" else (v >= target if kind == "min" else v <= target)
        rows.append({"metric": k, "untuned": before, "now": round(v, 3), "target": target, "met": "yes" if ok else "NO"})
    return pd.DataFrame(rows).set_index("metric")
