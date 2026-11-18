"""Submission formatting and the organiser's validate/score scripts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .models import OUT_COLS


def sanitize(pred: pd.DataFrame) -> pd.DataFrame:
    """Enforce the schema's consistency rules: ordered quantiles, valid probabilities."""
    p = pred.copy()
    q = np.sort(p[["p10", "p50", "p90"]].to_numpy(float), axis=1)
    p[["p10", "p50", "p90"]] = q
    d = np.clip(p[["p_up", "p_down", "p_none"]].to_numpy(float), 1e-4, 1)
    p[["p_up", "p_down", "p_none"]] = d / d.sum(1, keepdims=True)
    s = np.clip(p[["p_spike_high", "p_spike_low"]].to_numpy(float), 0, 1)
    tot = s.sum(1, keepdims=True)
    p[["p_spike_high", "p_spike_low"]] = np.where(tot > 1, s / tot, s)
    if not np.isfinite(p[OUT_COLS].to_numpy(float)).all():
        raise ValueError("forecast contains NaN or infinite values")
    return p


def to_frame(panel: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Forecast rows in submission column names (timestamps still as datetimes)."""
    df = pd.concat([panel[["origin", "delivery", "step"]], pred], axis=1).rename(
        columns={"origin": "forecast_origin_utc", "delivery": "delivery_start_utc"})
    # newest data used: the newest published imbalance price (start of its quarter-hour)
    df["data_cutoff_utc"] = panel["origin"] - pd.to_timedelta(panel["imb__staleness_min"].fillna(0), unit="min")
    if "top_drivers" not in df:
        df["top_drivers"] = ""
    return df


def write(df: pd.DataFrame, path: Path, template: Path) -> None:
    cols = pd.read_csv(template, nrows=0).columns.tolist()
    out = df.copy()
    for c in ("forecast_origin_utc", "delivery_start_utc", "data_cutoff_utc"):
        out[c] = pd.to_datetime(out[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(path, index=False, float_format="%.4f", lineterminator="\n")


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)], cwd=C.ROOT, capture_output=True, text=True)


def templates(days: tuple[str, str], out_dir: Path) -> Path:
    """Origins + template for a period, via the organiser's make_templates.py."""
    if not (out_dir / "sample_submission.csv").exists():
        r = _run(C.ROOT / "make_templates.py", "--test-start", days[0], "--test-end", days[1], "--output-dir", out_dir)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
    return out_dir


def validate(path: Path, origins: Path) -> tuple[bool, str]:
    r = _run(C.ROOT / "validate_submission.py", "--backtest", path, "--origins", origins)
    return r.returncode == 0, r.stdout + r.stderr


def score(path: Path, actuals: Path, out_dir: Path) -> pd.Series:
    r = _run(C.ROOT / "score.py", "--submission", path, "--actuals", actuals,
             "--spike-margin", str(int(C.SPIKE_MARGIN)), "--output-dir", out_dir)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return pd.read_csv(out_dir / "score_summary.csv").iloc[0]
