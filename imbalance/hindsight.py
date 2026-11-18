"""Hindsight on the test period (5 Oct - 1 Nov 2026) for the frozen v3 submission.

Used only after the freeze (SUBMISSION_FREEZE.md, tag v3-frozen); it changes nothing in the
submission. It checks the forecast file against the frozen checksum, builds actual values for the
test period from the starter package data (the same Fingrid and Nord Pool series the organisers
use; their own actuals file may differ slightly) and produces the organisers' baseline for the test
period with their notebook in organiser mode.

    python -m imbalance hindsight            # official metrics + our targets, runs/hindsight/
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config as C
from .data import DIR, IMB, SPOT, load_raw

FROZEN_SHA = {"test_full": "57bd8de6792be79522fbbb6c6b27743e1cb8546f613c0012730c9cb63f1940ce",
              "test_pre_validation": "2fce4f56fef853d262eb489e2653017e026edf99387cf6f2c225084b19e68942"}
OUT = C.RUNS_DIR / "hindsight"
NOTE = ("Hindsight: computed after the submission was frozen (tag v3-frozen, 7 Nov 2026 13:50 UTC); "
        "it does not change the submission. Actual values rebuilt from the starter package data.")


def frozen_file(name: str = "test_full") -> Path:
    """The submission file, checked against the checksum recorded at the freeze."""
    path = C.RUNS_DIR / name / "combined_v3" / "forecast_backtest.csv"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != FROZEN_SHA[name]:
        raise SystemExit(f"{path} is not the frozen submission (sha256 {got}); rebuild with scripts/reproduce.sh")
    return path


def actuals_test() -> Path:
    sub = pd.read_csv(frozen_file())
    d = pd.to_datetime(sub["delivery_start_utc"].unique(), utc=True)
    w = load_raw().pivot(index="timestamp_utc", columns="series_id", values="value")
    a = pd.DataFrame({"delivery_start_utc": d.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "imbalance_price": w[IMB].reindex(d).to_numpy(), "spot_price": w[SPOT].reindex(d).to_numpy(),
                      "direction": w[DIR].reindex(d).map({1.0: "up", -1.0: "down", 0.0: "none"}).to_numpy()})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "actuals_test_ours.csv"
    a.to_csv(path, index=False)
    return path


def baseline_test() -> Path:
    """The organisers' baseline for the test period (their notebook, organiser mode, folder outside the package)."""
    org = C.ROOT.parent / f"{C.ROOT.name}-organizer"
    env = {**os.environ, "SINCEAI_ORGANIZER": "1", "SINCEAI_ORGANIZER_DIR": str(org)}
    exe = Path(sys.executable).with_name("jupyter")
    subprocess.run([str(exe), "nbconvert", "--to", "notebook", "--execute", "baseline.ipynb",
                    "--output-dir", str(OUT / "nb")], cwd=C.ROOT, env=env, check=True, capture_output=True)
    return org / "baseline_test.csv"


def main() -> pd.DataFrame:
    from . import submission as S
    from .targets import measure
    act, base = actuals_test(), baseline_test()
    rows = {"baseline (organiser notebook)": S.score(base, act, OUT / "score_baseline")}
    for name in FROZEN_SHA:
        rows[f"v3 {name}"] = S.score(frozen_file(name), act, OUT / f"score_{name}")
    cols = ["mae_p50", "calibration_pinball_mean", "coverage_p10_p90", "brier_spike_mean", "brier_direction"]
    table = pd.DataFrame(rows).T[cols].astype(float)
    table.to_csv(OUT / "hindsight_summary.csv")
    t = measure(pd.read_csv(frozen_file()), pd.read_csv(act), float(table.iloc[0]["mae_p50"]))
    pd.Series(t).to_csv(OUT / "targets_test_full.csv", header=["value"])
    print(NOTE)
    print(table.round(4).to_string())
    print(pd.Series(t).round(4).to_string())
    return table
