"""Actual values for scoring, rebuilt from public data (no Nord Pool file).

    python -m imbalance actuals validation        -> data/actuals_validation.csv

imbalance price (Fingrid 319) and direction (369) from the inputs of `data_dir`, spot price from the
ENTSO-E day-ahead price (spot_entsoe.csv). Same format and first_usable_origin_utc rule as the
organisers' file (build_dataset.build_actuals).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import build_dataset as B

from . import config as C
from .data import DIR, IMB, SPOT, load_raw

PERIODS = {"validation": B.VALIDATION_PERIOD, "test": B.TEST_PERIOD}


def build(period: str, data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or C.DATA_DIR
    raw = load_raw(data_dir)
    a, b = PERIODS[period]
    index = B.quarter_index(a, b + B.QUARTER)
    part = {sid: raw[raw["series_id"] == sid] for sid in (IMB, SPOT, DIR)}
    return B.build_actuals((a, b), index, part[IMB], part[SPOT], part[DIR])


def write(period: str, data_dir: Path | None = None, out: Path | None = None) -> Path:
    data_dir = data_dir or C.DATA_DIR
    out = out or data_dir / f"actuals_{period}.csv"
    B.write_actuals(build(period, data_dir), out)
    return out


def write_package_spot(data_dir: Path | None = None, source: Path | None = None) -> Path:
    """data/spot_entsoe.csv: the ENTSO-E day-ahead price clipped to the window of data/inputs (for the
    organisers' notebook, which reads its spot file from data/)."""
    from .data import SPOT_FILE, read_long
    data_dir = data_dir or C.DATA_DIR
    source = source or C.EXT_DATA_DIR / SPOT_FILE
    inputs = pd.read_parquet(data_dir / "inputs.parquet", columns=["timestamp_utc"])
    lo, hi = pd.to_datetime(inputs["timestamp_utc"], utc=True).agg(["min", "max"])
    spot = read_long(source)
    spot = spot[(spot["timestamp_utc"] >= lo) & (spot["timestamp_utc"] <= hi)].copy()
    for col in ("timestamp_utc", "first_usable_origin_utc"):
        spot[col] = spot[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = data_dir / SPOT_FILE
    spot.to_csv(out, index=False, float_format="%.10g", lineterminator="\n")
    return out
