"""Data loading and the leak-safe lookup that every feature and label goes through.

`Store.asof(series, timestamps, origin)` returns NaN for any row whose
first_usable_origin_utc is later than `origin`. Features and labels are built
only through it, so the data availability rule (README section 4) holds by
construction. tests/test_leakage.py checks this against truncated data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

IMB = "fi_imbalance_price"
SPOT = "fi_spot_price"
DIR = "fi_mfrr_direction"


def _ns(t) -> np.ndarray:
    """UTC timestamps (scalar or array-like) as int64 nanoseconds."""
    if isinstance(t, pd.Timestamp):
        return np.int64(t.tz_convert("UTC").as_unit("ns").value)
    return pd.DatetimeIndex(t).tz_convert("UTC").as_unit("ns").asi8


def read_long(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"dataset_id": str})
    for col in ("timestamp_utc", "first_usable_origin_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


SPOT_FILE = "spot_entsoe.csv"     # ENTSO-E day-ahead price FI (build_dataset.py, source entsoe_spot)


def load_spot(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """ENTSO-E day-ahead price FI for [start, end]: data_dir/spot_entsoe.csv, else the extended-history copy
    (data_ext/). Clipped to the inputs' window so the store's span does not change. The Nord Pool file of the
    hackathon is never read (its licence ended with the event)."""
    for path in (data_dir / SPOT_FILE, C.EXT_DATA_DIR / SPOT_FILE):
        if path.is_file():
            spot = read_long(path)
            return spot[(spot["timestamp_utc"] >= start) & (spot["timestamp_utc"] <= end)]
    raise FileNotFoundError(f"{SPOT_FILE} not found in {data_dir} or {C.EXT_DATA_DIR}: build it with "
                            "build_dataset.py --sources ...,entsoe_spot (REPRODUCE.md)")


REGIME_HOURLY = "regime_hourly_price"      # 1 while Fingrid's imbalance price was hourly (before 19 Mar 2025)


def _variant_raw(v: C.Variant) -> pd.DataFrame:
    """Store rows for an extended-history variant (config.VARIANTS).

    The starter package rows are kept unchanged from 1 Jun 2026, so a variant differs from A only by the
    added history: older rows come from data_ext/ (same series), from one day before the variant's history
    start (lag features of the first origins). With label "v2" the imbalance price before 1 Jun 2026 is the
    re-priced series (data_ext/imbalance_v2.parquet), under the same series_id, so features and labels use
    one consistent series. first_usable_origin_utc is always that of Fingrid 319.
    """
    pkg = load_raw(C.DATA_DIR, variant=False)
    pkg_start = pkg["timestamp_utc"].min()
    ext = read_long(C.EXT_DATA_DIR / "inputs.parquet")
    ext = ext[ext["series_id"].isin(set(pkg["series_id"]) - {SPOT})]
    ext = ext[(ext["timestamp_utc"] >= v.history_start - pd.Timedelta(days=1)) & (ext["timestamp_utc"] < pkg_start)]
    spot = read_long(C.EXT_DATA_DIR / SPOT_FILE)
    spot = spot[(spot["timestamp_utc"] >= ext["timestamp_utc"].min()) & (spot["timestamp_utc"] < pkg_start)]
    parts = [ext, spot, pkg]
    if v.label == "v2":
        v2 = pd.read_parquet(C.EXT_DATA_DIR / "imbalance_v2.parquet")
        v2["timestamp_utc"] = pd.to_datetime(v2["timestamp_utc"], utc=True)
        v2 = v2.set_index("timestamp_utc")["value"]
        old = ext["series_id"] == IMB
        ext.loc[old, "value"] = v2.reindex(ext.loc[old, "timestamp_utc"]).to_numpy()
        assert ext.loc[old, "value"].notna().all(), "re-priced series does not cover the variant's history"
    if v.regime_feature:
        ts = pd.date_range(ext["timestamp_utc"].min(), pkg["timestamp_utc"].max(), freq="15min")
        parts.append(pd.DataFrame({"series_id": REGIME_HOURLY, "source": "calendar", "dataset_id": "",
                                   "timestamp_utc": ts, "value": (ts < C.QH_PRICE_START).astype(float), "unit": "0/1",
                                   "first_usable_origin_utc": C.EXT_START}))     # known in advance
    return pd.concat(parts, ignore_index=True)


def load_raw(data_dir: Path | None = None, variant: bool = True) -> pd.DataFrame:
    """inputs (parquet, csv fallback) + ENTSO-E day-ahead spot price, long format.
    With TE_VARIANT set to an extended-history variant: that variant's rows (_variant_raw)."""
    if variant and data_dir is None and C.VARIANT is not None and C.VARIANT.history_start is not None:
        return _variant_raw(C.VARIANT)
    data_dir = data_dir or C.DATA_DIR
    try:
        inputs = read_long(data_dir / "inputs.parquet")
    except (ImportError, FileNotFoundError, ValueError):
        inputs = read_long(data_dir / "inputs.csv")
    inputs = inputs[inputs["series_id"] != SPOT]
    spot = load_spot(data_dir, inputs["timestamp_utc"].min(), inputs["timestamp_utc"].max())
    return pd.concat([inputs, spot], ignore_index=True)


class Store:
    """Per-series values and first-usable times, indexed by timestamp_utc."""

    def __init__(self, raw: pd.DataFrame):
        self.idx: dict[str, pd.DatetimeIndex] = {}
        self.val: dict[str, np.ndarray] = {}
        self.fu: dict[str, np.ndarray] = {}          # int64 ns UTC
        for sid, g in raw.groupby("series_id"):
            g = g.sort_values("timestamp_utc")
            self.idx[sid] = pd.DatetimeIndex(g["timestamp_utc"])
            self.val[sid] = g["value"].to_numpy(float)
            self.fu[sid] = _ns(g["first_usable_origin_utc"])
        self.start = min(i[0] for i in self.idx.values())
        self.end = max(i[-1] for i in self.idx.values())

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "Store":
        return cls(load_raw(data_dir))

    def __contains__(self, sid: str) -> bool:
        return sid in self.val

    def asof(self, sid: str, ts, origin) -> np.ndarray:
        """Value of `sid` at timestamps `ts` as known at `origin`; NaN where not yet usable.

        origin: one timestamp, or an array of the same length as ts.
        """
        ts = pd.DatetimeIndex(ts)
        out = np.full(len(ts), np.nan)
        if sid not in self.val:
            return out
        pos = self.idx[sid].get_indexer(ts)
        ok = pos >= 0
        org = np.broadcast_to(_ns(origin), (len(ts),))
        usable = np.zeros(len(ts), dtype=bool)
        usable[ok] = self.fu[sid][pos[ok]] <= org[ok]
        out[usable] = self.val[sid][pos[usable]]
        return out

    def first_usable(self, sid: str, ts) -> np.ndarray:
        """first_usable_origin_utc (int64 ns) of `sid` at timestamps `ts`; int64 max where there is no row."""
        pos = self.idx[sid].get_indexer(pd.DatetimeIndex(ts))
        out = np.full(len(pos), np.iinfo(np.int64).max)
        out[pos >= 0] = self.fu[sid][pos[pos >= 0]]
        return out

    def window(self, sid: str, origins: pd.DatetimeIndex, lags: int) -> np.ndarray:
        """Matrix (origins x lags): value at origin - k*15 min for k = 1..lags, as known at the origin."""
        return np.column_stack([self.asof(sid, origins - k * C.Q, origins) for k in range(1, lags + 1)])
