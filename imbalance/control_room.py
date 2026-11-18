"""Dashboard with two linked views of the validation month: Overview (forecast against actual over days and
weeks, scores per model, by forecast hour and by day) and Control room (the next 10 hours around "now",
alerts, exposure, replay and a live simulation of any day).

    python -m imbalance control-room              -> runs/dashboard/control_room_B.html        (real spot: own use only)
    python -m imbalance control-room --share      -> runs/dashboard/control_room_B_share.html  (synthetic spot)

Everything shown comes from the forecast files and Fingrid's published prices. The share version contains no
day-ahead price: the spot line is synthetic (synthetic_spot.py) and spikes are carried as flags judged
against the real spot, so every score on the page is the real one.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .synthetic_spot import SPOT_HISTORY_DAYS, synthetic_spot

TEMPLATE = Path(__file__).with_name("web") / "control_room.html"
TZ = "Europe/Helsinki"
FIELDS = [("p10", 10), ("p50", 10), ("p90", 10), ("p_up", 1000), ("p_down", 1000), ("p_spike_high", 1000), ("p_spike_low", 1000)]
MEMBERS = ["xgb", "lgbm", "qrf"]


def model_files(variant: str, share: bool) -> list[tuple[str, str, Path]]:
    vr = C.ROOT / "runs" / "variants"
    out = [("combined_v3", f"Blend ({variant})", vr / variant / "validation" / "combined_v3"),
           *[(m, f"{n} ({variant})", vr / variant / "validation" / f"tuned_{m}") for m, n in
             (("xgb", "XGBoost"), ("lgbm", "LightGBM"), ("qrf", "QRF"))],
           ("A", "Blend, 4 months (A)", vr / "A" / "validation" / "combined_v3"),
           ("hack", "Hackathon submission", C.ROOT / "runs" / "validation" / "combined_v3")]
    if not share:   # the baseline's best guess is close to the spot price itself, so it stays out of the share version
        out.append(("baseline", "Official baseline", C.ROOT / "baseline_output" / "validation"))
    return [(k, lab, d / "forecast_backtest.csv") for k, lab, d in out]


# ----------------------------------------------------------------------------- data
def build_data(variant: str, share: bool) -> dict:
    files = model_files(variant, share)
    first = pd.read_csv(files[0][2], usecols=["forecast_origin_utc"], parse_dates=["forecast_origin_utc"])
    origins = pd.DatetimeIndex(sorted(first["forecast_origin_utc"].unique()))
    d0 = origins[0]                                            # m = 0: first forecast origin (local midnight)
    assert d0.tz_convert(TZ).hour == 0, "the first origin should be a local midnight"
    no = len(origins)
    mins = lambda ts: ((pd.DatetimeIndex(ts) - d0) / pd.Timedelta(minutes=1)).astype(int)

    models, labels = {}, []
    for key, label, path in files:
        f = pd.read_csv(path, parse_dates=["forecast_origin_utc"])
        oi = np.searchsorted(origins, f["forecast_origin_utc"].to_numpy())
        ok = (oi < no) & (origins[np.minimum(oi, no - 1)] == f["forecast_origin_utc"].to_numpy())
        arr = np.zeros((no, 40, len(FIELDS)), dtype=np.int32)
        for j, (col, sc) in enumerate(FIELDS):
            arr[oi[ok], f["step"].to_numpy()[ok] - 1, j] = np.round(f[col].to_numpy(float)[ok] * sc).astype(np.int32)
        models[key] = arr.ravel().tolist()
        labels.append([key, label])

    # quarter-hour grid from one day before the first origin to the end of the last forecast
    q0 = d0 - pd.Timedelta(days=1)
    grid = pd.date_range(q0, origins[-1] + pd.Timedelta(minutes=15 * 39), freq="15min")
    act = pd.read_csv(C.ROOT / "data" / "actuals_validation.csv", parse_dates=["delivery_start_utc"])
    act = act.set_index("delivery_start_utc").reindex(grid)
    price, real_spot = act["imbalance_price"], act["spot_price"]
    direction = act["direction"].map(lambda v: {"up": 1, "down": -1, "none": 0}.get(str(v).strip().lower(), v)).astype(float)
    spk = np.select([price > real_spot + C.SPIKE_MARGIN, price < real_spot - C.SPIKE_MARGIN], [1, -1], 0).astype(float)
    spk[(price.isna() | real_spot.isna()).to_numpy()] = np.nan

    if share:
        inp = pd.read_parquet(C.ROOT / "data" / "inputs.parquet", columns=["series_id", "timestamp_utc", "value"])
        inp = inp[inp["series_id"].isin(["fi_imbalance_price", "fi_mfrr_direction"])]
        wide = inp.pivot_table(index="timestamp_utc", columns="series_id", values="value")
        spot = synthetic_spot(grid, wide["fi_imbalance_price"], wide["fi_mfrr_direction"], days=SPOT_HISTORY_DAYS)
    else:
        spot = real_spot.to_numpy(float)

    r2 = lambda x: [None if not np.isfinite(v) else round(float(v), 2) for v in np.asarray(x, float)]
    ri = lambda x: [None if not np.isfinite(v) else int(v) for v in np.asarray(x, float)]
    loc0 = d0.tz_convert(TZ)
    hi = int(mins([origins[-1]])[0])
    ndays = hi // 1440
    alarm = json.loads((C.ROOT / "config" / f"alarm_levels_{variant}.json").read_text())
    tag = f"VALIDATION MONTH {origins[0].tz_convert(TZ):%-d %b} – {origins[-1].tz_convert(TZ):%-d %b %Y} · REPLAY".upper()
    notice = ("Spot on this page is synthetic: built from Fingrid's public imbalance prices, not the day-ahead price. "
              "Spikes and every accuracy number are judged against the real day-ahead price. "
              if share else "Contains ENTSO-E day-ahead prices: for your own use only, do not share or publish this file. ")
    notice += (f"Imbalance prices and regulation direction: Fingrid open data, CC BY 4.0. Forecasts: own model, variant {variant} "
               "(validation month, never used for training or tuning).")
    head = {}
    for key, path in (("B", C.ROOT / "runs" / "variants" / variant / "validation" / "combined_v3"), ("A", C.ROOT / "runs" / "variants" / "A" / "validation" / "combined_v3"),
                      ("hack", C.ROOT / "runs" / "validation" / "combined_v3"), ("baseline", C.ROOT / "baseline_output" / "validation")):
        s = pd.read_csv(path / "score" / "score_summary.csv").iloc[0]
        head[key] = {k: round(float(s[k]), 4) for k in ("mae_p50", "calibration_pinball_mean", "brier_spike_mean", "brier_direction", "coverage_p10_p90")}
    return {
        "headline": head, "variant": variant,
        "om0": 0, "no": no, "qm0": int(mins([q0])[0]), "lo": 1440, "hi": hi,
        "days": [f"{loc0 + pd.Timedelta(days=d):%a %-d %b}" for d in range(ndays + 1)],
        "dayFirst": 1, "dayLast": ndays - 1 if ndays * 1440 + 720 > hi else ndays,
        "act": r2(price), "spot": r2(spot), "dir": ri(direction), "spk": ri(spk),
        "models": models, "modelList": labels, "members": MEMBERS, "ref": "A", "refName": "A (4 months)",
        "alarm": alarm, "modelTag": f"variant {variant} · combined v3", "tag": tag, "notice": notice,
        "spotLabel": "Spot (synthetic)" if share else "Spot now", "share": share,
    }


# ----------------------------------------------------------------------------- page
def build_html(data: dict) -> str:
    """The page (web/control_room.html: Vue template and view logic) with the data inlined."""
    page = TEMPLATE.read_text(encoding="utf-8")
    assert page.count("__DATA__") == 1
    return page.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="B")
    ap.add_argument("--share", action="store_true", help="synthetic spot, no baseline: safe to publish")
    a = ap.parse_args(argv)
    data = build_data(a.variant, a.share)
    out = C.ROOT / "runs" / "dashboard" / f"control_room_{a.variant}{'_share' if a.share else ''}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(data), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
