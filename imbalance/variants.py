"""Compare extended-history variants (config.VARIANTS) with the reference variant A.

    TE_VARIANT=B python -m imbalance variants export     after the variant's tune/combine/blend chain
    python -m imbalance variants compare B C E F          -> runs/variants/comparison.md

export (run inside a variant) writes runs/variants/<v>/compare/:
    folds.parquet       every fold-test row (origin, step, labels), each member's out-of-fold prediction and a
                        forward-chained blend (weights fitted on the earlier folds, range calibration on the
                        previous fold; fold 0 has no earlier fold and keeps only the members)
    validation.parquet  the variant's validation-month forecast (combined_<tag>) with its labels

compare aligns rows by (origin, step), so every variant is scored on exactly the same quarter-hours, and
gives each metric's difference to A with a 90% block bootstrap over days (whole days resampled, so the
strong correlation within a day is kept). An interval that includes zero is reported as no clear difference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .metrics import all_metrics

KEY = ["origin", "step"]
METRICS = ["mae_p50", "pinball_p10", "pinball_p90", "calibration_pinball_mean", "coverage_p10_p90",
           "brier_spike_high", "brier_spike_low", "brier_spike_mean", "brier_direction"]
LOWER_IS_BETTER = {m: m != "coverage_p10_p90" for m in METRICS}
VARIANTS_DIR = C.ROOT / "runs" / "variants"


def export(tag: str = "v3", log=print) -> Path:
    from . import cv
    from .blend import MEMBERS, Blend, load_period
    from .data import Store
    from .models import OUT_COLS
    from .stack import STACK_DIR, fold_ids, labels

    out = C.RUNS_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    store = Store.load()
    u = cv.universe(store, "pre_validation")
    y = labels(pd.concat([f.test for f in cv.folds(u, "pre_validation")]))
    fold = fold_ids(y, "pre_validation")
    P = {m: pd.read_parquet(STACK_DIR / f"oof_pre_validation_{m}.parquet").loc[y.index][OUT_COLS] for m in MEMBERS}
    rows = y.reset_index(drop=True).assign(fold=fold)
    for m in MEMBERS:
        rows = rows.join(P[m].reset_index(drop=True).add_prefix(f"{m}__"))
    blend = pd.DataFrame(np.nan, index=rows.index, columns=OUT_COLS)
    for k in range(1, fold.max() + 1):
        tr, cal, te = fold < k, fold == k - 1, fold == k
        b = Blend(MEMBERS).fit_weights({m: v[tr] for m, v in P.items()}, y[tr])
        b.fit_calibration({m: v[cal] for m, v in P.items()}, y[cal])
        blend.loc[te] = b.predict({m: v[te] for m, v in P.items()}, y["step"][te]).to_numpy()
    rows = rows.join(blend.add_prefix("blend__"))
    rows.to_parquet(out / "folds.parquet")

    V, ref, yv = load_period("validation", with_labels=True)
    f = pd.read_csv(C.RUNS_DIR / "validation" / f"combined_{tag}" / "forecast_backtest.csv")
    val = f.rename(columns={"forecast_origin_utc": "origin"})
    val["origin"] = pd.to_datetime(val["origin"], utc=True)
    val = val.join(yv[["y_imb", "y_spot", "y_dir"]].reset_index(drop=True))
    val.to_parquet(out / "validation.parquet")
    log(f"exported {out} ({len(rows):,} fold rows, {len(val):,} validation rows)")
    return out


def _metrics(df: pd.DataFrame, prefix: str = "") -> dict:
    pred = df[[f"{prefix}{c}" for c in ("p10", "p50", "p90", "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low")]]
    pred.columns = [c[len(prefix):] for c in pred.columns]
    return all_metrics(pred, df["y_imb"], df["y_spot"], df["y_dir"])


def _day(origins: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(origins).tz_convert("Europe/Helsinki").normalize().asi8


def bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, prefix: str, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Metric(b) - metric(a) on the same rows, with a 90% day-block bootstrap interval."""
    day = _day(a["origin"])
    days, inv = np.unique(day, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(days))]
    point = {k: _metrics(b, prefix)[k] - _metrics(a, prefix)[k] for k in METRICS}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in METRICS}
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(days), len(days))])
        ma, mb = _metrics(a.iloc[idx], prefix), _metrics(b.iloc[idx], prefix)
        for k in METRICS:
            draws[k].append(mb[k] - ma[k])
    rows = []
    for k in METRICS:
        lo, hi = np.quantile(draws[k], [0.05, 0.95])
        better = (hi < 0) if LOWER_IS_BETTER[k] else None
        verdict = "no clear difference" if lo <= 0 <= hi else (
            ("better" if better else "worse") if LOWER_IS_BETTER[k] else "different")
        rows.append({"metric": k, "A": _metrics(a, prefix)[k], "variant": _metrics(b, prefix)[k],
                     "difference": point[k], "90% low": lo, "90% high": hi, "verdict": verdict})
    return pd.DataFrame(rows).set_index("metric")


def _load(v: str, what: str) -> pd.DataFrame:
    df = pd.read_parquet(VARIANTS_DIR / v / "compare" / f"{what}.parquet")
    df["origin"] = pd.to_datetime(df["origin"], utc=True)
    return df


def _align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = a[KEY].merge(b[KEY], on=KEY)
    a = keys.merge(a, on=KEY, how="left").reset_index(drop=True)
    b = keys.merge(b, on=KEY, how="left").reset_index(drop=True)
    labels_equal = np.allclose(a["y_imb"], b["y_imb"], equal_nan=True) and np.allclose(a["y_spot"], b["y_spot"], equal_nan=True)
    assert labels_equal, "variants are scored on different labels"
    known = a["y_imb"].notna().to_numpy()            # quarter-hours without a published price are not scored (as score.py)
    return a[known].reset_index(drop=True), b[known].reset_index(drop=True)


def _note(v: str) -> str:
    """A snapshot <V>_qrfA (scripts/qrf_followup.sh) is variant V with the quantile forest settings of A."""
    base, _, suffix = v.partition("_")
    note = C.VARIANTS[base].note if base in C.VARIANTS else v
    return note + (", quantile forest settings from A" if suffix == "qrfA" else "")


def compare(variants: list[str], ref: str = "A", n_boot: int = 1000, log=print) -> Path:
    md = ["# Extended-history variants compared with A", "",
          "Differences are variant minus A (negative = better for every metric except coverage, where 0.80 is the "
          "target). 90% intervals from a block bootstrap over days. 'Folds' are the rolling pre-validation folds "
          "(test windows 12 Jul - 6 Sep 2026, unchanged); 'blend' is forward-chained (weights from earlier folds), so "
          "fold 0 has members only. 'Validation' is the full chain's validation-month forecast scored on the "
          "rebuilt actuals (Fingrid 319/369 + ENTSO-E spot).", ""]
    for v in variants:
        md += [f"## {v}: {_note(v)}", ""]
        fa, fb = _align(_load(ref, "folds"), _load(v, "folds"))
        members = [c[:-len("__p50")] for c in fb.columns if c.endswith("__p50") and c != "blend__p50"]
        for prefix in ["blend__"] + [f"{m}__" for m in members]:
            sel = fa[f"{prefix}p50"].notna() & fb[f"{prefix}p50"].notna()
            if not sel.any():
                continue
            t = bootstrap_diff(fa[sel].reset_index(drop=True), fb[sel].reset_index(drop=True), prefix, n_boot)
            md += [f"### Folds, {prefix[:-2]} ({int(sel.sum()):,} rows)", "", _table(t), ""]
        # per fold and per forecast hour, the blend where it exists, else xgb
        per = []
        for k in sorted(fa["fold"].unique()):
            for prefix in ("blend__", "xgb__"):
                s = (fa["fold"] == k) & fa[f"{prefix}p50"].notna()
                if s.any():
                    ma, mb = _metrics(fa[s], prefix), _metrics(fb[s], prefix)
                    per.append({"fold": k, "model": prefix[:-2], "rows": int(s.sum()),
                                **{f"{m} A": ma[m] for m in ("mae_p50", "calibration_pinball_mean")},
                                **{f"{m} {v}": mb[m] for m in ("mae_p50", "calibration_pinball_mean")}})
        md += ["### Per fold", "", _table(pd.DataFrame(per).set_index("fold")), ""]
        va, vb = _align(_load(ref, "validation"), _load(v, "validation"))
        t = bootstrap_diff(va, vb, "", n_boot)
        md += ["### Validation month (official metrics)", "", _table(t), ""]
        hours = []
        for h in range(10):
            s = (va["step"] - 1) // 4 == h
            ma, mb = _metrics(va[s]), _metrics(vb[s])
            hours.append({"forecast hour": h + 1, **{f"{m} A": ma[m] for m in ("mae_p50", "brier_spike_mean", "brier_direction")},
                          **{f"{m} {v}": mb[m] for m in ("mae_p50", "brier_spike_mean", "brier_direction")}})
        md += ["### Validation month by forecast hour", "", _table(pd.DataFrame(hours).set_index("forecast hour")), ""]
        log(f"{v} done")
    path = VARIANTS_DIR / "comparison.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    return path


def _table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [f"{x:.4f}" if isinstance(x, float) else str(x) for x in r]) + " |")
    return "\n".join(out)
