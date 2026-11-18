"""Reference models without machine learning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .base import Context, Forecaster


class Baseline(Forecaster):
    """Organiser baseline: last usable imbalance price + per-step historical error quantiles."""
    name = "baseline"

    def fit(self, train, ctx):
        err = (train["y_imb"] - train["fi_imbalance_price__last"]).groupby(train["step"])
        self.q10 = np.minimum(err.quantile(0.1), 0)
        self.q90 = np.maximum(err.quantile(0.9), 0)
        d, pr = train["y_dir"], train["y_prem"]
        self.probs = dict(p_up=(d == 1).mean(), p_down=(d == -1).mean(), p_none=(d == 0).mean(),
                          p_spike_high=(pr > C.SPIKE_MARGIN).mean(), p_spike_low=(pr < -C.SPIKE_MARGIN).mean())
        return self

    def predict(self, panel, ctx):
        last = panel["fi_imbalance_price__last"].to_numpy()
        return pd.DataFrame(dict(p10=last + self.q10.reindex(panel["step"]).to_numpy(), p50=last,
                                 p90=last + self.q90.reindex(panel["step"]).to_numpy(), **self.probs),
                            index=panel.index)


class Climatology(Forecaster):
    """Spot price + premium distribution looked up by (local hour, step bucket, recent direction).

    Tunable: step bucket edges, which state key to condition on, hour grouping, minimum group size.
    """
    name = "clim"
    DEFAULTS = dict(bucket_edges=(4, 12, 24), state="dir_last", hour_group=1, weekend=False, min_rows=100)

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULTS, **(params or {})}

    def _keys(self, p):
        prm = self.params
        edges = np.asarray(prm["bucket_edges"])
        k = {"h": (p["cal_hour_local"].fillna(0).astype(int) // prm["hour_group"]),
             "b": np.searchsorted(edges, p["step"].to_numpy(), side="left")}
        if prm["state"] == "dir_last":
            k["s"] = p["fi_mfrr_direction__last"].fillna(9).astype(int)
        elif prm["state"] == "prem_sign":
            k["s"] = np.sign(p["prem__mean4h"].fillna(0)).astype(int)
        elif prm["state"] == "share_up":
            k["s"] = np.digitize(p["dir__share_up4h"].fillna(0.5), [0.25, 0.75])
        else:
            k["s"] = 0
        if prm["weekend"]:
            k["w"] = (p["cal_weekday_local"].fillna(0) >= 5).astype(int)
        return pd.DataFrame(k, index=p.index)

    def fit(self, train, ctx):
        keys = self._keys(train)
        self.fine_keys = list(keys.columns)
        self.coarse_keys = [c for c in self.fine_keys if c != "h"]
        t = train.join(keys)
        t["up"], t["down"], t["none"] = t["y_dir"] == 1, t["y_dir"] == -1, t["y_dir"] == 0
        t["hi"], t["lo"] = t["y_prem"] > C.SPIKE_MARGIN, t["y_prem"] < -C.SPIKE_MARGIN

        def table(by):
            g = t.groupby(by)
            out = g["y_prem"].quantile([.1, .5, .9]).unstack()
            out.columns = ["q10", "q50", "q90"]
            return out.join(g[["up", "down", "none", "hi", "lo"]].mean()).join(g.size().rename("n"))
        self.fine, self.coarse = table(self.fine_keys), table(self.coarse_keys)
        self.overall = table(lambda _: 0).iloc[0]
        return self

    def predict(self, panel, ctx):
        k = self._keys(panel)
        e = k.join(self.fine, on=self.fine_keys)
        cols = list(self.coarse.columns)
        small = e["n"].isna() | (e["n"] < self.params["min_rows"])
        e.loc[small, cols] = k[small].join(self.coarse, on=self.coarse_keys)[cols].to_numpy()
        missing = e["n"].isna()
        e.loc[missing, cols] = self.overall[cols].to_numpy()
        spot = panel["fi_spot_price__d"].to_numpy()
        return pd.DataFrame(dict(p10=spot + e["q10"], p50=spot + e["q50"], p90=spot + e["q90"],
                                 p_up=e["up"], p_down=e["down"], p_none=e["none"],
                                 p_spike_high=e["hi"], p_spike_low=e["lo"]), index=panel.index).astype(float)
