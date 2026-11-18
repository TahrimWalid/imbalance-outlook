"""Classical and foundation time-series models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..data import IMB, SPOT
from .base import Forecaster, premium_windows, probs_from_quantiles


def fourier(ts: pd.DatetimeIndex, k: int) -> np.ndarray | None:
    """Daily Fourier terms (local time) with k harmonics; None when k == 0."""
    if k == 0:
        return None
    loc = pd.DatetimeIndex(ts).tz_convert("Europe/Helsinki")
    x = 2 * np.pi * (loc.hour * 4 + loc.minute // 15).to_numpy() / 96
    return np.column_stack([f(j * x) for j in range(1, k + 1) for f in (np.sin, np.cos)])


def by_origin_step(panel: pd.DataFrame, origins: pd.DatetimeIndex):
    """Index tuple selecting each panel row from an (origins x 40 steps) array."""
    return origins.get_indexer(panel["origin"]), panel["step"].to_numpy() - 1


class Sarimax(Forecaster):
    """ARMA(p, q) on the premium with optional daily Fourier seasonality.

    Fitted once at the cut-off; at each origin the state is re-filtered with the data
    published by then and the next 40 quarter-hours are forecast.
    """
    name = "sarimax"
    DEFAULTS = dict(p=2, q=1, k_fourier=0, clip=500.0, window=192)

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULTS, **(params or {})}

    def fit(self, train, ctx):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        prm = self.params
        grid = pd.date_range(ctx.store.start, ctx.train_end - C.Q, freq="15min")
        y = ctx.store.asof(IMB, grid, ctx.train_end) - ctx.store.asof(SPOT, grid, ctx.train_end)
        self.res = SARIMAX(np.clip(y, -prm["clip"], prm["clip"]), exog=fourier(grid, prm["k_fourier"]),
                           order=(prm["p"], 0, prm["q"]), trend="c").fit(disp=False, maxiter=200)
        return self

    def predict(self, panel, ctx):
        from scipy.stats import norm
        prm = self.params
        origins = pd.DatetimeIndex(panel["origin"].unique())
        win = np.clip(premium_windows(ctx.store, origins, prm["window"]), -prm["clip"], prm["clip"])
        mu, sd = [], []
        for o, w in zip(origins, win):
            past = o - np.arange(prm["window"], 0, -1) * C.Q
            fut = o + np.arange(C.HORIZON) * C.Q
            f = self.res.apply(w, exog=fourier(past, prm["k_fourier"])).get_forecast(
                C.HORIZON, exog=fourier(fut, prm["k_fourier"]))
            mu.append(f.predicted_mean)
            sd.append(np.sqrt(f.var_pred_mean))
        rows = by_origin_step(panel, origins)
        mu, sd = np.stack(mu)[rows], np.stack(sd)[rows]
        spot, z, m = panel["fi_spot_price__d"].to_numpy(), norm.ppf(0.9), C.SPIKE_MARGIN
        cdf = lambda x: norm.cdf(x, mu, sd)
        return pd.DataFrame(dict(p10=spot + mu - z * sd, p50=spot + mu, p90=spot + mu + z * sd,
                                 p_up=1 - cdf(1.0), p_down=cdf(-1.0), p_none=cdf(1.0) - cdf(-1.0),
                                 p_spike_high=1 - cdf(m), p_spike_low=cdf(-m)), index=panel.index)


TIMESFM_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"   # Hugging Face snapshot used for all results


class TimesFM(Forecaster):
    """TimesFM 2.5 (zero-shot, nothing trained).

    Tunable: context length, target (premium or raw imbalance price), and whether known-ahead
    covariates (spot, net-load forecast) are added through TimesFM's in-context regression.
    """
    name = "timesfm"
    DEFAULTS = dict(context=1024, target="premium", covariates=False, xreg_mode="xreg + timesfm")
    _cache: dict = {}

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULTS, **(params or {})}

    def fit(self, train, ctx):
        return self

    def __getstate__(self):                     # never pickle the torch model
        return {"params": self.params}

    def _model(self, context: int, backcast: bool):
        key = (context, backcast)
        if key not in TimesFM._cache:
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", revision=TIMESFM_REVISION)
            m.compile(timesfm.ForecastConfig(max_context=context, max_horizon=64, normalize_inputs=True,
                                             use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                             infer_is_positive=False, per_core_batch_size=32,
                                             return_backcast=backcast))
            TimesFM._cache[key] = m
        return TimesFM._cache[key]

    def _series(self, store, origins, L):
        if self.params["target"] == "premium":
            return premium_windows(store, origins, L)
        k = np.arange(L, 0, -1)
        return np.stack([store.asof(IMB, o - k * C.Q, o) for o in origins])

    def predict(self, panel, ctx):
        prm, store = self.params, ctx.store
        L = prm["context"]
        origins = pd.DatetimeIndex(panel["origin"].unique())
        inputs, gaps, starts = [], [], []
        for o, w in zip(origins, self._series(store, origins, L)):
            ok = np.flatnonzero(~np.isnan(w))
            first, last = ok[0], ok[-1]
            x = pd.Series(w[first:last + 1]).interpolate(limit_direction="both").to_numpy()
            inputs.append(x)
            gaps.append(L - 1 - last)
            starts.append(o - (L - first) * C.Q)          # timestamp of x[0]
        h = C.HORIZON + max(gaps)
        if prm["covariates"]:
            q = self._forecast_xreg(store, origins, inputs, starts, h, L)
        else:
            _, qf = self._model(L, False).forecast(horizon=h, inputs=inputs)
            q = qf[:, :, 1:]                               # drop the mean: q10..q90
        qs = np.stack([q[i, g:g + C.HORIZON] for i, g in enumerate(gaps)])       # (origins, 40, 9)
        qs = np.sort(qs[by_origin_step(panel, origins)], axis=1)
        spot = panel["fi_spot_price__d"].to_numpy()
        if prm["target"] == "premium":
            price_q, prem_q = qs + spot[:, None], qs
        else:
            price_q, prem_q = qs, qs - spot[:, None]
        return pd.DataFrame(dict(p10=price_q[:, 0], p50=price_q[:, 4], p90=price_q[:, 8],
                                 **probs_from_quantiles(np.arange(1, 10) / 10, prem_q)), index=panel.index)

    def _forecast_xreg(self, store, origins, inputs, starts, h, L):
        """Known-ahead covariates (spot, net load forecast, hour of day) over context + horizon."""
        cov = {"spot": [], "netload": [], "hsin": [], "hcos": []}
        for o, x, s in zip(origins, inputs, starts):
            ts = s + np.arange(len(x) + h) * C.Q
            spot = store.asof(SPOT, ts, o)
            net = (store.asof("fi_consumption_fc_da", ts, o) - store.asof("fi_wind_fc_da", ts, o)
                   - np.nan_to_num(store.asof("fi_solar_fc_da", ts, o)))
            fill = lambda a: pd.Series(a).ffill().bfill().fillna(0).to_numpy()
            hx = fourier(pd.DatetimeIndex(ts), 1)
            cov["spot"].append(fill(spot))
            cov["netload"].append(fill(net) / 1000)
            cov["hsin"].append(hx[:, 0])
            cov["hcos"].append(hx[:, 1])
        model = self._model(L, True)
        # returns (point outputs, quantile outputs); each quantile output is (h, mean + q10..q90)
        _, qout = model.forecast_with_covariates(inputs=inputs, dynamic_numerical_covariates=cov,
                                                 xreg_mode=self.params["xreg_mode"], ridge=1.0)
        return np.stack([np.asarray(q)[:h, 1:] for q in qout])
