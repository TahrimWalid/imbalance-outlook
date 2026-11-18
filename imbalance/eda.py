"""Exploratory summary of the training data before the validation period (config.EDA_END).

Prints aggregates only: rows of spot_nordpool.csv must not be printed or published.
"""
import pandas as pd

from . import config as C
from .data import DIR, IMB, SPOT, load_raw


def main() -> None:
    raw = load_raw()
    raw = raw.loc[raw["timestamp_utc"] < C.EDA_END]
    print(f"EDA on deliveries before {C.EDA_END:%Y-%m-%d %H:%M}Z (validation and test periods excluded)\n")
    wide = raw.pivot(index="timestamp_utc", columns="series_id", values="value")
    imb, spot, dirn = wide[IMB], wide[SPOT], wide[DIR]
    prem = imb - spot
    local = wide.index.tz_convert("Europe/Helsinki")

    print("== Target ==")
    print(imb.describe(percentiles=[.01, .05, .5, .95, .99]).round(1).to_string())
    print("premium quantiles:", prem.quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99]).round(1).to_dict())
    print(f"spike high share {(prem > C.SPIKE_MARGIN).mean():.4f}, low share {(prem < -C.SPIKE_MARGIN).mean():.4f}")
    print("direction shares:", dirn.value_counts(normalize=True).round(3).to_dict())

    print("\n== By local hour ==")
    print(pd.DataFrame({"up": dirn.eq(1), "spike_hi": prem > C.SPIKE_MARGIN, "spike_lo": prem < -C.SPIKE_MARGIN,
                        "abs_prem": prem.abs()}).groupby(local.hour).mean().round(3).to_string())
    print("\n== By month ==")
    print(pd.DataFrame({"up": dirn.eq(1), "abs_prem": prem.abs(), "imb": imb}).groupby(local.to_period("M"))
          .agg(up=("up", "mean"), abs_prem=("abs_prem", "mean"), imb_std=("imb", "std")).round(3).to_string())

    print("\n== Direction persistence P(same direction k quarter-hours later) ==")
    for k in (1, 5, 20, 40):
        t = pd.crosstab(dirn.shift(k), dirn, normalize="index")
        print(f"k={k:2d}: down->down {t.loc[-1, -1]:.3f}, up->up {t.loc[1, 1]:.3f}")
    print("premium autocorrelation:", {k: round(prem.autocorr(k), 3) for k in (1, 5, 8, 16, 40, 96)})

    print("\n== Same-quarter-hour correlations ==")
    cand = {"wind_fc": wide["fi_wind_fc_da"], "netload_fc": wide["fi_consumption_fc_da"] - wide["fi_wind_fc_da"],
            "spot": spot, "cons_err": wide["fi_consumption"] - wide["fi_consumption_fc_da"],
            "se1_imb_prem": wide["se1_imbalance_price_long"] - spot, "ee_imb_prem": wide["ee_imbalance_price_long"] - spot}
    print(pd.DataFrame({k: {"premium": v.corr(prem), "up": v.corr(dirn.eq(1).astype(float))}
                        for k, v in cand.items()}).T.round(3).to_string())


if __name__ == "__main__":
    main()
