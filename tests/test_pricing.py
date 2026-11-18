"""The imbalance price rules against Fingrid's documents.

These prove the formula, not the choice of input datasets (that is runs/pricing/reproduction.md).
"""
import numpy as np
import pytest

from imbalance.pricing import bounds, price, round_cents

# Fingrid's worked example for the new rule (1.6.2026 document)
#        dir  mFRR price  FI mFRR energy (up, down)  VWA aFRR  aFRR satisfied demand  expected  case
EXAMPLE = [
    (1, 50, (50, 0), 100, 20, 64.29, "weighted"),     # 15:00
    (-1, 40, (0, 30), 20, 20, 32.00, "weighted"),     # 15:15
    (-1, 30, (50, 0), 40, 30, 30.00, "marginal"),     # 15:30: Finland activated 50 MWh up only
    (1, 100, (0, 0), 50, 50, 100.00, "marginal"),     # 15:45: no Finnish mFRR at all
]


def _example_row(row, rule="new"):
    d, mp, (mv_up, mv_down), vwa, av, _, _ = row
    up, down = d == 1, d == -1
    return price([d], vwa_up=[vwa if up else np.nan], vwa_down=[vwa if down else np.nan],
                 mfrr_price_up=[mp if up else np.nan], mfrr_price_down=[mp if down else np.nan],
                 day_ahead=[10.0], afrr_vol_up=[av if up else 0], afrr_vol_down=[av if down else 0],
                 mfrr_vol_up=[mv_up], mfrr_vol_down=[mv_down], rule=rule).iloc[0]


@pytest.mark.parametrize("i", range(4), ids=["15:00", "15:15", "15:30", "15:45"])
def test_fingrid_example(i):
    r = _example_row(EXAMPLE[i])
    assert r["price"] == EXAMPLE[i][5]
    assert r["case"] == EXAMPLE[i][6]


def test_old_rule_is_max_up_min_down():
    p = price([1, -1, 1, -1], vwa_up=[100, np.nan, 40, np.nan], vwa_down=[np.nan, 20, np.nan, 35],
              mfrr_price_up=[50, np.nan, 60, np.nan], mfrr_price_down=[np.nan, 40, np.nan, 30],
              day_ahead=[0, 0, 0, 0], rule="old")
    assert p["price"].tolist() == [100, 20, 60, 30]
    assert (p["case"] == "marginal").all()


def test_old_rule_ignores_volumes():
    a = price([1], [100], [np.nan], [50], [np.nan], [0], [20], [0], [50], [0], rule="old")
    assert a["price"].iloc[0] == 100


def test_no_direction_is_day_ahead_even_with_afrr():
    for rule in ("old", "new"):
        p = price([0], vwa_up=[100], vwa_down=[5], mfrr_price_up=[90], mfrr_price_down=[8], day_ahead=[42.42],
                  afrr_vol_up=[10], afrr_vol_down=[10], mfrr_vol_up=[30], mfrr_vol_down=[30], rule=rule)
        assert p["price"].iloc[0] == 42.42 and p["case"].iloc[0] == "day_ahead"


def test_missing_afrr_component():
    # no aFRR activation up: old -> mFRR price; new weighted -> mFRR price (aFRR has zero weight)
    for rule in ("old", "new"):
        p = price([1], [np.nan], [np.nan], [77.7], [np.nan], [0], [0], [0], [40], [0], rule=rule)
        assert p["price"].iloc[0] == 77.7


def test_missing_everything_is_flagged():
    p = price([1], [np.nan], [np.nan], [np.nan], [np.nan], [0], rule="old")
    assert np.isnan(p["price"].iloc[0]) and p["case"].iloc[0] == "missing"


def test_rounding_half_away_from_zero():
    assert round_cents([1.005, -1.005, 2.344999, 64.285714]).tolist() == [1.01, -1.01, 2.34, 64.29]


def _random_inputs(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.choice([-1, 0, 1], n)
    gaps = lambda x: np.where(rng.random(n) < 0.2, np.nan, x)
    return dict(direction=d, vwa_up=gaps(rng.normal(80, 60, n)), vwa_down=gaps(rng.normal(20, 60, n)),
                mfrr_price_up=gaps(rng.normal(90, 80, n)), mfrr_price_down=gaps(rng.normal(10, 80, n)),
                day_ahead=rng.normal(50, 30, n),
                afrr_vol_up=rng.exponential(20, n), afrr_vol_down=rng.exponential(20, n),
                mfrr_vol_up=np.where(rng.random(n) < 0.5, 0, rng.exponential(50, n)),
                mfrr_vol_down=np.where(rng.random(n) < 0.5, 0, rng.exponential(50, n)))


def test_rules_differ_only_in_weighted_quarter_hours():
    x = _random_inputs()
    old, new = price(**x, rule="old"), price(**x, rule="new")
    differ = ~np.isclose(old["price"], new["price"], equal_nan=True)
    assert (new.loc[differ, "case"] == "weighted").all()
    # and the new price is never above the old one when up, never below when down
    d = x["direction"]
    priced = old["price"].notna().to_numpy()
    assert (new["price"].notna().to_numpy() == priced).all()
    up, down = priced & (d == 1), priced & (d == -1)
    assert (new["price"][up] <= old["price"][up] + 1e-9).all()
    assert (new["price"][down] >= old["price"][down] - 1e-9).all()


def test_new_price_lies_between_components():
    x = _random_inputs(seed=1)
    new = price(**x, rule="new", rounded=False)
    lo, hi = bounds(x["direction"], x["vwa_up"], x["vwa_down"], x["mfrr_price_up"], x["mfrr_price_down"])
    w = (new["case"] == "weighted").to_numpy()
    assert w.sum() > 1000
    assert (new["price"][w] >= lo[w] - 1e-9).all() and (new["price"][w] <= hi[w] + 1e-9).all()


def test_both_directions_only_dominating_volume_counts():
    # Finland activated 40 MWh up and 25 MWh down; direction up: only the up volume weights the price
    p = price([1], vwa_up=[100], vwa_down=[10], mfrr_price_up=[50], mfrr_price_down=[20], day_ahead=[0],
              afrr_vol_up=[10], afrr_vol_down=[99], mfrr_vol_up=[40], mfrr_vol_down=[25], rule="new")
    assert p["case"].iloc[0] == "weighted" and p["price"].iloc[0] == 60.0      # (100*10 + 50*40) / 50
