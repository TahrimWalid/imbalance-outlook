"""Paths, constants and experiment definitions. The single place where dates live."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TE_DATA_DIR", ROOT / "data"))
RUNS_DIR = Path(os.environ.get("TE_RUNS_DIR", ROOT / "runs"))

Q = pd.Timedelta(minutes=15)
HORIZON = 40                 # quarter-hours per forecast (10 h)
SPIKE_MARGIN = 50.0          # EUR/MWh, same as score.py

# Training cut-offs: labels are read as known at the cut-off (first_usable_origin_utc <= cut-off).
TRAIN_END_PRE_VALIDATION = pd.Timestamp("2026-09-06T21:00:00Z")   # 7 Sep 00:00 Finnish time
TRAIN_END_FULL = pd.Timestamp("2026-10-04T21:00:00Z")             # 5 Oct 00:00, official training cut-off

# Exploratory analysis must not look at the validation period (keeps validation scores honest).
EDA_END = TRAIN_END_PRE_VALIDATION


@dataclass(frozen=True)
class Experiment:
    """A model trained up to `train_end` and evaluated on one period."""
    name: str
    train_end: pd.Timestamp
    period_days: tuple[str, str]          # local dates, for make_templates.py
    actuals: Path | None                  # None = no scoring (test period)

    @property
    def first_origin(self) -> pd.Timestamp:
        return pd.Timestamp(self.period_days[0], tz="Europe/Helsinki").tz_convert("UTC")


VALIDATION_DAYS = ("2026-09-07", "2026-10-04")
TEST_DAYS = ("2026-10-05", "2026-11-01")

EXPERIMENTS = {
    # model selection: trained before the validation month, scored on it
    "validation": Experiment("validation", TRAIN_END_PRE_VALIDATION, VALIDATION_DAYS, DATA_DIR / "actuals_validation.csv"),
    # test forecasts, variant A: validation month never used for training
    "test_pre_validation": Experiment("test_pre_validation", TRAIN_END_PRE_VALIDATION, TEST_DAYS, None),
    # test forecasts, variant B: all training data allowed by the rules (up to 4 Oct)
    "test_full": Experiment("test_full", TRAIN_END_FULL, TEST_DAYS, None),
}


# --------------------------------------------------------------------------------------------------
# Extended history (data_ext/, built by build_dataset.py --catalogue extended) and the market regimes.
# One source of truth for every date the re-pricing and the extended-history experiments use.
# --------------------------------------------------------------------------------------------------
EXT_DATA_DIR = Path(os.environ.get("TE_EXT_DATA_DIR", ROOT / "data_ext"))
EXT_START = pd.Timestamp("2024-06-11T21:00:00Z")          # 12 Jun 2024 00:00 Finnish time


@dataclass(frozen=True)
class Regime:
    start_utc: pd.Timestamp
    name: str
    source: str           # where the date was checked


REGIMES = [
    Regime(pd.Timestamp("2024-06-11T21:00:00Z"), "aFRR energy market; imbalance price = max/min(VWA aFRR, mFRR), hourly",
           "Fingrid, Imbalance price in Finland 29.8.2024 and 1.6.2026 (timeline: 12.6.2024); data: 347-350 start "
           "2024-06-12. Dataset 369 (direction) is missing for 983 hours between 2024-06-12 and 2024-08-08 and 319 is unrounded until 2024-08-07"),
    Regime(pd.Timestamp("2025-03-04T12:00:00Z"), "Nordic mFRR energy activation market (15 min)",
           "Fingrid 1.6.2026 document timeline (4.3.2025); data: 375/376 and 385-394 start 2025-03-04T12:00Z; "
           "33/34 (hourly ordered regulation) end on 4.3.2025"),
    Regime(pd.Timestamp("2025-03-18T23:00:00Z"), "quarter-hour imbalance price; 15 min intraday",
           "Fingrid 1.6.2026 document timeline (19.3.2025); data: first hour in which 319 differs between quarter-hours "
           "is 2025-03-18T23:00Z (19.3.2025 00:00 CET); 14-18.3.2025 had a few overlapping quarter-hour rows"),
    Regime(pd.Timestamp("2025-03-25T22:00:00Z"), "Fingrid connected to PICASSO (aFRR cross-border platform)",
           "Fingrid developer portal: 'Fingrid joined the European PICASSO platform in March 2025'; press reports "
           "give 26 or 27 March. Data: local-selection prices 351/352 start 2025-03-26. Dataset 411 (PICASSO status) "
           "only starts 2025-12-31, so it cannot date the accession. Start of the local day 26.3.2025 used"),
    Regime(pd.Timestamp("2025-09-02T21:00:00Z"), "mFRR price: last bid activated in Finland when selected bids are left unactivated",
           "Fingrid mFRR and imbalance price Q&A 28.8.2025, question 1 ('Alkaen 3.9.2025 ...'); local day start used"),
    Regime(pd.Timestamp("2025-09-30T22:00:00Z"), "day-ahead market 15 min MTU",
           "not yet checked against data: to be confirmed from the ENTSO-E day-ahead resolution (Phase 1)"),
    Regime(pd.Timestamp("2026-05-31T21:00:00Z"), "volume-weighted imbalance price when Finland activated mFRR in the dominating direction",
           "Fingrid, Imbalance price in Finland 1.6.2026; news 27.4.2026"),
]

# Known Fingrid corrections and exclusions found when reproducing 319 (runs/pricing/reproduction.md).
PRICE_EXCEPTIONS = [
    ("2025-05-05T21:00:00Z", "2025-08-25T20:45:00Z", "Fingrid news 5.9.2025 'Error in imbalance pricing': avoided-activation "
     "value in VWA aFRR had the wrong sign; corrected in eSett's data. Open data 319 and 347/348 agree with each other "
     "in this period (no reproduction mismatches)."),
    ("2025-10-22T22:00:00Z", "2025-10-23T08:45:00Z", "aFRR component removed from the imbalance price (PICASSO price "
     "errors): 319 equals the mFRR price"),
]


# --------------------------------------------------------------------------------------------------
# Extended-history experiments (Phase 5). Selected with the environment variable TE_VARIANT; each variant
# writes to runs/variants/<name>/ and reads/writes config/*_<name>.json, so the frozen submission's
# files are never touched. Fold test windows stay as in cv.py (12 Jul - 6 Sep 2026): older data only
# extends the training side.
# --------------------------------------------------------------------------------------------------
QH_PRICE_START = pd.Timestamp("2025-03-18T23:00:00Z")      # first quarter-hour-priced hour (Fingrid 319)
NEW_RULE_START = pd.Timestamp("2026-05-31T21:00:00Z")      # 1 Jun 2026, volume-weighted imbalance price


@dataclass(frozen=True)
class Variant:
    name: str
    history_start: pd.Timestamp | None   # None: starter package data only (1 Jun 2026)
    label: str                            # "published" (Fingrid 319) or "v2" (re-priced, imbalance_v2.parquet)
    regime_feature: bool = False          # add regime indicators as features
    decay_halflife_days: float | None = None   # time-decay sample weights (None = equal weights)
    note: str = ""


FAST_MEMBERS = ["xgb", "lgbm", "qrf"]
VARIANTS = {
    "A": Variant("A", None, "published", note="current model, 1 Jun 2026 onward, ENTSO-E spot (reference)"),
    "B": Variant("B", QH_PRICE_START, "v2", note="history from 19 Mar 2025 (quarter-hour prices), re-priced labels"),
    "C": Variant("C", pd.Timestamp("2025-09-02T21:00:00Z"), "v2", note="history from 3 Sep 2025, re-priced labels"),
    "E": Variant("E", QH_PRICE_START, "published", note="B without re-pricing (published old-rule prices)"),
    "F": Variant("F", EXT_START, "v2", regime_feature=True,
                 note="B plus the hourly-price year 12 Jun 2024 - 18 Mar 2025 (published, not re-priced), regime flag"),
    # D (regime feature and/or time decay on the better of B and C) is defined once B and C are scored.
}
VARIANT = VARIANTS.get(os.environ.get("TE_VARIANT", ""))
if VARIANT is not None and "TE_RUNS_DIR" not in os.environ:
    RUNS_DIR = ROOT / "runs" / "variants" / VARIANT.name
