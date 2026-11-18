"""Model registry: name -> factory taking optional tuned parameters."""
from .base import OUT_COLS, Context, Forecaster
from .gbm import GBM
from .qrf import QRF
from .simple import Baseline, Climatology
from .timeseries import Sarimax, TimesFM

REGISTRY = {
    "baseline": lambda p=None: Baseline(),
    "clim": lambda p=None: Climatology(p),
    "sarimax": lambda p=None: Sarimax(p),
    "timesfm": lambda p=None: TimesFM(p),
    "qrf": lambda p=None: QRF(p),
    "xgb": lambda p=None: GBM("xgb", p),
    "lgbm": lambda p=None: GBM("lgbm", p),
    "catboost": lambda p=None: GBM("catboost", p),
}


def make(name: str, params: dict | None = None) -> Forecaster:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'; choose from {', '.join(REGISTRY)}")
    return REGISTRY[name](params)
