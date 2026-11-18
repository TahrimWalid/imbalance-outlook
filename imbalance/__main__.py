"""Command line.

  python -m imbalance run validation --models baseline,clim,sarimax,timesfm,lgbm,xgb
  python -m imbalance run test_pre_validation test_full --models xgb
  python -m imbalance eda
  python -m imbalance targets runs/validation/xgb/forecast_backtest.csv
  python -m imbalance explain validation test_full  # why spikes / extreme lows are forecast
  python -m imbalance control-room [--share]    # dashboard, runs/dashboard/control_room_B[_share].html
  python -m imbalance reprice                   # reproduce Fingrid's imbalance price (data_ext/)
  python -m imbalance tune --families clim,sarimax,timesfm,xgb,lgbm,catboost,qrf --trials 50
"""
import argparse
import sys

import pandas as pd

from . import config as C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m imbalance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fit, forecast, validate and (validation only) score")
    r.add_argument("experiments", nargs="+", choices=list(C.EXPERIMENTS))
    r.add_argument("--models", default="baseline,clim,lgbm,xgb")
    sub.add_parser("eda", help="summary statistics of the pre-validation training data")
    t = sub.add_parser("tune", help="Optuna tuning on the rolling pre-validation folds")
    t.add_argument("--families", default="clim,sarimax,timesfm,xgb,lgbm,catboost,qrf")
    t.add_argument("--trials", type=int, default=50)
    g = sub.add_parser("targets", help="measure our own minimum targets on a validation forecast file")
    g.add_argument("forecast")
    fg = sub.add_parser("figures", help="forecast vs actual figures for a validation forecast file")
    fg.add_argument("forecast")
    fg.add_argument("--title", default="Imbalance price: forecast vs actual")
    ex = sub.add_parser("explain", help="per-row drivers of the spike probabilities (separate file, forecasts unchanged)")
    ex.add_argument("experiments", nargs="+", choices=list(C.EXPERIMENTS))
    cr = sub.add_parser("control-room", help="dashboard: Overview + Control room, one HTML file in runs/dashboard/")
    cr.add_argument("--variant", default="B")
    cr.add_argument("--share", action="store_true", help="synthetic spot, no baseline forecast: safe to publish")
    sub.add_parser("hindsight", help="test-period scores of the frozen v3 submission (after the freeze)")
    cb = sub.add_parser("combine", help="stack the tuned models and score the combined forecast")
    cb.add_argument("--tag", default="v1")
    cb.add_argument("--models", default="baseline,clim,sarimax,timesfm,xgb,lgbm,qrf,catboost")
    cb.add_argument("--experiments", default="validation")
    bl = sub.add_parser("blend", help="per-step blend + recent range calibration on the combine outputs")
    bl.add_argument("--tag", default="v3")
    sub.add_parser("alarms", help="choose spike alarm levels on the pre-validation folds")
    ac = sub.add_parser("actuals", help="rebuild actuals_<period>.csv from Fingrid 319/369 and the ENTSO-E spot price")
    ac.add_argument("period", choices=["validation", "test"])
    ac.add_argument("--data-dir", help="data folder (default: data/); test actuals go to data_ext/ by default")
    va = sub.add_parser("variants", help="extended-history experiments: export (inside TE_VARIANT) or compare with A")
    va.add_argument("action", choices=["export", "compare"])
    va.add_argument("names", nargs="*", help="variants to compare with A")
    va.add_argument("--tag", default="v3")
    rp = sub.add_parser("reprice", help="reproduce Fingrid's imbalance price from data_ext/ -> runs/pricing/reproduction.md")
    rp.add_argument("--v2", action="store_true", help="also build the re-priced series data_ext/imbalance_v2.parquet")
    args = ap.parse_args(argv)

    if args.cmd == "actuals":
        from pathlib import Path
        from .actuals import write, write_package_spot
        data_dir = Path(args.data_dir) if args.data_dir else (C.EXT_DATA_DIR if args.period == "test" else C.DATA_DIR)
        if data_dir == C.DATA_DIR and not (data_dir / "spot_entsoe.csv").is_file():
            print(write_package_spot(data_dir))
        print(write(args.period, data_dir))
        return 0

    if args.cmd == "variants":
        from .variants import compare, export
        log = lambda m: print(m, flush=True)
        export(args.tag, log) if args.action == "export" else compare(args.names, log=log)
        return 0

    if args.cmd == "reprice":
        from .reprice import quality_by_regime, reproduce
        quality_by_regime()
        reproduce(log=lambda m: print(m, flush=True))
        if args.v2:
            from .reprice import write_v2
            write_v2(log=lambda m: print(m, flush=True))
        return 0

    if args.cmd == "alarms":
        from .blend import choose_alarm_levels
        choose_alarm_levels()
        return 0

    if args.cmd == "blend":
        from .blend import run as blend
        blend(args.tag, log=lambda m: print(m, flush=True))
        return 0

    if args.cmd == "combine":
        from .combine import run as combine
        combine(args.models.split(","), args.tag, args.experiments.split(","), log=lambda m: print(m, flush=True))
        return 0

    if args.cmd == "figures":
        from pathlib import Path
        from .figures import main as figures
        for path in figures(Path(args.forecast), args.title):
            print(path)
        return 0

    if args.cmd == "hindsight":
        from .hindsight import main as hindsight
        hindsight()
        return 0

    if args.cmd == "explain":
        from .explain import run as explain
        for e in args.experiments:
            explain(e, log=lambda m: print(m, flush=True))
        return 0

    if args.cmd == "control-room":
        from .control_room import main as control_room
        control_room(["--variant", args.variant] + (["--share"] if args.share else []))
        return 0

    if args.cmd == "targets":
        from pathlib import Path
        from .targets import report
        print(report(Path(args.forecast)).to_string())
        return 0

    if args.cmd == "eda":
        from .eda import main as eda
        eda()
        return 0

    if args.cmd == "tune":
        import logging
        import optuna
        from .data import Store
        from .tune import Tuner
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger("jax").setLevel(logging.ERROR)
        log = lambda msg: print(msg, flush=True)
        Tuner(Store.load(), log).run(args.families.split(","), args.trials)
        return 0

    from .data import Store
    from .pipeline import run
    store = Store.load()
    for name in args.experiments:
        table = run(name, args.models.split(","), store)
        if len(table):
            with pd.option_context("display.width", 220, "display.max_columns", 20):
                print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
