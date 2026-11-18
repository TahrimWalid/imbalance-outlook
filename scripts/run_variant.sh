#!/usr/bin/env bash
# One extended-history variant through the whole chain (imbalance/config.py VARIANTS), fast members only:
#   scripts/run_variant.sh B
# tuning (40 Optuna trials per head, rolling pre-validation folds) -> config/best_params_<v>.json,
# out-of-fold predictions + members for every experiment, per-step blend + range calibration,
# alarm levels -> config/alarm_levels_<v>.json, export for `python -m imbalance variants compare`.
# Outputs: runs/variants/<v>/. Studies are stored in runs/variants/<v>/tuning/optuna.db, so an
# interrupted run resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
V="$1"; MODELS=xgb,lgbm,qrf; TRIALS="${TRIALS:-40}"
export TE_VARIANT="$V"
PY=".venv/bin/python -W ignore"
step() { printf '\n[%s] %s %s\n' "$(date -u +%H:%M:%S)" "$V" "$*"; }
# QRF_FROM=A: tune only xgb/lgbm here and take the quantile forest's settings from variant A (a full qrf
# search on the long histories needs more than a day per variant). SKIP_TUNE=1: the studies are finished.
if [ -n "${QRF_FROM:-}" ]; then
  if [ -z "${SKIP_TUNE:-}" ]; then
    step "tune (xgb,lgbm, $TRIALS trials; qrf from $QRF_FROM)"
    $PY -m imbalance tune --families xgb,lgbm --trials "$TRIALS"
  fi
  $PY - "$V" "$QRF_FROM" <<'PYEOF'
import json, sys
v, src = sys.argv[1:]
best = json.load(open(f"runs/variants/{v}/tuning/best_params.json"))
best["qrf"] = json.load(open(f"config/best_params_{src}.json"))["qrf"]
json.dump(best, open(f"config/best_params_{v}.json", "w"), indent=2)
print(f"config/best_params_{v}.json: xgb, lgbm tuned on {v}; qrf from {src}")
PYEOF
else
  step "tune ($MODELS, $TRIALS trials)"
  $PY -m imbalance tune --families "$MODELS" --trials "$TRIALS"
  cp "runs/variants/$V/tuning/best_params.json" "config/best_params_$V.json"
fi
step "members: out-of-fold predictions, then each experiment"
$PY -m imbalance combine --tag v3 --models "$MODELS" --experiments validation,test_pre_validation,test_full
step "blend"
$PY -m imbalance blend --tag v3
step "alarm levels"
$PY -m imbalance alarms && cp "runs/variants/$V/tuning/alarm_levels.json" "config/alarm_levels_$V.json"
step "export for comparison"
$PY -m imbalance variants export
$PY score.py --submission "runs/variants/$V/validation/combined_v3/forecast_backtest.csv" \
    --actuals data/actuals_validation.csv --spike-margin 50 --output-dir "runs/variants/$V/validation/combined_v3/score" | tail -12
step "done"
