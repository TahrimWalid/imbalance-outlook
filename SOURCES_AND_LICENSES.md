# Data sources, external services, models and software

Everything used to produce the submitted forecast (`runs/test_full/combined_v3/forecast_backtest.csv`)
and its usage rights. Package licences are taken from each installed package's own metadata;
data licences from the organisers' starter package (`README.md`, section 9).

## 1. Data

| Source | What we use | Licence / usage rights |
|---|---|---|
| **Fingrid Open Data** (via the starter package) | Imbalance price (the target), mFRR and aFRR prices and activations, dominating mFRR direction, consumption and production, cross-border flows (SE1, SE3, NO4, EE), day-ahead consumption, wind and solar forecasts | CC BY 4.0 |
| **ENTSO-E Transparency Platform** (via the starter package) | Imbalance prices of SE1, SE3 and EE; day-ahead load forecasts of SE1, SE3 and EE; offered FI–EE day-ahead capacity | CC BY 4.0. Only data on ENTSO-E's list of data available for free re-use; converted by the organisers to UTC and 15-minute series |
| **Nord Pool** day-ahead spot price, Finland (via the starter package) | Spot price at each delivery quarter-hour; the forecast is spot + predicted premium | Licensed to Turku Energia and shared with the teams **for the hackathon only**: not shared further, not uploaded to any repository, deleted after the event. Applies to `spot_nordpool.csv` and the `spot_price` column of `actuals_validation.csv` |
| **ENTSO-E Transparency Platform, day-ahead price FI** (branch `post-hack-tree` only, replaces Nord Pool) | Spot price at each delivery quarter-hour (document A44, bidding zone 10YFI-1--------U); hourly until 30 Sep 2025, 15 min from 1 Oct 2025 | Not on ENTSO-E's list of data available for free re-use: fetched at run time with the user's own token into git-ignored files (`data/spot_entsoe.csv`, `data_ext/spot_entsoe.csv`), never committed or shared |
| **Calendar** (via the starter package) | Local weekday, hour, Finnish public holidays (plus Midsummer Eve), daylight saving time | Built by the organisers from the official calendar |

**Extended history (branch `post-hack-tree`, experiments after the hackathon; not used in the submission).**
Fetched by `build_dataset.py --catalogue extended` into the git-ignored folder `data_ext/`
(12 Jun 2024 onward). All from **Fingrid Open Data**, CC BY 4.0: the starter package's 17 Fingrid
series, and the inputs of Fingrid's imbalance price formula: 349/350 and 353/354 (aFRR activation
volumes, marginal price and local selection), 351/352 (aFRR local-selection prices), 407/408 (avoided
aFRR activation), 411/412 (PICASSO and LFC status), 385/386 and 389/390 (activated mFRR balancing
regulation, SA and DA), 387/388 and 391-394 (mFRR special regulation), 377 (mFRR need), 381 (mFRR net
export), 400-402 (mFRR real-time prices), and the older series 33/34, 252/253 and 118/119 (hourly
ordered regulation, activated regulation power, special regulation). Documents used to implement the
price rules: Fingrid, "Imbalance price in Finland" (29.8.2024 and 1.6.2026) and the mFRR and imbalance
price Q&A (28.8.2025). The Nord Pool file is not used on this branch (see the ENTSO-E day-ahead row above).
The extended history also has the starter package's ENTSO-E series (CC BY 4.0, free re-use list) from
12 Jun 2024.

The data files are git-ignored and are not part of this repository. They come from the organisers'
`starter_package_v1.zip` (checksums in `SHA256SUMS.txt`).

**Tested but not used in the submission:** Open-Meteo Single Runs API (ECMWF IFS and MET Nordic
forecast runs, wind speed and temperature), used only in the weather experiment on branch
`walid-weather`. Licence: CC BY 4.0, "Weather data by Open-Meteo.com"; the free tier is for
non-commercial use. Weather features are off in the submitted pipeline.

## 2. Ready-made model

| Model | Use | Licence |
|---|---|---|
| **TimesFM 2.5 (200M)**, Google Research, Hugging Face `google/timesfm-2.5-200m-pytorch`, snapshot `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | One of the eight ensemble members, used zero-shot (not retrained) | Apache-2.0 (licence stated on the model page). The page notes it is not an officially supported Google product. Paper: "A decoder-only foundation model for time-series forecasting", ICML 2024, arXiv:2310.10688 |

All other models (XGBoost, LightGBM, CatBoost, quantile regression forest, SARIMAX, the lookup
table and the baseline) are trained by us on the starter package data only.

## 3. Organiser-provided files

`score.py`, `validate_submission.py`, `make_templates.py`, `build_dataset.py`, `baseline.ipynb`,
`forecast_origins.csv`, `sample_submission.csv` and `availability.csv` come unchanged from the
starter package (verified against `SHA256SUMS.txt`). They are used as provided for the challenge;
no separate licence is stated for them. On branch `post-hack-tree`, `build_dataset.py` and
`availability.csv` are extended (longer window, more Fingrid datasets, rules for them); the package
build with the default options is unchanged.

## 4. Software components

Python 3.14 (Python Software Foundation License). Main libraries, with the versions used:

| Package | Version | Licence | Used for |
|---|---|---|---|
| pandas | 3.1.1 | BSD-3-Clause | data handling |
| numpy | 2.5.4 | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 parts) | numerics |
| pyarrow | 26.0.1 | Apache-2.0 | reading `inputs.parquet` |
| scipy | 1.18.1 | BSD | optimisation of blend weights, statistics |
| scikit-learn | 1.9.1 | BSD-3-Clause | isotonic regression, utilities |
| statsmodels | 0.15.0 | BSD-3-Clause | SARIMAX model |
| xgboost | 3.4.1 | Apache-2.0 | gradient-boosted trees |
| lightgbm | 4.7.0 | MIT | gradient-boosted trees |
| catboost | 1.2.10 | Apache-2.0 | gradient-boosted trees |
| quantile-forest | 1.4.2 | Apache-2.0 | quantile regression forest |
| optuna | 5.0.0 | MIT | hyperparameter tuning |
| torch (CPU build) | 2.15.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib (as declared in the package metadata) | running TimesFM |
| timesfm | 3.0.2 | Apache-2.0 | TimesFM model code |
| jax, jaxlib | 0.12.0.1 | Apache-2.0 | TimesFM covariate regression |
| huggingface_hub, safetensors | 2.4.0, 0.8.0 | Apache-2.0 | downloading and loading the TimesFM weights |
| matplotlib | 3.11.2 | matplotlib licence (PSF-based) | figures |
| pytest | 9.2.0 | MIT | rule and metric tests |
| notebook, nbconvert | 7.6.5, 7.17.2 | BSD-3-Clause | running the organisers' baseline notebook |
| requests | 2.34.2 | Apache-2.0 | listed in the starter package `requirements.txt`, needed only by the organisers' `build_dataset.py` (not run by us) |
| tzdata | 2026.6 | Apache-2.0 | time zones |

The complete pinned list (including indirect dependencies) is `requirements-lock.txt`.

## 5. External services

| Service | Use |
|---|---|
| Hugging Face Hub | One-time download of the TimesFM weights (pinned snapshot) |
| PyPI and the PyTorch package index | Installing the packages above |
| GitHub (private repository) | Source code hosting; no data files |
| Open-Meteo | Weather experiment only (not in the submission) |
