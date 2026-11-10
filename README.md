# Classical ML Dataset Ecosystem

**Two fictional, reproducible datasets for classical (non-neural) machine learning education.** Not clinical or commercial evidence.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/generate_datasets.py
python src/inject_defects.py
python src/validate_datasets.py
python -m pytest -q
python src/model_examples.py
python src/clean_data.py      # instructor reference solution for the cleaning exercises
jupyter notebook
```

Launch from the repository root and open `notebooks/01_python_numpy_foundations.ipynb` first. The notebooks locate project files relative to the current working directory, whether run from the root or notebooks folder.

> **Never open and re-save the raw CSVs in Excel or Google Sheets.** Spreadsheet apps silently rewrite dates (`2026-01-05` becomes `1/5/2026`, and a deliberate day-first defect such as `05/01/2026` becomes a *wrong* date), drop trailing zeros and change line endings. That destroys the intended exercises. If it happens, run `python src/inject_defects.py`; `pytest` detects the damage (`test_raw_files_match_injector`).

## Contents and scope

- `data/clean`: `healthcare_clean.csv`, `retail_clean.csv` (1,000 rows each), the answer key
- `data/raw`: intentionally imperfect versions (1,000 physical rows each), what learners clean
- `data/metadata`: per-feature dictionaries, generator config and **instructor-only** defect manifest
- `src`: deterministic generators, defect injection, validation, leakage-safe model examples and a reference cleaning solution
- `notebooks`: nine short, executable reference notebooks
- `tests`: integrity, traceability, deterministic regeneration, raw-file integrity, defect presence, cleaning and split checks
- `docs`: dataset design, twenty-stage roadmap, instructor guide, limitations

There are 2,000 independent underlying observations, not 4,000. Healthcare unit = one fictional patient assessment; target = fictional binary `cancer_status`. Retail unit = one product-store-day; target = realized inventory-capped `units_sold`. Do not merge the tasks. The clean/raw pairs correspond to the same observations, with two deliberate duplicate replacements per raw dataset fully traceable in the manifest.

Healthcare uses stratified 700/150/150 train/validation/test. Retail calendar uses chronological 700/150/150 rows; after removing ten initial rows with unavailable lag1, the reference forecasting model uses 690/150/150. Both models fit preprocessing only on training. Holdout test data are reserved and not used for model selection. Reference benchmark outputs are validation metrics only; see `src/model_examples.py`.

## The defects are the lesson

The raw files keep every defect on purpose. Learners find and repair them during data preparation:

| Defect | Healthcare | Retail | Typical repair |
|---|---|---|---|
| Messy category text (`" NEVER "`) | `smoking_status` (12) | `category` (12) | strip + lower case |
| Invalid category (`unknown_typo`) | `smoking_status` (5) | `category` (5) | validate against dictionary → NaN |
| Sentinel value (`-999`) | `age` (7) | `inventory_available` (7) | dictionary range → NaN |
| Text in a numeric column (`not recorded`) | `bmi` (8) | `advertising_spend` (8) | `pd.to_numeric(errors='coerce')` |
| Corrupted outlier (`9999`) | `bmi` (5) | `advertising_spend` (5) | dictionary range → NaN |
| Impossible value | `symptom_count = -3` (5) | `return_count > units_sold` (5) | logical rule → NaN |
| Mixed date formats | — | 20 day-first dates | parse ISO, then `%d/%m/%Y` |
| MCAR / MAR / MNAR missingness | `bmi` / `biomarker_level` (age > 60) / `inflammation_marker` (high values) | `advertising_spend` / `customer_rating` (promotion days) / `advertising_spend` (high values) | diagnose the mechanism; impute later inside a Pipeline |
| Exact duplicate rows | 2 | 2 | `drop_duplicates()` |
| ID collisions | 4 | 4 | keep first, re-key the rest |

`src/clean_data.py` is the instructor reference solution. It uses only the raw file and the data dictionary, never imputes, logs every repair and then grades the result against the clean file. Current result for both datasets: **0 wrong values** outside the ID column; the remaining NaN cells are left for leakage-safe imputation after the split.

## Reference results (validation set)

| Task | Model | Metric |
|---|---|---|
| Healthcare classification | Dummy / Logistic / Random forest | ROC AUC 0.500 / 0.901 / 0.905 |
| Retail, scenario A (history only: product, calendar, lag1, rolling7) | Dummy / Ridge / Forest | MAE 10.55 / 10.26 / 11.31 |
| Retail, scenario B (same-day plan: + opening inventory, price, discount, advertising, promotion) | Ridge / Forest | MAE 6.47 / 4.04 |
| Retail, scenario C, **invalid on purpose** (+ `net_sales_revenue`) | Forest | MAE 1.15 (post-outcome leakage) |

Scenario A shows that yesterday's sales alone barely beat the mean, because units sold are capped by stock on most days. Scenario B assumes the day's plan, including opening inventory, is fixed before the store opens. Scenario C exists only to demonstrate leakage: revenue is computed from units sold after the day ends.

## Learner path

Begin with clean healthcare, NumPy and pandas, then inspect raw data and use the private defect manifest only after completing repairs. Study descriptive statistics and visualizations, build leakage-safe pipelines, compare dummy versus learned models, explore clustering and finish with a stakeholder narrative. Retail exercises introduce chronological splits and product-wise historical features. See `docs/instructor_guide.md` for inspection, types, missingness, duplicates, imputation, encoding, scaling, model comparison and presentation guidance, and `docs/curriculum_mapping.md` for the full 20-stage progression.

## Key caveats

This is fictional synthetic data. The healthcare label is not a real diagnosis, and its model must never be used for patient decisions. Retail observations cover 100 days and cannot establish annual seasonality or validated demand forecasting. Future sales, returns and revenue are not legitimate presales predictors. Defects include MCAR, MAR and MNAR missingness; MNAR is not reliably repaired by generic imputation. See `docs/limitations_and_ethics.md`.
