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
jupyter notebook
```

Launch from the repository root and open `notebooks/01_python_numpy_foundations.ipynb` first. The notebooks locate project files relative to the current working directory, whether run from the root or notebooks folder.

## Contents and scope

- `data/clean`: `healthcare_clean.csv`, `retail_clean.csv` (1,000 rows each)
- `data/raw`: intentionally imperfect versions (1,000 physical rows each)
- `data/metadata`: per-feature dictionaries, generator config and **instructor-only** defect manifest
- `src`: deterministic generators, defect injection, validation and leakage-safe model examples
- `notebooks`: nine short, executable reference notebooks
- `tests`: integrity, traceability, deterministic regeneration and split checks
- `docs`: dataset design, twenty-stage roadmap, instructor guide, limitations

There are 2,000 independent underlying observations, not 4,000. Healthcare unit = one fictional patient assessment; target = fictional binary `cancer_status`. Retail unit = one product-store-day; target = realized inventory-capped `units_sold`. Do not merge the tasks. The clean/raw pairs correspond to the same observations, with two deliberate duplicate replacements per raw dataset fully traceable in the manifest.

Healthcare uses stratified 700/150/150 train/validation/test. Retail calendar uses chronological 700/150/150 rows; after removing ten initial rows with unavailable lag1, the reference forecasting model uses 690/150/150. Both models fit preprocessing only on training. Holdout test data are reserved and not used for model selection. Reference benchmark outputs are validation metrics only; see `src/model_examples.py`.

## Learner path

Begin with clean healthcare, NumPy and pandas, then inspect raw data and use the private defect manifest only after completing repairs. Study descriptive statistics and visualizations, build leakage-safe pipelines, compare dummy versus learned models, explore clustering and finish with a stakeholder narrative. Retail exercises introduce chronological splits and product-wise historical features. See `docs/instructor_guide.md` for inspection, types, missingness, duplicates, imputation, encoding, scaling, model comparison and presentation guidance, and `docs/curriculum_mapping.md` for the full 20-stage progression.

## Key caveats

This is fictional synthetic data. The healthcare label is not a real diagnosis, and its model must never be used for patient decisions. Retail observations cover 100 days and cannot establish annual seasonality or validated demand forecasting. Future sales, returns and revenue are not legitimate presales predictors. Defects include MCAR, MAR and MNAR missingness; MNAR is not reliably repaired by generic imputation. See `docs/limitations_and_ethics.md`.
