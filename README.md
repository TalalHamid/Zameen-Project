# Zameen House Price Prediction (AIC354 Lab Project)

End-to-end ML project: scrape Lahore listings from Zameen.com → clean → train 6 regressors → predict price.

## Features used (stable fields only)

| Column | Description |
|--------|-------------|
| `price` | Target (PKR) |
| `area_sqft` | Area in square feet |
| `bedrooms` | Count |
| `bathrooms` | Count |
| `city` | Lahore |
| `location` | Area/neighbourhood text |
| `property_type` | House, Flat, Plot, etc. |

Optional amenity fields (built year, kitchens, etc.) are **not** scraped to avoid heavy missing data.

## Setup

```bash
cd "d:\University\6th Sem\ML\Zameen Project"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Scrape data

Scrape enough pages so that after cleaning you still have **≥300 rows** (aim for 400–500 raw rows).

```bash
python scraper_zameen.py --pages 45 --output zameen_lahore_raw.csv
```

## 2. Clean data

```bash
python preprocess.py --input zameen_lahore_raw.csv --output zameen_lahore_clean.csv --min-rows 300
```

## 3. Train models

```bash
python train_ml.py --data zameen_lahore_clean.csv
```

Outputs in `artifacts/`:

- `model_comparison.csv` — MAE, MSE, RMSE, R² for all models
- `best_model.pkl` — best pipeline for prediction
- `training_summary.json`

Or clean + train in one step:

```bash
python train_ml.py --raw zameen_lahore_raw.csv --data zameen_lahore_clean.csv
```

## 4. Predict

**CLI:**

```bash
python predict.py
```

**Streamlit GUI:**

```bash
streamlit run app.py
```

Opens a browser with inputs for area, bedrooms, bathrooms, location, and property type.

## Deliverables checklist

1. `scraper_zameen.py`
2. `zameen_lahore_clean.csv` (and optionally raw CSV)
3. `train_ml.py`, `preprocess.py`, `predict.py`
4. Report: problem statement, dataset stats, algorithms, metrics table, challenges

## Notes

- If Zameen HTML changes, update selectors in `scraper_zameen.py`.
- Respect site terms; use delays (`--delay-min` / `--delay-max`).
- Plots for the report: use `artifacts/model_comparison.csv` in Excel or a short notebook.
