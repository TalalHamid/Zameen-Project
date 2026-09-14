"""
Train and evaluate regression models for house price prediction.
Models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from preprocess import clean_dataset, load_raw_csv

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


FEATURE_COLS = ["area_sqft", "bedrooms", "bathrooms", "city", "location", "property_type"]
TARGET_COL = "price"
CATEGORICAL = ["city", "location", "property_type"]
NUMERIC = ["area_sqft", "bedrooms", "bathrooms"]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": rmse(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }


def build_preprocessor():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("cat", ohe, CATEGORICAL),
        ]
    )


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=12),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_state=42,
        )
    return models


def train_all(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()
    results = []
    best_name = None
    best_model = None
    best_r2 = -np.inf

    for name, reg in get_models().items():
        pipe = Pipeline([("prep", preprocessor), ("model", reg)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(name, pipe, X_test, y_test)
        results.append(metrics)
        print(
            f"{name}: MAE={metrics['MAE']:,.0f} | RMSE={metrics['RMSE']:,.0f} | R2={metrics['R2']:.4f}"
        )
        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_name = name
            best_model = pipe

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    return results_df, best_name, best_model, X_test, y_test


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="zameen_lahore_clean.csv")
    parser.add_argument("--raw", default=None, help="If set, clean from raw CSV first")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.raw:
        df_raw = load_raw_csv(args.raw)
        df = clean_dataset(df_raw)
        df.to_csv(args.data, index=False)
    else:
        df = pd.read_csv(args.data)

    if len(df) < 50:
        raise SystemExit(
            f"Need more data: only {len(df)} rows. Run scraper and preprocess first."
        )

    print(f"[INFO] Training on {len(df)} rows")
    results_df, best_name, best_model, _, _ = train_all(df)

    results_path = out_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Results saved to {results_path}")
    print(f"\n[INFO] Best model: {best_name}")

    with open(out_dir / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(out_dir / "best_model_name.txt", "w") as f:
        f.write(best_name)

    summary = {
        "best_model": best_name,
        "n_samples": len(df),
        "metrics": results_df.to_dict(orient="records"),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Saved best_model.pkl for predict.py")


if __name__ == "__main__":
    main()
