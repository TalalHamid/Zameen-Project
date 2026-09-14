"""
Data cleaning and preprocessing for Zameen house price prediction.
Uses only stable features: price, area_sqft, bedrooms, bathrooms, city, location, property_type
"""

import pandas as pd
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = [
    "price",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "city",
    "location",
    "property_type",
]


def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def clean_dataset(df: pd.DataFrame, min_rows: int = 300) -> pd.DataFrame:
    """Clean raw scraped data. Returns dataframe with at least min_rows if possible."""
    # Keep only required columns (drop extras if present)
    available = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing_cols = set(REQUIRED_COLUMNS) - set(available)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()

    # Numeric coercion
    for col in ["price", "area_sqft", "bedrooms", "bathrooms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # String cleanup
    for col in ["city", "location", "property_type"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["", "nan", "None", "unknown"], np.nan)

    # Normalize property type labels
    df["property_type"] = df["property_type"].str.title().str.strip()
    df["property_type"] = df["property_type"].replace(
        {"Unknown": np.nan, "Apartment": "Flat", "Penthouse": "Flat"}
    )

    # Plots often have no bed/bath on Zameen — use 0 instead of dropping
    is_plot = df["property_type"].fillna("").str.lower().str.contains("plot")
    df.loc[is_plot, "bedrooms"] = df.loc[is_plot, "bedrooms"].fillna(0)
    df.loc[is_plot, "bathrooms"] = df.loc[is_plot, "bathrooms"].fillna(0)
    if df["property_type"].isna().any():
        df["property_type"] = df["property_type"].fillna("House")

    # Drop rows missing target or core numeric features
    df = df.dropna(subset=["price", "area_sqft", "bedrooms", "bathrooms"])
    df = df[df["price"] > 0]
    df = df[df["area_sqft"] > 0]
    df = df[df["bedrooms"] >= 0]
    df = df[df["bathrooms"] >= 0]

    # Fill categorical missing with mode or default
    if df["city"].isna().any():
        df["city"] = df["city"].fillna("Lahore")
    if df["location"].isna().any():
        df["location"] = df["location"].fillna("Lahore")
    if df["property_type"].isna().any():
        df["property_type"] = df["property_type"].fillna(
            df["property_type"].mode().iloc[0] if len(df["property_type"].mode()) else "House"
        )

    # Remove duplicates
    df = df.drop_duplicates()

    # Outlier cap (optional but helps training)
    price_q99 = df["price"].quantile(0.99)
    area_q99 = df["area_sqft"].quantile(0.99)
    df = df[df["price"] <= price_q99]
    df = df[df["area_sqft"] <= area_q99]

    if len(df) < min_rows:
        print(
            f"[WARN] Only {len(df)} rows after cleaning (target >= {min_rows}). "
            "Scrape more pages with scraper_zameen.py --pages 50"
        )

    return df.reset_index(drop=True)


def save_cleaned(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved {len(df)} cleaned rows to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="zameen_lahore_raw.csv")
    parser.add_argument("--output", default="zameen_lahore_clean.csv")
    parser.add_argument("--min-rows", type=int, default=300)
    args = parser.parse_args()

    df = load_raw_csv(args.input)
    print(f"[INFO] Loaded {len(df)} raw rows")
    df_clean = clean_dataset(df, min_rows=args.min_rows)
    save_cleaned(df_clean, args.output)
    print(df_clean.describe(include="all"))


if __name__ == "__main__":
    main()
