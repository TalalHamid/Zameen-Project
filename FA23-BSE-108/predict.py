"""
Interactive house price prediction using trained best model.
User inputs: area, bedrooms, bathrooms, location (city/property_type optional defaults).
"""

import pickle
from pathlib import Path

import pandas as pd


DEFAULT_CITY = "Lahore"
DEFAULT_PROPERTY_TYPE = "House"
MODEL_PATH = Path("artifacts/best_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python train_ml.py"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_price(
    area_sqft: float,
    bedrooms: int,
    bathrooms: int,
    location: str,
    city: str = DEFAULT_CITY,
    property_type: str = DEFAULT_PROPERTY_TYPE,
) -> float:
    model = load_model()
    row = pd.DataFrame(
        [
            {
                "area_sqft": area_sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "city": city,
                "location": location,
                "property_type": property_type,
            }
        ]
    )
    price = model.predict(row)[0]
    return float(price)


def format_pkr(amount: float) -> str:
    if amount >= 1e7:
        return f"PKR {amount/1e7:.2f} Crore ({amount:,.0f})"
    if amount >= 1e5:
        return f"PKR {amount/1e5:.2f} Lakh ({amount:,.0f})"
    return f"PKR {amount:,.0f}"


def main():
    print("=" * 50)
    print("  Zameen House Price Prediction (Lahore)")
    print("=" * 50)
    try:
        area = float(input("Area (sq ft): ").strip())
        beds = int(input("Bedrooms: ").strip())
        baths = int(input("Bathrooms: ").strip())
        location = input("Location (e.g. DHA Phase 5, Lahore): ").strip() or "Lahore"
        ptype = input(f"Property type [{DEFAULT_PROPERTY_TYPE}]: ").strip() or DEFAULT_PROPERTY_TYPE

        price = predict_price(area, beds, baths, location, property_type=ptype)
        print("\n--- Estimated Price ---")
        print(format_pkr(price))
    except FileNotFoundError as e:
        print(e)
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Invalid input or cancelled: {e}")


if __name__ == "__main__":
    main()
