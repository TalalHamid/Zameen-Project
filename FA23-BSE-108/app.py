"""
Streamlit GUI for Zameen house price prediction (Lahore).
Run: streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from predict import format_pkr, load_model

DATA_PATH = Path("zameen_lahore_clean.csv")
MODEL_NAME_PATH = Path("artifacts/best_model_name.txt")
METRICS_PATH = Path("artifacts/model_comparison.csv")

PROPERTY_TYPES = ["House", "Flat", "Plot", "Farm House", "Penthouse"]


@st.cache_data
def load_location_options() -> list[str]:
    if DATA_PATH.exists():
        locs = pd.read_csv(DATA_PATH)["location"].dropna().astype(str).unique().tolist()
        return sorted(locs)
    return ["Lahore"]


@st.cache_resource
def get_model():
    return load_model()


def get_best_model_name() -> str:
    if MODEL_NAME_PATH.exists():
        return MODEL_NAME_PATH.read_text(encoding="utf-8").strip()
    return "Trained model"


def main():
    st.set_page_config(
        page_title="Zameen Price Predictor",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 House Price Prediction System")
    st.caption("Lahore property listings · Machine Learning · AIC354 Lab Project")

    with st.sidebar:
        st.header("About")
        st.write(
            "Enter property details to get an estimated price in PKR. "
            "The model was trained on scraped Zameen.com listings."
        )
        try:
            st.success(f"**Model:** {get_best_model_name()}")
            get_model()
            st.info("Model loaded successfully.")
        except FileNotFoundError:
            st.error("Model not found. Run `python train_ml.py` first.")
            st.stop()

        if METRICS_PATH.exists():
            st.subheader("Model metrics")
            metrics = pd.read_csv(METRICS_PATH)
            best = metrics.iloc[0]
            st.metric("R² Score", f"{best['R2']:.4f}")
            st.metric("RMSE", f"PKR {best['RMSE']:,.0f}")

        st.divider()
        st.markdown("**Run locally**")
        st.code("streamlit run app.py", language="bash")

    locations = load_location_options()

    col1, col2 = st.columns(2)

    with col1:
        area_sqft = st.number_input(
            "Area (sq ft)",
            min_value=100.0,
            max_value=500000.0,
            value=2250.0,
            step=50.0,
            help="Total covered or plot area in square feet.",
        )
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=20,
            value=6,
            step=1,
        )

    with col2:
        property_type = st.selectbox(
            "Property type",
            options=PROPERTY_TYPES,
            index=0,
        )

        use_custom_location = st.checkbox("Enter custom location", value=False)

        if use_custom_location:
            location = st.text_input(
                "Location",
                value="DHA Phase 5, DHA Defence",
                placeholder="e.g. DHA Phase 5, DHA Defence",
            )
        else:
            location = st.selectbox(
                "Location",
                options=locations,
                index=0,
                help="Choose a neighbourhood from the training dataset.",
            )

        city = st.text_input("City", value="Lahore", disabled=True)

    st.divider()

    if st.button("Predict Price", type="primary", use_container_width=True):
        if not location or not str(location).strip():
            st.warning("Please enter a location.")
            return

        try:
            with st.spinner("Calculating estimate..."):
                model = get_model()
                row = pd.DataFrame(
                    [
                        {
                            "area_sqft": area_sqft,
                            "bedrooms": int(bedrooms),
                            "bathrooms": int(bathrooms),
                            "city": city,
                            "location": str(location).strip(),
                            "property_type": property_type,
                        }
                    ]
                )
                price = float(model.predict(row)[0])

            st.success("Estimated house price")
            st.markdown(
                f"<h1 style='text-align:center;color:#1f77b4;'>{format_pkr(price)}</h1>",
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Area", f"{area_sqft:,.0f} sq ft")
            c2.metric("Bedrooms", bedrooms)
            c3.metric("Bathrooms", bathrooms)
            c4.metric("Type", property_type)

            st.caption(f"Location: {location}")

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
