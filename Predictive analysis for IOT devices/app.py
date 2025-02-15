import time
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_css(file_name):
    """Load external CSS file."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load the CSS file
load_css("styles.css")

# Load the model and feature names
model = joblib.load("predictive_maintenance_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title and description
st.title("EDGE SENSE")
st.write(
    "This app predicts engine failure based on sensor data using NASA's "
    "turbofan engine degradation simulation dataset. Enter the values below and "
    "click **Predict**."
)

# Sidebar with model info
with st.sidebar:
    st.write("**Model Info**")
    st.write("Model: XGBoost")
    st.write(f"Features: {len(feature_names)}")
    st.markdown("---")

# Input fields for all features
st.header("Input Sensor Data")
st.write("Enter the sensor values below:")

input_data = {}
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    with col1 if i % 2 == 0 else col2:
        input_data[feature] = st.number_input(
            label=f"{feature}",
            value=0.0,
            format="%.6f",
            key=feature,
            help=f"Enter the value for {feature}"
        )

# Predict button
if st.button("Predict", key="predict_button"):
    with st.spinner("Predicting..."):
        time.sleep(2)  # Simulate prediction time

        # Convert input data to DataFrame and ensure correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Make predictions
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display prediction result
        st.success(f"### Prediction: **{'Failure' if prediction[0] == 1 else 'No Failure'}**")
        st.write(f"**Confidence:** {max(prediction_proba[0]):.2%}")

        # Display feature importance
        st.write("### Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
        sns.set_style("darkgrid")
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, palette="viridis")
        ax.set_title("Feature Importance", fontsize=16, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=14)
        ax.set_ylabel("Feature", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig)

        # Additional visualizations
        st.write("### Confidence Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(prediction_proba[0], bins=10, kde=True, ax=ax, color="dodgerblue")
        ax.set_title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Confidence Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        st.pyplot(fig)

# Footer with GitHub repo link
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; background-color: black; color: white; padding: 10px;">
        Developed by <strong>Arjun Sridhar</strong> | 
        <a href="https://github.com/smooth-glitch/smooth-glitch" target="_blank" style="color: white;">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)