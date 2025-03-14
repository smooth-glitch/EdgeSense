import time
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_css(file_name):
    """Load external CSS file."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Default styles will be used.")


# Load the CSS file
load_css("styles/styles.css")

# Load the model and feature names with error handling
@st.cache_resource
def load_model_and_features():
    """Load the model and feature names."""
    try:
        model = joblib.load("models/predictive_maintenance_model.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Error loading model or feature names: {e}")
        st.stop()


# Load model and features
model, feature_names = load_model_and_features()

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
st.write("Enter the sensor values below. Disable sensors if you want the model to ignore them.")

input_data = {}
disabled_sensors = set()  # Track disabled sensors
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    with col1 if i % 2 == 0 else col2:
        disable_feature = st.checkbox(f"Disable {feature}", key=f"disable_{feature}_{i}")
        if disable_feature:
            disabled_sensors.add(feature)
            input_data[feature] = None  # Placeholder value for disabled sensors
        else:
            input_data[feature] = st.number_input(
                label=f"{feature}",
                value=0.0,
                format="%.6f",
                key=f"{feature}_{i}",
                help=f"Enter the value for {feature}"
            )

# Predict button
if st.button("Predict", key="predict_button"):
    with st.spinner("Predicting..."):
        time.sleep(2)  # Simulate prediction time

        # Convert input data to DataFrame and ensure correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Handle disabled sensors by replacing None with the mean (or zero as fallback)
        for feature in disabled_sensors:
            input_df[feature] = 0.0  # Replace with a neutral/default value

        # Ensure all values are in float32
        input_df = input_df.astype('float32')

        # Convert to NumPy array for prediction
        input_array = input_df.values.astype('float32')

        # Make predictions
        try:
            prediction = model.predict(input_array)
            prediction_proba = model.predict_proba(input_array) if hasattr(model, "predict_proba") else None
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Display prediction result
        st.success(f"### Prediction: **{'Failure' if prediction[0] == 1 else 'No Failure'}**")
        if prediction_proba is not None:
            st.write(f"**Confidence:** {max(prediction_proba[0]):.2%}")
        else:
            st.write("**Confidence:** Not available")

        # Display feature importance
        st.write("### Feature Importance")
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            # Highlight disabled sensors in gray
            importance_df["Color"] = importance_df["Feature"].apply(
                lambda x: "gray" if x in disabled_sensors else "viridis"
            )

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            sns.set_style("darkgrid")
            sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax, hue="Color", dodge=False, legend=False)
            ax.set_title("Feature Importance", fontsize=16, fontweight="bold")
            ax.set_xlabel("Importance", fontsize=14)
            ax.set_ylabel("Feature", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(axis="x", linestyle="--", alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.write("Feature importance is not available for this model.")

        # Additional visualizations
        if prediction_proba is not None:
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
