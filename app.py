import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load external CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS file
load_css("styles/styles.css")

# Load the model
model = joblib.load('models/predictive_maintenance_model.pkl')

# Load the feature names
feature_names = joblib.load('models/feature_names.pkl')

# JavaScript to toggle theme and update footer color
toggle_theme_js = """
<script>
function toggleTheme() {
    const body = document.body;
    if (body.getAttribute('data-theme') === 'dark') {
        body.setAttribute('data-theme', 'bright');
    } else {
        body.setAttribute('data-theme', 'dark');
    }
    // Update footer color
    const footer = document.querySelector('.footer');
    if (footer) {
        footer.style.backgroundColor = body.getAttribute('data-theme') === 'dark' ? '#2d2d2d' : '#f0f2f6';
    }
}
</script>
"""

# Inject JavaScript into the app
st.components.v1.html(toggle_theme_js)

# Function to toggle theme
def toggle_theme():
    if st.session_state.get("theme") == "dark":
        st.session_state.theme = "bright"
    else:
        st.session_state.theme = "dark"

# Initialize session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "bright"

# Apply theme
st.markdown(f'<body data-theme="{st.session_state.theme}">', unsafe_allow_html=True)

# Title and description
st.title('EDGE SENSE')
st.write("This app predicts engine failure based on sensor data using Nasa's very own turbofan engine degradation simulation dataset. Enter the values below and click **Predict**.")

# Sidebar for additional options
with st.sidebar:
    # Theme toggle
    st.write("**Theme**")
    if st.button("Toggle Theme (Dark/Bright)"):
        toggle_theme()
        st.rerun()  # Force rerun to apply theme changes

    st.markdown("---")
    st.write("**Model Info**")
    st.write(f"Model: XGBoost")
    st.write(f"Features: {len(feature_names)}")
    st.markdown("---")

    # About section link
    if st.button("About This App"):
        st.session_state.page = "about"

# About page
if st.session_state.get("page") == "about":
    st.title("About This App")
    st.write("""
    ### What Does This App Do?
    This app is designed for **predictive maintenance** in IoT devices. It uses machine learning to predict engine failure based on sensor data. By analyzing sensor readings, the app can identify potential failures before they occur, helping you save time and resources.

    ### How Does It Work?
    1. **Input Sensor Data**: Enter the sensor values in the input fields.
    2. **Predict**: Click the **Predict** button to get the engine's health status.
    3. **Results**: The app will display whether the engine is likely to fail and provide a confidence score.

    ### Why Use This App?
    - **Proactive Maintenance**: Identify issues before they become critical.
    - **Cost-Effective**: Reduce downtime and maintenance costs.
    - **User-Friendly**: Simple and intuitive interface.

    ### Color Scheme
    The app uses a **comfortable color scheme** designed to be easy on the eyes. You can switch between **dark** and **bright** themes in the settings.

    ### Developer
    Developed by **Arjun Sridhar**. Check out the [GitHub Repo](https://github.com/smooth-glitch/smooth-glitch) for more details.
    """)
    if st.button("Back to Main App"):
        st.session_state.page = "main"

# Main app page
else:
    # Input fields for all features
    st.header("Input Sensor Data")
    st.write("Enter the sensor values below:")

    input_data = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:
            input_data[feature] = st.number_input(
                f'{feature}', 
                value=0.0, 
                format="%.6f",  # Allow up to 6 decimal places
                key=feature,
                help=f"Enter the value for {feature}"
            )

    # Predict button
    if st.button('Predict', key='predict_button'):
        with st.spinner('Predicting...'):
            time.sleep(2)  # Simulate prediction time

            # Convert input data to a DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure the columns are in the correct order
            input_df = input_df[feature_names]
            
            # Make a prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # Display prediction result
            st.success(f'### Prediction: **{"Failure" if prediction[0] == 1 else "No Failure"}**')
            st.write(f"**Confidence:** {max(prediction_proba[0]):.2%}")

            # Display feature importance
            st.write('### Feature Importance')
            feature_importance = model.named_steps['xgb'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            sns.set_style("darkgrid")
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette="viridis")
            ax.set_title("Feature Importance", fontsize=16, fontweight='bold')
            ax.set_xlabel("Importance", fontsize=14)
            ax.set_ylabel("Feature", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig)

            # Additional Visualizations
            st.write("### Confidence Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(prediction_proba[0], bins=10, kde=True, ax=ax, color="dodgerblue")
            ax.set_title("Prediction Confidence Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("Confidence Score", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            st.pyplot(fig)

            st.write("### Correlation Heatmap of Input Data")
            input_corr = input_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(input_corr, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
            ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
            st.pyplot(fig)

    # Footer with GitHub repo button
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
